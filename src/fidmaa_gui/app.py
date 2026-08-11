import math
import os
import sys
from functools import partial
from textwrap import dedent

import numpy as np
from PIL import ImageFile
from portrait_analyser.depth_sampling import (
    measure_filtered_surface_length,
    median_filter_depthmap,
    sample_points_along_line,
)
from portrait_analyser.exceptions import (
    ExifValidationFailed,
    NoDepthMapFound,
    UnknownExtension,
)
from portrait_analyser.incisor import (
    depth_raw_to_distance_cm,
    pixel_to_mm,
    pixels_per_mm_at_distance,
    vector_length_3d,
)
from portrait_analyser.ios import IOSPortrait, load_image
from portrait_analyser.mouth import compute_mouth_measurement_from_facemesh
from portrait_analyser.neck import compute_neck_circumference
from portrait_analyser.pose import PortraitPose, detect_neck_midpoint
from portrait_analyser.tmd import compute_tmd_3d
from PySide6 import QtGui
from PySide6.QtCore import QObject, QPoint, QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import const, errors
from .calculations import findPoint
from .depth_visualization import (
    DepthDisplayMode,
    render_depth_contour_overlay,
    render_depth_visualization,
)
from .region_measurement import (
    MeasurementError,
    Region,
    RegionMask,
    RegionMeasurementEngine,
    SelectionMode,
)
from .utils import (
    UILoaderMixin,
    clamp,
    interpolate_pixels_along_line,
    translate_coordinates_to_other_image,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

tr = QObject.tr


def _show_3d_view(image, depthmap, float_min, float_max):
    """Show 3D view in a separate process to avoid VTK/Qt conflicts on macOS."""
    from fidmaa_simple_viewer.core import pyvista_show

    pyvista_show(image, depthmap, float_min, float_max)


class MainWindow(UILoaderMixin, QMainWindow):
    uifile_name = "form.ui"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.region_a = None
        self.region_b = None
        self.region_measurement_result = None
        self._region_target = None
        self._last_region_target = "a"
        self._region_drag_action = None
        self._region_drag_start = None
        self._region_drag_original = None
        self.last_zoom_x = 0
        self.last_zoom_y = 0
        self.load_ui()
        self._create_zoom_panel()
        self._create_region_measurement_panel()
        self._create_tools_menu()

        self.filename = None

        self.smallImage = None
        self.portrait: IOSPortrait = None
        self.depthmap = None
        self.filtered_depthmap = None
        self.teethmap = None
        self.neck_measurement_auto = None

        self.float_max_value = self.float_min_value = None

        self.last_click_x = None
        self.last_click_y = None
        self.last_angle = None
        self.last_depth = None
        self.last_show_centroids = None
        self.last_show_neck_arc = None
        self.last_show_landmarks = None
        self.last_show_segmentation = None
        self.hairless_depth_image = None

        self.last_5_distances_vect = []
        self.last_5_distances_srfc = []

        self.redrawImage()
        self.redrawZoom()

    def _create_region_measurement_panel(self):
        self.regionDock = QDockWidget("Region measurement", self)
        self.regionDock.setObjectName("regionMeasurementDock")
        self.regionDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        self.regionSelectionButton = QPushButton("Show measurement controls")
        self.regionSelectionButton.setCheckable(True)
        self.regionSelectionButton.setToolTip(
            "Click to place each fixed-radius patch, or drag an existing patch to move it."
        )
        layout.addWidget(self.regionSelectionButton)

        self.regionModeACombo, self.regionMaskACombo = self._create_region_options(
            layout, "Region A", SelectionMode.LOCAL_PEAK
        )
        self.regionModeBCombo, self.regionMaskBCombo = self._create_region_options(
            layout, "Region B", SelectionMode.LOCAL_VALLEY
        )

        settings_group = QGroupBox("Sampling")
        settings_form = QFormLayout(settings_group)
        self.regionRadiusSpin = QSpinBox()
        self.regionRadiusSpin.setRange(10, 120)
        self.regionRadiusSpin.setValue(40)
        self.regionRadiusSpin.setSuffix(" px")
        self.regionRadiusSpin.setToolTip(
            "Shared radius of both click-to-place measurement patches."
        )
        settings_form.addRow("Patch radius:", self.regionRadiusSpin)
        self.regionPercentileSpin = QSpinBox()
        self.regionPercentileSpin.setRange(5, 20)
        self.regionPercentileSpin.setValue(10)
        self.regionPercentileSpin.setSuffix(" %")
        settings_form.addRow("Candidate pool:", self.regionPercentileSpin)
        self.regionVectorCountSpin = QSpinBox()
        self.regionVectorCountSpin.setRange(5, 10)
        self.regionVectorCountSpin.setValue(10)
        settings_form.addRow("Profiles:", self.regionVectorCountSpin)
        layout.addWidget(settings_group)

        depth_group = QGroupBox("Depth display")
        depth_form = QFormLayout(depth_group)
        self.depthDisplayCombo = QComboBox()
        for mode in DepthDisplayMode:
            self.depthDisplayCombo.addItem(mode.value, mode)
        self.depthDisplayCombo.setCurrentIndex(
            self.depthDisplayCombo.findData(DepthDisplayMode.COLOR)
        )
        self.depthDisplayCombo.setToolTip(
            "Display-only enhancement; measurement data is never remapped."
        )
        depth_form.addRow("Mode:", self.depthDisplayCombo)
        self.depthContourStepSpin = QSpinBox()
        self.depthContourStepSpin.setRange(1, 8)
        self.depthContourStepSpin.setValue(3)
        self.depthContourStepSpin.setSuffix(" levels")
        self.depthContourStepSpin.setToolTip(
            "Draw one contour for every N raw depth levels. Measurements are unchanged."
        )
        self.depthContourStepSpin.setEnabled(False)
        depth_form.addRow("Contour step:", self.depthContourStepSpin)
        layout.addWidget(depth_group)

        clear_button = QPushButton("Clear regions")
        clear_button.clicked.connect(self.clear_region_measurement)
        layout.addWidget(clear_button)

        self.regionResultEdit = QPlainTextEdit()
        self.regionResultEdit.setReadOnly(True)
        self.regionResultEdit.setMinimumWidth(270)
        self.regionResultEdit.setPlainText(
            "Enable measurement controls, then click once to place A and once to place B."
        )
        layout.addWidget(self.regionResultEdit, stretch=1)

        self.regionDock.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.regionDock)

        self.regionSelectionButton.toggled.connect(self._set_region_selection_enabled)
        for widget in (
            self.regionModeACombo,
            self.regionMaskACombo,
            self.regionModeBCombo,
            self.regionMaskBCombo,
        ):
            widget.currentIndexChanged.connect(self._region_settings_changed)
        self.regionPercentileSpin.valueChanged.connect(self._region_settings_changed)
        self.regionVectorCountSpin.valueChanged.connect(self._region_settings_changed)
        self.regionRadiusSpin.valueChanged.connect(self._region_radius_changed)
        self.depthDisplayCombo.currentIndexChanged.connect(self._depth_display_changed)
        self.depthContourStepSpin.valueChanged.connect(self._depth_display_changed)

    def _create_region_options(self, parent_layout, title, default_mode):
        group = QGroupBox(title)
        form = QFormLayout(group)
        mode_combo = QComboBox()
        for mode in SelectionMode:
            mode_combo.addItem(mode.value, mode)
        mode_tooltips = {
            SelectionMode.HIGHEST: (
                "Select the globally nearest surface points in the entire patch."
            ),
            SelectionMode.LOWEST: (
                "Select the globally farthest surface points in the entire patch."
            ),
            SelectionMode.LOCAL_PEAK: (
                "Find a local anatomical bump after removing camera-facing tilt."
            ),
            SelectionMode.LOCAL_VALLEY: (
                "Find a local anatomical depression after removing camera-facing tilt."
            ),
            SelectionMode.FLATTEST: (
                "Reject extreme depths, then select the smallest local 3D plane error."
            ),
        }
        for index, mode in enumerate(SelectionMode):
            mode_combo.setItemData(index, mode_tooltips[mode], Qt.ToolTipRole)
        mode_combo.setCurrentIndex(mode_combo.findData(default_mode))
        mask_combo = QComboBox()
        for region_mask in RegionMask:
            mask_combo.addItem(region_mask.value, region_mask)
        mask_combo.setToolTip("Optionally restrict candidate pixels to the detected teeth mask.")
        form.addRow("Selector:", mode_combo)
        form.addRow("Mask:", mask_combo)
        parent_layout.addWidget(group)
        return mode_combo, mask_combo

    def _create_tools_menu(self):
        tools_menu = self.menuBar().addMenu("&Tools")
        self.pixelToolAction = tools_menu.addAction("Pixel tool")
        self.pixelToolAction.setShortcut(QtGui.QKeySequence(Qt.Key_P))
        self.pixelToolAction.setShortcutContext(Qt.WindowShortcut)
        self.pixelToolAction.setStatusTip(
            "Hide measurement controls and return to the original single-pixel tool"
        )
        self.pixelToolAction.triggered.connect(self._activate_pixel_tool)

        self.measurementControlsAction = tools_menu.addAction("Show measurement controls")
        self.measurementControlsAction.setCheckable(True)
        self.measurementControlsAction.setStatusTip(
            "Show and directly manipulate the two measurement circles"
        )
        self.measurementControlsAction.toggled.connect(self.regionSelectionButton.setChecked)
        self.regionSelectionButton.toggled.connect(self.measurementControlsAction.setChecked)

        tools_menu.addSeparator()
        action_details = (
            (SelectionMode.HIGHEST, Qt.Key_H, "Highest region tool"),
            (SelectionMode.LOWEST, Qt.Key_L, "Lowest region tool"),
            (SelectionMode.FLATTEST, Qt.Key_F, "Flattest region tool"),
            (SelectionMode.LOCAL_PEAK, None, "Local peak region tool"),
            (SelectionMode.LOCAL_VALLEY, None, "Local valley region tool"),
        )
        self.regionToolActions = {}
        for mode, key, text in action_details:
            action = tools_menu.addAction(text)
            if key is not None:
                action.setShortcut(QtGui.QKeySequence(key))
                action.setShortcutContext(Qt.WindowShortcut)
            action.setStatusTip(f"Use {mode.value.lower()} candidates for the active region")
            action.triggered.connect(partial(self._activate_region_selector, mode))
            self.regionToolActions[mode] = action

    def _activate_pixel_tool(self, *args):
        self.regionSelectionButton.setChecked(False)

    def _activate_region_selector(self, mode, *args):
        target = self._region_target
        if target is None:
            if self.region_a is None:
                target = "a"
            elif self.region_b is None:
                target = "b"
            else:
                target = self._last_region_target

        self.regionSelectionButton.setChecked(True)
        if target == "a":
            combo = self.regionModeACombo
        else:
            combo = self.regionModeBCombo
        self._region_target = target
        self._last_region_target = target
        combo.setCurrentIndex(combo.findData(mode))

    def _set_region_selection_enabled(self, enabled):
        if enabled:
            if self.region_a is None:
                self._region_target = "a"
            elif self.region_b is None:
                self._region_target = "b"
            else:
                self._region_target = self._last_region_target
        else:
            self._region_target = None
            self._region_drag_action = None
        self.ui.imageLabel.setCursor(Qt.CursorShape.CrossCursor)
        self._redraw_region_overlay()

    def _next_missing_region(self):
        if self.region_a is None:
            return "a"
        if self.region_b is None:
            return "b"
        return None

    def _active_region(self):
        if self._region_target == "a":
            return self.region_a
        if self._region_target == "b":
            return self.region_b
        return None

    def _set_active_region(self, region):
        if self._region_target == "a":
            self.region_a = region
        elif self._region_target == "b":
            self.region_b = region

    def _find_region_at_point(self, x, y):
        inside_hits = []
        for key, region in (("a", self.region_a), ("b", self.region_b)):
            if region is None:
                continue
            center_x, center_y = region.center
            distance = math.hypot(x - center_x, y - center_y)
            if region.contains(x, y, tolerance=4):
                inside_hits.append((distance, key, region, "move"))

        if inside_hits:
            return min(inside_hits, key=lambda hit: (hit[0], hit[1]))[1:]
        return None

    def _fixed_radius_region(self, center_x, center_y):
        radius = float(self.regionRadiusSpin.value())
        center_x = float(clamp(center_x, radius, const.MAIN_IMAGE_WIDTH - radius))
        center_y = float(clamp(center_y, radius, const.MAIN_IMAGE_HEIGHT - radius))
        return Region(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )

    def _region_radius_changed(self, *args):
        if self.region_a is not None:
            self.region_a = self._fixed_radius_region(*self.region_a.center)
        if self.region_b is not None:
            self.region_b = self._fixed_radius_region(*self.region_b.center)
        self._region_settings_changed()

    def _update_region_cursor(self, point):
        if self._region_target is None:
            self.ui.imageLabel.setCursor(Qt.CursorShape.CrossCursor)
            return

        if self._region_drag_action == "move":
            cursor = Qt.CursorShape.SizeAllCursor
        elif self._region_drag_action == "place":
            cursor = Qt.CursorShape.CrossCursor
        else:
            hit = self._find_region_at_point(point.x(), point.y())
            if hit is not None:
                key, _region, _action = hit
                self._region_target = key
                self._last_region_target = key
                cursor = Qt.CursorShape.SizeAllCursor
            elif self._next_missing_region() is not None:
                cursor = Qt.CursorShape.CrossCursor
            else:
                cursor = Qt.CursorShape.ArrowCursor
        self.ui.imageLabel.setCursor(cursor)

    def _begin_region_drag(self, point):
        if self._region_target is None:
            return
        x = float(clamp(point.x(), 0, const.MAIN_IMAGE_WIDTH))
        y = float(clamp(point.y(), 0, const.MAIN_IMAGE_HEIGHT))
        hit = self._find_region_at_point(x, y)
        if hit is not None:
            target, region, action = hit
            self._region_target = target
            self._last_region_target = target
        else:
            target = self._next_missing_region()
            if target is None:
                self._region_drag_action = None
                self._update_region_cursor(point)
                return
            self._region_target = target
            self._last_region_target = target
            region = None
            action = "place"

        self._region_drag_start = (x, y)
        self._region_drag_original = region

        if action == "move":
            self._region_drag_action = "move"
        else:
            self._region_drag_action = "place"
            self._set_active_region(self._fixed_radius_region(x, y))
        self._update_region_cursor(point)

    def _drag_region(self, point):
        if self._region_drag_action is None or self._region_target is None:
            return
        x = float(clamp(point.x(), 0, const.MAIN_IMAGE_WIDTH))
        y = float(clamp(point.y(), 0, const.MAIN_IMAGE_HEIGHT))

        if self._region_drag_action == "move":
            original = self._region_drag_original
            start_x, start_y = self._region_drag_start
            dx = x - start_x
            dy = y - start_y
            region = self._fixed_radius_region(
                original.center[0] + dx,
                original.center[1] + dy,
            )
        else:
            region = self._fixed_radius_region(x, y)

        self._set_active_region(region)
        self.region_measurement_result = None
        self._redraw_region_overlay()

    def _finish_region_drag(self, point):
        if self._region_drag_action is None or self._region_target is None:
            return
        self._drag_region(point)
        region = self._active_region()
        if region is None or region.radius < 2:
            self._set_active_region(self._region_drag_original)

        completed_target = self._region_target
        self._region_drag_action = None
        self._region_drag_start = None
        self._region_drag_original = None

        if completed_target == "a" and self.region_a is not None and self.region_b is None:
            self._region_target = "b"
            self._last_region_target = "b"

        self._calculate_region_measurement()
        self._redraw_region_overlay()
        self._update_region_cursor(point)

    def _region_settings_changed(self, *args):
        self.region_measurement_result = None
        self._calculate_region_measurement()
        self._redraw_region_overlay()

    def _depth_display_changed(self, *args):
        self.depthContourStepSpin.setEnabled(
            self.depthDisplayCombo.currentData() == DepthDisplayMode.COLOR_CONTOURS
        )
        self.redrawZoom()

    def clear_region_measurement(self, *args, repaint=True):
        self.region_a = None
        self.region_b = None
        self.region_measurement_result = None
        self._region_target = (
            "a"
            if hasattr(self, "regionSelectionButton") and self.regionSelectionButton.isChecked()
            else None
        )
        self._last_region_target = "a"
        self._region_drag_action = None
        if hasattr(self, "regionResultEdit"):
            self.regionResultEdit.setPlainText(
                "Click once to place region A, then once to place region B."
            )
        if repaint:
            self._redraw_region_overlay()

    def _calculate_region_measurement(self):
        if not hasattr(self, "regionResultEdit"):
            return
        if self.region_a is None or self.region_b is None:
            missing = "A" if self.region_a is None else "B"
            self.regionResultEdit.setPlainText(
                f"Click once to place region {missing} and calculate a measurement."
            )
            return
        if self.filtered_depthmap is None or not hasattr(self, "image"):
            self.regionResultEdit.setPlainText("Load an image with depth data first.")
            return

        mode_a = SelectionMode(self.regionModeACombo.currentText())
        mode_b = SelectionMode(self.regionModeBCombo.currentText())
        mask_a = RegionMask(self.regionMaskACombo.currentText())
        mask_b = RegionMask(self.regionMaskBCombo.currentText())
        engine = RegionMeasurementEngine(
            filtered_depthmap=self.filtered_depthmap,
            display_size=(const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT),
            image_size=self.image.size,
            depth_to_cm=self.get_depthmap_distance,
            pixels_per_mm=pixels_per_mm_at_distance,
            surface_length=lambda start, end: self.surface_vector_filtered(
                start[0], start[1], end[0], end[1], 3
            ),
            teethmap=self.portrait.teethmap if self.portrait else None,
        )
        try:
            result = engine.measure(
                self.region_a,
                self.region_b,
                mode_a=mode_a,
                mode_b=mode_b,
                mask_a=mask_a,
                mask_b=mask_b,
                percentile=self.regionPercentileSpin.value(),
                vector_count=self.regionVectorCountSpin.value(),
            )
        except MeasurementError as error:
            self.region_measurement_result = None
            self.regionResultEdit.setPlainText(f"Cannot calculate measurement:\n{error}")
            return

        self.region_measurement_result = result
        requested = self.regionVectorCountSpin.value()
        linear = result.linear_stats()
        surface = result.surface_stats()
        valid_surface_count = sum(sample.surface_mm is not None for sample in result.samples)
        lines = [
            f"A: {mode_a.value}, mask={mask_a.value}, radius={self.region_a.radius:.0f} px",
            f"B: {mode_b.value}, mask={mask_b.value}, radius={self.region_b.radius:.0f} px",
            f"Candidate pool: {self.regionPercentileSpin.value()}%",
            f"Requested profiles: {requested}",
            "",
        ]
        if linear is None:
            lines.append(f"Linear 3D: insufficient profiles (n={len(result.samples)})")
        else:
            lines.append(
                "Linear 3D: "
                f"{linear.mean_mm / 10:.2f} ± {linear.sample_sd_mm / 10:.2f} cm "
                f"(n={linear.count})"
            )
        if surface is None:
            lines.append(f"Surface 3D: insufficient valid profiles (n={valid_surface_count})")
        else:
            lines.append(
                "Surface 3D: "
                f"{surface.mean_mm / 10:.2f} ± {surface.sample_sd_mm / 10:.2f} cm "
                f"(n={surface.count}, step=3 px)"
            )
        if valid_surface_count < requested:
            lines.extend(
                (
                    "",
                    f"Warning: {requested - valid_surface_count} surface profile(s) crossed invalid depth data.",
                )
            )
        self.regionResultEdit.setPlainText("\n".join(lines))

    def _redraw_region_overlay(self):
        if self.portrait:
            self.redrawImage(force=True)
        if hasattr(self, "zoomedDepthMapLabel"):
            self.redrawZoom()

    def _paint_region_overlay(self, painter):
        if self._region_target is None:
            return
        painter.save()
        try:
            colors = {
                "a": QColor(0, 190, 255, 220),
                "b": QColor(255, 0, 170, 220),
            }
            result = self.region_measurement_result
            if result is not None:
                line_pen = QtGui.QPen(QColor(255, 255, 255, 125), 1)
                painter.setPen(line_pen)
                painter.setBrush(Qt.NoBrush)
                for sample in result.samples:
                    painter.drawLine(
                        QPoint(sample.start.x, sample.start.y),
                        QPoint(sample.end.x, sample.end.y),
                    )
                for key, candidates in (
                    ("a", result.candidates_a),
                    ("b", result.candidates_b),
                ):
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(colors[key])
                    for candidate in candidates:
                        painter.drawEllipse(QPoint(candidate.x, candidate.y), 3, 3)

            for key, region in (("a", self.region_a), ("b", self.region_b)):
                if region is None:
                    continue
                color = colors[key]
                painter.setPen(QtGui.QPen(color, 2))
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 35))
                painter.drawEllipse(
                    int(region.left),
                    int(region.top),
                    max(1, int(region.width)),
                    max(1, int(region.height)),
                )
                painter.setPen(color)
                painter.setBrush(Qt.NoBrush)
                painter.drawText(
                    QPoint(int(region.center[0]) + 6, int(region.center[1]) - 6),
                    key.upper(),
                )
        finally:
            painter.restore()

    def _create_zoom_panel(self):
        # Take the UI widget out of central and wrap it with a zoom panel below
        ui_widget = self.centralWidget()
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)
        wrapper_layout.addWidget(ui_widget)

        # Create the bottom zoom panel with 2 rows of always-visible labels
        zoom_panel = QWidget()
        zoom_panel.setFixedHeight(400)
        zoom_panel_layout = QVBoxLayout(zoom_panel)
        zoom_panel_layout.setContentsMargins(2, 2, 2, 2)
        zoom_panel_layout.setSpacing(2)

        # Row 1: depth map, teeth map, reconstruction
        row1 = QHBoxLayout()
        row1.setSpacing(2)

        self.zoomedDepthMapLabel = QLabel("Depth Map")
        self.zoomedDepthMapLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.zoomedDepthMapLabel.setStyleSheet("border:1px solid;")
        row1.addWidget(self.zoomedDepthMapLabel, stretch=1)

        self.zoomedTeethMapLabel = QLabel("Teeth Map")
        self.zoomedTeethMapLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.zoomedTeethMapLabel.setStyleSheet("border:1px solid;")
        row1.addWidget(self.zoomedTeethMapLabel, stretch=1)

        self.reconstructionLabel = QLabel("Reconstruction")
        self.reconstructionLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.reconstructionLabel.setStyleSheet("border:1px solid;")
        row1.addWidget(self.reconstructionLabel, stretch=1)

        zoom_panel_layout.addLayout(row1)

        # Row 2: photo, skin matte
        row2 = QHBoxLayout()
        row2.setSpacing(2)

        self.zoomedImageLabel = QLabel("Photo")
        self.zoomedImageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.zoomedImageLabel.setStyleSheet("border:1px solid;")
        row2.addWidget(self.zoomedImageLabel, stretch=1)

        self.zoomedSkinMapLabel = QLabel("Skin Matte")
        self.zoomedSkinMapLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.zoomedSkinMapLabel.setStyleSheet("border:1px solid;")
        row2.addWidget(self.zoomedSkinMapLabel, stretch=1)

        self.skinMaskLabel = QLabel("Depth (hairless)")
        self.skinMaskLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.skinMaskLabel.setStyleSheet("border:1px solid;")
        row2.addWidget(self.skinMaskLabel, stretch=1)

        zoom_panel_layout.addLayout(row2)

        wrapper_layout.addWidget(zoom_panel)
        self.setCentralWidget(wrapper)

    # ------------------------------------------------------------------
    # Zoom painting methods (moved from ZoomWindow)
    # ------------------------------------------------------------------

    @staticmethod
    def _zoom_display_point(point, source_size, crop_box, display_rect):
        source_width, source_height = source_size
        crop_left, crop_top, crop_right, crop_bottom = crop_box
        display_left, display_top, display_width, display_height = display_rect
        source_x = point[0] * source_width / const.MAIN_IMAGE_WIDTH
        source_y = point[1] * source_height / const.MAIN_IMAGE_HEIGHT
        return (
            display_left + (source_x - crop_left) * display_width / max(1, crop_right - crop_left),
            display_top + (source_y - crop_top) * display_height / max(1, crop_bottom - crop_top),
        )

    def _paint_zoom_region_overlay(self, painter, source_size, crop_box, display_rect):
        show_regions = self._region_target is not None
        show_neck = (
            self.ui.showNeckArcCheckBox.isChecked()
            and self.neck_measurement_auto is not None
            and self.portrait is not None
            and self.portrait.skinmap is not None
        )
        if not show_regions and not show_neck:
            return

        def display_point(point):
            x, y = self._zoom_display_point(point, source_size, crop_box, display_rect)
            return QPoint(round(x), round(y))

        painter.save()
        try:
            if show_regions:
                colors = {
                    "a": QColor(0, 190, 255, 230),
                    "b": QColor(255, 0, 170, 230),
                }
                result = self.region_measurement_result
                if result is not None:
                    painter.setPen(QtGui.QPen(QColor(255, 255, 255, 175), 1))
                    painter.setBrush(Qt.NoBrush)
                    for sample in result.samples:
                        painter.drawLine(
                            display_point((sample.start.x, sample.start.y)),
                            display_point((sample.end.x, sample.end.y)),
                        )
                    for key, candidates in (
                        ("a", result.candidates_a),
                        ("b", result.candidates_b),
                    ):
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(colors[key])
                        for candidate in candidates:
                            painter.drawEllipse(display_point((candidate.x, candidate.y)), 3, 3)

                for key, region in (("a", self.region_a), ("b", self.region_b)):
                    if region is None:
                        continue
                    top_left = display_point((region.left, region.top))
                    bottom_right = display_point((region.right, region.bottom))
                    color = colors[key]
                    painter.setPen(QtGui.QPen(color, 2))
                    painter.setBrush(QColor(color.red(), color.green(), color.blue(), 28))
                    painter.drawEllipse(
                        top_left.x(),
                        top_left.y(),
                        bottom_right.x() - top_left.x(),
                        bottom_right.y() - top_left.y(),
                    )
                    painter.setPen(color)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawText(
                        display_point((region.center[0] + 4, region.center[1] - 4)),
                        key.upper(),
                    )

            if show_neck:
                skin_size = self.portrait.skinmap.size

                def neck_display_point(point):
                    main_point = translate_coordinates_to_other_image(
                        point,
                        skin_size,
                        (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT),
                    )
                    return display_point(main_point)

                self._paint_neck_depth_overlay(painter, neck_display_point)
        finally:
            painter.restore()

    def _paint_neck_depth_overlay(self, painter, point_mapper, *, labels=False):
        """Paint skin-matte edges and their corrected stable depth samples."""
        measurement = self.neck_measurement_auto
        if measurement is None:
            return

        raw_points = (
            (measurement.mask_left_x, measurement.neck_y),
            (measurement.mask_right_x, measurement.neck_y),
        )
        depth_points = (
            (measurement.left_x, measurement.neck_y),
            (measurement.right_x, measurement.neck_y),
        )
        arc_points = [point_mapper(point) for point in measurement.arc_points_photo]
        raw_points = [point_mapper(point) if point[0] is not None else None for point in raw_points]
        depth_points = [point_mapper(point) for point in depth_points]

        painter.save()
        try:
            arc_color = QColor(80, 220, 130, 220)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QtGui.QPen(arc_color, 2))
            for first, second in zip(arc_points, arc_points[1:]):
                painter.drawLine(first, second)
            painter.setPen(Qt.NoPen)
            painter.setBrush(arc_color)
            for point in arc_points:
                painter.drawEllipse(point, 3, 3)

            skin_color = QColor(255, 190, 0, 240)
            depth_color = QColor(0, 230, 255, 240)
            connector_pen = QtGui.QPen(QColor(255, 255, 255, 210), 2, Qt.DashLine)
            for raw_point, depth_point in zip(raw_points, depth_points, strict=True):
                if raw_point is not None:
                    painter.setPen(connector_pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawLine(raw_point, depth_point)
                    painter.setPen(QtGui.QPen(skin_color, 2))
                    painter.drawEllipse(raw_point, 6, 6)
                    if labels:
                        painter.setPen(skin_color)
                        painter.drawText(raw_point + QPoint(7, -7), "skin")
                painter.setPen(QtGui.QPen(QColor(0, 70, 80, 255), 1))
                painter.setBrush(depth_color)
                painter.drawEllipse(depth_point, 5, 5)
                if labels:
                    painter.setPen(depth_color)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawText(depth_point + QPoint(7, 14), "depth")
        finally:
            painter.restore()

    @staticmethod
    def _paint_compact_zoom_info(painter, width, height, lines, text_color):
        lines = [line for line in lines if line][:2]
        if not lines:
            return
        painter.save()
        try:
            font = painter.font()
            font.setPixelSize(max(9, min(11, height // 18)))
            painter.setFont(font)
            metrics = painter.fontMetrics()
            line_height = metrics.height()
            box_height = line_height * len(lines) + 8
            box_top = max(5, height - box_height - 5)
            painter.fillRect(5, box_top, max(1, width - 10), box_height, QColor(0, 0, 0, 155))
            painter.setPen(text_color)
            available_width = max(1, width - 20)
            for index, line in enumerate(lines):
                text = metrics.elidedText(line, Qt.ElideRight, available_width)
                painter.drawText(10, box_top + 4 + metrics.ascent() + index * line_height, text)
        finally:
            painter.restore()

    def _paintZoomedMap(
        self,
        label_text,
        image_map,
        ui_element,
        mouse_x=None,
        mouse_y=None,
        ok_value_threshold=100,
        depth_visualization=False,
        source_size=None,
        crop_box=None,
    ):
        w = ui_element.width()
        h = ui_element.height()
        if w < 1 or h < 1:
            return

        displayed_map = image_map
        raw_range = None
        draw_contours = False
        if depth_visualization:
            mode = DepthDisplayMode(self.depthDisplayCombo.currentText())
            if mode != DepthDisplayMode.RAW:
                visualization = render_depth_visualization(image_map)
                displayed_map = visualization.image
                raw_range = (visualization.low_raw, visualization.high_raw)
                draw_contours = mode == DepthDisplayMode.COLOR_CONTOURS

        qimg = displayed_map.toqimage().scaled(w, h, Qt.KeepAspectRatio)
        canvas = QtGui.QPixmap(w, h)
        canvas.fill(Qt.black)
        painter = QtGui.QPainter(canvas)
        try:
            offset_x = (w - qimg.width()) // 2
            offset_y = (h - qimg.height()) // 2
            painter.drawImage(offset_x, offset_y, qimg)
            if draw_contours:
                contour_overlay = render_depth_contour_overlay(
                    image_map,
                    (qimg.width(), qimg.height()),
                    level_step=self.depthContourStepSpin.value(),
                )
                painter.drawImage(offset_x, offset_y, contour_overlay.toqimage())

            if source_size is not None and crop_box is not None:
                self._paint_zoom_region_overlay(
                    painter,
                    source_size,
                    crop_box,
                    (offset_x, offset_y, qimg.width(), qimg.height()),
                )

            painter.setPen(QColor(255, 0, 0, 255))
            half_w = w // 2
            half_h = h // 2
            painter.drawLine(QPoint(half_w, 0), QPoint(half_w, h))
            painter.drawLine(QPoint(0, half_h), QPoint(w, half_h))

            source_x = image_map.width // 2
            source_y = image_map.height // 2
            try:
                value = image_map.getpixel((source_x, source_y))[0]
            except TypeError:
                value = image_map.getpixel((source_x, source_y))

            if depth_visualization and raw_range is not None:
                text_color = QColor(255, 255, 255, 255)
            elif value < ok_value_threshold:
                text_color = QColor(255, 90, 90, 255)
            else:
                text_color = QColor(90, 255, 90, 255)
            lines = [f"{label_text} · raw {value}"]
            details = []
            if raw_range is not None and raw_range[0] is not None:
                details.append(f"range {raw_range[0]}–{raw_range[1]}")
            if mouse_x is not None and mouse_y is not None:
                details.append(f"px {int(mouse_x)},{int(mouse_y)}")
            lines.append(" · ".join(details))
            self._paint_compact_zoom_info(painter, w, h, lines, text_color)
        finally:
            painter.end()
        ui_element.setPixmap(canvas)

    def paintZoomedDepthmap(
        self,
        depthmap,
        mouse_x=None,
        mouse_y=None,
        source_size=None,
        crop_box=None,
    ):
        self._paintZoomedMap(
            "Depth map",
            depthmap,
            self.zoomedDepthMapLabel,
            mouse_x,
            mouse_y,
            depth_visualization=True,
            source_size=source_size,
            crop_box=crop_box,
        )

    def paintZoomedImage(self, zoomed, source_size=None, crop_box=None):
        w = self.zoomedImageLabel.width()
        h = self.zoomedImageLabel.height()
        if w < 1 or h < 1:
            return

        qimg = zoomed.toqimage().scaled(w, h, Qt.KeepAspectRatio)
        canvas = QtGui.QPixmap(w, h)
        canvas.fill(Qt.black)
        painter = QtGui.QPainter(canvas)
        try:
            offset_x = (w - qimg.width()) // 2
            offset_y = (h - qimg.height()) // 2
            painter.drawImage(offset_x, offset_y, qimg)

            if source_size is not None and crop_box is not None:
                self._paint_zoom_region_overlay(
                    painter,
                    source_size,
                    crop_box,
                    (offset_x, offset_y, qimg.width(), qimg.height()),
                )

            painter.setPen(QColor(255, 0, 0, 255))
            half_w = w // 2
            half_h = h // 2
            painter.drawLine(QPoint(half_w, 0), QPoint(half_w, h))
            painter.drawLine(QPoint(0, half_h), QPoint(w, half_h))
        finally:
            painter.end()
        self.zoomedImageLabel.setPixmap(canvas)

    def paintZoomedSkinmap(
        self,
        skinmap,
        mouse_x=None,
        mouse_y=None,
        source_size=None,
        crop_box=None,
    ):
        w = self.zoomedSkinMapLabel.width()
        h = self.zoomedSkinMapLabel.height()
        if w < 1 or h < 1:
            return

        qimg = skinmap.toqimage().scaled(w, h, Qt.KeepAspectRatio)
        canvas = QtGui.QPixmap(w, h)
        canvas.fill(Qt.black)
        painter = QtGui.QPainter(canvas)
        try:
            offset_x = (w - qimg.width()) // 2
            offset_y = (h - qimg.height()) // 2
            painter.drawImage(offset_x, offset_y, qimg)

            if source_size is not None and crop_box is not None:
                self._paint_zoom_region_overlay(
                    painter,
                    source_size,
                    crop_box,
                    (offset_x, offset_y, qimg.width(), qimg.height()),
                )

            painter.setPen(QColor(255, 0, 0, 255))
            half_w = w // 2
            half_h = h // 2
            painter.drawLine(QPoint(half_w, 0), QPoint(half_w, h))
            painter.drawLine(QPoint(0, half_h), QPoint(w, half_h))

            try:
                value = skinmap.getpixel((skinmap.width // 2, skinmap.height // 2))[0]
            except TypeError:
                value = skinmap.getpixel((skinmap.width // 2, skinmap.height // 2))

            if value < 100:
                text_color = QColor(255, 90, 90, 255)
            else:
                text_color = QColor(90, 255, 90, 255)
            lines = [f"Skin matte · raw {value}"]
            if mouse_x is not None and mouse_y is not None:
                lines.append(f"px {int(mouse_x)},{int(mouse_y)}")

            self._paint_compact_zoom_info(painter, w, h, lines, text_color)
        finally:
            painter.end()
        self.zoomedSkinMapLabel.setPixmap(canvas)

    def _build_hairless_depth_image(self):
        """Build hairless depth map (depth with hair zeroed out) as a PIL Image."""
        from PIL import Image as PILImage

        if not self.portrait or self.portrait.depthmap is None:
            self.hairless_depth_image = None
            return

        depth_arr = np.array(self.portrait.depthmap)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]
        depth_arr = depth_arr.copy()
        dh, dw = depth_arr.shape[:2]

        if self.portrait.hairmap is not None:
            import cv2

            hair_arr = np.array(self.portrait.hairmap)
            if hair_arr.ndim == 3:
                hair_arr = hair_arr[:, :, 0]
            if hair_arr.shape[:2] != (dh, dw):
                hair_arr = cv2.resize(hair_arr, (dw, dh), interpolation=cv2.INTER_LINEAR)
            depth_arr[hair_arr >= 5] = 0

        self.hairless_depth_image = PILImage.fromarray(depth_arr, mode="L")

    def paintHairlessDepthFull(self):
        """Display the full hairless depth map scaled to fit the label."""
        if self.hairless_depth_image is None:
            return
        w = self.skinMaskLabel.width()
        h = self.skinMaskLabel.height()
        if w < 1 or h < 1:
            return

        displayed_map = self.hairless_depth_image
        raw_range = None
        mode = DepthDisplayMode(self.depthDisplayCombo.currentText())
        draw_contours = mode == DepthDisplayMode.COLOR_CONTOURS
        if mode != DepthDisplayMode.RAW:
            visualization = render_depth_visualization(self.hairless_depth_image)
            displayed_map = visualization.image
            raw_range = (visualization.low_raw, visualization.high_raw)

        qimg = displayed_map.toqimage().scaled(w, h, Qt.KeepAspectRatio)
        canvas = QtGui.QPixmap(w, h)
        canvas.fill(Qt.black)
        painter = QtGui.QPainter(canvas)
        try:
            offset_x = (w - qimg.width()) // 2
            offset_y = (h - qimg.height()) // 2
            painter.drawImage(offset_x, offset_y, qimg)
            if draw_contours:
                contour_overlay = render_depth_contour_overlay(
                    self.hairless_depth_image,
                    (qimg.width(), qimg.height()),
                    level_step=self.depthContourStepSpin.value(),
                )
                painter.drawImage(offset_x, offset_y, contour_overlay.toqimage())

            self._paint_zoom_region_overlay(
                painter,
                self.hairless_depth_image.size,
                (0, 0, *self.hairless_depth_image.size),
                (offset_x, offset_y, qimg.width(), qimg.height()),
            )
            lines = ["Depth (hairless)"]
            if raw_range is not None and raw_range[0] is not None:
                lines.append(f"range {raw_range[0]}–{raw_range[1]}")
            self._paint_compact_zoom_info(
                painter,
                w,
                h,
                lines,
                QColor(255, 255, 255, 255),
            )
        finally:
            painter.end()
        self.skinMaskLabel.setPixmap(canvas)

    def paintZoomedHairlessDepth(
        self,
        zoomed,
        mouse_x=None,
        mouse_y=None,
        source_size=None,
        crop_box=None,
    ):
        """Display zoomed hairless depth map with value readout."""
        self._paintZoomedMap(
            "Depth (hairless)",
            zoomed,
            self.skinMaskLabel,
            mouse_x,
            mouse_y,
            depth_visualization=True,
            source_size=source_size,
            crop_box=crop_box,
        )

    def paintZoomedTeethmap(
        self,
        teethmap,
        mouse_x=None,
        mouse_y=None,
        centroids=None,
        crop_origin=None,
        crop_size=None,
        source_size=None,
        crop_box=None,
    ):
        w = self.zoomedTeethMapLabel.width()
        h = self.zoomedTeethMapLabel.height()
        if w < 1 or h < 1:
            return

        self._paintZoomedMap(
            "Teeth map",
            teethmap,
            self.zoomedTeethMapLabel,
            mouse_x,
            mouse_y,
            ok_value_threshold=200,
            source_size=source_size,
            crop_box=crop_box,
        )

        if centroids is None or crop_origin is None or crop_size is None:
            return

        canvas = self.zoomedTeethMapLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        try:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 255, 255, 200))
            for centroid in (centroids.upper_centroid, centroids.lower_centroid):
                if centroid is None:
                    continue
                local_x = centroid[0] - crop_origin[0]
                local_y = centroid[1] - crop_origin[1]
                display_x = local_x * w / crop_size[0]
                display_y = local_y * h / crop_size[1]
                if 0 <= display_x <= w and 0 <= display_y <= h:
                    painter.drawEllipse(QPoint(int(display_x), int(display_y)), 5, 5)
        finally:
            painter.end()
        self.zoomedTeethMapLabel.setPixmap(canvas)

    def paintReconstruction(self, values):
        if not values:
            return

        w = self.reconstructionLabel.width()
        h = self.reconstructionLabel.height()
        if w < 1 or h < 1:
            return

        canvas = QtGui.QPixmap(w, h)
        canvas.fill(Qt.yellow)
        painter = QtGui.QPainter(canvas)
        try:
            painter.setPen(QColor(0, 0, 0, 255))
            for a in range(w):
                idx = min(int(a * len(values) / float(w)), len(values) - 1)
                v = values[idx]
                painter.drawLine(
                    QPoint(a, h),
                    QPoint(a, h - v),
                )
        finally:
            painter.end()
        self.reconstructionLabel.setPixmap(canvas)

    # ------------------------------------------------------------------
    # Core application logic
    # ------------------------------------------------------------------

    def get_depthmap_distance(self, value):
        """Returns a distance from a given depthMap value in centimeters
        using EXIF data from TrueDepth[tm] camera

        :returns: distance in centimeters
        """
        if self.float_min_value is None or self.float_max_value is None:
            return value

        return depth_raw_to_distance_cm(value, self.float_min_value, self.float_max_value)

    def redrawZoom(self, *args, **kw):
        if args:
            event = args[0]
            mouse_x = event.x()
            mouse_y = event.y()
            self.last_zoom_x = mouse_x
            self.last_zoom_y = mouse_y
        else:
            mouse_x = self.last_zoom_x
            mouse_y = self.last_zoom_y

        img_w = const.MAIN_IMAGE_WIDTH
        img_h = const.MAIN_IMAGE_HEIGHT
        zoom_w = const.ZOOM_WIDTH
        zoom_h = const.ZOOM_HEIGHT

        # Crop radii — 3/4 of full zoom size for moderate zoom
        crop_w = zoom_w * 3 // 4
        crop_h = zoom_h * 3 // 4

        if self.smallImage:
            big_image_x = mouse_x * self.image.size[0] / img_w
            big_image_y = mouse_y * self.image.size[1] / img_h
            photo_crop = (
                big_image_x - crop_w,
                big_image_y - crop_h,
                big_image_x + crop_w,
                big_image_y + crop_h,
            )
            zoomed = self.image.crop(photo_crop).resize((zoom_w, zoom_h))
            self.paintZoomedImage(
                zoomed,
                source_size=self.image.size,
                crop_box=photo_crop,
            )

        if self.portrait and self.portrait.skinmap:
            skinmap_x, skinmap_y = translate_coordinates_to_other_image(
                (mouse_x, mouse_y), (img_w, img_h), (self.portrait.skinmap.size)
            )
            skin_crop = (
                skinmap_x - crop_w,
                skinmap_y - crop_h,
                skinmap_x + crop_w,
                skinmap_y + crop_h,
            )
            zoomed = self.portrait.skinmap.crop(skin_crop).resize((zoom_w, zoom_h))
            self.paintZoomedSkinmap(
                zoomed,
                mouse_x=skinmap_x,
                mouse_y=skinmap_y,
                source_size=self.portrait.skinmap.size,
                crop_box=skin_crop,
            )

        if self.hairless_depth_image is not None:
            hd_x, hd_y = translate_coordinates_to_other_image(
                (mouse_x, mouse_y),
                (img_w, img_h),
                self.hairless_depth_image.size,
            )
            hairless_crop = (hd_x - 72, hd_y - 48, hd_x + 72, hd_y + 48)
            zoomed = self.hairless_depth_image.crop(hairless_crop)
            self.paintZoomedHairlessDepth(
                zoomed,
                mouse_x=hd_x,
                mouse_y=hd_y,
                source_size=self.hairless_depth_image.size,
                crop_box=hairless_crop,
            )
        else:
            self.paintHairlessDepthFull()

        if self.portrait and self.portrait.teethmap:
            teethmap_x, teethmap_y = translate_coordinates_to_other_image(
                (mouse_x, mouse_y), (img_w, img_h), (self.portrait.teethmap.size)
            )

            teeth_crop = (
                teethmap_x - crop_w,
                teethmap_y - crop_h,
                teethmap_x + crop_w,
                teethmap_y + crop_h,
            )
            zoomed = self.portrait.teethmap.crop(teeth_crop).resize((zoom_w, zoom_h))

            centroids = None
            if self.ui.showCentroidsCheckBox.isChecked() and self.portrait.incisor_measurement:
                centroids = self.portrait.incisor_measurement

            self.paintZoomedTeethmap(
                zoomed,
                mouse_x=teethmap_x,
                mouse_y=teethmap_y,
                centroids=centroids,
                crop_origin=(
                    teethmap_x - crop_w,
                    teethmap_y - crop_h,
                ),
                crop_size=(crop_w * 2, crop_h * 2),
                source_size=self.portrait.teethmap.size,
                crop_box=teeth_crop,
            )

        if self.depthmap:
            depth_crop = (mouse_x - 72, mouse_y - 48, mouse_x + 72, mouse_y + 48)
            zoomed = self.depthmap.crop(depth_crop)

            self.paintZoomedDepthmap(
                zoomed,
                mouse_x=mouse_x,
                mouse_y=mouse_y,
                source_size=self.depthmap.size,
                crop_box=depth_crop,
            )

    def redrawImage(self, *args, **kw):
        if not self.portrait:
            return
        force = kw.pop("force", False)

        mouse_x = x = self.ui.xValue.value()
        y = mouse_y = self.ui.yValue.value()
        angle = self.ui.angleValue.value()

        mouse_x = clamp(mouse_x, 0, const.MAIN_IMAGE_WIDTH)
        mouse_y = clamp(mouse_y, 0, const.MAIN_IMAGE_HEIGHT)

        show_centroids = self.ui.showCentroidsCheckBox.isChecked()
        show_neck_arc = self.ui.showNeckArcCheckBox.isChecked()
        show_landmarks = self.ui.showMediaPipeLandmarksCheckBox.isChecked()
        show_face_mesh = self.ui.showFaceMeshLandmarksCheckBox.isChecked()
        show_segmentation = self.ui.showSegmentationDebugCheckBox.isChecked()

        if (
            not force
            and self.last_click_x is not None
            and (
                self.last_click_x == mouse_x
                and self.last_click_y == mouse_y
                and self.last_angle == angle
                and self.last_show_centroids == show_centroids
                and self.last_show_neck_arc == show_neck_arc
                and self.last_show_landmarks == show_landmarks
                and self.last_show_face_mesh == show_face_mesh
                and self.last_show_segmentation == show_segmentation
            )
        ):
            return

        self.last_angle = angle
        self.last_show_centroids = show_centroids
        self.last_show_neck_arc = show_neck_arc
        self.last_show_landmarks = show_landmarks
        self.last_show_face_mesh = show_face_mesh
        self.last_show_segmentation = show_segmentation

        img_w = const.MAIN_IMAGE_WIDTH
        img_h = const.MAIN_IMAGE_HEIGHT

        canvas = self.ui.imageLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        try:
            painter.setPen(QColor(0, 0, 255, 127))
            painter.pen().setDashOffset(2)
            canvas.fill(Qt.white)
            if self.smallImage:
                painter.drawImage(0, 0, self.smallImage.toqimage())

            if self.portrait:
                if self.portrait.teeth_bbox:
                    tx, ty, twi, the = self.portrait.teeth_bbox_translated(img_w, img_h)
                    painter.setPen(QColor(255, 255, 0, 127))
                    painter.drawRect(tx, ty, twi, the)

                    if self.portrait.incisor_distance:
                        auto_id_click_1 = translate_coordinates_to_other_image(
                            (
                                self.portrait.incisor_distance[0],
                                self.portrait.incisor_distance[1],
                            ),
                            self.portrait.teethmap.size,
                            (img_w, img_h),
                        )
                        auto_id_click_2 = translate_coordinates_to_other_image(
                            (
                                self.portrait.incisor_distance[2],
                                self.portrait.incisor_distance[3],
                            ),
                            self.portrait.teethmap.size,
                            (img_w, img_h),
                        )
                        painter.drawLine(
                            QPoint(*auto_id_click_1),
                            QPoint(*auto_id_click_2),
                        )

                if self.ui.showCentroidsCheckBox.isChecked() and self.portrait.incisor_measurement:
                    im = self.portrait.incisor_measurement
                    for centroid in (im.upper_centroid, im.lower_centroid):
                        if centroid is None:
                            continue
                        cx, cy = translate_coordinates_to_other_image(
                            centroid,
                            self.portrait.teethmap.size,
                            (img_w, img_h),
                        )
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QColor(0, 255, 255, 200))
                        painter.drawEllipse(QPoint(int(cx), int(cy)), 5, 5)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QColor(255, 255, 0, 127))

                # FaceMesh mouth opening fallback (green circles + dashed line)
                if self.ui.showCentroidsCheckBox.isChecked() and self.mouth_measurement:
                    mm = self.mouth_measurement
                    fm_mouth_color = QColor(0, 200, 0, 200)
                    points = []
                    for pt in (mm.upper_point, mm.lower_point):
                        dx, dy = translate_coordinates_to_other_image(
                            pt,
                            self.image.size,
                            (img_w, img_h),
                        )
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(fm_mouth_color)
                        painter.drawEllipse(QPoint(int(dx), int(dy)), 5, 5)
                        points.append(QPoint(int(dx), int(dy)))
                    if len(points) == 2:
                        dash_pen = QtGui.QPen(fm_mouth_color, 2, Qt.DashLine)
                        painter.setPen(dash_pen)
                        painter.setBrush(Qt.NoBrush)
                        painter.drawLine(points[0], points[1])
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QColor(255, 255, 0, 127))

            if self.ui.showNeckArcCheckBox.isChecked() and self.neck_measurement_auto:
                skin_size = self.portrait.skinmap.size

                def neck_display_point(point):
                    x, y = translate_coordinates_to_other_image(
                        point,
                        skin_size,
                        (img_w, img_h),
                    )
                    return QPoint(round(x), round(y))

                self._paint_neck_depth_overlay(
                    painter,
                    neck_display_point,
                    labels=True,
                )
                painter.setPen(QColor(0, 0, 255, 127))

            print(
                f"Neck midpoint draw check: "
                f"checkbox={self.ui.showNeckArcCheckBox.isChecked()}, "
                f"midpoint={self.neck_midpoint}"
            )
            if (
                self.ui.showNeckArcCheckBox.isChecked()
                and self.neck_midpoint
                and self.neck_midpoint.x is not None
                and self.neck_midpoint.pose == PortraitPose.NEUTRAL_NECK
            ):
                mx, my = translate_coordinates_to_other_image(
                    (self.neck_midpoint.x, self.neck_midpoint.y),
                    self.image.size,
                    (img_w, img_h),
                )
                midpoint_color = QColor(255, 0, 255, 200)
                painter.setPen(Qt.NoPen)
                painter.setBrush(midpoint_color)
                painter.drawEllipse(QPoint(int(mx), int(my)), 5, 5)
                pen = QtGui.QPen(midpoint_color, 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(int(mx) - 10, int(my), int(mx) + 10, int(my))
                painter.drawLine(int(mx), int(my) - 10, int(mx), int(my) + 10)

            # Draw chin point and TMD line (always, no checkbox)
            if (
                self.neck_midpoint
                and self.neck_midpoint.x is not None
                and self.neck_midpoint.pose
                in (PortraitPose.EXTENDED_NECK, PortraitPose.NEUTRAL_NECK)
            ):
                midpoint_color = QColor(255, 0, 255, 200)
                mx, my = translate_coordinates_to_other_image(
                    (self.neck_midpoint.x, self.neck_midpoint.y),
                    self.image.size,
                    (img_w, img_h),
                )
                painter.setPen(Qt.NoPen)
                painter.setBrush(midpoint_color)
                painter.drawEllipse(QPoint(int(mx), int(my)), 5, 5)
                pen = QtGui.QPen(midpoint_color, 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(int(mx) - 10, int(my), int(mx) + 10, int(my))
                painter.drawLine(int(mx), int(my) - 10, int(mx), int(my) + 10)

                cx, cy = translate_coordinates_to_other_image(
                    self.neck_midpoint.chin,
                    self.image.size,
                    (img_w, img_h),
                )
                painter.setPen(Qt.NoPen)
                painter.setBrush(midpoint_color)
                painter.drawEllipse(QPoint(int(cx), int(cy)), 5, 5)
                pen = QtGui.QPen(midpoint_color, 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(int(cx) - 10, int(cy), int(cx) + 10, int(cy))
                painter.drawLine(int(cx), int(cy) - 10, int(cx), int(cy) + 10)

                # Dashed line from chin to neck midpoint
                dash_pen = QtGui.QPen(midpoint_color, 2, Qt.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(QPoint(int(cx), int(cy)), QPoint(int(mx), int(my)))

            if self.ui.showMediaPipeLandmarksCheckBox.isChecked() and self.mediapipe_debug:
                landmark_color = QColor(0, 255, 0, 200)
                painter.setPen(Qt.NoPen)
                painter.setBrush(landmark_color)
                font = painter.font()
                font.setPointSize(7)
                painter.setFont(font)
                for idx, (lx, ly) in enumerate(self.mediapipe_debug.landmarks):
                    dx, dy = translate_coordinates_to_other_image(
                        (lx, ly),
                        self.image.size,
                        (img_w, img_h),
                    )
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(landmark_color)
                    painter.drawEllipse(QPoint(int(dx), int(dy)), 3, 3)
                    painter.setPen(QColor(0, 255, 0))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawText(int(dx) + 5, int(dy) - 2, str(idx))
                painter.setBrush(Qt.NoBrush)

            if self.ui.showFaceMeshLandmarksCheckBox.isChecked() and self.face_mesh_debug:
                fm_color = QColor(255, 165, 0, 180)
                painter.setPen(Qt.NoPen)
                painter.setBrush(fm_color)
                font = painter.font()
                font.setPointSize(5)
                painter.setFont(font)
                for idx, (lx, ly) in enumerate(self.face_mesh_debug.landmarks):
                    dx, dy = translate_coordinates_to_other_image(
                        (lx, ly),
                        self.image.size,
                        (img_w, img_h),
                    )
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(fm_color)
                    painter.drawEllipse(QPoint(int(dx), int(dy)), 2, 2)
                    painter.setPen(QColor(255, 165, 0))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawText(int(dx) + 4, int(dy) - 2, str(idx))
                painter.setBrush(Qt.NoBrush)

            # Segmentation debug overlay
            if self.ui.showSegmentationDebugCheckBox.isChecked() and self.segmentation_debug:
                sd = self.segmentation_debug

                # 1. Semi-transparent mask overlay (light blue at 40% opacity)
                mask = sd.mask.astype(np.uint8) * 255
                mask_h, mask_w = mask.shape
                mask_rgba = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
                mask_rgba[mask > 0, 0] = 100  # R
                mask_rgba[mask > 0, 1] = 180  # G
                mask_rgba[mask > 0, 2] = 255  # B
                mask_rgba[mask > 0, 3] = 100  # alpha ~40%
                mask_qimage = QtGui.QImage(
                    mask_rgba.data,
                    mask_w,
                    mask_h,
                    mask_w * 4,
                    QtGui.QImage.Format_RGBA8888,
                )
                scaled_mask = mask_qimage.scaled(
                    img_w, img_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )
                painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
                painter.drawImage(0, 0, scaled_mask)

                # 1b. Semi-transparent skin mask overlay (green at 40%)
                if sd.skin_mask is not None:
                    smask = sd.skin_mask.astype(np.uint8) * 255
                    smask_h, smask_w = smask.shape
                    smask_rgba = np.zeros((smask_h, smask_w, 4), dtype=np.uint8)
                    smask_rgba[smask > 0, 0] = 50  # R
                    smask_rgba[smask > 0, 1] = 255  # G
                    smask_rgba[smask > 0, 2] = 50  # B
                    smask_rgba[smask > 0, 3] = 80  # alpha ~30%
                    smask_qimage = QtGui.QImage(
                        smask_rgba.data,
                        smask_w,
                        smask_h,
                        smask_w * 4,
                        QtGui.QImage.Format_RGBA8888,
                    )
                    scaled_smask = smask_qimage.scaled(
                        img_w,
                        img_h,
                        Qt.IgnoreAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    painter.drawImage(0, 0, scaled_smask)

                # 2. Width profile edges (left/right silhouette in cyan)
                if sd.width_profile is not None and len(sd.width_profile) > 0:
                    # Recompute left/right edges from mask within ROI
                    roi_top = sd.roi_top
                    roi_h = len(sd.width_profile)
                    roi_mask = sd.mask[roi_top : roi_top + roi_h]
                    edge_pen = QtGui.QPen(QColor(0, 255, 255, 180), 1)
                    painter.setPen(edge_pen)
                    painter.setBrush(Qt.NoBrush)
                    prev_left = prev_right = None
                    for row_idx in range(roi_h):
                        row_pixels = np.where(roi_mask[row_idx])[0]
                        if len(row_pixels) == 0:
                            prev_left = prev_right = None
                            continue
                        left_x = float(row_pixels[0])
                        right_x = float(row_pixels[-1])
                        abs_y = roi_top + row_idx
                        lx, ly = translate_coordinates_to_other_image(
                            (left_x, abs_y), self.image.size, (img_w, img_h)
                        )
                        rx, ry = translate_coordinates_to_other_image(
                            (right_x, abs_y), self.image.size, (img_w, img_h)
                        )
                        if prev_left is not None:
                            painter.drawLine(
                                QPoint(int(prev_left[0]), int(prev_left[1])),
                                QPoint(int(lx), int(ly)),
                            )
                            painter.drawLine(
                                QPoint(int(prev_right[0]), int(prev_right[1])),
                                QPoint(int(rx), int(ry)),
                            )
                        prev_left = (lx, ly)
                        prev_right = (rx, ry)

                # 3. Neck line (yellow-green dashed)
                print(
                    f"Segmentation debug draw: neck_y={sd.neck_y}, "
                    f"chin_y={sd.chin_y}, midline_x={sd.midline_x}, "
                    f"image.size={self.image.size}, display=({img_w},{img_h})"
                )
                if sd.neck_y is not None:
                    _, ny = translate_coordinates_to_other_image(
                        (0, sd.neck_y), self.image.size, (img_w, img_h)
                    )
                    neck_pen = QtGui.QPen(QColor(180, 255, 0, 200), 2, Qt.DashLine)
                    painter.setPen(neck_pen)
                    painter.drawLine(0, int(ny), img_w, int(ny))
                    painter.setPen(QColor(180, 255, 0))
                    font = painter.font()
                    font.setPointSize(9)
                    painter.setFont(font)
                    painter.drawText(5, int(ny) - 4, "neck")

                # 4. Chin line (yellow dashed)
                if sd.chin_y is not None:
                    _, cy = translate_coordinates_to_other_image(
                        (0, sd.chin_y), self.image.size, (img_w, img_h)
                    )
                    chin_pen = QtGui.QPen(QColor(255, 255, 0, 200), 2, Qt.DashLine)
                    painter.setPen(chin_pen)
                    painter.drawLine(0, int(cy), img_w, int(cy))
                    painter.setPen(QColor(255, 255, 0))
                    font = painter.font()
                    font.setPointSize(9)
                    painter.setFont(font)
                    painter.drawText(5, int(cy) - 4, "chin")

                # 5. Shoulder line (orange dashed)
                if sd.shoulder_y is not None:
                    _, sy = translate_coordinates_to_other_image(
                        (0, sd.shoulder_y), self.image.size, (img_w, img_h)
                    )
                    shoulder_pen = QtGui.QPen(QColor(255, 165, 0, 200), 2, Qt.DashLine)
                    painter.setPen(shoulder_pen)
                    painter.drawLine(0, int(sy), img_w, int(sy))
                    painter.setPen(QColor(255, 165, 0))
                    font = painter.font()
                    font.setPointSize(9)
                    painter.setFont(font)
                    painter.drawText(5, int(sy) - 4, "shoulder")

                # 5b. Ear/jaw level line (magenta dashed)
                if hasattr(sd, "ear_y") and sd.ear_y is not None:
                    _, ey = translate_coordinates_to_other_image(
                        (0, sd.ear_y), self.image.size, (img_w, img_h)
                    )
                    ear_pen = QtGui.QPen(QColor(255, 0, 255, 200), 2, Qt.DashLine)
                    painter.setPen(ear_pen)
                    painter.drawLine(0, int(ey), img_w, int(ey))
                    painter.setPen(QColor(255, 0, 255))
                    font = painter.font()
                    font.setPointSize(9)
                    painter.setFont(font)
                    painter.drawText(5, int(ey) - 4, "ear/jaw")

                # 6. ROI search band boundaries (faint lines at 25% and 75%)
                if sd.width_profile is not None and len(sd.width_profile) > 0:
                    roi_h = len(sd.width_profile)
                    band_25_y = sd.roi_top + int(roi_h * 0.25)
                    band_75_y = sd.roi_top + int(roi_h * 0.75)
                    band_pen = QtGui.QPen(QColor(255, 255, 255, 80), 1, Qt.DotLine)
                    painter.setPen(band_pen)
                    for band_y in (band_25_y, band_75_y):
                        _, by = translate_coordinates_to_other_image(
                            (0, band_y), self.image.size, (img_w, img_h)
                        )
                        painter.drawLine(0, int(by), img_w, int(by))

                painter.setBrush(Qt.NoBrush)

            self._paint_region_overlay(painter)
            painter.setPen(QColor(0, 0, 255, 127))

            # Calculate 2 points at the edge of the image, using the angle.

            p1 = findPoint(x, y, direction=-1, angle=angle)
            p2 = findPoint(x, y, direction=1, angle=angle)
            painter.drawLine(p1, p2)

            if self.last_click_x is not None:
                painter.setPen(QColor(255, 0, 0, 127))
                painter.drawLine(
                    QPoint(mouse_x, mouse_y),
                    QPoint(self.last_click_x, self.last_click_y),
                )
        finally:
            painter.end()
        self.ui.imageLabel.setPixmap(canvas)

        # Now the right image -- the depths:

        canvas = self.ui.chartLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        try:
            canvas.fill(Qt.red)

            if self.depthmap:
                point_beg = p2
                point_end = p1

                if p1.y() < p2.y():
                    point_beg = p1
                    point_end = p2

                for x, y in interpolate_pixels_along_line(point_beg.x(), 0, point_end.x(), 639):
                    painter.drawLine(
                        0,
                        y,
                        self.get_depthmap_value(x, y),
                        y,
                    )

                if self.last_click_x is not None:
                    painter.setPen(QColor(0, 255, 0, 127))

                    z1 = self.get_depthmap_value(mouse_x, mouse_y)
                    z2 = self.get_depthmap_value(self.last_click_x, self.last_click_y)

                    painter.drawLine(
                        QPoint(z1, mouse_y),
                        QPoint(z2, self.last_click_y),
                    )

                    values = []
                    for x, y in interpolate_pixels_along_line(
                        mouse_x, mouse_y, self.last_click_x, self.last_click_y
                    ):
                        values.append(self.get_depthmap_value(x, y))
                    self.paintReconstruction(values)

                # Draw chin and neck midpoint horizontal markers (extended neck only)
                if (
                    self.neck_midpoint
                    and self.neck_midpoint.x is not None
                    and self.neck_midpoint.pose == PortraitPose.EXTENDED_NECK
                ):
                    pink_pen = QtGui.QPen(QColor(255, 105, 180, 200), 1)
                    painter.setPen(pink_pen)
                    _, chin_dy = translate_coordinates_to_other_image(
                        self.neck_midpoint.chin,
                        self.image.size,
                        (img_w, img_h),
                    )
                    _, neck_dy = translate_coordinates_to_other_image(
                        (self.neck_midpoint.x, self.neck_midpoint.y),
                        self.image.size,
                        (img_w, img_h),
                    )
                    painter.drawLine(0, int(chin_dy), 255, int(chin_dy))
                    painter.drawLine(0, int(neck_dy), 255, int(neck_dy))
        finally:
            painter.end()
        self.ui.chartLabel.setPixmap(canvas)

        if self.depthmap:
            mouse_x = clamp(mouse_x, 0, img_w)
            mouse_y = clamp(mouse_y, 0, img_h)

            if self.last_click_x is None:
                line_len = 0
            else:
                line_x = abs(self.last_click_x - mouse_x)
                line_y = abs(self.last_click_y - mouse_y)
                line_len = self.calculate_line_length(line_x, line_y)

            surface_length_3d = vector_length_3d = 0
            filtered_surface_lengths = {step: 0.0 for step in range(2, 8)}
            if (
                self.last_click_x != mouse_x or self.last_click_y != mouse_y
            ) and self.last_click_x is not None:
                vector_length_3d = self.vector_length_between_two_clicks(
                    mouse_x, mouse_y, self.last_click_x, self.last_click_y
                )

                surface_length_3d = self.vector_length_surface(
                    mouse_x, mouse_y, self.last_click_x, self.last_click_y
                )

                filtered_surface_lengths = {
                    step: self.surface_vector_filtered(
                        mouse_x,
                        mouse_y,
                        self.last_click_x,
                        self.last_click_y,
                        step,
                    )
                    for step in range(2, 8)
                }

            self.last_click_x = mouse_x
            self.last_click_y = mouse_y

            closeness = self.depthmap.getpixel((mouse_x, mouse_y))[0]

            depth_mm = self.get_depthmap_distance(closeness)

            closeness_delta = 0
            closeness_delta_mm = 0.0

            if self.last_depth:
                closeness_delta = self.last_depth - closeness
                closeness_delta_mm = self.get_depthmap_distance(self.last_depth) - depth_mm

            self.last_depth = closeness

            self.last_5_distances_srfc.append(surface_length_3d)
            if len(self.last_5_distances_srfc) > 5:
                self.last_5_distances_srfc = self.last_5_distances_srfc[1:6]

            self.last_5_distances_vect.append(vector_length_3d)
            if len(self.last_5_distances_vect) > 5:
                self.last_5_distances_vect = self.last_5_distances_vect[1:6]

            self.ui.dataOutputEdit.clear()
            pose_label = self.neck_midpoint.pose.value.upper() if self.neck_midpoint else "UNKNOWN"
            txt = f"*** {pose_label} ***\n\n"
            txt += dedent(f"""\
            Single click:
              Depth map coords: {mouse_x, mouse_y}
              Depth map value: {closeness}
              Distance from camera: {depth_mm:.2f} cm

            Last two clicks:
              Delta: {closeness_delta} pixels, {closeness_delta_mm:.1f} cm
              Line length (2D): {line_len:.2f} px
              Vector length (3D): {vector_length_3d / 10.0:.2f} cm
              Surface length (3D): {(surface_length_3d / 10.0):.2f} cm
            """)
            txt += "\n  Surface vector filtered (median 3x3):"
            for step, length_mm in filtered_surface_lengths.items():
                if length_mm is None:
                    formatted_length = "n/a (invalid depth)"
                else:
                    formatted_length = f"{length_mm / 10.0:.2f} cm"
                txt += f"\n    N={step} px: {formatted_length}"
            txt += dedent(f"""

                Last 5 clicks:
                  Sum vector length (3D): {sum(self.last_5_distances_vect) / 10.0:.2f} cm
                  Sum surface length (3D): {sum(self.last_5_distances_srfc) / 10.0:.2f} cm\
                """)

            # Gate measurement display by pose
            pose = self.neck_midpoint.pose if self.neck_midpoint else None

            if pose == PortraitPose.OPEN_MOUTH:
                if self.portrait.teeth_bbox:
                    txt += "\n\nTeeth detected."
                    if self.portrait.incisor_distance:
                        txt += "\nAutomatic incisor distance (legacy): "
                        txt += "%.2f cm" % (
                            self.vector_length_between_two_clicks(
                                *auto_id_click_1, *auto_id_click_2
                            )
                            / 10.0
                        )
                    im = self.portrait.incisor_measurement
                    if im and im.distance_3d_mm is not None:
                        txt += "\nAutomatic incisor distance (centroid): "
                        txt += "%.2f cm" % (im.distance_3d_mm / 10.0)
                if self.mouth_measurement and self.mouth_measurement.distance_3d_mm is not None:
                    txt += "\n\n[FaceMesh] Mouth opening: "
                    txt += "%.2f cm" % (self.mouth_measurement.distance_3d_mm / 10.0)

            elif pose == PortraitPose.NEUTRAL_NECK:
                if self.neck_measurement_auto:
                    nma = self.neck_measurement_auto
                    txt += "\n\nAutomatic neck circumference (3D arc): "
                    txt += f"{nma.circumference_mm / 10.0:.2f} cm"
                    txt += f"\n  Front arc: {nma.front_arc_length_mm:.2f} mm"
                    txt += f"\n  Multiplier: {nma.circumference_multiplier:.1f}"
                    txt += f"\n  Arc points: {len(nma.arc_points_photo):d}"
                    txt += (
                        "\n  Skin → stable depth:"
                        f" L {nma.mask_left_x} → {nma.left_x} px,"
                        f" R {nma.mask_right_x} → {nma.right_x} px"
                    )

                    # Neck circumference approximation from diameter (left_x → right_x) * pi
                    left_display = translate_coordinates_to_other_image(
                        (nma.left_x, nma.neck_y),
                        self.portrait.skinmap.size,
                        (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT),
                    )
                    right_display = translate_coordinates_to_other_image(
                        (nma.right_x, nma.neck_y),
                        self.portrait.skinmap.size,
                        (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT),
                    )
                    diameter_mm = self.vector_length_between_two_clicks(
                        *left_display, *right_display
                    )
                    circumference_from_radius = diameter_mm * math.pi
                    txt += (
                        "\n\nNeck approximation from radius:"
                        f" {circumference_from_radius / 10.0:.2f} cm"
                        f"\n  Diameter (3D): {diameter_mm / 10.0:.2f} cm"
                    )

                if self.portrait.teethmap:
                    teeth_bin = self.portrait.teethmap.point(lambda v: 255 if v > 30 else 0)
                    bbox = teeth_bin.getbbox()
                    if bbox:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        # Require a meaningful teeth region (at least 20x10 pixels)
                        if bw >= 20 and bh >= 10:
                            txt += (
                                "\n\nMouth closed, neutral neck, teeth visible"
                                " -- possible positive ULBT test?"
                                f"\n  Teeth bbox: {bw}x{bh} px"
                            )

            elif pose == PortraitPose.EXTENDED_NECK:
                if self.segmentation_debug and self.segmentation_debug.neck_width_front_arc_mm:
                    txt += (
                        f"\n\nNeck width (3D front arc):"
                        f" {self.segmentation_debug.neck_width_front_arc_mm / 10:.2f} cm"
                    )
                    txt += (
                        f"\nNeck width (3D straight):"
                        f" {self.segmentation_debug.neck_width_straight_mm / 10:.2f} cm"
                    )

            if self.tmd_mm is not None:
                txt += f"\n\nThyromental distance (3D): {self.tmd_mm / 10.0:.2f} cm"

            # Portrait picture

            # Try neck measurement

            self.ui.dataOutputEdit.appendPlainText(txt.strip())

    def get_depthmap_value(self, x, y):
        x = int(clamp(x, 0, self.depthmap.size[0]))
        y = int(clamp(y, 0, self.depthmap.size[1]))
        return self.depthmap.getpixel((x, y))[0]

    def distance_for_click(self, x, y):
        return self.get_depthmap_distance(self.get_depthmap_value(x, y))

    def translate_click_to_mm(self, distance_cm, x, y):
        """Translate a display-space click to camera-centred X/Y millimetres.

        Pixel coordinates are measured from the top-left corner in display
        (480x640) space; portrait_analyser.pixel_to_mm() converts to physical
        millimetres centred on the image's principal point.
        """
        image_x = x * self.image.size[0] / const.MAIN_IMAGE_WIDTH
        image_y = y * self.image.size[1] / const.MAIN_IMAGE_HEIGHT

        return (
            pixel_to_mm(image_x, distance_cm, self.image.size[0]),
            pixel_to_mm(image_y, distance_cm, self.image.size[1]),
        )

    def click_to_3d(self, x, y):
        z_cm = self.distance_for_click(x, y)
        x, y = self.translate_click_to_mm(z_cm, x, y)
        z_mm = z_cm * 10
        return x, y, z_mm

    def vector_length_surface(
        self,
        mouse_x,
        mouse_y,
        last_click_x,
        last_click_y,
    ):
        """Calculate length iterating over the surface of 3D data"""

        pixels = list(interpolate_pixels_along_line(mouse_x, mouse_y, last_click_x, last_click_y))
        s = []

        for (x1, y1), (x2, y2) in zip(pixels, pixels[1:], strict=False):
            z1 = self.distance_for_click(x1, y1) * 10  # cm → mm
            x1, y1 = self.translate_click_to_mm(z1 / 10, x1, y1)

            z2 = self.distance_for_click(x2, y2) * 10  # cm → mm
            x2, y2 = self.translate_click_to_mm(z2 / 10, x2, y2)

            line_len_3d = self.vector_length_simple(x1, y1, z1, x2, y2, z2)
            s.append(line_len_3d)
        return sum(s)

    def surface_vector_filtered(
        self,
        mouse_x,
        mouse_y,
        last_click_x,
        last_click_y,
        step,
    ):
        """Measure a median-filtered surface profile every ``step`` pixels.

        Coordinates between samples are measured in the 480x640 display space,
        then mapped to photo-space for portrait_analyser's
        measure_filtered_surface_length(), which handles the fractional
        depth-map sampling and mm conversion. A zero disparity invalidates
        the measurement rather than introducing a foreground-to-background
        depth wall.
        """
        if step <= 0:
            raise ValueError("step must be greater than zero")

        if self.filtered_depthmap is None:
            self.filtered_depthmap = median_filter_depthmap(self.depthmap, size=3)

        photo_width, photo_height = self.image.size
        points_photo = [
            (
                x * photo_width / const.MAIN_IMAGE_WIDTH,
                y * photo_height / const.MAIN_IMAGE_HEIGHT,
            )
            for x, y in sample_points_along_line(
                mouse_x,
                mouse_y,
                last_click_x,
                last_click_y,
                step,
            )
        ]

        return measure_filtered_surface_length(
            self.filtered_depthmap,
            points_photo,
            photo_width,
            photo_height,
            self.float_min_value,
            self.float_max_value,
        )

    def vector_length_simple(self, x1, y1, z1, x2, y2, z2):
        """Simple mathematical lenght of the vector"""
        return vector_length_3d(x1, y1, z1, x2, y2, z2)

    def vector_length_between_two_clicks(self, x1, y1, x2, y2):
        z1 = self.get_depthmap_value(x1, y1)
        z2 = self.get_depthmap_value(x2, y2)

        distance_z1 = self.get_depthmap_distance(z1) * 10  # cm → mm
        distance_z2 = self.get_depthmap_distance(z2) * 10  # cm → mm

        distance_x1, distance_y1 = self.translate_click_to_mm(
            distance_z1 / 10,
            x1,
            y1,
        )
        distance_x2, distance_y2 = self.translate_click_to_mm(
            distance_z2 / 10,
            x2,
            y2,
        )
        return self.vector_length_simple(
            distance_x1,
            distance_y1,
            distance_z1,
            distance_x2,
            distance_y2,
            distance_z2,
        )

    def calculate_line_length(self, dist_x, dist_y):
        line_len = math.sqrt(abs(dist_x * dist_x) + abs(dist_y * dist_y))
        return line_len

    def _loadImage(self, fileName):
        self.filename = fileName
        self.clear_region_measurement(repaint=False)

        try:
            self.portrait: IOSPortrait = load_image(self.filename, use_exif=False)
            self.image = self.portrait.photo
            self.depthmap = self.portrait.depthmap
            self.filtered_depthmap = median_filter_depthmap(self.depthmap, size=3)
            self.teethmap = self.portrait.teethmap
            self.float_min_value = self.portrait.floatValueMin
            self.float_max_value = self.portrait.floatValueMax

            if self.float_max_value is not None:
                self.float_max_value = float(self.float_max_value)

            if self.float_min_value is not None:
                self.float_min_value = float(self.float_min_value)

        except ExifValidationFailed as e:
            QMessageBox.critical(
                self,
                tr("FIDMAA notification"),
                errors.NO_FRONT_CAMERA_NOTIFICATION.format(exif_camera_description=e),
            )
            return

        except NoDepthMapFound:
            self.critical_error(errors.NO_DEPTH_DATA_ERROR)
            return

        except UnknownExtension as e:
            self.critical_error(QObject.tr(f"Unknown file extension ({e})"))
            return

        self.smallImage = self.image.resize((const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT))

        self._build_hairless_depth_image()

        self._run_detection_and_update(force_extended=False)

    def _run_detection_and_update(self, force_extended=False):
        """Run face/neck detection and update UI accordingly.

        When force_extended is True, skip MediaPipe and use only the
        segmentation/depth-based extended neck analysis.
        """
        self.neck_measurement_auto = None
        self.neck_midpoint = None
        self.mediapipe_debug = None
        self.face_mesh_debug = None
        self.tmd_mm = None
        self.mouth_measurement = None
        self.segmentation_debug = None

        img_w = self.image.size[0]

        if not force_extended:
            # Detect MediaPipe pose landmarks (mouth + shoulders -> search bounds)
            try:
                self.neck_midpoint, self.mediapipe_debug, self.face_mesh_debug = (
                    detect_neck_midpoint(self.image)
                )
                if self.neck_midpoint:
                    print(f"Neck midpoint detected: {self.neck_midpoint}")
                    print(f"  x={self.neck_midpoint.x}, y={self.neck_midpoint.y}")
            except Exception as e:
                print(f"detect_neck_midpoint failed: {type(e).__name__}: {e}")
                self.neck_midpoint = None
                self.mediapipe_debug = None
                self.face_mesh_debug = None

        if not self.neck_midpoint:
            try:
                from portrait_analyser.extended_neck import (
                    detect_neck_midpoint_from_dual_mask,
                    detect_neck_midpoint_from_segmentation,
                )

                if self.portrait.skinmap is not None and self.portrait.depthmap is not None:
                    self.neck_midpoint, self.segmentation_debug = (
                        detect_neck_midpoint_from_dual_mask(
                            self.image,
                            self.portrait.skinmap,
                            self.portrait.depthmap,
                            hairmap=self.portrait.hairmap,
                            float_min=self.float_min_value,
                            float_max=self.float_max_value,
                        )
                    )

                if not self.neck_midpoint:
                    self.neck_midpoint, self.segmentation_debug = (
                        detect_neck_midpoint_from_segmentation(self.image)
                    )
            except Exception as e:
                print(f"Segmentation fallback failed: {e}")

        if not self.neck_midpoint:
            self.critical_error(errors.FACE_NOT_DETECTED)

        if self.neck_midpoint:
            pose = self.neck_midpoint.pose

            neck_arc_available = pose == PortraitPose.NEUTRAL_NECK
            self.ui.showNeckArcCheckBox.setEnabled(neck_arc_available)
            self.ui.showNeckArcCheckBox.setChecked(neck_arc_available)
            self.ui.showNeckArcCheckBox.setToolTip(
                (
                    "Yellow: skin-matte boundary; cyan: first stable depth sample; "
                    "green: measured neck arc"
                )
                if neck_arc_available
                else "Neck edge measurement is only available for neutral neck pose"
            )

            centroids_available = pose == PortraitPose.OPEN_MOUTH
            self.ui.showCentroidsCheckBox.setEnabled(centroids_available)
            self.ui.showCentroidsCheckBox.setChecked(centroids_available)
            self.ui.showCentroidsCheckBox.setToolTip(
                ""
                if centroids_available
                else "Teeth centroids are only available for open mouth pose"
            )
        else:
            self.ui.showNeckArcCheckBox.setEnabled(False)
            self.ui.showNeckArcCheckBox.setChecked(False)
            self.ui.showNeckArcCheckBox.setToolTip("No face/pose detected in this picture")

            self.ui.showCentroidsCheckBox.setEnabled(False)
            self.ui.showCentroidsCheckBox.setChecked(False)
            self.ui.showCentroidsCheckBox.setToolTip("No face/pose detected in this picture")

        # Enable/disable debug checkboxes based on available data
        has_pose = self.mediapipe_debug is not None
        self.ui.showMediaPipeLandmarksCheckBox.setEnabled(has_pose)
        if not has_pose:
            self.ui.showMediaPipeLandmarksCheckBox.setChecked(False)
            self.ui.showMediaPipeLandmarksCheckBox.setToolTip(
                "Pose landmarks were not detected in this picture"
            )
        else:
            self.ui.showMediaPipeLandmarksCheckBox.setToolTip("")

        has_face_mesh = self.face_mesh_debug is not None
        self.ui.showFaceMeshLandmarksCheckBox.setEnabled(has_face_mesh)
        if not has_face_mesh:
            self.ui.showFaceMeshLandmarksCheckBox.setChecked(False)
            self.ui.showFaceMeshLandmarksCheckBox.setToolTip(
                "FaceMesh data were not detected in this picture"
            )
        else:
            self.ui.showFaceMeshLandmarksCheckBox.setToolTip("")

        has_segmentation = self.segmentation_debug is not None
        self.ui.showSegmentationDebugCheckBox.setEnabled(has_segmentation)
        if not has_segmentation:
            self.ui.showSegmentationDebugCheckBox.setChecked(False)
            self.ui.showSegmentationDebugCheckBox.setToolTip(
                "Segmentation data were not detected in this picture"
            )
        else:
            self.ui.showSegmentationDebugCheckBox.setToolTip("")

        # Enable the force button now that an image is loaded
        self.ui.forceExtendedNeckButton.setEnabled(True)

        # Set initial measurement point from MediaPipe nose/mouth
        if self.neck_midpoint:
            nose_x, nose_y = self.neck_midpoint.nose
            self.ui.xValue.setValue(
                int(round(nose_x / self.image.size[0] * (const.MAIN_IMAGE_WIDTH - 1)))
            )
            mouth_y = max(
                self.neck_midpoint.mouth_left[1],
                self.neck_midpoint.mouth_right[1],
            )
            self.ui.yValue.setValue(
                int(round(mouth_y / self.image.size[1] * (const.MAIN_IMAGE_HEIGHT - 1)))
            )

        # Compute MediaPipe search bounds: mouth -> neck midpoint.
        # The narrowest neck is between the chin and mid-cervical level,
        # NOT between mouth and shoulders (that range includes the collar
        # line where skin disappears -- the global minimum, not the neck).
        scan_start_y = None
        scan_end_y = None
        if self.neck_midpoint and self.neck_midpoint.y is not None:
            mouth_y = max(
                self.neck_midpoint.mouth_left[1],
                self.neck_midpoint.mouth_right[1],
            )
            scan_start_y = round(mouth_y)
            scan_end_y = round(self.neck_midpoint.y)

        if (
            self.portrait.skinmap
            and self.portrait.depthmap
            and self.neck_midpoint
            and self.neck_midpoint.x is not None
            and self.neck_midpoint.pose == PortraitPose.NEUTRAL_NECK
        ):
            self.neck_measurement_auto = compute_neck_circumference(
                skinmap=self.portrait.skinmap,
                depthmap=self.portrait.depthmap,
                photo_width=img_w,
                photo_height=self.image.size[1],
                float_min=self.portrait.floatValueMin,
                float_max=self.portrait.floatValueMax,
                scan_start_y=scan_start_y,
                scan_end_y=scan_end_y,
                neck_midpoint_y=self.neck_midpoint.y,
                circumference_multiplier=2.7,
                n_samples=10,
            )

        # Compute thyromental distance (chin -> neck midpoint)
        if (
            self.neck_midpoint
            and self.portrait.depthmap
            and self.neck_midpoint.x is not None
            and self.neck_midpoint.pose in (PortraitPose.EXTENDED_NECK, PortraitPose.NEUTRAL_NECK)
        ):
            chin_x, chin_y = self.neck_midpoint.chin
            neck_x, neck_y = self.neck_midpoint.x, self.neck_midpoint.y
            dm_w, dm_h = self.portrait.depthmap.size
            ph_w, ph_h = self.image.size
            # Translate photo-space coords to depthmap-space for depth sampling
            chin_dm = translate_coordinates_to_other_image(
                (chin_x, chin_y), (ph_w, ph_h), (dm_w, dm_h)
            )
            neck_dm = translate_coordinates_to_other_image(
                (neck_x, neck_y), (ph_w, ph_h), (dm_w, dm_h)
            )
            chin_depth = self.portrait.depthmap.getpixel(
                (
                    int(clamp(chin_dm[0], 0, dm_w - 1)),
                    int(clamp(chin_dm[1], 0, dm_h - 1)),
                )
            )[0]
            neck_depth = self.portrait.depthmap.getpixel(
                (
                    int(clamp(neck_dm[0], 0, dm_w - 1)),
                    int(clamp(neck_dm[1], 0, dm_h - 1)),
                )
            )[0]
            tmd_result = compute_tmd_3d(
                chin_coord=(chin_x, chin_y),
                neck_coord=(neck_x, neck_y),
                chin_depth_raw=chin_depth,
                neck_depth_raw=neck_depth,
                float_min=self.portrait.floatValueMin,
                float_max=self.portrait.floatValueMax,
                image_width=ph_w,
                image_height=ph_h,
            )
            if tmd_result:
                self.tmd_mm = tmd_result[0]
                print(f"TMD (3D): {self.tmd_mm / 10:.2f} cm")
            else:
                print("TMD computation failed (depth unavailable)")

        # FaceMesh mouth opening fallback: when pose is OPEN_MOUTH but
        # teethmap-based incisor detection failed (e.g. no upper teeth)
        if (
            self.neck_midpoint
            and self.neck_midpoint.pose == PortraitPose.OPEN_MOUTH
            and self.portrait.incisor_measurement is None
            and self.face_mesh_debug
            and self.portrait.depthmap
        ):
            self.mouth_measurement = compute_mouth_measurement_from_facemesh(
                landmarks=self.face_mesh_debug.landmarks,
                depthmap=self.portrait.depthmap,
                photo_w=self.image.size[0],
                photo_h=self.image.size[1],
                float_min=self.portrait.floatValueMin,
                float_max=self.portrait.floatValueMax,
            )
            if self.mouth_measurement and self.mouth_measurement.distance_3d_mm is not None:
                print(
                    f"FaceMesh mouth opening (3D): "
                    f"{self.mouth_measurement.distance_3d_mm / 10:.2f} cm"
                )
            else:
                print("FaceMesh mouth measurement: depth conversion failed")

        self.last_click_y = self.last_click_x = None
        self.last_clicks = []
        self.redrawImage()
        self.updateWindowTitle()

    def forceExtendedNeckAnalysis(self):
        """Re-run detection using only segmentation/depth method, skipping MediaPipe."""
        if not hasattr(self, "image") or self.image is None:
            return
        print("--- Force extended neck analysis ---")
        self._run_detection_and_update(force_extended=True)

    def getWindowTitle(self, fileName=None, fun=None):
        ret = "FIDMAA GUI"
        if fileName:
            ret += " - " + fileName
        if fun:
            ret += " - " + fun
        return ret

    def updateWindowTitle(self):
        fn = self.filename
        if fn is not None:
            fn = os.path.basename(fn)
        self.setWindowTitle(self.getWindowTitle(fn))

    def critical_error(self, err):
        QMessageBox.critical(
            self,
            tr("FIDMAA error"),
            tr(err),
            QMessageBox.Cancel,
        )

    def loadJPEG(self, *args, **kw):
        settings = QSettings("FIDMAA - open file")
        last_directory_used = settings.value(
            const.LAST_DIRECTORY_USED, os.path.expanduser("~/Downloads")
        )
        if self.filename:
            last_directory_used = self.filename

        dlg = QFileDialog(
            self,
            QObject.tr("Open File"),
            last_directory_used,
            QObject.tr("Images (*.heic)"),
        )
        dlg.setFileMode(QFileDialog.ExistingFile)

        ret = dlg.exec_()
        if ret and len(dlg.selectedFiles()) == 1:
            fileName = dlg.selectedFiles()[0]
            settings.setValue(const.LAST_DIRECTORY_USED, os.path.dirname(fileName))
            self._loadImage(fileName)

    def setMidlinePoint(self, point, *args, **kw):
        if self._region_target is not None:
            return
        self.ui.xValue.setValue(point.x())
        self.ui.yValue.setValue(point.y())
        self.redrawImage()

    def setMidlineY(self, point, *args, **kw):
        # self.ui.xValue.setValue(point.x())
        self.ui.yValue.setValue(point.y())
        self.redrawImage()

    def open3DView(self):
        import multiprocessing

        new_depthmap = self.depthmap.convert("L")

        p = multiprocessing.Process(
            target=_show_3d_view,
            args=(
                self.image,
                new_depthmap,
                self.float_min_value,
                self.float_max_value,
            ),
        )
        p.start()
        self._3d_process = p

    def closeEvent(self, event):
        if hasattr(self, "_3d_process") and self._3d_process.is_alive():
            self._3d_process.terminate()
            self._3d_process.join(timeout=2)
        super().closeEvent(event)

    def connect_ui(self):
        canvas = QtGui.QPixmap(const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT)
        self.ui.imageLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(const.DEPTH_CHART_WIDTH, const.DEPTH_CHART_HEIGHT)
        self.ui.chartLabel.setPixmap(canvas)

        self.ui.loadJPEGButton.clicked.connect(self.loadJPEG)
        self.ui.open3DViewButton.clicked.connect(self.open3DView)
        self.ui.imageLabel.clicked.connect(self.setMidlinePoint)
        self.ui.imageLabel.setMouseTracking(True)
        self.ui.imageLabel.pointerMoved.connect(self.redrawZoom)
        self.ui.imageLabel.pointerMoved.connect(self._update_region_cursor)
        self.ui.imageLabel.pressed.connect(self._begin_region_drag)
        self.ui.imageLabel.dragged.connect(self._drag_region)
        self.ui.imageLabel.released.connect(self._finish_region_drag)
        self.ui.imageLabel.setCursor(Qt.CursorShape.CrossCursor)
        self.ui.chartLabel.clicked.connect(self.setMidlineY)

        self.ui.angleValue.valueChanged.connect(self.redrawImage)
        self.ui.showCentroidsCheckBox.stateChanged.connect(self.redrawImage)
        self.ui.showNeckArcCheckBox.stateChanged.connect(self.redrawImage)
        self.ui.showMediaPipeLandmarksCheckBox.stateChanged.connect(self.redrawImage)
        self.ui.showFaceMeshLandmarksCheckBox.stateChanged.connect(self.redrawImage)
        self.ui.showSegmentationDebugCheckBox.stateChanged.connect(self.redrawImage)
        self.ui.forceExtendedNeckButton.clicked.connect(self.forceExtendedNeckAnalysis)

        self.ui.angleValue.setValue(90)
        self.ui.angleSlider.setValue(90)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FIDMAA GUI")
    app.setApplicationDisplayName("FIDMAA GUI")

    mainWindow = MainWindow()
    mainWindow.updateWindowTitle()
    mainWindow.show()

    screen = app.primaryScreen().geometry()
    mainWindow.move(
        (screen.width() - mainWindow.width()) // 2,
        (screen.height() - mainWindow.height()) // 2,
    )

    try:
        if sys.argv[1]:
            mainWindow._loadImage(os.path.expanduser(sys.argv[1]))
    except IndexError:
        mainWindow.loadJPEG()

    sys.exit(app.exec())
