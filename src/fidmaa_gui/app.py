import importlib
import logging
import math
import os
import signal
import sys
from functools import partial

import numpy as np
import portrait_analyser
from PIL import ImageDraw, ImageFile
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
    raw_depth_to_distance_cm,
    vector_length_3d,
)
from portrait_analyser.ios import IOSPortrait, load_image
from portrait_analyser.mouth import compute_mouth_measurement_from_facemesh
from portrait_analyser.neck import (
    compute_neck_circumference,
    neck_search_bounds_from_face_landmarks,
)
from portrait_analyser.neck_width import EDGE_INSET_MM
from portrait_analyser.pose import PortraitPose, detect_neck_midpoint
from PySide6 import QtGui
from PySide6.QtCore import QObject, QPoint, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from . import const, errors
from .arc_circle import (
    ARC_CIRCLE_FLAT,
    ARC_CIRCLE_FLAT_TEXT,
    ARC_CIRCLE_OK,
    arc_circle,
    inset_polyline_ends,
)
from .calculations import findPoint, lower_jaw_x_bounds, neck_to_jaw_width_ratio
from .depth_source import (
    IMPLAUSIBLE_DEPTH_NA,
    combine_warnings,
    depth_is_implausible,
    depth_source_status,
    depth_zero_is_invalid,
    float_depth,
    portrait_camera,
)
from .depth_visualization import (
    DepthDisplayMode,
    boost_matte,
    matte_display_high,
    render_depth_contour_overlay,
    render_depth_visualization,
)
from .measurement_panel import (
    MeasurementPanel,
    format_cm,
    format_decimal,
    format_neck_circumference,
)
from .measurement_tools import (
    HistoryEntry,
    MeasurementTool,
    PolylineEditor,
    measure_polyline,
    neck_history_surface_mm,
)
from .neck_arc import compute_neck_arc
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

logger = logging.getLogger(__name__)

# portrait_analyser NeckWidthResult.status on success.
NECK_WIDTH_OK = "ok"
NECK_TOOL_LEGACY_NA = "n/a (needs capture-app photo)"
NECK_TOOL_LEGACY_DETAIL = "Neck tool needs a TrueDepth capture-app photo"
# Fixed order for every Neck tool state, so rows never swap while clicking.
NECK_TOOL_VALUE_LABELS = ("Circumference (arc circle):", "Width (3D):")
NECK_TOOL_NEEDS_3_POINTS = "— (needs ≥ 3 points)"
# The arc circle's chord and arc start this far inside the clicked edges, the
# same inset the library reads the width's edge depth at: within ~1 mm of the
# silhouette the depth mixes with the background (see the "Width bias" notes
# in portrait_analyser.neck_width).
NECK_ARC_EDGE_INSET_MM = EDGE_INSET_MM
NECK_TOOL_HINT_SHORT = (
    "Use the Neck tool (N): click the left neck edge, points across the front of the neck "
    "and the right edge."
)


# The library's no-camera messages address programmers ("pass camera=").
NECK_WIDTH_NO_CAMERA_TEXT = (
    "No camera calibration data in this file; a metric neck width needs the camera intrinsics"
)
NECK_WIDTH_UNUSABLE_CAMERA_TEXT = (
    "The file's camera calibration data are not usable here (depth not absolute, "
    "implausible or not aligned with the photo)"
)
NECK_WIDTH_RELATIVE_DEPTH_TEXT = "Neck width needs absolute TrueDepth depth"
NECK_WIDTH_QUALITY_LOW = "low"
# Statuses where clicking the edges by hand (Neck tool) can still help.
NECK_WIDTH_TOOL_HELPS = ("edges-occluded", "no-face")


def _library_advice_texts():
    """portrait_analyser.neck_width's ADVICE_* sentences, longest first.

    Empty for library versions without them (then the whole message is shown).
    """
    try:
        module = importlib.import_module("portrait_analyser.neck_width")
    except ImportError:
        # Older portrait-analyser (e.g. the pinned 0.8.0) has no neck width
        # module; there is no advice to pick out, which is not an error.
        return []
    advice = [v for k, v in vars(module).items() if k.startswith("ADVICE_") and isinstance(v, str)]
    return sorted(advice, key=len, reverse=True)


def _sentence(text):
    text = str(text).replace(" -- ", " — ")
    return text[:1].upper() + text[1:]


def neck_width_message(result, portrait=None, *, full=False):
    """User-facing text of a NeckWidthResult's ``message``.

    The library writes e.g. "neck edges hidden by the collar or beard -- open
    or lower the collar ... (52 rows with skin edges; ... 2 mm needed)"; the
    panel shows only the leading advice sentence ("Neck edges hidden by the
    collar or beard — open or lower the collar ..."), ``full=True`` (the
    diagnostics) the whole message. ``no-camera`` distinguishes missing
    intrinsics from intrinsics the library could not use (the file has a
    focal length, as the library checks).
    """
    status = getattr(result, "status", None)
    if status == "no-camera":
        if getattr(portrait, "focal_length_px", None) is not None:
            return NECK_WIDTH_UNUSABLE_CAMERA_TEXT
        return NECK_WIDTH_NO_CAMERA_TEXT
    if status == "relative-depth":
        return NECK_WIDTH_RELATIVE_DEPTH_TEXT
    message = str(getattr(result, "message", None) or status or "unknown")
    if not full and status == "edges-occluded":
        for advice in _library_advice_texts():
            if message.startswith(advice):
                return _sentence(advice)
    return _sentence(message)


def neck_width_quality_warning(result):
    """ "Neck width quality: low — reason; reason", or None when not low."""
    if getattr(result, "quality", None) != NECK_WIDTH_QUALITY_LOW:
        return None
    reasons = getattr(result, "quality_reasons", None) or []
    text = "Neck width quality: low"
    if reasons:
        text += " — " + "; ".join(reasons)
    return text


def neck_width_other_warnings(result):
    """Library warnings not already listed as quality reasons."""
    reasons = set(getattr(result, "quality_reasons", None) or [])
    return [w for w in (getattr(result, "warnings", None) or []) if w not in reasons]


def neck_width_position_text(result, *, quality=True):
    """ "measured 1,2 cm below the chin · quality: good" (parts left out when
    unknown; ``quality=False`` leaves the quality out)."""
    parts = []
    height_mm = getattr(result, "height_below_chin_mm", None)
    if height_mm is not None:
        parts.append(f"measured {format_cm(height_mm, 1)} below the chin")
    quality = getattr(result, "quality", None) if quality else None
    if quality is not None:
        parts.append(f"quality: {quality}")
    return " · ".join(parts) if parts else None


tr = QObject.tr

IMPLAUSIBLE_DEPTH_DETAIL = "Depth looks implausible — measurements disabled."


def _show_3d_view(image, depthmap, float_min, float_max):
    """Show 3D view in a separate process to avoid VTK/Qt conflicts on macOS."""
    from fidmaa_simple_viewer.core import pyvista_show

    pyvista_show(image, depthmap, float_min, float_max)


def window_float_depth(window):
    """Float ``DepthMap`` of the window's photo (capture-app files), or None.

    Module-level (not only a method) so the measurement helpers also work on
    the lightweight stand-in windows used by unit tests.
    """
    return float_depth(getattr(window, "portrait", None))


def window_integration_depth(window):
    """The library's integration map of the window's float depth, or None.

    ``portrait.depth.integration_map(camera)``: portrait-analyser's own
    smoothing for anything integrated along a path or region on float depth
    (3x3 median + edge-preserving 2 mm Gaussian, cached by the library). The
    GUI never filters float depth itself.
    """
    depth = window_float_depth(window)
    if depth is None:
        return None
    return depth.integration_map(portrait_camera(window))


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
        self.current_tool = MeasurementTool.VECTOR
        self.polyline_editor = PolylineEditor(max_points=2)
        self._tool_history_index = None
        self._point_drag_index = None
        self._suppress_next_click = False
        self.load_ui()
        self._create_zoom_panel()
        self._create_measurement_panel()
        self._create_toolbar_and_menus()
        self._create_analysis_menu()
        self._create_advanced_menu()
        self._create_face_mesh_menu()

        self.filename = None

        self.smallImage = None
        self.portrait: IOSPortrait = None
        self.depthmap = None
        self.filtered_depthmap = None
        self.teethmap = None
        self.neck_measurement_auto = None
        self.neck_search_bounds = None
        self.neck_lateral_bounds = None
        self.neck_confidence_ratio = None
        self.neck_arc = None
        self.neck_midpoint = None
        # portrait.neck_width (capture-app photos): a NeckWidthResult, or
        # None (legacy photo / not computed); neck_width_error holds the text
        # of an unexpected library failure shown in the panel instead.
        self.neck_width_auto = None
        self.neck_width_error = None
        self.mouth_measurement = None
        self.segmentation_debug = None
        # _compute_mouth_measurement() reads these; it can run (via the
        # mouth-opening opt-in toggle) before any photo has been detected.
        self.mediapipe_debug = None
        self.face_mesh_debug = None

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

        self.set_tool(MeasurementTool.VECTOR)

        self.redrawImage()
        self.redrawZoom()
        self._ui_ready = True

    def _create_measurement_panel(self):
        self.measurementPanel = MeasurementPanel()
        self.measurementPanel.historyCleared.connect(self._on_history_cleared)

        self.lineOptionsWidget = QGroupBox("Surface sampling")
        line_form = QFormLayout(self.lineOptionsWidget)
        self.surfaceStepSpin = QSpinBox()
        self.surfaceStepSpin.setRange(2, 7)
        self.surfaceStepSpin.setValue(3)
        self.surfaceStepSpin.setSuffix(" px")
        self.surfaceStepSpin.setToolTip(
            "Median-filtered surface profile is sampled every N display pixels."
        )
        line_form.addRow("Step:", self.surfaceStepSpin)
        self.surfaceStepSpin.valueChanged.connect(self._update_tool_result)
        self.measurementPanel.optionsLayout.addWidget(self.lineOptionsWidget)

        self.regionOptionsWidget = QWidget()
        layout = QVBoxLayout(self.regionOptionsWidget)
        layout.setContentsMargins(0, 0, 0, 0)

        regions_group = QGroupBox("Regions")
        regions_grid = QGridLayout(regions_group)
        regions_grid.addWidget(QLabel("Selector"), 0, 1)
        regions_grid.addWidget(QLabel("Mask"), 0, 2)
        self.regionModeACombo, self.regionMaskACombo = self._create_region_options(
            SelectionMode.LOCAL_PEAK
        )
        self.regionModeBCombo, self.regionMaskBCombo = self._create_region_options(
            SelectionMode.LOCAL_VALLEY
        )
        for row, (name, mode_combo, mask_combo) in enumerate(
            (
                ("A", self.regionModeACombo, self.regionMaskACombo),
                ("B", self.regionModeBCombo, self.regionMaskBCombo),
            ),
            start=1,
        ):
            regions_grid.addWidget(QLabel(name), row, 0)
            regions_grid.addWidget(mode_combo, row, 1)
            regions_grid.addWidget(mask_combo, row, 2)
        regions_grid.setColumnStretch(1, 2)
        regions_grid.setColumnStretch(2, 1)
        layout.addWidget(regions_group)

        settings_group = QGroupBox("Sampling")
        settings_form = QFormLayout(settings_group)
        self.regionRadiusSpin = QSpinBox()
        self.regionRadiusSpin.setRange(10, 120)
        self.regionRadiusSpin.setValue(20)
        self.regionRadiusSpin.setSuffix(" px")
        self.regionRadiusSpin.setToolTip(
            "Shared radius of both click-to-place measurement patches."
        )
        settings_form.addRow("Patch radius:", self.regionRadiusSpin)
        self.regionPercentileSpin = QSpinBox()
        self.regionPercentileSpin.setRange(5, 20)
        self.regionPercentileSpin.setValue(5)
        self.regionPercentileSpin.setSuffix(" %")
        settings_form.addRow("Candidate pool:", self.regionPercentileSpin)
        self.regionVectorCountSpin = QSpinBox()
        self.regionVectorCountSpin.setRange(5, 30)
        self.regionVectorCountSpin.setValue(10)
        settings_form.addRow("Profiles:", self.regionVectorCountSpin)
        layout.addWidget(settings_group)

        clear_button = QPushButton("Clear regions")
        clear_button.clicked.connect(self.clear_region_measurement)
        layout.addWidget(clear_button)

        self.regionResultEdit = QPlainTextEdit()
        self.regionResultEdit.setReadOnly(True)
        self.regionResultEdit.setMinimumWidth(270)
        self.regionResultEdit.setMinimumHeight(120)
        self.regionResultEdit.setPlainText(
            "Click once to place region A, then once to place region B."
        )
        layout.addWidget(self.regionResultEdit, stretch=1)
        self.measurementPanel.optionsLayout.addWidget(self.regionOptionsWidget)

        # Scroll instead of squeezing: the panel's option groups and history
        # need more height than the window may offer.
        self.measurementPanelScroll = QScrollArea()
        self.measurementPanelScroll.setWidget(self.measurementPanel)
        self.measurementPanelScroll.setWidgetResizable(True)
        self.measurementPanelScroll.setFrameShape(QFrame.NoFrame)
        self.measurementPanelScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.measurementPanelScroll.setMinimumWidth(
            self.measurementPanel.minimumWidth()
            + self.measurementPanelScroll.verticalScrollBar().sizeHint().width()
        )
        column = self.ui.findChild(QVBoxLayout, "verticalLayout")
        column.insertWidget(column.indexOf(self.ui.dataOutputEdit), self.measurementPanelScroll)

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

    def _create_region_options(self, default_mode):
        mode_combo = QComboBox()
        for mode in SelectionMode:
            mode_combo.addItem(mode.value, mode)
        mode_tooltips = {
            SelectionMode.AREA: (
                "Ignore shape/depth ranking and sample uniformly across the whole patch."
            ),
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
        return mode_combo, mask_combo

    def _create_toolbar_and_menus(self):
        self.toolBar = QToolBar("Tools", self)
        self.toolBar.setObjectName("mainToolBar")
        self.toolBar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.openAction = QAction("Open…", self)
        self.openAction.setShortcut(QtGui.QKeySequence.Open)
        self.openAction.triggered.connect(self.loadJPEG)
        self.open3DAction = QAction("3D view", self)
        self.open3DAction.triggered.connect(self.open3DView)
        self.open3DAction.setEnabled(False)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.openAction)
        file_menu.addAction(self.open3DAction)

        self.toolBar.addAction(self.openAction)
        self.toolBar.addAction(self.open3DAction)
        self.toolBar.addSeparator()

        tools_menu = self.menuBar().addMenu("&Tools")
        self.toolActionGroup = QActionGroup(self)
        self.toolActionGroup.setExclusive(True)
        self.toolActions = {}
        tool_details = (
            (MeasurementTool.VECTOR, Qt.Key_V, "Vector: two clicks, linear and surface length"),
            (
                MeasurementTool.POLYLINE,
                Qt.Key_M,
                "Polyline: click points, finish with Enter or right-click",
            ),
            (MeasurementTool.REGION, Qt.Key_R, "Region: two patches A and B"),
            (
                MeasurementTool.NECK,
                Qt.Key_N,
                "Neck: click the left neck edge, across the front, and the right edge; "
                "finish with Enter (capture-app photos)",
            ),
        )
        for tool, key, tip in tool_details:
            action = QAction(tool.value, self)
            action.setCheckable(True)
            action.setShortcut(QtGui.QKeySequence(key))
            action.setShortcutContext(Qt.WindowShortcut)
            action.setStatusTip(tip)
            action.setToolTip(tip)
            action.triggered.connect(partial(self.set_tool, tool))
            self.toolActionGroup.addAction(action)
            self.toolActions[tool] = action
            tools_menu.addAction(action)
            self.toolBar.addAction(action)

        tools_menu.addSeparator()
        action_details = (
            (SelectionMode.HIGHEST, Qt.Key_H, "Highest region tool"),
            (SelectionMode.LOWEST, Qt.Key_L, "Lowest region tool"),
            (SelectionMode.FLATTEST, Qt.Key_F, "Flattest region tool"),
            (SelectionMode.LOCAL_PEAK, None, "Local peak region tool"),
            (SelectionMode.LOCAL_VALLEY, None, "Local valley region tool"),
            (SelectionMode.AREA, None, "Uniform area region tool"),
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

        self.toolBar.addSeparator()
        self.toolBar.addWidget(QLabel(" Depth: "))
        self.depthDisplayCombo = QComboBox()
        for mode in DepthDisplayMode:
            self.depthDisplayCombo.addItem(mode.value, mode)
        self.depthDisplayCombo.setCurrentIndex(
            self.depthDisplayCombo.findData(DepthDisplayMode.COLOR)
        )
        self.depthDisplayCombo.setToolTip(
            "Display-only enhancement; measurement data is never remapped."
        )
        self.toolBar.addWidget(self.depthDisplayCombo)
        self.depthContourStepSpin = QSpinBox()
        self.depthContourStepSpin.setRange(1, 8)
        self.depthContourStepSpin.setValue(4)
        self.depthContourStepSpin.setSuffix(" levels")
        self.depthContourStepSpin.setToolTip(
            "Draw one contour for every N raw depth levels. Measurements are unchanged."
        )
        self.depthContourStepSpin.setEnabled(False)
        self.toolBar.addWidget(self.depthContourStepSpin)

        self.viewMenu = self.menuBar().addMenu("&View")
        depth_menu = self.viewMenu.addMenu("Depth display")
        depth_group = QActionGroup(self)
        depth_group.setExclusive(True)
        self.depthDisplayActions = {}
        for mode in DepthDisplayMode:
            action = QAction(mode.value, self)
            action.setCheckable(True)
            action.triggered.connect(
                partial(
                    lambda m, *args: self.depthDisplayCombo.setCurrentIndex(
                        self.depthDisplayCombo.findData(m)
                    ),
                    mode,
                )
            )
            depth_group.addAction(action)
            depth_menu.addAction(action)
            self.depthDisplayActions[mode] = action
        self.depthDisplayActions[DepthDisplayMode.COLOR].setChecked(True)

        self.boostTeethMapAction = self.viewMenu.addAction("Boost teeth map (display only)")
        self.boostTeethMapAction.setCheckable(True)
        self.boostTeethMapAction.setChecked(True)
        self.boostTeethMapAction.setStatusTip(
            "Brighten weak teeth-matte confidence in the zoom view; measurements use raw values"
        )
        # redrawZoom treats a positional argument as a pointer event.
        self.boostTeethMapAction.toggled.connect(lambda *_: self.redrawZoom())

        self.neckWidthOverlayAction = self.viewMenu.addAction("Neck width overlay")
        self.neckWidthOverlayAction.setCheckable(True)
        self.neckWidthOverlayAction.setChecked(True)
        self.neckWidthOverlayAction.setStatusTip(
            "Show the automatic neck width row and its edges (capture-app photos)"
        )
        self.neckWidthOverlayAction.toggled.connect(lambda *_: self._redraw_tool_overlay())

        self.depthDisplayCombo.currentIndexChanged.connect(self._depth_display_changed)
        self.depthContourStepSpin.valueChanged.connect(self._depth_display_changed)

    def _create_analysis_menu(self):
        analysis_menu = self.menuBar().addMenu("&Analysis")
        self.forceExtendedNeckAction = analysis_menu.addAction("Force extended neck analysis")
        self.forceExtendedNeckAction.setStatusTip(
            "Skip MediaPipe and re-run analysis using segmentation/depth profile method"
        )
        self.forceExtendedNeckAction.triggered.connect(self.forceExtendedNeckAnalysis)
        analysis_menu.addSeparator()
        self.mouthOpeningAction = analysis_menu.addAction(
            "Mouth opening from FaceMesh (experimental)"
        )
        self.mouthOpeningAction.setCheckable(True)
        self.mouthOpeningAction.setChecked(False)
        self.mouthOpeningAction.setStatusTip(
            "FaceMesh lip landmarks can be wrong; only used when teeth detection fails"
        )
        self.mouthOpeningAction.toggled.connect(self._mouth_opening_toggled)

    def _mouth_opening_toggled(self, *args):
        if self.portrait:
            self._compute_mouth_measurement()
            self.redrawImage(force=True)
            self.ui.dataOutputEdit.setPlainText(self._diagnostics_text())
        self._update_automatic_results()

    def _create_advanced_menu(self):
        advanced = self.viewMenu.addMenu("Advanced")
        self.depthProfileAction = advanced.addAction("Depth profile chart (blue line, x/y/angle)")
        self.depthProfileAction.setCheckable(True)
        self.depthProfileAction.toggled.connect(self._set_depth_profile_visible)
        self.diagnosticsAction = advanced.addAction("Diagnostics text")
        self.diagnosticsAction.setCheckable(True)
        self.diagnosticsAction.toggled.connect(self.ui.dataOutputEdit.setVisible)
        self._set_depth_profile_visible(False)
        self.ui.dataOutputEdit.setVisible(False)

    def _create_face_mesh_menu(self):
        menu = self.viewMenu.addMenu("FaceMesh landmarks")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.faceMeshModeActions = {}
        for key, text in (
            ("off", "Off"),
            ("main", "Main photo only"),
            ("everywhere", "Main photo and zoom views"),
        ):
            action = QAction(text, self)
            action.setCheckable(True)
            action.triggered.connect(partial(self._face_mesh_mode_selected, key))
            group.addAction(action)
            menu.addAction(action)
            self.faceMeshModeActions[key] = action
        self.faceMeshModeActions["off"].setChecked(True)
        self.ui.showFaceMeshLandmarksCheckBox.toggled.connect(self._face_mesh_checkbox_toggled)

    def _face_mesh_mode_selected(self, key, *args):
        self.faceMeshModeActions[key].setChecked(True)
        self.ui.showFaceMeshLandmarksCheckBox.setChecked(key != "off")
        self.redrawZoom()

    def _face_mesh_checkbox_toggled(self, checked):
        if not checked:
            self.faceMeshModeActions["off"].setChecked(True)
        elif self.faceMeshModeActions["off"].isChecked():
            self.faceMeshModeActions["main"].setChecked(True)
        self.redrawZoom()

    def _face_mesh_in_zoom(self):
        return (
            self.faceMeshModeActions["everywhere"].isChecked()
            and self.ui.showFaceMeshLandmarksCheckBox.isChecked()
            and self.face_mesh_debug is not None
        )

    def _legacy_profile_widgets(self):
        return [
            self.ui.chartLabel,
            self.ui.label_3,
            self.ui.xValue,
            self.ui.label_4,
            self.ui.yValue,
            self.ui.label_5,
            self.ui.angleValue,
            self.ui.angleSlider,
            self.reconstructionLabel,
        ]

    def _set_depth_profile_visible(self, visible):
        for widget in self._legacy_profile_widgets():
            widget.setVisible(visible)
        if hasattr(self, "portrait") and self.portrait:
            self.redrawImage(force=True)

    TOOL_HINTS = {
        MeasurementTool.VECTOR: "Click two points. Drag a point to adjust it. Esc clears.",
        MeasurementTool.POLYLINE: (
            "Click points; finish with Enter or right-click. "
            "Backspace removes the last point, Esc clears."
        ),
        MeasurementTool.REGION: "Click to place region A, then region B. Drag to move.",
        MeasurementTool.NECK: (
            "Click the left neck edge, points across the front of the neck, and the right "
            "edge; finish with Enter or right-click. 2 points = width only."
        ),
    }

    # Tools drawing two-point measurements with the PolylineEditor.
    TWO_POINT_TOOLS = (MeasurementTool.VECTOR,)
    # Tools finished explicitly with Enter / right-click.
    OPEN_POLYLINE_TOOLS = (MeasurementTool.POLYLINE, MeasurementTool.NECK)
    # Tools using the surface-sampling options (the Neck tool's arc).
    SURFACE_TOOLS = (MeasurementTool.VECTOR, MeasurementTool.POLYLINE, MeasurementTool.NECK)

    def set_tool(self, tool, *args):
        self.current_tool = tool
        self.toolActions[tool].setChecked(True)
        self.polyline_editor = PolylineEditor(
            max_points=2 if tool in self.TWO_POINT_TOOLS else None
        )
        self._tool_history_index = None
        self._point_drag_index = None
        self.regionOptionsWidget.setVisible(tool == MeasurementTool.REGION)
        self.lineOptionsWidget.setVisible(tool in self.SURFACE_TOOLS)
        self._set_region_selection_enabled(tool == MeasurementTool.REGION)
        self._update_tool_result()

    def _on_history_cleared(self, *args):
        """Clearing the history table must not leave a stale row index behind.

        Without this, the next _record_history() call (e.g. from changing
        surfaceStepSpin) would replace() into the now-empty history and raise
        IndexError.
        """
        self._tool_history_index = None

    def _record_history(self, tool, points, linear_mm, surface_mm):
        entry = HistoryEntry(
            filename=os.path.basename(self.filename) if self.filename else "",
            tool=tool,
            points=points,
            linear_mm=linear_mm,
            surface_mm=surface_mm,
        )
        if self._tool_history_index is None:
            self._tool_history_index = self.measurementPanel.add_history(entry)
        else:
            self.measurementPanel.replace_history(self._tool_history_index, entry)

    def _measure_current_polyline(self, points=None):
        if points is None:
            points = self.polyline_editor.points
        if self.depthmap is None or not hasattr(self, "image") or len(points) < 2:
            return None
        step = self.surfaceStepSpin.value()
        return measure_polyline(
            points,
            linear_mm=lambda a, b: self.linear_length_between_two_clicks_or_none(
                a[0], a[1], b[0], b[1]
            ),
            surface_mm=lambda a, b: self.surface_vector_filtered(a[0], a[1], b[0], b[1], step),
        )

    def _update_tool_result(self, *args):
        if self.current_tool == MeasurementTool.REGION:
            self._show_region_tool_result()
            return
        if self.current_tool == MeasurementTool.NECK:
            self._show_neck_tool_result()
            return
        hint = self.TOOL_HINTS[self.current_tool]
        if self.depthmap is None:
            hint = "Open a photo with depth data first. " + hint
        result = self._measure_current_polyline()
        if result is None or not result.segments:
            self.measurementPanel.set_tool_result(self.current_tool.value, None, None, [], hint)
            return
        if depth_is_implausible(self.portrait):
            # Nothing measured on this photo, so nothing goes to the history.
            self.measurementPanel.set_tool_result(
                self.current_tool.value,
                None,
                None,
                [IMPLAUSIBLE_DEPTH_DETAIL],
                hint,
                unavailable=IMPLAUSIBLE_DEPTH_NA,
            )
            return

        details = []
        if self.current_tool == MeasurementTool.VECTOR:
            (ax, ay), (bx, by) = self.polyline_editor.points[:2]
            if result.linear_mm is None:
                # Same invalid-depth rule as the linear/surface totals: a
                # point outside the calibrated range (or zero disparity)
                # makes the whole vector meaningless, not just its 3D length.
                details.append("Δ depth: n/a")
            else:
                depth_a_cm = self.distance_for_click(ax, ay)
                depth_b_cm = self.distance_for_click(bx, by)
                details.append(f"Depth A: {depth_a_cm:.1f} cm · B: {depth_b_cm:.1f} cm")
                details.append(f"Δ depth: {(depth_b_cm - depth_a_cm) * 10:+.1f} mm")
                # Length ignoring the depth difference: each point projected at
                # its own depth, only the X/Y components of the 3D vector.
                ax_mm, ay_mm = self.translate_click_to_mm(depth_a_cm, ax, ay)
                bx_mm, by_mm = self.translate_click_to_mm(depth_b_cm, bx, by)
                in_plane_mm = math.hypot(bx_mm - ax_mm, by_mm - ay_mm)
                details.append(f"In image plane (no Δ depth): {format_cm(in_plane_mm)}")
        else:
            state = "finished" if self.polyline_editor.finished else "in progress"
            details.append(f"{len(result.segments)} segments ({state})")
            for number, segment in enumerate(result.segments, start=1):
                details.append(
                    f"  {number}. {format_cm(segment.linear_mm)} / {format_cm(segment.surface_mm)}"
                )
        details.append(f"Surface step: {self.surfaceStepSpin.value()} px (median filtered)")
        self.measurementPanel.set_tool_result(
            self.current_tool.value, result.linear_mm, result.surface_mm, details, hint
        )
        if self.polyline_editor.finished:
            self._record_history(
                self.current_tool,
                len(self.polyline_editor.points),
                result.linear_mm,
                result.surface_mm,
            )

    def _neck_tool_unavailable(self, hint, unavailable, detail):
        self.measurementPanel.set_tool_result(
            MeasurementTool.NECK.value,
            None,
            None,
            [detail],
            hint,
            unavailable=unavailable,
            labels=NECK_TOOL_VALUE_LABELS,
        )

    def _show_neck_tool_result(self):
        """Neck tool: a polyline from the left neck edge across the front of the
        neck to the right edge.

        Width: between the first and last point (neck_width_from_edges).
        Circumference (3 or more points): the circle through the 3D chord and
        the surface arc along the polyline (arc_circle.py).

        History: one row per finished measurement, linear = width; the surface
        column holds the arc-circle circumference when available, otherwise
        the midpoint of the width model's ellipse range, or stays empty, see
        ``NECK_HISTORY_SURFACE_IS_CIRCUMFERENCE`` in measurement_tools.
        """
        hint = self.TOOL_HINTS[MeasurementTool.NECK]
        labels = NECK_TOOL_VALUE_LABELS
        points = self.polyline_editor.points
        portrait = getattr(self, "portrait", None)
        if portrait is None or not hasattr(self, "image"):
            self.measurementPanel.set_tool_result(
                MeasurementTool.NECK.value,
                None,
                None,
                [],
                "Open a photo with depth data first. " + hint,
                labels=labels,
            )
            return
        if self._float_depth() is None:
            # Legacy Camera-app photo: no metric float depth / intrinsics.
            self._neck_tool_unavailable(hint, NECK_TOOL_LEGACY_NA, NECK_TOOL_LEGACY_DETAIL)
            return
        if depth_is_implausible(portrait):
            self._neck_tool_unavailable(hint, IMPLAUSIBLE_DEPTH_NA, IMPLAUSIBLE_DEPTH_DETAIL)
            return
        if len(points) < 2:
            self.measurementPanel.set_tool_result(
                MeasurementTool.NECK.value, None, None, [], hint, labels=labels
            )
            return
        measure = getattr(portrait_analyser, "neck_width_from_edges", None)
        if measure is None:
            self._neck_tool_unavailable(
                hint,
                "n/a",
                "This portrait-analyser version has no neck width support (neck_width_from_edges).",
            )
            return
        (ax, ay), (bx, by) = points[0], points[-1]
        first = self.display_to_photo_coordinates(ax, ay)
        second = self.display_to_photo_coordinates(bx, by)
        if first == second:
            self.measurementPanel.set_tool_result(
                MeasurementTool.NECK.value,
                None,
                None,
                ["The two edge points coincide."],
                hint,
                texts=("—", "—"),
                labels=labels,
            )
            return
        # The width depends on the first/last point only: dragging a middle
        # point or finishing the polyline must not re-run the library.
        cache_key = (measure, id(portrait), first, second)
        cached = getattr(self, "_neck_width_tool_cache", None)
        if cached is not None and cached[0] == cache_key and cached[1] is portrait:
            result = cached[2]
        else:
            try:
                result = measure(portrait, first, second)
            except ValueError as exc:
                logger.warning("neck_width_from_edges failed", exc_info=True)
                self._neck_width_tool_cache = None
                self._neck_tool_unavailable(hint, "n/a", f"Neck width failed: {exc}")
                return
            self._neck_width_tool_cache = (cache_key, portrait, result)
        if result is None:
            self._neck_tool_unavailable(hint, NECK_TOOL_LEGACY_NA, NECK_TOOL_LEGACY_DETAIL)
            return

        status = getattr(result, "status", None)
        if status not in (NECK_WIDTH_OK, "no-depth"):
            # no-camera, relative-depth, error: this tool cannot measure it.
            self._neck_tool_unavailable(hint, "n/a", neck_width_message(result, portrait))
            return

        width_details = []
        width_mm = None
        ellipse_mm = None
        if status == NECK_WIDTH_OK:
            width_mm = result.width_mm
            ellipse_mm = result.circumference_ellipse_mm
            width_model_text = format_neck_circumference(ellipse_mm, result.circumference_circle_mm)
            # Quality once: the low-quality warning (with reasons), else a
            # plain "Quality: good" line.
            quality_warning = neck_width_quality_warning(result)
            if quality_warning is not None:
                width_details.append(f"⚠ {quality_warning}")
            elif getattr(result, "quality", None) is not None:
                width_details.append(f"Quality: {result.quality}")
            position = neck_width_position_text(result, quality=False)
            if position is not None:
                width_details.append(position[:1].upper() + position[1:])
            left_cm = result.left_depth_cm
            right_cm = result.right_depth_cm
            if left_cm is not None and right_cm is not None:
                width_details.append(f"Edge depth: L {left_cm:.1f} cm · R {right_cm:.1f} cm")
        else:
            width_model_text = format_cm(None)
            width_details.append(neck_width_message(result, portrait))
        width_warnings = [f"⚠ {warning}" for warning in neck_width_other_warnings(result)]
        width_text = format_cm(width_mm)

        arc_circumference_mm = None
        if len(points) < 3:
            # Two points: the width model only; the arc circle needs a point
            # on the front of the neck.
            details = [f"Width model: {width_model_text}"]
            details.extend(width_details)
            if status == NECK_WIDTH_OK:
                details.append(
                    "Ellipse assumes b/a 0.85–0.90; circle = π·W (a model, not a bound)."
                )
            details.extend(width_warnings)
            texts = (NECK_TOOL_NEEDS_3_POINTS, width_text)
        else:
            fit = self._neck_arc_circle(points)
            if fit.status == ARC_CIRCLE_OK:
                arc_circumference_mm = fit.circumference_mm
                circumference_text = format_cm(fit.circumference_mm, 1)
            elif fit.status == ARC_CIRCLE_FLAT:
                circumference_text = f"n/a ({ARC_CIRCLE_FLAT_TEXT})"
            else:
                circumference_text = format_cm(None)
            state = "finished" if self.polyline_editor.finished else "in progress"
            details = [f"{len(points)} points ({state})"]
            details.append(f"Chord (3D): {format_cm(fit.chord_mm)} · arc: {format_cm(fit.arc_mm)}")
            if fit.ratio is not None:
                details.append(f"Arc/chord: {format_decimal(fit.ratio, 3)}")
            if fit.status == ARC_CIRCLE_OK:
                details.append(
                    f"θ: {format_decimal(fit.theta_deg, 1)}° · R: {format_cm(fit.radius_mm)}"
                )
            details.extend(f"⚠ {warning[:1].upper()}{warning[1:]}" for warning in fit.warnings)
            details.append(f"Width model: {width_model_text}")
            details.extend(width_details)
            details.extend(width_warnings)
            details.append(
                f"Arc: ends sampled {format_decimal(NECK_ARC_EDGE_INSET_MM, 0)} mm inside "
                f"the clicked edges; surface step {self.surfaceStepSpin.value()} px "
                "(median filtered); circle through the chord and the arc."
            )
            texts = (circumference_text, width_text)

        self.measurementPanel.set_tool_result(
            MeasurementTool.NECK.value,
            width_mm,
            None,
            details,
            hint,
            texts=texts,
            labels=labels,
        )
        if self.polyline_editor.finished:
            surface_mm = neck_history_surface_mm(ellipse_mm, arc_circumference_mm)
            self._record_history(MeasurementTool.NECK, len(points), width_mm, surface_mm)

    def _neck_arc_circle(self, points):
        """Arc-circle fit for a Neck polyline (display points, >= 3).

        The first and last point are first moved ``NECK_ARC_EDGE_INSET_MM``
        inward along their segments (see ``_neck_arc_points``). Chord: the 3D
        distance between the moved ends, the same depth/camera path as the
        Vector tool's linear length. Arc: the sum of the segments' surface
        lengths, the same filtered path as the Polyline tool (the library's
        ``surface_length_mm`` for float depth).
        """
        points = self._neck_arc_points(points)
        if points is None:
            return arc_circle(None, None)
        (ax, ay), (bx, by) = points[0], points[-1]
        chord_mm = self.linear_length_between_two_clicks_or_none(ax, ay, bx, by)
        polyline = self._measure_current_polyline(points)
        arc_mm = None
        if polyline is not None and polyline.segments:
            arc_mm = polyline.surface_mm
        return arc_circle(chord_mm, arc_mm)

    def _neck_arc_points(self, points):
        """The Neck polyline (display points) with both ends moved
        ``NECK_ARC_EDGE_INSET_MM`` inward, or None (no camera / no depth).

        Like the library's edge depth: the inset in photo pixels is
        ``inset_mm * fx / z`` with ``z`` read first at the neighbouring point,
        then refined at the inset point itself (two passes).
        """
        camera = portrait_camera(self)
        depth = self._float_depth()
        if camera is None or depth is None:
            return None
        photo_points = [self.display_to_photo_coordinates(x, y) for x, y in points]

        def inset_px(end, unit, neighbour):
            z_cm = depth.distance_cm(*neighbour, radius=1)
            if z_cm is None:
                z_cm = depth.distance_cm(*end, radius=1)
            if z_cm is None:
                return None
            offset = NECK_ARC_EDGE_INSET_MM * camera.fx / (z_cm * 10.0)
            inside = depth.distance_cm(
                end[0] + offset * unit[0], end[1] + offset * unit[1], radius=1
            )
            if inside is not None:
                offset = NECK_ARC_EDGE_INSET_MM * camera.fx / (inside * 10.0)
            return offset

        moved = inset_polyline_ends(photo_points, inset_px)
        if moved is None:
            return None
        return [self._photo_to_main_display(point) for point in moved]

    def _show_region_tool_result(self):
        if depth_is_implausible(getattr(self, "portrait", None)):
            self.measurementPanel.set_tool_result(
                MeasurementTool.REGION.value,
                None,
                None,
                [IMPLAUSIBLE_DEPTH_DETAIL],
                self.TOOL_HINTS[MeasurementTool.REGION],
                unavailable=IMPLAUSIBLE_DEPTH_NA,
            )
            return
        result = self.region_measurement_result
        linear = result.linear_stats() if result is not None else None
        surface = result.surface_stats() if result is not None else None
        self.measurementPanel.set_tool_result(
            MeasurementTool.REGION.value,
            linear.mean_mm if linear is not None else None,
            surface.mean_mm if surface is not None else None,
            ["Mean of region profiles; details below."] if result is not None else [],
            self.TOOL_HINTS[MeasurementTool.REGION],
        )

    def _refresh_region_tool_result(self):
        """Keep the panel's Region result in sync outside the calculation's success path.

        No-op during __init__ (before measurementPanel exists) and whenever a
        different tool is active, so it never steals the panel from Vector/Polyline.
        """
        if getattr(self, "current_tool", None) != MeasurementTool.REGION:
            return
        if not hasattr(self, "measurementPanel"):
            return
        self._show_region_tool_result()

    POINT_GRAB_TOLERANCE = 6
    FACE_MESH_ZOOM_COLOR = QColor(255, 165, 0, 220)

    @staticmethod
    def _clamped_display_point(point):
        return (
            float(clamp(point.x(), 0, const.MAIN_IMAGE_WIDTH)),
            float(clamp(point.y(), 0, const.MAIN_IMAGE_HEIGHT)),
        )

    def _on_image_clicked(self, point, *args):
        # Keep Backspace/Enter/Esc reaching MainWindow.keyPressEvent even when
        # a spin box had focus (e.g. after editing the surface step).
        self.ui.imageLabel.setFocus()
        if self.current_tool == MeasurementTool.REGION:
            return
        if self._suppress_next_click:
            self._suppress_next_click = False
            return
        x, y = self._clamped_display_point(point)
        if self.polyline_editor.add_point((x, y)):
            self._tool_history_index = None
        self.ui.xValue.setValue(int(x))
        self.ui.yValue.setValue(int(y))
        self._update_tool_result()
        self._redraw_tool_overlay()

    def _on_image_right_clicked(self, point, *args):
        self._finish_polyline()

    def _finish_polyline(self):
        if self.current_tool not in self.OPEN_POLYLINE_TOOLS:
            return
        if self.polyline_editor.finished or not self.polyline_editor.finish():
            return
        self._update_tool_result()
        self._redraw_tool_overlay()

    def _begin_point_drag(self, point, *args):
        if self.current_tool == MeasurementTool.REGION:
            return
        index = self.polyline_editor.point_near(
            self._clamped_display_point(point), self.POINT_GRAB_TOLERANCE
        )
        self._point_drag_index = index
        # A press on an existing point is a drag; the click emitted on release
        # (when the pointer moved <= 3 px) must not add a new point.
        self._suppress_next_click = index is not None

    def _drag_point(self, point, *args):
        if self._point_drag_index is None:
            return
        self.polyline_editor.move_point(self._point_drag_index, self._clamped_display_point(point))
        self._update_tool_result()
        self._redraw_tool_overlay()

    def _finish_point_drag(self, point, *args):
        if self._point_drag_index is None:
            return
        self._drag_point(point)
        self._point_drag_index = None

    def _redraw_tool_overlay(self):
        if self.portrait:
            self.redrawImage(force=True)
        self.redrawZoom()

    def keyPressEvent(self, event):
        """Keep the single-key measurement tools reliable across Qt platforms."""
        if event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
            tool_by_key = {
                Qt.Key_V: MeasurementTool.VECTOR,
                Qt.Key_M: MeasurementTool.POLYLINE,
                Qt.Key_R: MeasurementTool.REGION,
                Qt.Key_N: MeasurementTool.NECK,
            }
            tool = tool_by_key.get(event.key())
            if tool is not None:
                self.toolActions[tool].trigger()
                event.accept()
                return

            mode_by_key = {
                Qt.Key_H: SelectionMode.HIGHEST,
                Qt.Key_L: SelectionMode.LOWEST,
                Qt.Key_F: SelectionMode.FLATTEST,
            }
            mode = mode_by_key.get(event.key())
            if mode is not None:
                self.regionToolActions[mode].trigger()
                event.accept()
                return

        if self.current_tool != MeasurementTool.REGION:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._finish_polyline()
                event.accept()
                return
            if event.key() in (Qt.Key_Backspace, Qt.Key_Escape):
                if event.key() == Qt.Key_Backspace:
                    self.polyline_editor.undo()
                else:
                    self.polyline_editor.clear()
                    self._tool_history_index = None
                self._update_tool_result()
                self._redraw_tool_overlay()
                event.accept()
                return
        elif event.key() == Qt.Key_Escape:
            self.clear_region_measurement()
            event.accept()
            return

        super().keyPressEvent(event)

    def _activate_region_selector(self, mode, *args):
        target = self._region_target
        if target is None:
            if self.region_a is None:
                target = "a"
            elif self.region_b is None:
                target = "b"
            else:
                target = self._last_region_target

        if self.current_tool != MeasurementTool.REGION:
            self.set_tool(MeasurementTool.REGION)
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
        both_uniform_area = (
            self.regionModeACombo.currentText() == SelectionMode.AREA.value
            and self.regionModeBCombo.currentText() == SelectionMode.AREA.value
        )
        self.regionPercentileSpin.setEnabled(not both_uniform_area)
        self.regionPercentileSpin.setToolTip(
            "Ignored because uniform Area samples the complete disk."
            if both_uniform_area
            else "Percentage of ranked pixels used as the candidate pool."
        )
        self.region_measurement_result = None
        self._calculate_region_measurement()
        self._redraw_region_overlay()

    def _depth_display_changed(self, *args):
        mode = self.depthDisplayCombo.currentData()
        self.depthContourStepSpin.setEnabled(mode == DepthDisplayMode.COLOR_CONTOURS)
        self.depthDisplayActions[mode].setChecked(True)
        self.redrawZoom()

    def clear_region_measurement(self, *args, repaint=True):
        self.region_a = None
        self.region_b = None
        self.region_measurement_result = None
        self._region_target = (
            "a" if getattr(self, "current_tool", None) == MeasurementTool.REGION else None
        )
        self._tool_history_index = None
        self._last_region_target = "a"
        self._region_drag_action = None
        if hasattr(self, "regionResultEdit"):
            self.regionResultEdit.setPlainText(
                "Click once to place region A, then once to place region B."
            )
        self._refresh_region_tool_result()
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
            self.region_measurement_result = None
            self._refresh_region_tool_result()
            return
        if self.filtered_depthmap is None or not hasattr(self, "image"):
            self.regionResultEdit.setPlainText("Load an image with depth data first.")
            self.region_measurement_result = None
            self._refresh_region_tool_result()
            return
        if depth_is_implausible(self.portrait):
            self.regionResultEdit.setPlainText(
                f"{IMPLAUSIBLE_DEPTH_NA}\n{IMPLAUSIBLE_DEPTH_DETAIL}"
            )
            self.region_measurement_result = None
            self._refresh_region_tool_result()
            return

        mode_a = SelectionMode(self.regionModeACombo.currentText())
        mode_b = SelectionMode(self.regionModeBCombo.currentText())
        mask_a = RegionMask(self.regionMaskACombo.currentText())
        mask_b = RegionMask(self.regionMaskBCombo.currentText())
        # Float (capture-app) depth: the engine samples the library's
        # integration map in centimetres; legacy photos keep the 8-bit codes.
        filtered_float = window_integration_depth(self)
        engine = RegionMeasurementEngine(
            filtered_depthmap=(
                filtered_float if filtered_float is not None else self.filtered_depthmap
            ),
            display_size=(const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT),
            image_size=self.image.size,
            depth_to_cm=self.get_depthmap_distance,
            pixels_per_mm=pixels_per_mm_at_distance,
            camera=portrait_camera(self),
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
            self._refresh_region_tool_result()
            return

        self.region_measurement_result = result
        requested = self.regionVectorCountSpin.value()
        linear = result.linear_stats()
        surface = result.surface_stats()
        valid_surface_count = sum(sample.surface_mm is not None for sample in result.samples)
        if mode_a == SelectionMode.AREA and mode_b == SelectionMode.AREA:
            candidate_pool_text = "Candidate pool: full area (uniform)"
        elif SelectionMode.AREA in (mode_a, mode_b):
            candidate_pool_text = (
                f"Candidate pool: {self.regionPercentileSpin.value()}% (ignored for Area selector)"
            )
        else:
            candidate_pool_text = f"Candidate pool: {self.regionPercentileSpin.value()}%"
        lines = [
            f"A: {mode_a.value}, mask={mask_a.value}, radius={self.region_a.radius:.0f} px",
            f"B: {mode_b.value}, mask={mask_b.value}, radius={self.region_b.radius:.0f} px",
            candidate_pool_text,
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
        self._record_history(
            MeasurementTool.REGION,
            2,
            linear.mean_mm if linear is not None else None,
            surface.mean_mm if surface is not None else None,
        )
        self._refresh_region_tool_result()

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

    POLYLINE_COLOR = QColor(255, 220, 0, 230)

    def _paint_polyline_overlay(self, painter, point_mapper):
        points = self.polyline_editor.points
        if self.current_tool == MeasurementTool.REGION or not points:
            return
        painter.save()
        try:
            mapped = [point_mapper(point) for point in points]
            painter.setPen(QtGui.QPen(self.POLYLINE_COLOR, 2))
            painter.setBrush(Qt.NoBrush)
            for start, end in zip(mapped, mapped[1:], strict=False):
                painter.drawLine(start, end)
            for number, point in enumerate(mapped, start=1):
                painter.setPen(Qt.NoPen)
                painter.setBrush(self.POLYLINE_COLOR)
                painter.drawEllipse(point, 4, 4)
                painter.setPen(self.POLYLINE_COLOR)
                painter.drawText(point + QPoint(6, -6), str(number))
        finally:
            painter.restore()

    NECK_WIDTH_COLOR = QColor(40, 230, 120, 235)
    NECK_WIDTH_BAND_COLOR = QColor(40, 230, 120, 130)

    def _neck_width_overlay_visible(self):
        result = getattr(self, "neck_width_auto", None)
        return (
            self.neckWidthOverlayAction.isChecked()
            and result is not None
            and getattr(result, "status", None) == NECK_WIDTH_OK
            and getattr(self, "portrait", None) is not None
            and hasattr(self, "image")
            and self._neck_width_shown()
        )

    def _photo_to_main_display(self, point):
        """Full-resolution photo point -> 480x640 display point (floats)."""
        return translate_coordinates_to_other_image(
            point, self.image.size, (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT)
        )

    def _paint_neck_width_overlay(self, painter, point_mapper, *, labels=False):
        """Automatic neck width: the measuring row between its two edges.

        ``point_mapper`` maps a full-resolution photo point to a QPoint on the
        painted canvas. The searched band is drawn as two faint dashed lines.
        """
        result = self.neck_width_auto
        left = point_mapper((result.left_x, result.row_y))
        right = point_mapper((result.right_x, result.row_y))
        painter.save()
        try:
            if result.band is not None:
                margin = 0.15 * (result.right_x - result.left_x)
                painter.setPen(QtGui.QPen(self.NECK_WIDTH_BAND_COLOR, 1, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                for band_y in result.band:
                    painter.drawLine(
                        point_mapper((result.left_x - margin, band_y)),
                        point_mapper((result.right_x + margin, band_y)),
                    )
            painter.setPen(QtGui.QPen(self.NECK_WIDTH_COLOR, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(left, right)
            for point in (left, right):
                painter.drawLine(point + QPoint(0, -6), point + QPoint(0, 6))
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.NECK_WIDTH_COLOR)
            for point in (left, right):
                painter.drawEllipse(point, 3, 3)
            if labels:
                painter.setPen(self.NECK_WIDTH_COLOR)
                painter.setBrush(Qt.NoBrush)
                painter.drawText(left + QPoint(4, -8), format_cm(result.width_mm))
        finally:
            painter.restore()

    def _paint_face_mesh_zoom(self, painter, display_point):
        """Paint FaceMesh on an already scaled zoom canvas (sharp dots and labels)."""
        main_size = (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT)
        bounds = painter.device().rect() if painter.device() is not None else None
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        for index, landmark in enumerate(self.face_mesh_debug.landmarks):
            main_point = translate_coordinates_to_other_image(landmark, self.image.size, main_size)
            point = display_point(main_point)
            if bounds is not None and not bounds.contains(point):
                continue
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.FACE_MESH_ZOOM_COLOR)
            painter.drawEllipse(point, 3, 3)
            painter.setPen(self.FACE_MESH_ZOOM_COLOR)
            painter.setBrush(Qt.NoBrush)
            painter.drawText(point + QPoint(5, -3), str(index))

    ZOOM_COLUMN_MIN_WIDTH = 260

    def _create_zoom_panel(self):
        # Put the UI widget next to a right-hand column of zoom views,
        # stacked vertically; a splitter lets the user widen the column.
        ui_widget = self.centralWidget()

        zoom_column = QWidget()
        zoom_column.setMinimumWidth(self.ZOOM_COLUMN_MIN_WIDTH)
        zoom_layout = QVBoxLayout(zoom_column)
        zoom_layout.setContentsMargins(2, 2, 2, 2)
        zoom_layout.setSpacing(2)

        def zoom_label(text):
            label = QLabel(text)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            label.setMinimumHeight(80)
            label.setStyleSheet("border:1px solid;")
            zoom_layout.addWidget(label, stretch=1)
            return label

        self.zoomedImageLabel = zoom_label("Photo")
        self.zoomedDepthMapLabel = zoom_label("Depth Map")
        self.skinMaskLabel = zoom_label("Depth (hairless)")
        self.zoomedSkinMapLabel = zoom_label("Skin Matte")
        self.zoomedTeethMapLabel = zoom_label("Teeth Map")
        self.reconstructionLabel = zoom_label("Reconstruction")

        self.zoomSplitter = QSplitter(Qt.Horizontal)
        self.zoomSplitter.addWidget(ui_widget)
        self.zoomSplitter.addWidget(zoom_column)
        self.zoomSplitter.setStretchFactor(0, 0)
        self.zoomSplitter.setStretchFactor(1, 1)
        self.zoomSplitter.setCollapsible(0, False)
        self.zoomSplitter.setSizes([ui_widget.sizeHint().width(), self.ZOOM_COLUMN_MIN_WIDTH])
        self.zoomSplitter.splitterMoved.connect(self._zoom_views_resized)
        self.setCentralWidget(self.zoomSplitter)

    def _zoom_views_resized(self, *args):
        self.redrawZoom()
        if self.hairless_depth_image is None:
            self.paintHairlessDepthFull()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Zoom views paint pixmaps at their current size; repaint after a resize.
        # Resize events can arrive while __init__ is still building the window.
        if getattr(self, "_ui_ready", False):
            self._zoom_views_resized()

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
        show_polyline = self.current_tool != MeasurementTool.REGION and bool(
            self.polyline_editor.points
        )
        show_face_mesh = self._face_mesh_in_zoom()
        show_neck_width = self._neck_width_overlay_visible()
        if not (show_regions or show_neck or show_polyline or show_face_mesh or show_neck_width):
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

            if show_neck_width:
                self._paint_neck_width_overlay(
                    painter, lambda point: display_point(self._photo_to_main_display(point))
                )

            if show_polyline:
                self._paint_polyline_overlay(painter, display_point)

            if show_face_mesh:
                self._paint_face_mesh_zoom(painter, display_point)
        finally:
            painter.restore()

    def _paint_geometric_neck_arc(self, painter, point_mapper, *, labels=False):
        """Paint the jaw-parallel measuring arc and its silhouette endpoints."""
        result = self.neck_arc
        if result is None:
            return

        points = [point_mapper(point) for point in result.arc_points_photo]
        if len(points) < 2:
            return

        painter.save()
        try:
            arc_color = QColor(255, 60, 200, 235)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QtGui.QPen(arc_color, 3))
            for first, second in zip(points, points[1:]):
                painter.drawLine(first, second)
            painter.setPen(Qt.NoPen)
            painter.setBrush(arc_color)
            for point in points:
                painter.drawEllipse(point, 4, 4)
            if labels:
                painter.setPen(arc_color)
                painter.setBrush(Qt.NoBrush)
                painter.drawText(
                    points[0] + QPoint(4, -8),
                    f"{result.circumference_mm / 10.0:.1f} cm",
                )
        finally:
            painter.restore()

    def _paint_neck_depth_overlay(self, painter, point_mapper, *, labels=False):
        """Paint skin-matte edges and their corrected stable depth samples."""
        self._paint_geometric_neck_arc(painter, point_mapper, labels=labels)

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
        display_map=None,
        value_text=None,
    ):
        """Paint a zoomed map; ``display_map`` (if given) is drawn instead of
        ``image_map`` while the value readout still uses the raw ``image_map``.

        ``value_text`` replaces the "raw <code>" readout (float depth shows the
        distance in cm; its 8-bit display codes mean nothing)."""
        w = ui_element.width()
        h = ui_element.height()
        if w < 1 or h < 1:
            return

        displayed_map = image_map if display_map is None else display_map
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
            if value_text is not None:
                lines = [f"{label_text} · {value_text}"]
            else:
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
            value_text=self._zoom_depth_value_text(mouse_x, mouse_y),
        )

    def _zoom_depth_value_text(self, depth_x, depth_y):
        """cm readout for float depth at the zoom centre (native depth
        pixels), None for legacy photos (raw code readout)."""
        if depth_x is None or depth_y is None:
            return None
        return self.depth_readout_at_native(depth_x, depth_y)

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

        if not self.portrait or self.depthmap is None:
            self.hairless_depth_image = None
            return

        # Display image (portrait.depth.to_display_image()), so repaired
        # capture-app depth shows correctly.
        depth_arr = np.array(self.depthmap)
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
            value_text=self._zoom_depth_value_text(mouse_x, mouse_y),
        )

    def _teeth_display_high(self):
        """Per-photo brightness reference for the boosted teeth map (cached)."""
        teethmap = self.portrait.teethmap if self.portrait else None
        if teethmap is None:
            return None
        cached = getattr(self, "_teeth_display_high_cache", None)
        if cached is None or cached[0] is not teethmap:
            self._teeth_display_high_cache = (teethmap, matte_display_high(teethmap))
        return self._teeth_display_high_cache[1]

    def paintZoomedTeethmap(
        self,
        teethmap,
        mouse_x=None,
        mouse_y=None,
        centroids=None,
        source_size=None,
        crop_box=None,
    ):
        w = self.zoomedTeethMapLabel.width()
        h = self.zoomedTeethMapLabel.height()
        if w < 1 or h < 1:
            return

        display_map = None
        high = self._teeth_display_high()
        if self.boostTeethMapAction.isChecked() and high is not None:
            display_map = boost_matte(teethmap, high)

        self._paintZoomedMap(
            "Teeth map",
            teethmap,
            self.zoomedTeethMapLabel,
            mouse_x,
            mouse_y,
            ok_value_threshold=200,
            source_size=source_size,
            crop_box=crop_box,
            display_map=display_map,
        )

        if centroids is None or crop_box is None:
            return

        canvas = self.zoomedTeethMapLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        try:
            scaled = teethmap.toqimage().scaled(w, h, Qt.KeepAspectRatio)
            offset_x = (w - scaled.width()) / 2
            offset_y = (h - scaled.height()) / 2
            crop_left, crop_top, crop_right, crop_bottom = crop_box
            crop_width = max(1, crop_right - crop_left)
            crop_height = max(1, crop_bottom - crop_top)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 255, 255, 200))
            for centroid in (centroids.upper_centroid, centroids.lower_centroid):
                if centroid is None:
                    continue
                display_x = offset_x + (centroid[0] - crop_left) * scaled.width() / crop_width
                display_y = offset_y + (centroid[1] - crop_top) * scaled.height() / crop_height
                if (
                    offset_x <= display_x <= offset_x + scaled.width()
                    and offset_y <= display_y <= offset_y + scaled.height()
                ):
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

        :returns: distance in centimeters, or None when the depth must not be
            used: the photo's depth is implausible, or ``value`` is code 0 of
            a capture-app depth map ("no depth", which would otherwise read
            as the farthest distance)
        """
        if self.float_min_value is None or self.float_max_value is None:
            return value

        portrait = getattr(self, "portrait", None)
        if depth_is_implausible(portrait):
            return None
        if np.ndim(value) == 0:
            if depth_zero_is_invalid(portrait) and value == 0:
                return None
            return raw_depth_to_distance_cm(
                value, self.float_min_value, self.float_max_value, portrait_camera(self)
            )
        # Arrays (region engine): vectorised legacy conversion; the engine
        # already treats raw 0 as invalid before converting.
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

        if self.smallImage:
            photo_crop = self._zoom_crop_box(
                mouse_x, mouse_y, self.image.size, self.zoomedImageLabel
            )
            zoomed = self.image.crop(photo_crop).resize(
                self._zoom_label_size(self.zoomedImageLabel)
            )
            self.paintZoomedImage(
                zoomed,
                source_size=self.image.size,
                crop_box=photo_crop,
            )

        if self.portrait and self.portrait.skinmap:
            skin_size = self.portrait.skinmap.size
            skin_crop = self._zoom_crop_box(mouse_x, mouse_y, skin_size, self.zoomedSkinMapLabel)
            skinmap_x, skinmap_y = self._zoom_crop_center(skin_crop)
            zoomed = self.portrait.skinmap.crop(skin_crop).resize(
                self._zoom_label_size(self.zoomedSkinMapLabel)
            )
            self.paintZoomedSkinmap(
                zoomed,
                mouse_x=skinmap_x,
                mouse_y=skinmap_y,
                source_size=skin_size,
                crop_box=skin_crop,
            )

        if self.hairless_depth_image is not None:
            hairless_size = self.hairless_depth_image.size
            hairless_crop = self._zoom_crop_box(mouse_x, mouse_y, hairless_size, self.skinMaskLabel)
            hd_x, hd_y = self._zoom_crop_center(hairless_crop)
            zoomed = self.hairless_depth_image.crop(hairless_crop)
            self.paintZoomedHairlessDepth(
                zoomed,
                mouse_x=hd_x,
                mouse_y=hd_y,
                source_size=hairless_size,
                crop_box=hairless_crop,
            )
        else:
            self.paintHairlessDepthFull()

        if self.portrait and self.portrait.teethmap:
            teeth_size = self.portrait.teethmap.size
            teeth_crop = self._zoom_crop_box(mouse_x, mouse_y, teeth_size, self.zoomedTeethMapLabel)
            teethmap_x, teethmap_y = self._zoom_crop_center(teeth_crop)
            zoomed = self.portrait.teethmap.crop(teeth_crop).resize(
                self._zoom_label_size(self.zoomedTeethMapLabel)
            )

            centroids = None
            if self.ui.showCentroidsCheckBox.isChecked() and self.portrait.incisor_measurement:
                centroids = self.portrait.incisor_measurement

            self.paintZoomedTeethmap(
                zoomed,
                mouse_x=teethmap_x,
                mouse_y=teethmap_y,
                centroids=centroids,
                source_size=teeth_size,
                crop_box=teeth_crop,
            )

        if self.depthmap:
            depth_crop = self._zoom_crop_box(
                mouse_x, mouse_y, self.depthmap.size, self.zoomedDepthMapLabel
            )
            depth_x, depth_y = self._zoom_crop_center(depth_crop)
            zoomed = self.depthmap.crop(depth_crop)

            self.paintZoomedDepthmap(
                zoomed,
                mouse_x=depth_x,
                mouse_y=depth_y,
                source_size=self.depthmap.size,
                crop_box=depth_crop,
            )

    # Half-height of every zoom view, in 480x640 display pixels (smaller = more zoom).
    ZOOM_DISPLAY_HALF_HEIGHT = 32

    @staticmethod
    def _zoom_label_size(label):
        return max(1, label.width()), max(1, label.height())

    def _zoom_crop_box(self, mouse_x, mouse_y, source_size, label):
        """Crop box around the pointer, in ``source_size`` pixels.

        The same display-space area is shown in every zoom view (so all maps
        zoom equally), and the box has the label's aspect ratio so the scaled
        crop fills the whole label.
        """
        label_w, label_h = self._zoom_label_size(label)
        half_h = self.ZOOM_DISPLAY_HALF_HEIGHT
        half_w = half_h * label_w / label_h
        center_x, center_y = translate_coordinates_to_other_image(
            (mouse_x, mouse_y),
            (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT),
            source_size,
        )
        scale_x = source_size[0] / const.MAIN_IMAGE_WIDTH
        scale_y = source_size[1] / const.MAIN_IMAGE_HEIGHT
        return (
            center_x - half_w * scale_x,
            center_y - half_h * scale_y,
            center_x + half_w * scale_x,
            center_y + half_h * scale_y,
        )

    @staticmethod
    def _zoom_crop_center(crop_box):
        left, top, right, bottom = crop_box
        return (left + right) / 2, (top + bottom) / 2

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

            if self._neck_width_overlay_visible():

                def neck_width_display_point(point):
                    x, y = self._photo_to_main_display(point)
                    return QPoint(round(x), round(y))

                self._paint_neck_width_overlay(painter, neck_width_display_point, labels=True)

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
            self._paint_polyline_overlay(painter, lambda p: QPoint(round(p[0]), round(p[1])))
            painter.setPen(QColor(0, 0, 255, 127))

            # Calculate 2 points at the edge of the image, using the angle.

            p1 = findPoint(x, y, direction=-1, angle=angle)
            p2 = findPoint(x, y, direction=1, angle=angle)
            if self.depthProfileAction.isChecked():
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
            self.last_click_x = mouse_x
            self.last_click_y = mouse_y

    def display_to_depthmap_coordinates(self, x, y):
        """Map a 480x640 display-space point to the depth map's native resolution.

        Depth maps are not always 480x640 (e.g. 576x768), so display coordinates
        must be scaled before sampling or cropping, like every other aux map.
        """
        return translate_coordinates_to_other_image(
            (x, y),
            (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT),
            self.depthmap.size,
        )

    def get_depthmap_value(self, x, y):
        """Raw depth value at a display-space (480x640) point."""
        x, y = self.display_to_depthmap_coordinates(x, y)
        x = int(clamp(x, 0, self.depthmap.size[0] - 1))
        y = int(clamp(y, 0, self.depthmap.size[1] - 1))
        value = self.depthmap.getpixel((x, y))
        # Legacy Camera-app depth maps are RGB; capture-app ones are 8-bit "L".
        return value[0] if isinstance(value, tuple) else value

    def _float_depth(self):
        """The photo's float ``DepthMap`` (capture-app files), or None (legacy)."""
        return window_float_depth(self)

    def display_to_photo_coordinates(self, x, y):
        """480x640 display point -> full-resolution photo point (per axis),
        clamped to the photo."""
        photo_width, photo_height = self.image.size
        photo_x = x * photo_width / const.MAIN_IMAGE_WIDTH
        photo_y = y * photo_height / const.MAIN_IMAGE_HEIGHT
        return (
            min(max(photo_x, 0.0), photo_width - 1),
            min(max(photo_y, 0.0), photo_height - 1),
        )

    def distance_for_click(self, x, y):
        """Camera distance (cm) at a display-space point, or None if invalid.

        Float (capture-app) depth is read at full precision from
        ``portrait.depth`` as the NaN-aware median of the 3x3 depth pixels
        around the point (``radius=1``): unfiltered TrueDepth depth has ~1 mm
        per-pixel jitter and isolated holes, so a single missing pixel does
        not turn a click into "n/a"; only a click with no valid pixel in its
        3x3 window does. Legacy 8-bit photos keep the historical single
        nearest-pixel read and conversion, so their values are unchanged.
        """
        depth = self._float_depth()
        if depth is None:
            return self.get_depthmap_distance(self.get_depthmap_value(x, y))
        if depth_is_implausible(getattr(self, "portrait", None)):
            return None
        return depth.distance_cm(*self.display_to_photo_coordinates(x, y), radius=1)

    def depth_readout_at_native(self, depth_x, depth_y):
        """Zoom-view readout for float depth at a native depth pixel, or None
        for legacy photos (they show the raw 8-bit code instead)."""
        depth = self._float_depth()
        if depth is None:
            return None
        rows, cols = depth.shape
        if not (0 <= depth_y < rows and 0 <= depth_x < cols):
            return "n/a"
        cached = getattr(self, "_float_cm_cache", None)
        if cached is None or cached[0] is not depth:
            cached = (depth, depth.to_cm_array())
            self._float_cm_cache = cached
        value = cached[1][int(depth_y), int(depth_x)]
        return "n/a" if not np.isfinite(value) else f"{value:.1f} cm"

    def translate_click_to_mm(self, distance_cm, x, y):
        """Translate a display-space click to camera-centred X/Y millimetres.

        Pixel coordinates are measured from the top-left corner in display
        (480x640) space; portrait_analyser.pixel_to_mm() converts to physical
        millimetres centred on the image's principal point.
        """
        image_x = x * self.image.size[0] / const.MAIN_IMAGE_WIDTH
        image_y = y * self.image.size[1] / const.MAIN_IMAGE_HEIGHT

        camera = portrait_camera(self)
        if camera is None:
            # Legacy Camera-app photo: calibration polynomial, image centre.
            return (
                pixel_to_mm(image_x, distance_cm, self.image.size[0]),
                pixel_to_mm(image_y, distance_cm, self.image.size[1]),
            )
        # TrueDepth capture-app photo: pinhole model with the file intrinsics.
        return (
            pixel_to_mm(
                image_x,
                distance_cm,
                self.image.size[0],
                focal_px=camera.fx,
                principal_px=camera.cx,
            ),
            pixel_to_mm(
                image_y,
                distance_cm,
                self.image.size[1],
                focal_px=camera.fy,
                principal_px=camera.cy,
            ),
        )

    def click_to_3d(self, x, y):
        """Display point -> (x_mm, y_mm, z_mm), or None for invalid depth."""
        z_cm = self.distance_for_click(x, y)
        if z_cm is None:
            return None
        x, y = self.translate_click_to_mm(z_cm, x, y)
        if x is None or y is None:
            return None
        z_mm = z_cm * 10
        return x, y, z_mm

    def vector_length_surface(
        self,
        mouse_x,
        mouse_y,
        last_click_x,
        last_click_y,
    ):
        """Calculate length iterating over the surface of 3D data.

        Legacy/diagnostic only. Samples every pixel of the raw depth map, so
        each 8-bit quantisation step is summed as real surface displacement
        and the result reads roughly 10% high. Use surface_vector_filtered()
        for measurement; this is kept to show what the filtering removes.
        """

        pixels = list(interpolate_pixels_along_line(mouse_x, mouse_y, last_click_x, last_click_y))
        s = []

        for (x1, y1), (x2, y2) in zip(pixels, pixels[1:], strict=False):
            point_1 = self.click_to_3d(x1, y1)
            point_2 = self.click_to_3d(x2, y2)
            if point_1 is None or point_2 is None:
                return None  # invalid depth along the line
            s.append(self.vector_length_simple(*point_1, *point_2))
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

        if depth_is_implausible(getattr(self, "portrait", None)):
            return None

        depth = window_float_depth(self)
        if depth is None and self.filtered_depthmap is None:
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

        if depth is not None:
            # Capture-app photo: the library integrates full-precision float
            # depth over its own smoothed integration map (with the camera).
            return depth.surface_length_mm(points_photo, camera=portrait_camera(self))

        return measure_filtered_surface_length(
            self.filtered_depthmap,
            points_photo,
            photo_width,
            photo_height,
            self.float_min_value,
            self.float_max_value,
            camera=portrait_camera(self),
        )

    def vector_length_simple(self, x1, y1, z1, x2, y2, z2):
        """Simple mathematical lenght of the vector"""
        return vector_length_3d(x1, y1, z1, x2, y2, z2)

    def linear_length_between_two_clicks_or_none(self, x1, y1, x2, y2):
        """3-D distance between two display points, or None when either
        point's depth is invalid: zero disparity, code 0 of a capture-app
        depth map ("no depth"), implausible depth, or a distance outside the
        range pixel_to_mm() accepts (15-80 cm polynomial, 10-150 cm pinhole).
        Used by the Vector/Polyline tools and the diagnostics text, which
        show "n/a" for a bad point rather than crash.
        """
        distance_z1 = self.distance_for_click(x1, y1)
        distance_z2 = self.distance_for_click(x2, y2)
        if distance_z1 is None or distance_z2 is None:
            return None
        distance_z1 *= 10  # cm → mm
        distance_z2 *= 10  # cm → mm

        distance_x1, distance_y1 = self.translate_click_to_mm(distance_z1 / 10, x1, y1)
        distance_x2, distance_y2 = self.translate_click_to_mm(distance_z2 / 10, x2, y2)
        if None in (distance_x1, distance_y1, distance_x2, distance_y2):
            return None

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
        # Load into locals first: on failure (ExifValidationFailed /
        # NoDepthMapFound / UnknownExtension) the previous photo stays on
        # screen, so self.filename and the in-progress measurement must be
        # left untouched rather than pointing at a photo that never loaded.
        try:
            portrait: IOSPortrait = load_image(fileName, use_exif=False)
            image = portrait.photo
            # Measurements read portrait.depth (float for capture-app files);
            # this 8-bit image is for display (and the legacy code path).
            depth = getattr(portrait, "depth", None)
            depthmap = depth.to_display_image() if depth is not None else portrait.depthmap
            filtered_depthmap = median_filter_depthmap(depthmap, size=3)
            teethmap = portrait.teethmap
            float_min_value = portrait.floatValueMin
            float_max_value = portrait.floatValueMax

            if float_max_value is not None:
                float_max_value = float(float_max_value)

            if float_min_value is not None:
                float_min_value = float(float_min_value)

        except ExifValidationFailed as e:
            QMessageBox.critical(
                self,
                tr("FIDMAA notification"),
                errors.NO_FRONT_CAMERA_NOTIFICATION.format(exif_camera_description=e),
            )
            return

        except NoDepthMapFound as e:
            # Also raised for TrueDepth capture-app files when the macOS depth
            # reader is unavailable (e.g. on Linux/Windows); the library's
            # message says which case it is, so show it with the generic hint.
            self.critical_error(errors.no_depth_data_error(e))
            return

        except UnknownExtension as e:
            self.critical_error(QObject.tr(f"Unknown file extension ({e})"))
            return

        self.filename = fileName
        self.clear_region_measurement(repaint=False)
        self.set_tool(self.current_tool)
        self.open3DAction.setEnabled(True)

        self.portrait: IOSPortrait = portrait
        self.image = image
        self.depthmap = depthmap
        self.filtered_depthmap = filtered_depthmap
        self._float_cm_cache = None
        self.teethmap = teethmap
        self.float_min_value = float_min_value
        self.float_max_value = float_max_value

        self.smallImage = self.image.resize((const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT))

        self._build_hairless_depth_image()

        self._run_detection_and_update(force_extended=False)

    def _run_detection_and_update(self, force_extended=False):
        """Run face/neck detection and update UI accordingly.

        When force_extended is True, skip MediaPipe and use only the
        segmentation/depth-based extended neck analysis.
        """
        self.neck_measurement_auto = None
        self.neck_lateral_bounds = None
        self.neck_confidence_ratio = None
        self.neck_arc = None
        self.neck_midpoint = None
        self.neck_width_auto = None
        self.neck_width_error = None
        self.mediapipe_debug = None
        self.face_mesh_debug = None
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
                            camera=portrait_camera(self),
                            **self._library_depth_kwargs(),
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
            # Capture-app photos report the width-based neck measurement
            # instead (Neck width overlay); the older skin-matte / geometric
            # arc overlay stays available but starts hidden.
            self.ui.showNeckArcCheckBox.setChecked(
                neck_arc_available and self._float_depth() is None
            )
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

        # Bound the neck search to an anatomical band directly below the
        # lowest FaceMesh row. Pose may shorten this range but cannot extend it
        # toward the shoulders/collar.
        scan_start_y = None
        scan_end_y = None
        arc_midpoint_y = None
        face_mesh_landmarks = None
        self.neck_search_bounds = None
        if self.neck_midpoint:
            face_mesh_landmarks = (
                self.face_mesh_debug.landmarks if self.face_mesh_debug is not None else None
            )
            scan_start_y, scan_end_y = neck_search_bounds_from_face_landmarks(
                chin=self.neck_midpoint.chin,
                nose=self.neck_midpoint.nose,
                image_height=self.image.size[1],
                face_mesh_landmarks=face_mesh_landmarks,
                pose_neck_y=self.neck_midpoint.y,
            )
            self.neck_search_bounds = (scan_start_y, scan_end_y)
            if self.neck_midpoint.y is not None and scan_start_y < scan_end_y:
                arc_midpoint_y = clamp(
                    self.neck_midpoint.y,
                    scan_start_y,
                    scan_end_y - 1,
                )

        neck_skinmap = self.portrait.skinmap
        jaw_bounds = lower_jaw_x_bounds(face_mesh_landmarks)
        if jaw_bounds is not None and neck_skinmap is not None and scan_start_y < scan_end_y:
            jaw_left = max(0, jaw_bounds[0])
            jaw_right = min(img_w - 1, jaw_bounds[1])
            if jaw_left < jaw_right:
                self.neck_lateral_bounds = (jaw_left, jaw_right)
                neck_skinmap = neck_skinmap.copy()
                mask_draw = ImageDraw.Draw(neck_skinmap)
                if jaw_left > 0:
                    mask_draw.rectangle(
                        (0, scan_start_y, jaw_left - 1, scan_end_y - 1),
                        fill=0,
                    )
                if jaw_right < img_w - 1:
                    mask_draw.rectangle(
                        (jaw_right + 1, scan_start_y, img_w - 1, scan_end_y - 1),
                        fill=0,
                    )

        # Implausible depth (e.g. inverted capture-app map): measure nothing.
        depth_measurable = not depth_is_implausible(self.portrait)
        if (
            depth_measurable
            and self.portrait.skinmap
            and self.portrait.depthmap
            and self.neck_midpoint
            and self.neck_midpoint.x is not None
            and self.neck_midpoint.pose == PortraitPose.NEUTRAL_NECK
            and scan_start_y < scan_end_y
        ):
            self.neck_measurement_auto = compute_neck_circumference(
                skinmap=neck_skinmap,
                depthmap=self.portrait.depthmap,
                photo_width=img_w,
                photo_height=self.image.size[1],
                float_min=self.portrait.floatValueMin,
                float_max=self.portrait.floatValueMax,
                scan_start_y=scan_start_y,
                scan_end_y=scan_end_y,
                neck_midpoint_y=arc_midpoint_y,
                circumference_multiplier=3.0,
                n_samples=7,
                hairmap=self.portrait.hairmap,
                camera=portrait_camera(self),
                **self._library_depth_kwargs(),
            )

        if self.neck_measurement_auto is not None:
            self.neck_confidence_ratio = neck_to_jaw_width_ratio(
                self.neck_measurement_auto.mask_left_x,
                self.neck_measurement_auto.mask_right_x,
                self.neck_lateral_bounds,
            )

        # Geometric neck arc: parallel to the FaceMesh lower jaw, at the height
        # where the depth surface stops receding below the chin. Independent of
        # the skin matte, so it is not fooled by bare skin running continuously
        # from jaw to chest (see NECK_DETECTOR_STATUS.md).
        if (
            depth_measurable
            and face_mesh_landmarks is not None
            and self.filtered_depthmap is not None
            and self.neck_midpoint is not None
            and self.neck_midpoint.pose == PortraitPose.NEUTRAL_NECK
        ):
            try:
                self.neck_arc = compute_neck_arc(
                    self.filtered_depthmap,
                    jaw_points=[face_mesh_landmarks[i] for i in const.FACE_MESH_LOWER_JAW_INDICES],
                    chin_xy=face_mesh_landmarks[const.FACE_MESH_CHIN_INDEX],
                    nose_xy=face_mesh_landmarks[const.FACE_MESH_NOSE_TIP_INDEX],
                    lower_lip_xy=face_mesh_landmarks[const.FACE_MESH_LOWER_LIP_INDEX],
                    photo_width=img_w,
                    photo_height=self.image.size[1],
                    float_min=self.portrait.floatValueMin,
                    float_max=self.portrait.floatValueMax,
                    camera=portrait_camera(self),
                    zero_is_invalid=depth_zero_is_invalid(self.portrait),
                    depth=window_float_depth(self),
                )
            except (IndexError, ValueError) as e:
                print(f"compute_neck_arc failed: {type(e).__name__}: {e}")
                self.neck_arc = None

        if depth_measurable:
            self._compute_neck_width()

        self._compute_mouth_measurement()

        self.last_click_y = self.last_click_x = None
        self.last_clicks = []
        self.redrawImage()
        self._update_automatic_results()
        self.ui.dataOutputEdit.setPlainText(self._diagnostics_text())
        self.updateWindowTitle()
        self._update_tool_result()

    def _compute_neck_width(self):
        """Automatic width-based neck measurement (capture-app photos only).

        ``portrait.neck_width`` is lazy (about 1 s on first access), so it is
        only read for float-depth photos in the neutral-neck pose, where the
        panel shows it. Legacy photos never touch it.
        """
        self.neck_width_auto = None
        self.neck_width_error = None
        if self._float_depth() is None:
            return
        if self.neck_midpoint is None or self.neck_midpoint.pose != PortraitPose.NEUTRAL_NECK:
            return
        try:
            self.neck_width_auto = getattr(self.portrait, "neck_width", None)
        except Exception as exc:
            # Unexpected library failure: keep the rest of the analysis and
            # say so in the panel instead of aborting the photo load.
            logger.exception("portrait.neck_width failed")
            self.neck_width_error = f"{type(exc).__name__}: {exc}"

    def _neck_width_shown(self):
        """True when the panel reports the width-based neck result."""
        return (
            self._float_depth() is not None
            and self.neck_midpoint is not None
            and self.neck_midpoint.pose == PortraitPose.NEUTRAL_NECK
            and not depth_is_implausible(self.portrait)
        )

    def _neck_width_lines(self):
        """``(lines, warning)`` of the automatic width-based neck result."""
        result = self.neck_width_auto
        if self.neck_width_error is not None:
            return [f"Neck width: error ({self.neck_width_error})", NECK_TOOL_HINT_SHORT], None
        if result is None:
            return ["Neck width: not available"], None
        if result.status != NECK_WIDTH_OK:
            lines = [f"Neck width: n/a — {neck_width_message(result, self.portrait)}"]
            if result.status in NECK_WIDTH_TOOL_HELPS:
                lines.append(NECK_TOOL_HINT_SHORT)
            return lines, None
        circumference = format_neck_circumference(
            result.circumference_ellipse_mm, result.circumference_circle_mm
        )
        lines = [f"Neck width: {format_cm(result.width_mm)} · circumference {circumference}"]
        position = neck_width_position_text(result)
        if position is not None:
            lines.append(f"  {position}")
        warnings = [neck_width_quality_warning(result)]
        warnings += [f"Neck width: {warning}" for warning in neck_width_other_warnings(result)]
        return lines, combine_warnings(*warnings)

    def _neck_width_diagnostics(self):
        """Diagnostics block for the width-based neck result (capture-app only)."""
        result = self.neck_width_auto
        txt = "\n\n[Neck width] "
        if self.neck_width_error is not None:
            return txt + f"error: {self.neck_width_error}"
        if result is None:
            return txt + "not available"
        txt += f"status: {result.status}"
        if result.status != NECK_WIDTH_OK:
            txt += f"\n  {neck_width_message(result, self.portrait, full=True)}"
        else:
            low, high = result.circumference_ellipse_mm
            txt += (
                f"\n  Width (3D): {result.width_mm / 10.0:.2f} cm"
                f"\n  Circumference: ellipse {low / 10.0:.1f}–{high / 10.0:.1f} cm"
                f" (b/a 0.85–0.90 assumed), circle (π·W)"
                f" {result.circumference_circle_mm / 10.0:.1f} cm"
                f"\n  Quality: {result.quality}"
            )
            for reason in result.quality_reasons or []:
                txt += f"\n    - {reason}"
            txt += (
                f"\n  Row Y {result.row_y:.0f} px, edges X {result.left_x:.0f} → "
                f"{result.right_x:.0f} px"
            )
            for label, value, unit in (
                ("Below the chin", result.height_below_chin_mm, "mm"),
                ("Support", result.support_mm, "mm"),
                ("Head roll", result.roll_deg, "deg"),
                ("Neck roll", result.neck_roll_deg, "deg"),
            ):
                if value is not None:
                    txt += f"\n  {label}: {value:.1f} {unit}"
            if result.left_depth_cm is not None and result.right_depth_cm is not None:
                txt += (
                    f"\n  Edge depth: L {result.left_depth_cm:.1f} cm,"
                    f" R {result.right_depth_cm:.1f} cm"
                )
            txt += f"\n  Rows used: {len(result.rows_used)}"
        if result.band is not None:
            txt += (
                f"\n  Band: Y {result.band[0]:.0f}–{result.band[1]:.0f} px ({result.band_source})"
            )
        reject_counts = getattr(result, "reject_counts", None)
        if reject_counts:
            counts = ", ".join(f"{code} {count}" for code, count in reject_counts.items())
            txt += f"\n  Rejected rows: {counts}"
        for warning in result.warnings or []:
            txt += f"\n  ⚠ {warning}"
        return txt

    def _library_depth_kwargs(self):
        """``{"depth": portrait.depth}`` for portrait_analyser measurement calls.

        Empty when the portrait has no ``depth`` (older portrait-analyser or
        test stand-ins), so the positional 8-bit arguments are used as before.
        """
        depth = getattr(getattr(self, "portrait", None), "depth", None)
        return {} if depth is None else {"depth": depth}

    def _compute_mouth_measurement(self):
        """FaceMesh mouth opening fallback, only when the user opted in."""
        self.mouth_measurement = None
        if not self.mouthOpeningAction.isChecked():
            return
        if depth_is_implausible(self.portrait):
            return
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
                camera=portrait_camera(self),
                zero_is_invalid=getattr(self.portrait, "depth_code_zero_is_invalid", None),
                **self._library_depth_kwargs(),
            )
            if self.mouth_measurement is None or self.mouth_measurement.distance_3d_mm is None:
                print("FaceMesh mouth measurement: depth conversion failed")

    def _update_automatic_results(self):
        pose = self.neck_midpoint.pose if self.neck_midpoint else None
        if self.portrait is None and pose is None and self.filename is None:
            self.measurementPanel.set_automatic_results(
                ["Open a photo to run the automatic analysis."], None
            )
            return
        depth_line, depth_warning = depth_source_status(self.portrait)
        if pose is None:
            self.measurementPanel.set_automatic_results(
                ["Pose: not detected", depth_line],
                combine_warnings(
                    "Face not detected. Try Analysis → Force extended neck analysis.",
                    depth_warning,
                ),
            )
            return

        lines = [f"Pose: {pose.value.upper()}"]
        if depth_is_implausible(self.portrait):
            # No automatic millimetre values from depth that cannot be trusted.
            lines.append(f"Automatic measurements: {IMPLAUSIBLE_DEPTH_NA}")
            lines.append(depth_line)
            self.measurementPanel.set_automatic_results(lines, depth_warning)
            return
        warning = None
        if pose == PortraitPose.NEUTRAL_NECK and self._neck_width_shown():
            neck_lines, warning = self._neck_width_lines()
            lines.extend(neck_lines)
        elif pose == PortraitPose.NEUTRAL_NECK:
            if self.neck_arc is not None:
                lines.append(f"Neck circumference: {format_cm(self.neck_arc.circumference_mm)}")
            else:
                lines.append("Neck circumference: unavailable (no stable neck level)")
            if self.neck_measurement_auto is not None:
                lines.append(
                    "Neck (skin-matte arc): "
                    f"{format_cm(self.neck_measurement_auto.circumference_mm)}"
                )
                if (
                    self.neck_confidence_ratio is not None
                    and self.neck_confidence_ratio >= const.NECK_LOW_CONFIDENCE_RATIO_THRESHOLD
                ):
                    warning = (
                        "Skin-matte neck arc: low confidence (may include shoulder/chest skin)."
                    )
        elif pose == PortraitPose.OPEN_MOUTH:
            incisor = self.portrait.incisor_measurement if self.portrait else None
            if incisor is not None and incisor.distance_3d_mm is not None:
                line = f"Incisor distance: {format_cm(incisor.distance_3d_mm)}"
                notes = []
                if incisor.weak_arch:
                    notes.append(f"weak {incisor.weak_arch} arch")
                if incisor.depth_assumed:
                    notes.append(f"{incisor.depth_assumed} depth taken from the other arch")
                if notes:
                    line += f" ({', '.join(notes)})"
                lines.append(line)
            else:
                lines.append("Incisor distance: teeth not detected")
            if (
                self.mouthOpeningAction.isChecked()
                and self.mouth_measurement is not None
                and self.mouth_measurement.distance_3d_mm is not None
            ):
                lines.append(
                    "Mouth opening (FaceMesh, experimental): "
                    f"{format_cm(self.mouth_measurement.distance_3d_mm)}"
                )
        elif pose == PortraitPose.EXTENDED_NECK:
            sd = self.segmentation_debug
            if sd is not None and sd.neck_width_front_arc_mm:
                lines.append(f"Neck width (3D front arc): {format_cm(sd.neck_width_front_arc_mm)}")
                lines.append(f"Neck width (3D straight): {format_cm(sd.neck_width_straight_mm)}")
        lines.append(depth_line)
        self.measurementPanel.set_automatic_results(lines, combine_warnings(warning, depth_warning))

    def _diagnostics_text(self):
        pose = self.neck_midpoint.pose if self.neck_midpoint else None
        txt = f"*** {pose.value.upper() if pose else 'UNKNOWN'} ***"
        if self._float_depth() is not None:
            # Legacy photos keep their diagnostics text unchanged.
            txt += "\nDepth: full-precision float map (capture app)"
            repair = getattr(self.portrait, "depth_repaired", None)
            if repair:
                txt += f"; repaired: {repair}"
        if pose == PortraitPose.OPEN_MOUTH:
            if self.portrait.teeth_bbox:
                txt += "\n\nTeeth detected."
                if self.portrait.incisor_distance:
                    img_size = (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT)
                    teeth_size = self.portrait.teethmap.size
                    start = translate_coordinates_to_other_image(
                        self.portrait.incisor_distance[:2], teeth_size, img_size
                    )
                    end = translate_coordinates_to_other_image(
                        self.portrait.incisor_distance[2:4], teeth_size, img_size
                    )
                    txt += "\nAutomatic incisor distance (legacy): "
                    legacy_mm = self.linear_length_between_two_clicks_or_none(*start, *end)
                    txt += "n/a" if legacy_mm is None else "%.2f cm" % (legacy_mm / 10.0)
                im = self.portrait.incisor_measurement
                if im and im.distance_3d_mm is not None:
                    txt += "\nAutomatic incisor distance (centroid): "
                    txt += "%.2f cm" % (im.distance_3d_mm / 10.0)
            if self.mouthOpeningAction.isChecked():
                if self.mouth_measurement and self.mouth_measurement.distance_3d_mm is not None:
                    txt += "\n\n[FaceMesh] Mouth opening: "
                    txt += "%.2f cm" % (self.mouth_measurement.distance_3d_mm / 10.0)

        elif pose == PortraitPose.NEUTRAL_NECK:
            if self._neck_width_shown():
                txt += self._neck_width_diagnostics()
            if self.neck_measurement_auto:
                nma = self.neck_measurement_auto
                low_confidence = (
                    self.neck_confidence_ratio is not None
                    and self.neck_confidence_ratio >= const.NECK_LOW_CONFIDENCE_RATIO_THRESHOLD
                )
                txt += "\n\nAutomatic neck circumference (3D arc): "
                txt += f"{nma.circumference_mm / 10.0:.2f} cm"
                if low_confidence:
                    txt += "  ⚠ LOW CONFIDENCE"
                txt += f"\n  Front surface arc (3D polyline): {nma.front_arc_length_mm:.2f} mm"
                if nma.front_chord_length_mm is not None:
                    txt += (
                        f"\n  Arc endpoint chord (3D Euclidean): {nma.front_chord_length_mm:.2f} mm"
                    )
                txt += f"\n  Multiplier: {nma.circumference_multiplier:.1f}"
                txt += f"\n  Arc points: {len(nma.arc_points_photo):d}"
                if self.neck_search_bounds is not None:
                    txt += (
                        "\n  Anatomical search band:"
                        f" Y {self.neck_search_bounds[0]}–{self.neck_search_bounds[1]} px"
                    )
                if self.neck_lateral_bounds is not None:
                    txt += (
                        "\n  FaceMesh jaw guard:"
                        f" X {self.neck_lateral_bounds[0]}–{self.neck_lateral_bounds[1]} px"
                    )
                txt += (
                    "\n  Skin → stable depth:"
                    f" L {nma.mask_left_x} → {nma.left_x} px,"
                    f" R {nma.mask_right_x} → {nma.right_x} px"
                )
                if self.neck_confidence_ratio is not None:
                    txt += (
                        "\n  Neck/jaw width ratio:"
                        f" {self.neck_confidence_ratio:.2f}"
                        f" (warn at ≥ {const.NECK_LOW_CONFIDENCE_RATIO_THRESHOLD:.2f})"
                    )
                if low_confidence:
                    txt += (
                        "\n  ⚠ No anatomical narrowing was found below the chin --"
                        " this measurement likely includes shoulder or chest skin,"
                        " not just the neck."
                    )

                if self.neck_arc is not None:
                    arc = self.neck_arc
                    txt += "\n\n[Geometric] Neck circumference: "
                    txt += f"{arc.circumference_mm / 10.0:.2f} cm"
                    txt += (
                        f"\n  Measuring line: {arc.offset_px} px below the jaw"
                        f" (Y={arc.measuring_y})"
                    )
                    txt += (
                        f"\n  Silhouette edges: X {arc.left_x} → {arc.right_x} px"
                        f"   width {arc.raw_width_mm:.1f} mm"
                    )
                    txt += (
                        f"\n  Width after erosion correction"
                        f" (×{const.NECK_SILHOUETTE_EROSION_CORRECTION:.3f}):"
                        f" {arc.corrected_width_mm:.1f} mm"
                    )
                    txt += f"\n  Visible 3D arc: {arc.front_arc_length_mm:.1f} mm"
                    txt += (
                        f"\n  Apex X {arc.apex_x} vs facial midline X {arc.midline_x}"
                        f" ({arc.apex_x - arc.midline_x:+d} px)"
                    )
                else:
                    txt += (
                        "\n\n[Geometric] Neck circumference: unavailable"
                        " — no stable neck level found below the chin"
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
                diameter_mm = nma.front_chord_length_mm
                if diameter_mm is None:
                    diameter_mm = self.linear_length_between_two_clicks_or_none(
                        *left_display, *right_display
                    )
                if diameter_mm is None:
                    txt += "\n\nNeck approximation from Euclidean diameter × π: n/a"
                else:
                    circumference_from_diameter_mm = diameter_mm * math.pi
                    txt += (
                        "\n\nNeck approximation from Euclidean diameter × π:"
                        f" {circumference_from_diameter_mm / 10.0:.2f} cm"
                        f"\n  Diameter (3D): {diameter_mm / 10.0:.2f} cm"
                    )
                txt += "\n  Straight-row surface vector (median depth):"
                for step in (2, 3, 5):
                    surface_mm = self.surface_vector_filtered(
                        *left_display,
                        *right_display,
                        step,
                    )
                    if surface_mm is None:
                        txt += f"\n    N={step} px: n/a"
                    else:
                        txt += f"\n    N={step} px: {surface_mm / 10.0:.2f} cm"

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
        return txt.strip()

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

    def setMidlineY(self, point, *args, **kw):
        # self.ui.xValue.setValue(point.x())
        self.ui.yValue.setValue(point.y())
        self.redrawImage()

    def open3DView(self):
        """Open the 3D view once per photo.

        The viewer runs in a separate process (VTK/Qt conflict on macOS). While
        it is still open for the current photo, further clicks do nothing; for
        a different photo the old viewer is replaced.
        """
        import multiprocessing

        process = getattr(self, "_3d_process", None)
        if process is not None and process.is_alive():
            if self._3d_filename == self.filename:
                return
            process.terminate()
            process.join(timeout=2)

        new_depthmap = self.depthmap.convert("L")
        float_min, float_max = self.float_min_value, self.float_max_value
        depth = self._float_depth()
        if depth is not None and hasattr(depth, "display_encoding"):
            # The viewer needs an 8-bit image plus its range: the float map's
            # display encoding (repaired depth included).
            display_image, float_min, float_max = depth.display_encoding()
            new_depthmap = display_image.convert("L")

        p = multiprocessing.Process(
            target=_show_3d_view,
            args=(
                self.image,
                new_depthmap,
                float_min,
                float_max,
            ),
        )
        p.start()
        self._3d_process = p
        self._3d_filename = self.filename

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

        self.ui.loadJPEGButton.hide()
        self.ui.open3DViewButton.hide()

        self.ui.loadJPEGButton.clicked.connect(self.loadJPEG)
        self.ui.open3DViewButton.clicked.connect(self.open3DView)
        self.ui.imageLabel.clicked.connect(self._on_image_clicked)
        self.ui.imageLabel.rightClicked.connect(self._on_image_right_clicked)
        self.ui.imageLabel.pressed.connect(self._begin_point_drag)
        self.ui.imageLabel.dragged.connect(self._drag_point)
        self.ui.imageLabel.released.connect(self._finish_point_drag)
        self.ui.imageLabel.setMouseTracking(True)
        self.ui.imageLabel.setFocusPolicy(Qt.ClickFocus)
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


INITIAL_WINDOW_WIDTH = 1500
INITIAL_WINDOW_HEIGHT = 860


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FIDMAA GUI")
    app.setApplicationDisplayName("FIDMAA GUI")

    mainWindow = MainWindow()
    mainWindow.updateWindowTitle()
    mainWindow.show()

    def handle_sigint(_signum, _frame):
        mainWindow.close()
        app.quit()

    signal.signal(signal.SIGINT, handle_sigint)

    # Qt's native event loop can otherwise keep Python from dispatching
    # SIGINT until another Python callback runs.
    sigint_timer = QTimer(app)
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(100)

    screen = app.primaryScreen().availableGeometry()
    # Start wide enough that the photo, the measurement panel and the zoom
    # column all get their preferred widths.
    hint = mainWindow.sizeHint()
    mainWindow.resize(
        min(max(hint.width(), INITIAL_WINDOW_WIDTH), screen.width()),
        min(max(hint.height(), INITIAL_WINDOW_HEIGHT), screen.height()),
    )
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
