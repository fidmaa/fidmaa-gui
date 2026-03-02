import math
import os
import sys
import traceback
from textwrap import dedent

from PIL import Image, ImageFile, ImageFilter
from portrait_analyser.exceptions import (
    ExifValidationFailed,
    MultipleFacesDetected,
    NoDepthMapFound,
    NoFacesDetected,
    UnknownExtension,
)
from portrait_analyser.face import (
    detect_eyes,
    get_face_parameters,
    translate_coordinates,
)
from portrait_analyser.ios import IOSPortrait, load_image
from portrait_analyser.neck import compute_neck_circumference
from portrait_analyser.pose import MediaPipeDebug, detect_neck_midpoint
from PySide6 import QtGui
from PySide6.QtCore import QObject, QPoint, QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import const, errors
from .calculations import findPoint
from .utils import (
    UILoaderMixin,
    clamp,
    get_circumference_of_circle,
    get_radius_of_circle_described_on_equilateral,
    get_radius_of_circle_described_on_square,
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
        self.load_ui()
        self._create_zoom_panel()

        self.filename = None
        self.face = None
        self.standalone_eyes = []

        self.smallImage = None
        self.portrait: IOSPortrait = None
        self.depthmap = None
        self.teethmap = None

        self.float_max_value = self.float_min_value = None

        self.last_click_x = None
        self.last_click_y = None
        self.last_angle = None
        self.last_depth = None
        self.last_show_centroids = None
        self.last_show_neck_arc = None
        self.last_show_landmarks = None
        self.face = None

        self.last_5_distances_vect = []
        self.last_5_distances_srfc = []

        self.redrawImage()
        self.redrawZoom()

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
        self.zoomedDepthMapLabel.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored
        )
        self.zoomedDepthMapLabel.setStyleSheet("border:1px solid;")
        row1.addWidget(self.zoomedDepthMapLabel, stretch=1)

        self.zoomedTeethMapLabel = QLabel("Teeth Map")
        self.zoomedTeethMapLabel.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored
        )
        self.zoomedTeethMapLabel.setStyleSheet("border:1px solid;")
        row1.addWidget(self.zoomedTeethMapLabel, stretch=1)

        self.reconstructionLabel = QLabel("Reconstruction")
        self.reconstructionLabel.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored
        )
        self.reconstructionLabel.setStyleSheet("border:1px solid;")
        row1.addWidget(self.reconstructionLabel, stretch=1)

        zoom_panel_layout.addLayout(row1)

        # Row 2: photo, skin matte
        row2 = QHBoxLayout()
        row2.setSpacing(2)

        self.zoomedImageLabel = QLabel("Photo")
        self.zoomedImageLabel.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored
        )
        self.zoomedImageLabel.setStyleSheet("border:1px solid;")
        row2.addWidget(self.zoomedImageLabel, stretch=1)

        self.zoomedSkinMapLabel = QLabel("Skin Matte")
        self.zoomedSkinMapLabel.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored
        )
        self.zoomedSkinMapLabel.setStyleSheet("border:1px solid;")
        row2.addWidget(self.zoomedSkinMapLabel, stretch=1)

        self.zoomedHairMapLabel = QLabel("Hair Matte")
        self.zoomedHairMapLabel.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored
        )
        self.zoomedHairMapLabel.setStyleSheet("border:1px solid;")
        row2.addWidget(self.zoomedHairMapLabel, stretch=1)

        zoom_panel_layout.addLayout(row2)

        wrapper_layout.addWidget(zoom_panel)
        self.setCentralWidget(wrapper)

    # ------------------------------------------------------------------
    # Zoom painting methods (moved from ZoomWindow)
    # ------------------------------------------------------------------

    def _paintZoomedMap(
        self,
        label_text,
        image_map,
        ui_element,
        mouse_x=None,
        mouse_y=None,
        ok_value_threshold=100,
    ):
        w = ui_element.width()
        h = ui_element.height()
        if w < 1 or h < 1:
            return

        qimg = image_map.toqimage().scaled(w, h, Qt.IgnoreAspectRatio)
        canvas = QtGui.QPixmap(w, h)
        painter = QtGui.QPainter(canvas)
        try:
            painter.drawImage(0, 0, qimg)

            painter.setPen(QColor(255, 0, 0, 255))
            half_w = w // 2
            half_h = h // 2
            painter.drawLine(QPoint(half_w, 0), QPoint(half_w, h))
            painter.drawLine(QPoint(0, half_h), QPoint(w, half_h))

            font = painter.font()
            font.setPixelSize(max(16, h // 10))
            painter.setFont(font)
            try:
                value = image_map.getpixel((half_w, half_h))[0]
            except TypeError:
                value = image_map.getpixel((half_w, half_h))

            if value < ok_value_threshold:
                painter.setPen(QColor(255, 0, 0, 255))
            else:
                painter.setPen(QColor(0, 255, 0, 255))
            painter.drawText(QPoint(50, 50), str(label_text))
            painter.drawText(QPoint(50, 100), str(value))
            if mouse_x is not None and mouse_y is not None:
                painter.drawText(
                    QPoint(50, 150), str(int(mouse_x)) + " x " + str(int(mouse_y))
                )
        finally:
            painter.end()
        ui_element.setPixmap(canvas)

    def paintZoomedDepthmap(self, depthmap, mouse_x=None, mouse_y=None):
        self._paintZoomedMap(
            "Depth map", depthmap, self.zoomedDepthMapLabel, mouse_x, mouse_y
        )

    def paintZoomedImage(self, zoomed):
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
        neck_arc_points=None,
        crop_origin=None,
        crop_size=None,
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

            painter.setPen(QColor(255, 0, 0, 255))
            half_w = w // 2
            half_h = h // 2
            painter.drawLine(QPoint(half_w, 0), QPoint(half_w, h))
            painter.drawLine(QPoint(0, half_h), QPoint(w, half_h))

            font = painter.font()
            font.setPixelSize(max(16, h // 10))
            painter.setFont(font)
            painter.setPen(QColor(0, 255, 0, 255))
            painter.drawText(QPoint(50, 50), "Skin matte")
            if mouse_x is not None and mouse_y is not None:
                painter.drawText(
                    QPoint(50, 100), str(int(mouse_x)) + " x " + str(int(mouse_y))
                )

            if neck_arc_points and crop_origin and crop_size:
                arc_color = QColor(255, 165, 0, 200)
                painter.setPen(Qt.NoPen)
                painter.setBrush(arc_color)
                display_pts = []
                for px, py in neck_arc_points:
                    local_x = px - crop_origin[0]
                    local_y = py - crop_origin[1]
                    display_x = offset_x + local_x * qimg.width() / crop_size[0]
                    display_y = offset_y + local_y * qimg.height() / crop_size[1]
                    if 0 <= display_x <= w and 0 <= display_y <= h:
                        pt = QPoint(int(display_x), int(display_y))
                        display_pts.append(pt)
                        painter.drawEllipse(pt, 3, 3)
                painter.setBrush(Qt.NoBrush)
                pen = QtGui.QPen(arc_color, 2)
                painter.setPen(pen)
                for i in range(1, len(display_pts)):
                    painter.drawLine(display_pts[i - 1], display_pts[i])
        finally:
            painter.end()
        self.zoomedSkinMapLabel.setPixmap(canvas)

    def paintZoomedHairmap(self, hairmap, mouse_x=None, mouse_y=None):
        w = self.zoomedHairMapLabel.width()
        h = self.zoomedHairMapLabel.height()
        if w < 1 or h < 1:
            return

        qimg = hairmap.toqimage().scaled(w, h, Qt.KeepAspectRatio)
        canvas = QtGui.QPixmap(w, h)
        canvas.fill(Qt.black)
        painter = QtGui.QPainter(canvas)
        try:
            offset_x = (w - qimg.width()) // 2
            offset_y = (h - qimg.height()) // 2
            painter.drawImage(offset_x, offset_y, qimg)

            painter.setPen(QColor(255, 0, 0, 255))
            half_w = w // 2
            half_h = h // 2
            painter.drawLine(QPoint(half_w, 0), QPoint(half_w, h))
            painter.drawLine(QPoint(0, half_h), QPoint(w, half_h))

            font = painter.font()
            font.setPixelSize(max(16, h // 10))
            painter.setFont(font)
            painter.setPen(QColor(0, 255, 0, 255))
            painter.drawText(QPoint(50, 50), "Hair matte")
            if mouse_x is not None and mouse_y is not None:
                painter.drawText(
                    QPoint(50, 100), str(int(mouse_x)) + " x " + str(int(mouse_y))
                )
        finally:
            painter.end()
        self.zoomedHairMapLabel.setPixmap(canvas)

    def paintZoomedTeethmap(
        self,
        teethmap,
        mouse_x=None,
        mouse_y=None,
        centroids=None,
        crop_origin=None,
        crop_size=None,
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

        return (
            100
            * 1.0
            / (
                self.float_max_value * value / 255
                + self.float_min_value * (1 - value / 255)
            )
        )

    def redrawZoom(self, *args, **kw):
        if args:
            event = args[0]
            mouse_x = event.x()
            mouse_y = event.y()
        else:
            mouse_x = mouse_y = 0

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
            zoomed = self.image.crop(
                (
                    big_image_x - crop_w,
                    big_image_y - crop_h,
                    big_image_x + crop_w,
                    big_image_y + crop_h,
                )
            ).resize((zoom_w, zoom_h))
            self.paintZoomedImage(zoomed)

        if self.portrait and self.portrait.skinmap:
            skinmap_x, skinmap_y = translate_coordinates_to_other_image(
                (mouse_x, mouse_y), (img_w, img_h), (self.portrait.skinmap.size)
            )
            zoomed = self.portrait.skinmap.crop(
                (
                    skinmap_x - crop_w,
                    skinmap_y - crop_h,
                    skinmap_x + crop_w,
                    skinmap_y + crop_h,
                )
            ).resize((zoom_w, zoom_h))
            neck_arc_points = None
            skin_crop_origin = None
            skin_crop_size = None
            if (
                self.ui.showNeckArcCheckBox.isChecked()
                and self.neck_measurement_auto
            ):
                neck_arc_points = self.neck_measurement_auto.arc_points_photo
                skin_crop_origin = (skinmap_x - crop_w, skinmap_y - crop_h)
                skin_crop_size = (crop_w * 2, crop_h * 2)

            self.paintZoomedSkinmap(
                zoomed,
                mouse_x=skinmap_x,
                mouse_y=skinmap_y,
                neck_arc_points=neck_arc_points,
                crop_origin=skin_crop_origin,
                crop_size=skin_crop_size,
            )

        if self.portrait and self.portrait.hairmap:
            hairmap_x, hairmap_y = translate_coordinates_to_other_image(
                (mouse_x, mouse_y), (img_w, img_h), self.portrait.hairmap.size
            )
            zoomed = self.portrait.hairmap.crop(
                (
                    hairmap_x - crop_w,
                    hairmap_y - crop_h,
                    hairmap_x + crop_w,
                    hairmap_y + crop_h,
                )
            ).resize((zoom_w, zoom_h))
            self.paintZoomedHairmap(zoomed, mouse_x=hairmap_x, mouse_y=hairmap_y)

        if self.portrait and self.portrait.teethmap:
            teethmap_x, teethmap_y = translate_coordinates_to_other_image(
                (mouse_x, mouse_y), (img_w, img_h), (self.portrait.teethmap.size)
            )

            zoomed = self.portrait.teethmap.crop(
                (
                    teethmap_x - crop_w,
                    teethmap_y - crop_h,
                    teethmap_x + crop_w,
                    teethmap_y + crop_h,
                )
            ).resize((zoom_w, zoom_h))

            centroids = None
            if (
                self.ui.showCentroidsCheckBox.isChecked()
                and self.portrait.incisor_measurement
            ):
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
            )

        if self.depthmap:
            zoomed = (
                self.depthmap.crop(
                    (mouse_x - 72, mouse_y - 48, mouse_x + 72, mouse_y + 48)
                )
                .resize((zoom_w, zoom_h), Image.HAMMING)
                .filter(ImageFilter.SHARPEN)
                .filter(ImageFilter.SHARPEN)
                .filter(ImageFilter.SHARPEN)
            )

            self.paintZoomedDepthmap(
                zoomed,
                mouse_x=mouse_x,
                mouse_y=mouse_y,
            )

    def redrawImage(self, *args, **kw):
        if not self.portrait:
            return

        mouse_x = x = self.ui.xValue.value()
        y = mouse_y = self.ui.yValue.value()
        angle = self.ui.angleValue.value()

        mouse_x = clamp(mouse_x, 0, const.MAIN_IMAGE_WIDTH)
        mouse_y = clamp(mouse_y, 0, const.MAIN_IMAGE_HEIGHT)

        show_centroids = self.ui.showCentroidsCheckBox.isChecked()
        show_neck_arc = self.ui.showNeckArcCheckBox.isChecked()
        show_landmarks = self.ui.showMediaPipeLandmarksCheckBox.isChecked()

        if self.last_click_x is not None:
            if (
                self.last_click_x == mouse_x
                and self.last_click_y == mouse_y
                and self.last_angle == angle
                and self.last_show_centroids == show_centroids
                and self.last_show_neck_arc == show_neck_arc
                and self.last_show_landmarks == show_landmarks
            ):
                return

        self.last_angle = angle
        self.last_show_centroids = show_centroids
        self.last_show_neck_arc = show_neck_arc
        self.last_show_landmarks = show_landmarks

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

                if (
                    self.ui.showCentroidsCheckBox.isChecked()
                    and self.portrait.incisor_measurement
                ):
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

            if self.ui.showNeckArcCheckBox.isChecked() and self.neck_measurement_auto:
                arc_color = QColor(255, 165, 0, 200)
                painter.setPen(Qt.NoPen)
                painter.setBrush(arc_color)
                arc_display_pts = []
                for px, py in self.neck_measurement_auto.arc_points_photo:
                    dx, dy = translate_coordinates_to_other_image(
                        (px, py),
                        self.portrait.skinmap.size,
                        (img_w, img_h),
                    )
                    arc_display_pts.append(QPoint(int(dx), int(dy)))
                    painter.drawEllipse(QPoint(int(dx), int(dy)), 3, 3)
                painter.setBrush(Qt.NoBrush)
                pen = QtGui.QPen(arc_color, 2)
                painter.setPen(pen)
                for i in range(1, len(arc_display_pts)):
                    painter.drawLine(arc_display_pts[i - 1], arc_display_pts[i])
                painter.setPen(QColor(0, 0, 255, 127))

            print(
                f"Neck midpoint draw check: "
                f"checkbox={self.ui.showNeckArcCheckBox.isChecked()}, "
                f"midpoint={self.neck_midpoint}"
            )
            if self.ui.showNeckArcCheckBox.isChecked() and self.neck_midpoint:
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

            if self.face:
                painter.setPen(QColor(0, 0, 255, 127))
                face_rect = self.face.translate_coordinates(img_w, img_h)
                painter.drawRect(*face_rect)

                for eye in self.face.eyes:
                    painter.setPen(QColor(0, 255, 0, 127))
                    rect = eye.translate_coordinates(img_w, img_h)
                    painter.drawRect(*rect)
            elif self.standalone_eyes:
                for eye in self.standalone_eyes:
                    painter.setPen(QColor(0, 255, 0, 127))
                    rx, ry, rw, rh = translate_coordinates(
                        eye, img_w, img_h,
                        self.image.size[0], self.image.size[1],
                    )
                    painter.drawRect(int(rx), int(ry), int(rw), int(rh))

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

                for pixels in interpolate_pixels_along_line(
                    point_beg.x(), 0, 0, point_end.x(), 639, 0
                ):
                    painter.drawLine(
                        0,
                        pixels[1],
                        self.get_depthmap_value(pixels[0], pixels[1]),
                        pixels[1],
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
                    for pixels in interpolate_pixels_along_line(
                        mouse_x, mouse_y, 0, self.last_click_x, self.last_click_y, 0
                    ):
                        values.append(self.get_depthmap_value(pixels[0], pixels[1]))
                    self.paintReconstruction(values)
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
            if (
                self.last_click_x != mouse_x or self.last_click_y != mouse_y
            ) and self.last_click_x is not None:
                vector_length_3d = self.vector_length_between_two_clicks(
                    mouse_x, mouse_y, self.last_click_x, self.last_click_y
                )

                surface_length_3d = self.vector_length_surface(
                    mouse_x, mouse_y, self.last_click_x, self.last_click_y
                )

            self.last_click_x = mouse_x
            self.last_click_y = mouse_y

            closeness = self.depthmap.getpixel((mouse_x, mouse_y))[0]

            depth_mm = self.get_depthmap_distance(closeness)

            closeness_delta = 0
            closeness_delta_mm = 0.0

            if self.last_depth:
                closeness_delta = self.last_depth - closeness
                closeness_delta_mm = (
                    self.get_depthmap_distance(self.last_depth) - depth_mm
                )

            self.last_depth = closeness

            self.last_5_distances_srfc.append(surface_length_3d)
            if len(self.last_5_distances_srfc) > 5:
                self.last_5_distances_srfc = self.last_5_distances_srfc[1:6]

            self.last_5_distances_vect.append(vector_length_3d)
            if len(self.last_5_distances_vect) > 5:
                self.last_5_distances_vect = self.last_5_distances_vect[1:6]

            self.ui.dataOutputEdit.clear()
            txt = dedent(
                f"""
            Depth map coords: {mouse_x, mouse_y}

            Depth map raw data: {closeness} (Δ: {closeness_delta})

            Depth map distance:
            {depth_mm:.2f} cm (Δ: {closeness_delta_mm:.1f} cm)

            Line length (2D, on flat picture):
            {line_len:.2f} pixels

            Vector length (3D) with depth:
            {vector_length_3d / 10.0:.2f} cm

            Sum of last 5 3D vector lengths:
            {sum(self.last_5_distances_vect) / 10.0:.2f} cm
            {[f"{x:.2f}" for x in self.last_5_distances_vect]}

            Vector length (3D) on surface:
            {(surface_length_3d / 10.0):.2f} cm

            Sum of last 5 3D surface vector lengths:
            {sum(self.last_5_distances_srfc) / 10.0:.2f} cm
            {[f"{x:.2f}" for x in self.last_5_distances_srfc]}
            """
            )

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
                    txt += (
                        "\nAutomatic incisor distance"
                        " (centroid): "
                    )
                    txt += "%.2f cm" % (
                        im.distance_3d_mm / 10.0
                    )

            if self.neck_measurement_auto:
                nma = self.neck_measurement_auto
                txt += (
                    "\n\nAutomatic neck circumference"
                    " (3D arc): "
                )
                txt += "%.2f cm" % (
                    nma.circumference_mm / 10.0
                )
                txt += "\n  Front arc: %.2f mm" % (
                    nma.front_arc_length_mm
                )
                txt += "\n  Multiplier: %.1f" % (
                    nma.circumference_multiplier
                )
                txt += "\n  Arc points: %d" % len(
                    nma.arc_points_photo
                )

            if (
                closeness_delta_mm is not None
                and vector_length_3d is not None
                and vector_length_3d > 0.0
            ):
                try:
                    txt += "\n\nAngle for last 2 clicks:\n%.2f°" % math.degrees(
                        math.acos(abs(closeness_delta_mm / (vector_length_3d / 10.0)))
                    )
                except ValueError:
                    pass

                txt += (
                    "\n\nNeck circumference estimation"
                    " for last 2 clicks"
                    " (equilateral triangle): %.2f"
                    % (
                        get_circumference_of_circle(
                            get_radius_of_circle_described_on_equilateral(
                                vector_length_3d
                            )
                        )
                        / 10.0
                    )
                )

                txt += (
                    "\n\nNeck circumference estimation"
                    " for last 2 clicks"
                    " (radius): %.2f"
                    % (
                        get_circumference_of_circle(
                            vector_length_3d
                        )
                        / 10.0
                    )
                )

                txt += (
                    "\n\nNeck circumference estimation"
                    " for last 2 clicks"
                    " (square): %.2f"
                    % (
                        get_circumference_of_circle(
                            get_radius_of_circle_described_on_square(vector_length_3d)
                        )
                        / 10.0
                    )
                )
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
        return (
            self.how_many_mm_per_pixels_at_distance_on_big_image(
                distance_cm, x * self.image.size[0] / const.MAIN_IMAGE_WIDTH
            ),
            self.how_many_mm_per_pixels_at_distance_on_big_image(
                distance_cm, y * self.image.size[1] / const.MAIN_IMAGE_HEIGHT
            ),
        )

    def click_to_3d(self, x, y):
        z = self.distance_for_click(x, y)
        x, y = self.translate_click_to_mm(z, x, y)
        return x, y, z

    def vector_length_surface(
        self,
        mouse_x,
        mouse_y,
        last_click_x,
        last_click_y,
    ):
        """Calculate length iterating over the surface of 3D data"""

        z1 = self.get_depthmap_value(mouse_x, mouse_y)
        z2 = self.get_depthmap_value(last_click_x, last_click_y)
        pixels = list(
            interpolate_pixels_along_line(
                mouse_x, mouse_y, z1, last_click_x, last_click_y, z2
            )
        )
        s = []

        for (x1, y1, z1), (x2, y2, z2) in zip(pixels, pixels[1:]):
            z1 = self.distance_for_click(x1, y1)
            x1, y1 = self.translate_click_to_mm(z1, x1, y1)

            z2 = self.distance_for_click(x2, y2)
            x2, y2 = self.translate_click_to_mm(z2, x2, y2)

            line_len_3d = self.vector_length_simple(x1, y1, z1, x2, y2, z2)
            s.append(line_len_3d)
        return sum(s)

    def vector_length_simple(self, x1, y1, z1, x2, y2, z2):
        """Simple mathematical lenght of the vector"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def vector_length_between_two_clicks(self, x1, y1, x2, y2):
        z1 = self.get_depthmap_value(x1, y1)
        z2 = self.get_depthmap_value(x2, y2)

        distance_z1 = self.get_depthmap_distance(z1)
        distance_z2 = self.get_depthmap_distance(z2)

        distance_x1, distance_y1 = self.translate_click_to_mm(
            distance_z1,
            x1,
            y1,
        )
        distance_x2, distance_y2 = self.translate_click_to_mm(
            distance_z2,
            x2,
            y2,
        )
        args = (
            distance_x1,
            distance_y1,
            distance_z1,
            distance_x2,
            distance_y2,
            distance_z2,
        )
        return self.vector_length_simple(*args)

    def calculate_line_length(self, dist_x, dist_y):
        line_len = math.sqrt(abs(dist_x * dist_x) + abs(dist_y * dist_y))
        return line_len

    def how_many_pixels_per_mm_at_distance_on_big_image(self, distance, mm):
        """Return pixel count per mm at a given distance from camera.

        Constants taken from own calibration data and a curve
        fitted by MyCurveFit.com. I strongly recommend their
        service, it is very easy to use and affordable.

        :param distance: distance in centimeters from the camera
        :param mm: millimeters (unused, kept for API compat)
        """
        return (
            30.79912
            - 1.346418 * distance
            + 0.03009753 * distance**2
            - 0.0003733656 * distance**3
            + 0.000002521213 * distance**4
            - 7.49986e-9 * distance**5
        )

    def how_many_mm_per_pixels_at_distance_on_big_image(self, distance, no_pixels):
        assert distance >= 15.0, "Distance must be bigger than 15 cms"
        pixels_per_mm = self.how_many_pixels_per_mm_at_distance_on_big_image(
            distance, 1
        )
        return no_pixels / pixels_per_mm

    def _loadImage(self, fileName):
        self.filename = fileName

        try:
            self.portrait: IOSPortrait = load_image(self.filename, use_exif=False)
            self.image = self.portrait.photo
            self.depthmap = self.portrait.depthmap
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
            self.critical_error(QObject.tr("Unknown file extension (%s)" % e))
            return

        self.smallImage = self.image.resize(
            (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT)
        )

        #
        # Get face position, if any:
        #

        self.neck_measurement_auto = None
        self.neck_midpoint = None
        self.mediapipe_debug = None
        self.face_proper = False

        # Always detect standalone eyes — used as fallback for neck search
        # when face has <2 eyes, and for drawing when face is absent.
        self.standalone_eyes = detect_eyes(self.image)
        img_w = self.image.size[0]

        try:
            self.face = get_face_parameters(self.image, raise_opencv_exceptions=True)
        except NoFacesDetected:
            self.face = None
            if not self.standalone_eyes:
                self.critical_error(errors.FACE_NOT_DETECTED)

        except MultipleFacesDetected:
            self.critical_error(errors.MULTIPLE_FACES_DETECTED)

        except Exception:
            tb_text = traceback.format_exc()
            self.critical_error(f"Exception: {tb_text}")
            print(tb_text)

        else:
            percent_width, percent_height = self.face.calculate_percentage_of_image()
            if (
                percent_width < const.MINIMUM_FACE_WIDTH_PERCENT
                or percent_height < const.MINIMUM_FACE_HEIGHT_PERCENT
            ):
                self.critical_error(
                    errors.FACE_TOO_SMALL.format(
                        percent_width=percent_width * 100,
                        percent_height=percent_height * 100,
                        minimum_width=const.MINIMUM_FACE_WIDTH_PERCENT * 100,
                        minimum_height=const.MINIMUM_FACE_HEIGHT_PERCENT * 100,
                    )
                )
            else:
                self.face_proper = True

            # Set lower point somewhere around mouth (below nose, above chin)

            self.ui.xValue.setValue(
                int(
                    round(
                        self.face.center_x
                        / self.image.size[0]
                        * (const.MAIN_IMAGE_WIDTH - 1)
                    )
                )
            )
            self.ui.yValue.setValue(
                int(
                    round(
                        (self.face.center_y + self.face.height / 4)
                        / self.image.size[1]
                        * (const.MAIN_IMAGE_HEIGHT - 1)
                    )
                )
            )

        # Detect MediaPipe pose landmarks (mouth + shoulders → search bounds)
        try:
            result = detect_neck_midpoint(self.image)
            if result:
                self.neck_midpoint, self.mediapipe_debug = result
                print(f"Neck midpoint detected: {self.neck_midpoint}")
                print(f"  x={self.neck_midpoint.x}, y={self.neck_midpoint.y}")
            else:
                self.neck_midpoint = None
                self.mediapipe_debug = None
        except Exception as e:
            print(f"detect_neck_midpoint failed: {type(e).__name__}: {e}")
            self.neck_midpoint = None
            self.mediapipe_debug = None

        # Compute MediaPipe search bounds: mouth → neck midpoint.
        # The narrowest neck is between the chin and mid-cervical level,
        # NOT between mouth and shoulders (that range includes the collar
        # line where skin disappears — the global minimum, not the neck).
        scan_start_y = None
        scan_end_y = None
        if self.neck_midpoint:
            mouth_y = max(
                self.neck_midpoint.mouth_left[1],
                self.neck_midpoint.mouth_right[1],
            )
            scan_start_y = round(mouth_y)
            scan_end_y = round(self.neck_midpoint.y)

        has_eyes = len(self.standalone_eyes) >= 2
        if self.portrait.skinmap and self.portrait.depthmap and (
            self.face_proper or has_eyes
        ):
            self.neck_measurement_auto = compute_neck_circumference(
                skinmap=self.portrait.skinmap,
                depthmap=self.portrait.depthmap,
                photo_width=img_w,
                photo_height=self.image.size[1],
                float_min=self.portrait.floatValueMin,
                float_max=self.portrait.floatValueMax,
                face_location=self.face,
                eyes=self.standalone_eyes,
                image_width=img_w,
                scan_start_y=scan_start_y,
                scan_end_y=scan_end_y,
                neck_midpoint_y=(
                    self.neck_midpoint.y
                    if self.neck_midpoint
                    else None
                ),
            )

        self.last_click_y = self.last_click_x = None
        self.last_clicks = []
        self.redrawImage()
        self.updateWindowTitle()

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
        self.ui.imageLabel.mouseMoveEvent = self.redrawZoom
        self.ui.imageLabel.setCursor(Qt.CursorShape.CrossCursor)
        self.ui.chartLabel.clicked.connect(self.setMidlineY)

        self.ui.angleValue.valueChanged.connect(self.redrawImage)
        self.ui.showCentroidsCheckBox.stateChanged.connect(self.redrawImage)
        self.ui.showNeckArcCheckBox.stateChanged.connect(self.redrawImage)
        self.ui.showMediaPipeLandmarksCheckBox.stateChanged.connect(self.redrawImage)

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
