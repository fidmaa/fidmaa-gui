from PIL import ImageFile
from PIL.Image import Image
from PySide6 import QtGui
from PySide6.QtCore import QObject, QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

from . import const
from .utils import UILoaderMixin

ImageFile.LOAD_TRUNCATED_IMAGES = True

tr = QObject.tr


class ZoomWindow(UILoaderMixin, QWidget):
    uifile_name = "zoom_window.ui"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_ui()

    def connect_ui(self):
        canvas = QtGui.QPixmap(const.ZOOM_WIDTH, const.ZOOM_HEIGHT)
        self.ui.zoomedImageLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(const.ZOOM_WIDTH, const.ZOOM_HEIGHT)
        self.ui.zoomedDepthMapLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(const.ZOOM_WIDTH, const.RECONSTRUCTION_HEIGHT)
        self.ui.reconstructionLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(const.ZOOM_WIDTH, const.ZOOM_HEIGHT)
        self.ui.zoomedSkinMapLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(const.ZOOM_WIDTH, const.ZOOM_HEIGHT)
        self.ui.zoomedTeethMapLabel.setPixmap(canvas)

    def _paintZoomedImage(self, image: Image, ui_element: QWidget):
        canvas = ui_element.pixmap()
        painter = QtGui.QPainter(canvas)
        try:
            canvas.fill(Qt.green)
            painter.drawImage(0, 0, image.toqimage())

            painter.setPen(QColor(255, 0, 0, 255))

            half_w = const.ZOOM_WIDTH // 2
            half_h = const.ZOOM_HEIGHT // 2
            painter.drawLine(
                QPoint(half_w, 0),
                QPoint(half_w, const.ZOOM_HEIGHT),
            )
            painter.drawLine(
                QPoint(0, half_h),
                QPoint(const.ZOOM_WIDTH, half_h),
            )
        finally:
            painter.end()
        ui_element.setPixmap(canvas)

    def paintZoomedImage(self, zoomed):
        return self._paintZoomedImage(zoomed, self.ui.zoomedImageLabel)

    def _paintZoomedMap(
        self,
        label: str,
        image_map: Image,
        ui_element: QWidget,
        mouse_x: int = None,
        mouse_y: int = None,
        ok_value_threshold: int = 100,
    ):
        canvas = ui_element.pixmap()
        painter = QtGui.QPainter(canvas)
        try:
            painter.drawImage(0, 0, image_map.toqimage())

            painter.setPen(QColor(255, 0, 0, 255))

            half_w = const.ZOOM_WIDTH // 2
            half_h = const.ZOOM_HEIGHT // 2
            painter.drawLine(
                QPoint(half_w, 0),
                QPoint(half_w, const.ZOOM_HEIGHT),
            )
            painter.drawLine(
                QPoint(0, half_h),
                QPoint(const.ZOOM_WIDTH, half_h),
            )

            font = painter.font()
            font.setPixelSize(32)
            painter.setFont(font)
            try:
                value = image_map.getpixel((half_w, half_h))[0]
            except TypeError:
                value = image_map.getpixel((half_w, half_h))

            if value < ok_value_threshold:
                painter.setPen(QColor(255, 0, 0, 255))
            else:
                painter.setPen(QColor(0, 255, 0, 255))
            painter.drawText(QPoint(50, 50), str(label))
            painter.drawText(QPoint(50, 100), str(value))
            painter.drawText(
                QPoint(50, 150), str(int(mouse_x)) + " x " + str(int(mouse_y))
            )
        finally:
            painter.end()
        ui_element.setPixmap(canvas)

    def paintZoomedDepthmap(self, depthmap, mouse_x=None, mouse_y=None):
        return self._paintZoomedMap(
            "Depth map", depthmap, self.ui.zoomedDepthMapLabel, mouse_x, mouse_y
        )

    def paintZoomedSkinmap(self, skinmap, mouse_x=None, mouse_y=None):
        return self._paintZoomedMap(
            "Skin map",
            skinmap,
            self.ui.zoomedSkinMapLabel,
            mouse_x,
            mouse_y,
            ok_value_threshold=200,
        )

    def paintZoomedTeethmap(self, teethmap, mouse_x=None, mouse_y=None):
        return self._paintZoomedMap(
            "Teeth map",
            teethmap,
            self.ui.zoomedTeethMapLabel,
            mouse_x,
            mouse_y,
            ok_value_threshold=200,
        )

    def paintReconstruction(self, values):
        if not values:
            return

        canvas = self.ui.reconstructionLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        try:
            canvas.fill(Qt.yellow)
            painter.setPen(QColor(0, 0, 0, 255))

            for a in range(const.ZOOM_WIDTH):
                idx = min(
                    int(a * len(values) / float(const.ZOOM_WIDTH)), len(values) - 1
                )
                v = values[idx]
                painter.drawLine(
                    QPoint(a, const.RECONSTRUCTION_HEIGHT),
                    QPoint(a, const.RECONSTRUCTION_HEIGHT - v),
                )
        finally:
            painter.end()
        self.ui.reconstructionLabel.setPixmap(canvas)
