from PIL import ImageFile
from PIL.Image import Image
from PySide6 import QtGui
from PySide6.QtCore import QObject, QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

ImageFile.LOAD_TRUNCATED_IMAGES = True

tr = QObject.tr
from PIL import ImageFile
from PySide6.QtCore import QObject

ImageFile.LOAD_TRUNCATED_IMAGES = True

tr = QObject.tr

from .utils import UILoaderMixin


class ZoomWindow(UILoaderMixin, QWidget):
    uifile_name = "zoom_window.ui"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_ui()

    def connect_ui(self):
        canvas = QtGui.QPixmap(480, 320)
        self.ui.zoomedImageLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(480, 320)
        self.ui.zoomedDepthMapLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(480, 256)
        self.ui.reconstructionLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(480, 320)
        self.ui.zoomedSkinMapLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(480, 320)
        self.ui.zoomedTeethMapLabel.setPixmap(canvas)

    def _paintZoomedImage(self, image: Image, ui_element: QWidget):
        canvas = ui_element.pixmap()
        painter = QtGui.QPainter(canvas)
        canvas.fill(Qt.green)
        painter.drawImage(0, 0, image.toqimage())

        painter.setPen(QColor(255, 0, 0, 255))

        painter.drawLine(
            QPoint(
                240,
                0,
            ),
            QPoint(240, 320),
        )
        painter.drawLine(QPoint(0, 160), QPoint(480, 160))

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
        painter.drawImage(0, 0, image_map.toqimage())

        painter.setPen(QColor(255, 0, 0, 255))

        painter.drawLine(
            QPoint(
                240,
                0,
            ),
            QPoint(240, 320),
        )
        painter.drawLine(QPoint(0, 160), QPoint(480, 160))

        font = painter.font()
        font.setPixelSize(32)
        painter.setFont(font)
        try:
            value = image_map.getpixel((240, 160))[0]
        except TypeError:
            value = image_map.getpixel((240, 160))

        if value < ok_value_threshold:
            painter.setPen(QColor(255, 0, 0, 255))
        else:
            painter.setPen(QColor(0, 255, 0, 255))
        painter.drawText(QPoint(50, 50), str(label))
        painter.drawText(QPoint(50, 100), str(value))
        painter.drawText(QPoint(50, 150), str(int(mouse_x)) + " x " + str(int(mouse_y)))

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
        canvas = self.ui.reconstructionLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        canvas.fill(Qt.yellow)
        painter.setPen(QColor(0, 0, 0, 255))

        for a in range(480):
            v = values[int(a * len(values) / 480.0)]
            painter.drawLine(
                QPoint(
                    a,
                    256,
                ),
                QPoint(a, 256 - v),
            )

        painter.end()
        self.ui.reconstructionLabel.setPixmap(canvas)
