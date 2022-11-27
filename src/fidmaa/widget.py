import math
import sys
from pathlib import Path
from typing import Optional

import PySide6
from PIL import Image, ImageFile
from PySide6 import QtGui
from PySide6.QtCore import QFile, QObject, Qt
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QWidget

from core import findPoint
from QClickableLabel import QClickableLabel

ImageFile.LOAD_TRUNCATED_IMAGES = True

tr = QObject.tr

NO_DEPTH_DATA_ERROR = """
Looks like this image has no depth data. Make sure you took the photo  without any 'Move furthrer from the subject' message on the phone.

This application currently supports selfies (photos taken with the front-facing camera) taken on the iPhone in **portrait** mode. Other kinds pictures contain no depth data.

If instead of JPEG your phone transfers a HEIC/HEIF file, this means you took the photo too close or too far away. Make sure there are no "Move away from the subject" messages.
"""


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_ui()

        self.chartWindow = None
        self.smallImage = None

        canvas = QtGui.QPixmap(480, 640)
        self.ui.imageLabel.setPixmap(canvas)

        self.redrawImage()

    def showChartWindow(self, *args, **kw):
        if self.chartWindow is not None:
            self.chartWindow.show()
            return

        self.chartWindow = ChartWindow()
        self.showChartWindow(*args, **kw)

    def redrawImage(self, *args, **kw):

        canvas = self.ui.imageLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        canvas.fill(Qt.white)
        if self.smallImage:
            painter.drawImage(0, 0, self.smallImage.toqimage())

        # Calculate 2 points at the edge of the image, using the angle.

        x = self.ui.xValue.value()
        y = self.ui.yValue.value()
        angle = self.ui.angleValue.value()
        area = self.ui.areaValue.value()

        p1 = findPoint(x, y, direction=-1, angle=angle)
        p2 = findPoint(x, y, direction=1, angle=angle)
        painter.drawLine(p1, p2)

        # Get a coefficient of a perpendicular function!

        perpendicular_coefficient = -1.0 / math.tan(math.radians(angle))

        # Paint a perpendicular line
        p1 = findPoint(x, y, direction=-1, linear_coefficient=perpendicular_coefficient)
        p2 = findPoint(
            x,
            y,
            direction=1,
            linear_coefficient=perpendicular_coefficient,
        )
        # Paint a perpendicular line
        painter.drawLine(p1, p2)

        #
        # Left line
        #

        # Get the midpoint of a paralell line to the left
        px1 = x + math.sin(math.radians(-angle)) * area
        py1 = y + math.cos(math.radians(-angle)) * area

        # Get the coordinates of paralell line to the left
        p1 = findPoint(px1, py1, direction=1, angle=angle)
        p2 = findPoint(px1, py1, direction=-1, angle=angle)

        painter.drawLine(p1, p2)

        #
        # Right line
        #

        # Get the midpoint of a paralell line to the right
        px1 = x - math.sin(math.radians(-angle)) * area
        py1 = y - math.cos(math.radians(-angle)) * area

        # Get the coordinates of paralell line to the right
        p1 = findPoint(px1, py1, direction=1, angle=angle)
        p2 = findPoint(px1, py1, direction=-1, angle=angle)

        painter.drawLine(p1, p2)

        # p1 = findPoint(dx1, dy1, direction=-1, angle=angle)
        # p2 = findPoint(dx2, dy2, direction=1, angle=angle)
        # painter.drawLine(p1, p2)

        # Now we need 2 points which lie on the perpendicular line

        # p1 = findPoint(x + 20, y, direction=-1)
        # p2 = findPoint(x + 20, y, direction=1)
        # painter.drawLine(p1, p2)

        # painter.drawLine(self.line_x, 0, self.line_x, 640)
        # painter.drawLine(self.line_x - 20, 0, self.line_x - 20, 640)
        # painter.drawLine(self.line_x + 20, 0, self.line_x + 20, 640)
        painter.end()
        self.ui.imageLabel.setPixmap(canvas)

    def loadJPEG(self, *args, **kw):
        fileName = QFileDialog.getOpenFileName(
            self, QObject.tr("Open File"), None, QObject.tr("Images (*.jpg)")
        )

        if fileName[0]:
            self.filename = fileName[0]
            image = Image.open(self.filename)

            smallImage = image.resize((480, 640))

            try:
                image.seek(1)
            except EOFError:
                QMessageBox.critical(
                    self,
                    tr("FIDMAA error"),
                    tr(NO_DEPTH_DATA_ERROR),
                    QMessageBox.Cancel,
                )
                return

            image.save("depthmap.jpg")

            self.image = image
            self.smallImage = smallImage
            self.depthmap = Image.open("depthmap.jpg")

            self.redrawImage()

        # XXX robienie wykresu wg linii i rysowanie linii

    def setMidlinePoint(self, point, *args, **kw):
        self.ui.xValue.setValue(point.x())
        self.ui.yValue.setValue(point.y())

    def load_ui(self):
        class MyQUiLoader(QUiLoader):
            def createWidget(
                self,
                className: str,
                parent: Optional[PySide6.QtWidgets.QWidget] = ...,
                name: str = ...,
            ) -> PySide6.QtWidgets.QWidget:
                if className == "QClickableLabel":
                    return QClickableLabel(parent=parent)
                return super(MyQUiLoader, self).createWidget(className, parent, name)

        loader = MyQUiLoader()

        path = Path(__file__).resolve().parent / "form.ui"
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()

        self.ui.showChartButton.clicked.connect(self.showChartWindow)
        self.ui.loadJPEGButton.clicked.connect(self.loadJPEG)
        self.ui.imageLabel.clicked.connect(self.setMidlinePoint)

        self.ui.xValue.valueChanged.connect(self.redrawImage)
        self.ui.yValue.valueChanged.connect(self.redrawImage)
        self.ui.angleValue.valueChanged.connect(self.redrawImage)
        self.ui.areaValue.valueChanged.connect(self.redrawImage)

        self.ui.angleValue.setValue(90.0)


class ChartWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_ui()

    def load_ui(self):
        loader = QUiLoader()
        path = Path(__file__).resolve().parent / "chart.ui"
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = Widget()
    widget.show()

    sys.exit(app.exec())
