import math
import sys
from pathlib import Path
from typing import Optional

import PySide6
from PIL import Image, ImageFile
from PySide6 import QtGui
from PySide6.QtCore import QFile, QObject, Qt
from PySide6.QtGui import QColor
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget
from src.fidmaa.calculations import findParalellPoint

from calculations import findPoint
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

        self.smallImage = None
        self.depthmap = None

        canvas = QtGui.QPixmap(480, 640)
        self.ui.imageLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(255, 640)
        self.ui.chartLabel.setPixmap(canvas)

        self.redrawImage()

    def redrawImage(self, *args, **kw):

        canvas = self.ui.imageLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        canvas.fill(Qt.white)
        if self.smallImage:
            painter.drawImage(0, 0, self.smallImage.toqimage())

        # Calculate 2 points at the edge of the image, using the angle.

        x = self.ui.xValue.value()
        chin_y = y = self.ui.yValue.value()
        angle = self.ui.angleValue.value()
        area = self.ui.areaValue.value()

        p1 = findPoint(x, y, direction=-1, angle=angle)
        p2 = findPoint(x, y, direction=1, angle=angle)
        painter.drawLine(p1, p2)

        # Get a coefficient of a perpendicular function!

        perpendicular_coefficient = -1.0 / math.tan(math.radians(angle))

        # Paint a perpendicular line
        pp1 = findPoint(
            x, y, direction=-1, linear_coefficient=perpendicular_coefficient
        )
        pp2 = findPoint(
            x,
            y,
            direction=1,
            linear_coefficient=perpendicular_coefficient,
        )
        # Paint a perpendicular line
        painter.drawLine(pp1, pp2)

        #
        # Left line
        #

        # Get the midpoint of a paralell line to the left
        lpx1, lpy1 = findParalellPoint(x, y, angle, distance=area)

        # Get the coordinates of paralell line to the left and draw it
        lp1 = findPoint(lpx1, lpy1, direction=1, angle=angle)
        lp2 = findPoint(lpx1, lpy1, direction=-1, angle=angle)

        painter.drawLine(lp1, lp2)

        #
        # Right line
        #

        # Get the midpoint of a paralell line to the right
        rpx1, rpy1 = findParalellPoint(x, y, angle, distance=area, direction=-1)

        # Get the coordinates of paralell line to the right and draw it
        rp1 = findPoint(rpx1, rpy1, direction=1, angle=angle)
        rp2 = findPoint(rpx1, rpy1, direction=-1, angle=angle)

        painter.drawLine(rp1, rp2)

        # Left image finished...
        painter.end()
        self.ui.imageLabel.setPixmap(canvas)

        # Now the right image -- the depths:

        canvas = self.ui.chartLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        canvas.fill(Qt.red)

        if self.depthmap:
            # debugCanvas = self.ui.imageLabel.pixmap()
            # debugPainter = QtGui.QPainter(debugCanvas)

            depthCutoff = self.ui.depthCutoffValue.value()
            depthMax = 255 - depthCutoff

            point_beg = p2
            point_end = p1

            if p1.y() < p2.y():
                point_beg = p1
                point_end = p2

            dx = (point_end.x() - point_beg.x()) / 640.0

            def mapCoordXToCutoff(data_value):
                return 255 * (data_value - depthCutoff) / depthMax

            chart_data = []
            for y in range(0, 640):

                sx = point_beg.x() + y * dx
                sy = y

                # find perpendicular points
                lpx, lpy = findParalellPoint(sx, sy, angle, distance=area, direction=1)
                rpx, rpy = findParalellPoint(sx, sy, angle, distance=area, direction=-1)

                # traverse the line between perpendicular points, gathering
                # depths, from left to right
                if lpx > rpx:
                    lpx, lpy = rpx, rpx

                dy = float(rpy - lpy) / float(rpx - lpx)

                depths = []

                # Debug paint
                # debugPainter.drawLine(lpx, lpy, rpx, rpy)

                for x in range(int(round(lpx)), int(round(rpx))):
                    depths.append(self.depthmap.getpixel((x, lpy + dy * x))[0])

                depth = sum(depths) / len(depths)

                if depth < depthCutoff:
                    depth = 0

                chart_data.append(depth)

                painter.drawLine(0, y, mapCoordXToCutoff(depth), y)

            # Get the lowest point downards from the chin and calculate it's
            # delta

            painter.setPen(QColor(0, 255, 0, 127))
            painter.drawLine(0, chin_y, 255, chin_y)

            minimum = (255, 0)

            for n in range(chin_y, len(chart_data)):
                if chart_data[n] < minimum[0]:
                    minimum = chart_data[n], n

            neck_x, neck_y = minimum

            maximum = (0, 0)

            for n in range(neck_y, 0, -1):
                if chart_data[n] > maximum[0]:
                    maximum = chart_data[n], n

            chin_x, chin_y = maximum

            # Lowest depth below midpoint
            painter.drawLine(
                mapCoordXToCutoff(neck_x),
                0,
                mapCoordXToCutoff(neck_x),
                640,
            )

            painter.drawLine(
                mapCoordXToCutoff(chin_x),
                0,
                mapCoordXToCutoff(chin_x),
                640,
            )

            # debugPainter.end()
            # self.ui.imageLabel.setPixmap(debugCanvas)

        painter.end()
        self.ui.chartLabel.setPixmap(canvas)

    def _loadJPEG(self, fileName):
        self.filename = fileName
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

    def loadJPEG(self, *args, **kw):
        fileName = QFileDialog.getOpenFileName(
            self, QObject.tr("Open File"), None, QObject.tr("Images (*.jpg)")
        )

        if fileName[0]:
            self._loadJPEG(fileName[0])

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

        self.ui.loadJPEGButton.clicked.connect(self.loadJPEG)
        self.ui.imageLabel.clicked.connect(self.setMidlinePoint)

        self.ui.xValue.valueChanged.connect(self.redrawImage)
        self.ui.yValue.valueChanged.connect(self.redrawImage)
        self.ui.angleValue.valueChanged.connect(self.redrawImage)
        self.ui.areaValue.valueChanged.connect(self.redrawImage)
        self.ui.depthCutoffValue.valueChanged.connect(self.redrawImage)

        self.ui.angleValue.setValue(90)
        self.ui.angleSlider.setValue(90)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = Widget()
    widget.show()

    try:
        if sys.argv[1]:
            widget._loadJPEG(sys.argv[1])
    except IndexError:
        pass

    sys.exit(app.exec())
