import io
import math
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Optional

import cv2
import numpy
import piexif
import pyheif
import PySide6
from bs4 import BeautifulSoup
from calculations import findParalellPoint, findPoint
from pi_heif import register_heif_opener
from piexif import InvalidImageDataError
from PIL import Image, ImageFile
from PySide6 import QtGui
from PySide6.QtCore import QFile, QObject, Qt
from PySide6.QtGui import QColor
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget
from QClickableLabel import QClickableLabel

from fidmaa import const, errors

register_heif_opener()


ImageFile.LOAD_TRUNCATED_IMAGES = True

tr = QObject.tr


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_ui()

        self.filename = None
        self.face = None

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
        painter.setPen(QColor(0, 0, 255, 127))
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
            depthMax = 256 - depthCutoff

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

            for n in range(chin_y, len(chart_data) - 10):
                # blur filter can give bad results if we go all the way down
                # (without -10)
                if chart_data[n] < minimum[0]:
                    minimum = chart_data[n], n

            neck_x, neck_y = minimum

            maximum = (0, 0)

            for n in range(neck_y, chin_y, -1):
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

            painter.drawLine(
                mapCoordXToCutoff(chin_x), chin_y, mapCoordXToCutoff(neck_x), neck_y
            )

            try:
                face_angle = "%.2f" % math.degrees(
                    math.asin((chin_y - neck_y) / chin_x)
                )
            except (ZeroDivisionError, ValueError):
                face_angle = "impossible to calc. "

            distance = 0
            for y in range(chin_y, neck_y - 1):
                delta = (
                    self.depthmap.getpixel((chin_x, y))[0]
                    - self.depthmap.getpixel((chin_x, y + 1))[0]
                )
                distance += delta

            self.ui.dataOutputEdit.clear()
            self.ui.dataOutputEdit.appendPlainText(
                dedent(
                    f"""
            Highest chin point at: {chin_x:.0f},
            Lowest neck point at: {neck_x:.0f},
            Difference: {chin_x - neck_x:.0f},
            Angle: {face_angle} deg,
            Distance: {distance} pixels.
            """
                )
            )

            # debugPainter.end()
            # self.ui.imageLabel.setPixmap(debugCanvas)

        painter.end()
        self.ui.chartLabel.setPixmap(canvas)

    def _loadImage(self, fileName):

        self.filename = fileName

        buf = io.BytesIO()

        if self.filename.lower().endswith("heic") or self.filename.lower().endswith(
            "heif"
        ):
            #
            # Get depth map from HEIC/HEIF container, then proceed normally:
            #
            heif_container = pyheif.open_container(open(self.filename, "rb"))

            primary_image = heif_container.primary_image

            for exif_metadata in [
                metadata
                for metadata in primary_image.image.load().metadata
                if metadata.get("type", "") == "Exif"
            ]:
                exif = piexif.load(exif_metadata["data"])
                if not self.check_exif_data(exif):
                    return

            depth_image = primary_image.depth_image.image.load()
            self.depthmap = Image.frombytes(
                depth_image.mode, depth_image.size, depth_image.data, "raw"
            )

            image = self.image = Image.open(self.filename)
        else:
            exif = None
            try:
                exif = piexif.load(self.filename)
            except InvalidImageDataError:
                pass

            if exif is not None:
                if not self.check_exif_data(exif):
                    return

            image = Image.open(self.filename, formats=["JPEG"])

            # XMP data
            f = open(self.filename, "rb")
            d = f.read()
            xmp_str = b""

            while d:

                xmp_start = d.find(b"<x:xmpmeta")
                xmp_end = d.find(b"</x:xmpmeta")
                xmp_str += d[xmp_start : xmp_end + 12]
                d = d[xmp_end + 12 :]

            xmpAsXML = BeautifulSoup(xmp_str, features="html.parser")

            found = False

            for _no, tag in enumerate(
                xmpAsXML.findAll("apdi:auxiliaryimagetype"), start=1
            ):
                if tag.text.find("disparity") >= 0:
                    found = True
                    break

            if not found:
                _no = 1

            try:
                image.seek(_no)
            except EOFError:
                self.critical_error(errors.NO_DEPTH_DATA_ERROR)
                return

            image.save(buf, format="png")
            self.depthmap = Image.open(buf)

            image.seek(0)
            self.image = image

        smallImage = image.resize((480, 640))
        self.smallImage = smallImage

        #
        # Guess face position
        #

        image = numpy.array(self.image.convert("RGB"))

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
        )
        try:
            face = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=4)
        except BaseException:
            face = []

        #
        # Update midline point to match detected face coords
        #

        if len(face) == 1:
            x, y, wi, he = face[0]

            print(x, y, wi, he)
            center_x = x + wi / 2
            center_y = y + he / 2

            # calculate percentage of the face
            img_wi, img_he = self.image.getbbox()[2:4]
            percent_width = float(wi) / float(img_wi)
            percent_height = float(he) / float(img_he)

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

            # Set lower point somewhere around mouth (below nose, above chin)

            self.ui.xValue.setValue(int(round(center_x / img_wi * 480)))
            self.ui.yValue.setValue(int(round((center_y + he / 4) / img_he * 640)))

        elif len(face) == 0:
            self.critical_error(errors.FACE_NOT_DETECTED)

        else:
            self.critical_error(errors.MULTIPLE_FACES_DETECTED)

        self.redrawImage()

    def critical_error(self, err):
        QMessageBox.critical(
            self,
            tr("FIDMAA error"),
            tr(err),
            QMessageBox.Cancel,
        )

    def check_exif_data(self, exif):
        data = exif.get("Exif", {})
        data = data.get(42036, "default")

        if type(data) == str:
            ret = data.find(const.TRUEDEPTH_EXIF_ID)
            print_data = data
        elif type(data) == bytes:
            ret = data.find(const.TRUEDEPTH_EXIF_ID.encode("ascii"))
            try:
                print_data = data.decode("ascii")
            except BaseException:
                print_data = "cannot encode"
        else:
            ret = -1

        if ret == -1:
            QMessageBox.critical(
                self,
                tr("FIDMAA notification"),
                errors.NO_FRONT_CAMERA_NOTIFICATION.format(
                    exif_camera_description=print_data
                ),
            )
            return False
        return True

    def loadJPEG(self, *args, **kw):
        fileName = QFileDialog.getOpenFileName(
            self,
            QObject.tr("Open File"),
            os.path.expanduser("~/Downloads"),
            QObject.tr("Images (*.jpg; *.jpeg; *.heic; *.heif)"),
        )

        if fileName[0]:
            self._loadImage(fileName[0])

    def setMidlinePoint(self, point, *args, **kw):
        self.ui.xValue.setValue(point.x())
        self.ui.yValue.setValue(point.y())

    def setMidlineY(self, point, *args, **kw):
        # self.ui.xValue.setValue(point.x())
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
        self.ui.chartLabel.clicked.connect(self.setMidlineY)

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
            widget._loadImage(sys.argv[1])
    except IndexError:
        pass

    sys.exit(app.exec())
