import math
import os
import sys
import traceback
from pathlib import Path
from textwrap import dedent
from typing import Optional

import PySide6
from fidmaa_simple_viewer.core import FIDMAA_to_pyvista_surface
from PIL import Image, ImageFile, ImageFilter
from portrait_analyser.exceptions import (
    ExifValidationFailed,
    MultipleFacesDetected,
    NoDepthMapFound,
    NoFacesDetected,
    UnknownExtension,
)
from portrait_analyser.face import get_face_parameters
from portrait_analyser.ios import IOSPortrait, load_image
from PySide6 import QtGui
from PySide6.QtCore import QFile, QObject, QPoint, QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from . import const, errors
from .calculations import findPoint
from .QClickableLabel import QClickableLabel

ImageFile.LOAD_TRUNCATED_IMAGES = True

tr = QObject.tr


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


def CV2_to_QImage(cv2_image):
    return QtGui.QImage(
        cv2_image.data,
        cv2_image.shape[1],
        cv2_image.shape[0],
        QtGui.QImage.Format_RGB888,
    ).rgbSwapped()


def interpolate_pixels_along_line(x1, y1, z1, x2, y2, z2):
    dist_x = x2 - x1
    dist_y = y2 - y1
    dist_z = z2 - z1

    # line_len = math.sqrt(dist_x**2 + dist_y**2 + dist_z**2)

    abs_dist_x = abs(dist_x)
    abs_dist_y = abs(dist_y)
    abs_dist_z = abs(dist_z)

    if abs_dist_x >= abs_dist_y and abs_dist_x >= abs_dist_z:
        no_steps = abs_dist_x
    elif abs_dist_y >= abs_dist_x and abs_dist_y >= abs_dist_z:
        no_steps = abs_dist_y
    elif abs_dist_z >= abs_dist_x and abs_dist_z >= abs_dist_y:
        no_steps = abs_dist_z
    else:
        raise NotImplementedError(dist_x, dist_y, dist_z)

    if no_steps == 0:
        return

    delta_x = dist_x / no_steps
    delta_y = dist_y / no_steps
    delta_z = dist_z / no_steps

    for a in range(no_steps + 1):
        yield (x1, y1, z1)
        x1 += delta_x
        y1 += delta_y
        z1 += delta_z


def clamp(n, minn, maxn):
    return max(min(maxn - 1, n), minn)


class UILoaderMixin:
    def load_ui(self):
        loader = MyQUiLoader(self)

        if hasattr(sys, "_MEIPASS"):
            path = os.path.join(sys._MEIPASS, self.uifile_name)
        else:
            path = Path(__file__).resolve().parent / self.uifile_name

        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()

        self.ui.show()
        self.connect_ui()

    def connect_ui(self):
        pass


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

    def paintZoomedImage(self, zoomed):
        canvas = self.ui.zoomedImageLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        canvas.fill(Qt.green)
        painter.drawImage(0, 0, zoomed.toqimage())

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
        self.ui.zoomedImageLabel.setPixmap(canvas)

    def paintZoomedDepthmap(self, depthmap):
        canvas = self.ui.zoomedDepthMapLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        canvas.fill(Qt.yellow)
        painter.drawImage(0, 0, depthmap.toqimage())

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
        value = depthmap.getpixel((240, 160))[0]
        if value < 100:
            painter.setPen(QColor(255, 0, 0, 255))
        else:
            painter.setPen(QColor(0, 255, 0, 255))
        painter.drawText(QPoint(50, 50), str(value))

        painter.end()
        self.ui.zoomedDepthMapLabel.setPixmap(canvas)

    def paintReconstruction(self, values):
        canvas = self.ui.reconstructionLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        canvas.fill(Qt.yellow)
        painter.setPen(QColor(0, 0, 0, 255))

        # values_min = min(values)
        # values_max = max(values)

        for a in range(480):
            v = values[int(a * len(values) / 480.0)]
            # v = 256 * ((v - values_min) / (values_max))
            painter.drawLine(
                QPoint(
                    a,
                    256,
                ),
                QPoint(a, 256 - v),
            )

        painter.end()
        self.ui.reconstructionLabel.setPixmap(canvas)


class MainWindow(UILoaderMixin, QWidget):
    uifile_name = "form.ui"

    def __init__(self, parent=None, zoomWindow=None):
        super().__init__(parent)
        self.load_ui()

        self.filename = None
        self.face = None

        self.smallImage = None
        self.portrait: IOSPortrait = None
        self.depthmap = None
        self.teethmap = None

        self.float_max_value = self.float_min_value = None

        self.zoomWindow = zoomWindow

        self.last_click_x = None
        self.last_click_y = None
        self.last_angle = None
        self.last_depth = None
        self.face = None

        self.redrawImage()
        self.redrawZoom()

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

        if self.zoomWindow:
            if self.smallImage:
                big_image_x = mouse_x * self.image.size[0] / 480
                big_image_y = mouse_y * self.image.size[1] / 640
                zoomed = self.image.crop(
                    (
                        big_image_x - 120,
                        big_image_y - 80,
                        big_image_x + 120,
                        big_image_y + 80,
                    )
                ).resize((480, 320))

                self.zoomWindow.paintZoomedImage(
                    zoomed,
                )

            if self.depthmap:
                zoomed = (
                    self.depthmap.crop(
                        (mouse_x - 144, mouse_y - 96, mouse_x + 144, mouse_y + 96)
                    )
                    .resize((480, 320), Image.HAMMING)
                    .filter(ImageFilter.SHARPEN)
                    .filter(ImageFilter.SHARPEN)
                    .filter(ImageFilter.SHARPEN)
                )
                self.zoomWindow.paintZoomedDepthmap(zoomed)

    def redrawImage(self, *args, **kw):
        mouse_x = x = self.ui.xValue.value()
        y = mouse_y = self.ui.yValue.value()
        angle = self.ui.angleValue.value()

        mouse_x = clamp(mouse_x, 0, 480)
        mouse_y = clamp(mouse_y, 0, 640)

        if self.last_click_x is not None:
            if (
                self.last_click_x == mouse_x
                and self.last_click_y == mouse_y
                and self.last_angle == angle
            ):
                return

        self.last_angle = angle

        canvas = self.ui.imageLabel.pixmap()
        painter = QtGui.QPainter(canvas)
        painter.setPen(QColor(0, 0, 255, 127))
        painter.pen().setDashOffset(2)
        canvas.fill(Qt.white)
        if self.smallImage:
            painter.drawImage(0, 0, self.smallImage.toqimage())

        # if self.teethmap:
        #     ni = self.teethmap.resize((480, 640)).filter(
        #         ImageFilter.MedianFilter(size=3)
        #     )
        #
        #     painter.drawImage(0, 0, ni.toqimage())

        if self.portrait:
            if self.portrait.teeth_bbox:
                tx, ty, twi, the = self.portrait.teeth_bbox_translated(480, 640)
                painter.setPen(QColor(255, 255, 0, 127))
                painter.drawRect(tx, ty, twi, the)

        if self.face:
            painter.setPen(QColor(0, 0, 255, 127))
            face_rect = self.face.translate_coordinates(480, 640)
            painter.drawRect(*face_rect)

            for eye in self.face.eyes:
                painter.setPen(QColor(0, 255, 0, 127))
                rect = eye.translate_coordinates(480, 640)
                painter.drawRect(*rect)

        painter.setPen(QColor(0, 0, 255, 127))

        # Calculate 2 points at the edge of the image, using the angle.

        p1 = findPoint(x, y, direction=-1, angle=angle)
        p2 = findPoint(x, y, direction=1, angle=angle)
        painter.drawLine(p1, p2)

        if self.last_click_x is not None:
            painter.setPen(QColor(255, 0, 0, 127))
            painter.drawLine(
                QPoint(mouse_x, mouse_y), QPoint(self.last_click_x, self.last_click_y)
            )

        if self.portrait and self.portrait.teeth_bbox:
            smx, smy, smwi, smhe = self.portrait.teeth_bbox_translated(480, 640)
            smy += 3
            smhe -= 6

            z1 = self.get_depthmap_value(smx + smwi / 2, smy)
            z2 = self.get_depthmap_value(smx + smwi / 2, smy + smhe)

            distance_z1 = self.get_depthmap_distance(z1)
            distance_z2 = self.get_depthmap_distance(z2)

            distance_x1, distance_y1 = self.translate_click_to_mm(
                distance_z1, smx + smwi / 2, smy
            )

            distance_x2, distance_y2 = self.translate_click_to_mm(
                distance_z2,
                smx + smwi / 2,
                smy + smhe,
            )

            teethbox_args = (
                distance_x1,
                distance_y1,
                distance_z1,
                distance_x2,
                distance_y2,
                distance_z2,
            )
            # print("Teethbox", args)
            painter.setPen(QColor(255, 255, 0, 255))
            painter.drawLine(smx + smwi / 2, smy, smx + smwi / 2, smhe + smy)

        painter.end()
        self.ui.imageLabel.setPixmap(canvas)

        # Now the right image -- the depths:

        canvas = self.ui.chartLabel.pixmap()
        painter = QtGui.QPainter(canvas)
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
                self.zoomWindow.paintReconstruction(values)

            painter.end()
            self.ui.chartLabel.setPixmap(canvas)

            mouse_x = clamp(mouse_x, 0, 480)
            mouse_y = clamp(mouse_y, 0, 640)

            if self.last_click_x is None:
                line_len = 0
            else:
                line_x = abs(self.last_click_x - mouse_x)
                line_y = abs(self.last_click_y - mouse_y)
                line_len = self.calculate_line_length(line_x, line_y)

            surface_length_3d = vector_len_voxels = vector_length_3d = 0
            if (
                self.last_click_x != mouse_x or self.last_click_y != mouse_y
            ) and self.last_click_x is not None:
                z1 = self.get_depthmap_value(mouse_x, mouse_y)
                z2 = self.get_depthmap_value(self.last_click_x, self.last_click_y)

                vector_len_voxels = self.vector_length_simple(
                    mouse_x, mouse_y, z1, self.last_click_x, self.last_click_y, z2
                )

                distance_z1 = self.get_depthmap_distance(z1)
                distance_z2 = self.get_depthmap_distance(z2)

                distance_x1, distance_y1 = self.translate_click_to_mm(
                    distance_z1, mouse_x, mouse_y
                )
                distance_x2, distance_y2 = self.translate_click_to_mm(
                    distance_z2, self.last_click_x, self.last_click_y
                )
                args = (
                    distance_x1,
                    distance_y1,
                    distance_z1,
                    distance_x2,
                    distance_y2,
                    distance_z2,
                )
                vector_length_3d = self.vector_length_simple(*args)

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

            self.ui.dataOutputEdit.clear()
            txt = dedent(
                f"""
            Depth map coords:
            {mouse_x, mouse_y}

            Depth map raw data:
            {closeness} (Δ: {closeness_delta})

            Depth map distance:
            {depth_mm:.2f} cm (Δ: {closeness_delta_mm:.1f} cm)

            Line length (2D, on flat picture):
            {line_len:.2f} pixels

            Vector length (3D) simple - on raw data:
            {vector_len_voxels:.2f} voxels

            Vector length (3D) with depth:
            {vector_length_3d / 10.0:.2f} cm

            Vector length (3D) on surface:
            {(surface_length_3d / 10.0):.2f} cm"""
            )

            if self.portrait.teeth_bbox:
                txt += "\n\nAutomatic incisor distance:\n"
                txt += "%.2f cm" % (self.vector_length_simple(*teethbox_args) / 10.0)

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

            self.ui.dataOutputEdit.appendPlainText(txt.strip())

    def get_depthmap_value(self, x, y):
        return self.depthmap.getpixel((x, y))[0]

    def translate_click_to_mm(
        self, distance_cm, x, y, SMALL_WIDTH=480, SMALL_HEIGHT=640
    ):
        return (
            self.how_many_mm_per_pixels_at_distance_on_big_image(
                distance_cm, x * self.image.size[0] / SMALL_WIDTH
            ),
            self.how_many_mm_per_pixels_at_distance_on_big_image(
                distance_cm, y * self.image.size[1] / SMALL_HEIGHT
            ),
        )

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
            z1 = self.get_depthmap_distance(self.get_depthmap_value(x1, y1))
            x1, y1 = self.translate_click_to_mm(z1, x1, y1)

            z2 = self.get_depthmap_distance(self.get_depthmap_value(x2, y2))
            x2, y2 = self.translate_click_to_mm(z2, x2, y2)

            line_len_3d = self.vector_length_simple(x1, y1, z1, x2, y2, z2)
            s.append(line_len_3d)
        return sum(s)

    def vector_length_simple(self, x1, y1, z1, x2, y2, z2):
        """Simple mathematical lenght of the vector"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def calculate_line_length(self, dist_x, dist_y):
        line_len = math.sqrt(abs(dist_x * dist_x) + abs(dist_y * dist_y))
        return line_len

    def how_many_pixels_per_mm_at_distance_on_big_image(self, distance, mm):
        """Returns how many pixels take up a 1 milimiter at a given distance (cm) from camera.

        Constants taken from own calibration data and a curve fitted by MyCurveFit.com,
        I strongly recommend their service, it is very easy to use and affordable.

        :param distance: The distance in centimeters from the camera
        :param no_pixels: line length in pixels, must be in original image size (2300x3000)
        """
        # return mm * -0.04378155 + (189.5944 - -0.04378155) / (
        #     1 + (distance / 1.81124) ** 1.056448
        # )
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
            self.portrait: IOSPortrait = load_image(self.filename)
            self.image = self.portrait.photo
            self.depthmap = self.portrait.depthmap
            self.teethmap = self.portrait.teethmap
            self.float_min_value = self.portrait.floatValueMin
            self.float_max_value = self.portrait.floatValueMax

            if self.float_max_value is not None:
                self.float_max_value = float(self.float_max_value)

            if self.float_min_value is not None:
                self.float_min_value = float(self.float_min_value)

            # self.depthmap = self.depthmap.filter(ImageFilter.GaussianBlur)
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

        self.smallImage = self.image.resize((480, 640))

        # If pictures taken with the back camera, the main miage should be mirrored to match
        # the depth map... then the depth map should be mirrored if printing in 3D... currently
        # I'm leaving this comment & not supporting it (the back camera).
        # self.depthmap = ImageOps.mirror(self.depthmap)

        #
        # Get face position, if any:
        #

        try:
            self.face = get_face_parameters(self.image, raise_opencv_exceptions=True)
        except NoFacesDetected:
            self.face = None
            self.critical_error(errors.FACE_NOT_DETECTED)

        except MultipleFacesDetected:
            self.critical_error(errors.MULTIPLE_FACES_DETECTED)

        except BaseException:
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

            # Set lower point somewhere around mouth (below nose, above chin)

            self.ui.xValue.setValue(
                int(round(self.face.center_x / self.image.size[0] * 479))
            )
            self.ui.yValue.setValue(
                int(
                    round(
                        (self.face.center_y + self.face.height / 4)
                        / self.image.size[1]
                        * 639
                    )
                )
            )

        self.last_click_x = None
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

    def showZoomWindow(self, *args, **kw):
        if self.zoomWindow:
            self.zoomWindow.show()
            self.zoomWindow.raise_()

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
        new_depthmap = self.depthmap.convert("L")
        #
        # for y in range(640):
        #     for x in range(480):
        #         pixel = new_depthmap.getpixel((x, y))
        #
        #         npixel = (
        #             100
        #             * 1.0
        #             / (
        #                 self.float_max_value * pixel / 255
        #                 + self.float_min_value * (1 - pixel / 255)
        #             )
        #         )
        #
        #         npixel = int(math.ceil(npixel))
        #
        #         new_depthmap.putpixel((x, y), npixel)

        surface, texture = FIDMAA_to_pyvista_surface(
            self.image,
            new_depthmap,  # self.depthmap.filter(ImageFilter.BLUR)
        )
        from pyvistaqt import BackgroundPlotter

        plotter = BackgroundPlotter(
            line_smoothing=True, title=self.getWindowTitle(self.filename, "3D view")
        )
        plotter.add_mesh(surface, texture=texture)
        plotter.add_text("FIDMAA (C) 2022-2024 Michal Pasternak & collaborators ")
        plotter.show()

    def connect_ui(self):
        canvas = QtGui.QPixmap(480, 640)
        self.ui.imageLabel.setPixmap(canvas)

        canvas = QtGui.QPixmap(255, 640)
        self.ui.chartLabel.setPixmap(canvas)

        self.ui.showZoomWindowButton.clicked.connect(self.showZoomWindow)
        self.ui.loadJPEGButton.clicked.connect(self.loadJPEG)
        self.ui.open3DViewButton.clicked.connect(self.open3DView)
        self.ui.imageLabel.clicked.connect(self.setMidlinePoint)
        self.ui.imageLabel.setMouseTracking(True)
        self.ui.imageLabel.mouseMoveEvent = self.redrawZoom
        self.ui.imageLabel.setCursor(Qt.CursorShape.CrossCursor)
        self.ui.chartLabel.clicked.connect(self.setMidlineY)

        self.ui.angleValue.valueChanged.connect(self.redrawImage)

        self.ui.angleValue.setValue(90)
        self.ui.angleSlider.setValue(90)


def main():
    app = QApplication(sys.argv)

    zoomWindow = ZoomWindow()
    zoomWindow.setWindowTitle("FIDMAA zoom")
    zoomWindow.show()
    zoomWindow.move(10, 10)

    mainWindow = MainWindow(zoomWindow=zoomWindow)
    mainWindow.updateWindowTitle()
    mainWindow.show()

    try:
        if sys.argv[1]:
            mainWindow._loadImage(os.path.expanduser(sys.argv[1]))
    except IndexError:
        mainWindow.loadJPEG()

    sys.exit(app.exec())
