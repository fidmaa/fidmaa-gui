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
from portrait_analyser.face import find_neck_measurement_point, get_face_parameters
from portrait_analyser.ios import IOSPortrait, load_image
from PySide6 import QtGui
from PySide6.QtCore import QObject, QPoint, QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

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
from .zoomWindow import ZoomWindow

ImageFile.LOAD_TRUNCATED_IMAGES = True

tr = QObject.tr


def _show_3d_view(image, depthmap, float_min, float_max):
    """Show 3D view in a separate process to avoid VTK/Qt conflicts on macOS."""
    from fidmaa_simple_viewer.core import pyvista_show

    pyvista_show(image, depthmap, float_min, float_max)


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

        self.last_5_distances_vect = []
        self.last_5_distances_srfc = []

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

        img_w = const.MAIN_IMAGE_WIDTH
        img_h = const.MAIN_IMAGE_HEIGHT
        zoom_w = const.ZOOM_WIDTH
        zoom_h = const.ZOOM_HEIGHT

        if self.zoomWindow:
            if self.smallImage:
                big_image_x = mouse_x * self.image.size[0] / img_w
                big_image_y = mouse_y * self.image.size[1] / img_h
                zoomed = self.image.crop(
                    (
                        big_image_x - zoom_w,
                        big_image_y - zoom_h,
                        big_image_x + zoom_w,
                        big_image_y + zoom_h,
                    )
                ).resize((zoom_w, zoom_h))

                self.zoomWindow.paintZoomedImage(
                    zoomed,
                )

            if self.portrait and self.portrait.skinmap:
                skinmap_x, skinmap_y = translate_coordinates_to_other_image(
                    (mouse_x, mouse_y), (img_w, img_h), (self.portrait.skinmap.size)
                )

                zoomed = self.portrait.skinmap.crop(
                    (
                        skinmap_x - zoom_w,
                        skinmap_y - zoom_h,
                        skinmap_x + zoom_w,
                        skinmap_y + zoom_h,
                    )
                ).resize((zoom_w, zoom_h))

                self.zoomWindow.paintZoomedSkinmap(
                    zoomed, mouse_x=skinmap_x, mouse_y=skinmap_y
                )

            if self.portrait and self.portrait.teethmap:
                teethmap_x, teethmap_y = translate_coordinates_to_other_image(
                    (mouse_x, mouse_y), (img_w, img_h), (self.portrait.teethmap.size)
                )

                zoomed = self.portrait.teethmap.crop(
                    (
                        teethmap_x - zoom_w,
                        teethmap_y - zoom_h,
                        teethmap_x + zoom_w,
                        teethmap_y + zoom_h,
                    )
                ).resize((zoom_w, zoom_h))

                self.zoomWindow.paintZoomedTeethmap(
                    zoomed, mouse_x=teethmap_x, mouse_y=teethmap_y
                )

            if self.depthmap:
                zoomed = (
                    self.depthmap.crop(
                        (mouse_x - 96, mouse_y - 64, mouse_x + 96, mouse_y + 64)
                    )
                    .resize((zoom_w, zoom_h), Image.HAMMING)
                    .filter(ImageFilter.SHARPEN)
                    .filter(ImageFilter.SHARPEN)
                    .filter(ImageFilter.SHARPEN)
                )

                self.zoomWindow.paintZoomedDepthmap(
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

        if self.last_click_x is not None:
            if (
                self.last_click_x == mouse_x
                and self.last_click_y == mouse_y
                and self.last_angle == angle
            ):
                return

        self.last_angle = angle

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
                        painter.drawLine(QPoint(*auto_id_click_1), QPoint(*auto_id_click_2))

            if self.neck_measurement:
                neck_click_1 = translate_coordinates_to_other_image(
                    (
                        self.neck_measurement[0],
                        self.neck_measurement[1],
                    ),
                    self.portrait.skinmap.size,
                    (img_w, img_h),
                )
                while self.get_depthmap_value(*neck_click_1) < 110:
                    neck_click_1 = (neck_click_1[0] + 1, neck_click_1[1])
                    if neck_click_1[0] > img_w:
                        neck_click_1 = None
                        break

                neck_click_2 = translate_coordinates_to_other_image(
                    (
                        self.neck_measurement[2],
                        self.neck_measurement[3],
                    ),
                    self.portrait.skinmap.size,
                    (img_w, img_h),
                )
                while self.get_depthmap_value(*neck_click_2) < 110:
                    neck_click_2 = (neck_click_2[0] - 1, neck_click_2[1])
                    if neck_click_2[0] < 0:
                        neck_click_2 = None
                        break

                if neck_click_2 is not None and neck_click_1 is not None:
                    painter.drawLine(QPoint(*neck_click_1), QPoint(*neck_click_2))

            if self.face:
                painter.setPen(QColor(0, 0, 255, 127))
                face_rect = self.face.translate_coordinates(img_w, img_h)
                painter.drawRect(*face_rect)

                for eye in self.face.eyes:
                    painter.setPen(QColor(0, 255, 0, 127))
                    rect = eye.translate_coordinates(img_w, img_h)
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
                    self.zoomWindow.paintReconstruction(values)
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
                txt += "\n\nTeeth detected. Automatic incisor distance:\n"
                txt += "%.2f cm" % (
                    self.vector_length_between_two_clicks(
                        *auto_id_click_1, *auto_id_click_2
                    )
                    / 10.0
                )

            if self.neck_measurement:
                if neck_click_2 is not None and neck_click_1 is not None:
                    txt += "\n\nFace detected, neck probably analysed. Automatic neck circumference:\n"
                    txt += "%.2f cm" % (
                        get_circumference_of_circle(
                            get_radius_of_circle_described_on_equilateral(
                                self.vector_length_between_two_clicks(
                                    *neck_click_1, *neck_click_2
                                )
                                / 10.0
                            )
                        )
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
                    "\n\nNeck circumference estimation for last 2 clicks (equilateral triangle): %.2f"
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
                    "\n\nNeck circumference estimation for last 2 clicks (radius): %.2f"
                    % (get_circumference_of_circle(vector_length_3d) / 10.0)
                )

                txt += (
                    "\n\nNeck circumference estimation for last 2 clicks (square): %.2f"
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

        self.smallImage = self.image.resize((const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT))

        # If pictures taken with the back camera, the main miage should be mirrored to match
        # the depth map... then the depth map should be mirrored if printing in 3D... currently
        # I'm leaving this comment & not supporting it (the back camera).
        # self.depthmap = ImageOps.mirror(self.depthmap)

        #
        # Get face position, if any:
        #

        self.neck_measurement = None
        self.face_proper = False
        try:
            self.face = get_face_parameters(self.image, raise_opencv_exceptions=True)
        except NoFacesDetected:
            self.face = None
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
                int(round(self.face.center_x / self.image.size[0] * (const.MAIN_IMAGE_WIDTH - 1)))
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

            if self.portrait.skinmap and self.face and self.face_proper:
                self.neck_measurement = find_neck_measurement_point(
                    self.portrait.skinmap, self.face
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
