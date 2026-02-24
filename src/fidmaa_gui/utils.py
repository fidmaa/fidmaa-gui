import math
import os
import sys
from pathlib import Path
from typing import Optional

import PySide6
from PIL import ImageFile
from PySide6 import QtGui
from PySide6.QtCore import QFile, QObject
from PySide6.QtUiTools import QUiLoader

from .QClickableLabel import QClickableLabel

ImageFile.LOAD_TRUNCATED_IMAGES = True

tr = QObject.tr


def translate_coordinates_to_other_image(point, current_image_size, new_image_size):
    return (
        point[0] * new_image_size[0] / current_image_size[0],
        point[1] * new_image_size[1] / current_image_size[1],
    )


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


def get_height_of_equilateral_triangle(length):
    return math.sqrt((length) ** 2 - (length / 2) ** 2)


def get_radius_of_circle_described_on_equilateral(length):
    return 2 / 3 * get_height_of_equilateral_triangle(length)


def get_radius_of_circle_described_on_square(length):
    return length * math.sqrt(2) / 2


def get_circumference_of_circle(radius):
    return radius * math.pi * 2
