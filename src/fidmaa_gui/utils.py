import math
import os
import sys
from pathlib import Path
from typing import Optional

import PySide6
from PIL import ImageFile, ImageFilter
from PySide6 import QtGui
from PySide6.QtCore import QFile, QObject
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QMainWindow, QVBoxLayout

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


def interpolate_pixels_along_line(x1, y1, x2, y2):
    """Yield (x, y) pixel coordinates along a line between two points."""
    dist_x = x2 - x1
    dist_y = y2 - y1

    no_steps = int(max(abs(dist_x), abs(dist_y)))

    if no_steps == 0:
        return

    delta_x = dist_x / no_steps
    delta_y = dist_y / no_steps

    for _ in range(no_steps + 1):
        yield (x1, y1)
        x1 += delta_x
        y1 += delta_y


def sample_points_along_line(x1, y1, x2, y2, step):
    """Yield points separated by approximately ``step`` pixels on a 2D line.

    The first and last points are always included and all intervals have equal
    length.  This also makes the returned set independent of line direction.
    """
    if step <= 0:
        raise ValueError("step must be greater than zero")

    dist_x = x2 - x1
    dist_y = y2 - y1
    line_length = math.hypot(dist_x, dist_y)

    if line_length == 0:
        yield (x1, y1)
        return

    interval_count = max(1, round(line_length / step))
    for index in range(interval_count + 1):
        ratio = index / interval_count
        yield (x1 + dist_x * ratio, y1 + dist_y * ratio)


def median_filter_depthmap(depthmap, size=3):
    """Return a same-size, single-channel median-filtered depth map."""
    if size < 3 or size % 2 == 0:
        raise ValueError("median filter size must be an odd number >= 3")
    if len(depthmap.getbands()) > 1:
        depthmap = depthmap.getchannel(0)
    else:
        depthmap = depthmap.convert("L")
    return depthmap.filter(ImageFilter.MedianFilter(size=size))


def bilinear_sample(image, x, y, invalid_value=None):
    """Sample a PIL image at fractional coordinates using bilinear interpolation.

    When ``invalid_value`` is provided, return ``None`` if a contributing pixel
    has that value.  This prevents interpolation across holes in a depth map.
    """
    if image.width == 0 or image.height == 0:
        return None

    x = min(max(float(x), 0.0), image.width - 1.0)
    y = min(max(float(y), 0.0), image.height - 1.0)
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = min(x0 + 1, image.width - 1)
    y1 = min(y0 + 1, image.height - 1)
    fraction_x = x - x0
    fraction_y = y - y0

    samples = (
        (image.getpixel((x0, y0)), (1.0 - fraction_x) * (1.0 - fraction_y)),
        (image.getpixel((x1, y0)), fraction_x * (1.0 - fraction_y)),
        (image.getpixel((x0, y1)), (1.0 - fraction_x) * fraction_y),
        (image.getpixel((x1, y1)), fraction_x * fraction_y),
    )

    weighted_value = 0.0
    for value, weight in samples:
        if weight == 0:
            continue
        if isinstance(value, tuple):
            value = value[0]
        if invalid_value is not None and value == invalid_value:
            return None
        weighted_value += value * weight
    return weighted_value


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

        if isinstance(self, QMainWindow):
            self.setCentralWidget(self.ui)
        else:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.ui)

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
