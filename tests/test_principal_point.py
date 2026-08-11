import math
from types import SimpleNamespace

import pytest

from fidmaa_gui.app import MainWindow


class MeasurementHarness:
    how_many_pixels_per_mm_at_distance_on_big_image = (
        MainWindow.how_many_pixels_per_mm_at_distance_on_big_image
    )
    how_many_mm_per_pixels_at_distance_on_big_image = (
        MainWindow.how_many_mm_per_pixels_at_distance_on_big_image
    )


def make_measurement_harness():
    window = MeasurementHarness()
    window.image = SimpleNamespace(size=(2320, 3087))
    return window


def test_image_centre_is_origin_at_every_depth():
    window = make_measurement_harness()

    at_30_cm = MainWindow.translate_click_to_mm(window, 30.0, 240, 320)
    at_36_cm = MainWindow.translate_click_to_mm(window, 36.0, 240, 320)

    assert at_30_cm == pytest.approx((0.0, 0.0))
    assert at_36_cm == pytest.approx((0.0, 0.0))


def test_same_depth_distance_keeps_calibrated_pixel_scale():
    window = make_measurement_harness()
    distance_cm = 30.0
    point_1 = MainWindow.translate_click_to_mm(window, distance_cm, 180, 280)
    point_2 = MainWindow.translate_click_to_mm(window, distance_cm, 300, 360)

    measured_dx = point_2[0] - point_1[0]
    measured_dy = point_2[1] - point_1[1]
    pixels_per_mm = MainWindow.how_many_pixels_per_mm_at_distance_on_big_image(
        window, distance_cm, 1
    )
    expected_dx = ((300 - 180) * window.image.size[0] / 480) / pixels_per_mm
    expected_dy = ((360 - 280) * window.image.size[1] / 640) / pixels_per_mm

    assert measured_dx == pytest.approx(expected_dx)
    assert measured_dy == pytest.approx(expected_dy)


def test_motion_along_optical_axis_has_no_phantom_xy_component():
    window = make_measurement_harness()
    point_1_xy = MainWindow.translate_click_to_mm(window, 30.0, 240, 320)
    point_2_xy = MainWindow.translate_click_to_mm(window, 36.0, 240, 320)
    point_1 = (*point_1_xy, 300.0)
    point_2 = (*point_2_xy, 360.0)

    measured_distance = math.dist(point_1, point_2)

    assert measured_distance == pytest.approx(60.0)


def test_points_equidistant_from_centre_are_symmetric():
    window = make_measurement_harness()

    left = MainWindow.translate_click_to_mm(window, 30.0, 180, 320)
    right = MainWindow.translate_click_to_mm(window, 30.0, 300, 320)

    assert left[0] == pytest.approx(-right[0])
    assert left[1] == pytest.approx(0.0)
    assert right[1] == pytest.approx(0.0)
