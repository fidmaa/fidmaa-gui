"""Tests for pure calculation functions that live in app.py (or, since the
principal-point fix, delegate from app.py to portrait_analyser).

Since app.py has heavy GUI/library dependencies that may not resolve in test
environments, we replicate the pure math functions here for testing. These
tests serve as regression guards -- if the logic in app.py changes, these
tests should be updated to match.
"""

import math

import pytest
from portrait_analyser.incisor import pixels_per_mm_at_distance

from fidmaa_gui import const
from fidmaa_gui.calculations import lower_jaw_x_bounds, neck_to_jaw_width_ratio


def vector_length_simple(x1, y1, z1, x2, y2, z2):
    """Replica of MainWindow.vector_length_simple for testing."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def calculate_line_length(dist_x, dist_y):
    """Replica of MainWindow.calculate_line_length for testing."""
    return math.sqrt(abs(dist_x * dist_x) + abs(dist_y * dist_y))


def get_depthmap_distance(value, float_min_value=None, float_max_value=None):
    """Replica of MainWindow.get_depthmap_distance for testing."""
    if float_min_value is None or float_max_value is None:
        return value
    return 100 * 1.0 / (float_max_value * value / 255 + float_min_value * (1 - value / 255))


def how_many_pixels_per_mm_at_distance_on_big_image(distance, mm):
    """how_many_pixels_per_mm_at_distance_on_big_image no longer exists on
    MainWindow -- the calibration polynomial now lives solely in
    portrait_analyser.incisor.pixels_per_mm_at_distance, tested there. Kept
    as a thin (distance, mm) -> float shim so the tests below don't need
    to change.
    """
    return pixels_per_mm_at_distance(distance)


class TestVectorLengthSimple:
    def test_zero_vector(self):
        assert vector_length_simple(0, 0, 0, 0, 0, 0) == 0

    def test_unit_x(self):
        assert vector_length_simple(0, 0, 0, 1, 0, 0) == pytest.approx(1.0)

    def test_unit_y(self):
        assert vector_length_simple(0, 0, 0, 0, 1, 0) == pytest.approx(1.0)

    def test_unit_z(self):
        assert vector_length_simple(0, 0, 0, 0, 0, 1) == pytest.approx(1.0)

    def test_3d_diagonal(self):
        result = vector_length_simple(0, 0, 0, 1, 1, 1)
        assert result == pytest.approx(math.sqrt(3))

    def test_345_triangle(self):
        result = vector_length_simple(0, 0, 0, 3, 4, 0)
        assert result == pytest.approx(5.0)

    def test_negative_coordinates(self):
        result = vector_length_simple(-1, -1, -1, 1, 1, 1)
        assert result == pytest.approx(2 * math.sqrt(3))


class TestCalculateLineLength:
    def test_zero_length(self):
        assert calculate_line_length(0, 0) == 0

    def test_horizontal(self):
        assert calculate_line_length(10, 0) == pytest.approx(10.0)

    def test_vertical(self):
        assert calculate_line_length(0, 10) == pytest.approx(10.0)

    def test_345(self):
        assert calculate_line_length(3, 4) == pytest.approx(5.0)


class TestGetDepthmapDistance:
    def test_no_calibration_returns_raw(self):
        assert get_depthmap_distance(128) == 128

    def test_with_calibration(self):
        result = get_depthmap_distance(128, float_min_value=0.5, float_max_value=2.0)
        assert isinstance(result, float)
        assert result > 0

    def test_calibration_at_zero(self):
        # Value 0 -> distance = 100 / float_min_value = 200 cm
        result = get_depthmap_distance(0, float_min_value=0.5, float_max_value=2.0)
        assert result == pytest.approx(200.0)

    def test_calibration_at_255(self):
        # Value 255 -> distance = 100 / float_max_value = 50 cm
        result = get_depthmap_distance(255, float_min_value=0.5, float_max_value=2.0)
        assert result == pytest.approx(50.0)


class TestHowManyPixelsPerMmAtDistance:
    def test_returns_float(self):
        result = how_many_pixels_per_mm_at_distance_on_big_image(20, 1)
        assert isinstance(result, float)

    def test_closer_means_more_pixels(self):
        close = how_many_pixels_per_mm_at_distance_on_big_image(20, 1)
        far = how_many_pixels_per_mm_at_distance_on_big_image(40, 1)
        assert close > far

    def test_positive_at_reasonable_distance(self):
        result = how_many_pixels_per_mm_at_distance_on_big_image(25, 1)
        assert result > 0


def test_lower_jaw_bounds_include_small_facemesh_tolerance():
    landmarks = [(0.0, 0.0)] * 478
    for index in const.FACE_MESH_LOWER_JAW_INDICES:
        landmarks[index] = (900.0, 1000.0)
    landmarks[const.FACE_MESH_LOWER_JAW_INDICES[0]] = (600.0, 900.0)
    landmarks[const.FACE_MESH_LOWER_JAW_INDICES[-1]] = (1200.0, 900.0)

    assert lower_jaw_x_bounds(landmarks) == (588, 1212)


def test_lower_jaw_bounds_require_complete_facemesh():
    assert lower_jaw_x_bounds([(0.0, 0.0)] * 100) is None


class TestNeckToJawWidthRatio:
    """Regression anchors from the real subjects investigated in
    NECK_DETECTOR_STATUS.md. IMG_6552/IMG_6572 have a genuine anatomical
    narrowing below the chin (ratio comfortably below threshold). IMG_6640
    has none -- the widest visible skin run there is nearly as wide as the
    jaw itself, which is exactly the signature of endpoints spilling onto
    the shoulder/chest instead of finding a real neck.
    """

    def test_basic_ratio(self):
        # neck width 600 / jaw width 750
        assert neck_to_jaw_width_ratio(700, 1300, (650, 1400)) == pytest.approx(0.8)

    def test_none_when_jaw_bounds_missing(self):
        assert neck_to_jaw_width_ratio(700, 1300, None) is None

    def test_none_when_jaw_width_not_positive(self):
        assert neck_to_jaw_width_ratio(700, 1300, (800, 800)) is None
        assert neck_to_jaw_width_ratio(700, 1300, (900, 800)) is None

    def test_known_good_case_img_6552_is_below_threshold(self):
        # mask_left_x=704, mask_right_x=1335, FaceMesh jaw guard (651, 1426)
        ratio = neck_to_jaw_width_ratio(704, 1335, (651, 1426))
        assert ratio == pytest.approx(0.8142, abs=1e-3)
        assert ratio < const.NECK_LOW_CONFIDENCE_RATIO_THRESHOLD

    def test_known_good_case_img_6572_is_below_threshold(self):
        # mask_left_x=737, mask_right_x=1407, FaceMesh jaw guard (668, 1416)
        ratio = neck_to_jaw_width_ratio(737, 1407, (668, 1416))
        assert ratio == pytest.approx(0.8957, abs=1e-3)
        assert ratio < const.NECK_LOW_CONFIDENCE_RATIO_THRESHOLD

    def test_known_bad_case_img_6640_is_at_or_above_threshold(self):
        # mask_left_x=579, mask_right_x=1317, FaceMesh jaw guard (554, 1317)
        # -- the detected "neck" run touches the jaw guard's own right
        # bound, i.e. no narrowing was found at all.
        ratio = neck_to_jaw_width_ratio(579, 1317, (554, 1317))
        assert ratio == pytest.approx(0.9672, abs=1e-3)
        assert ratio >= const.NECK_LOW_CONFIDENCE_RATIO_THRESHOLD
