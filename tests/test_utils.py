import math

import pytest

from fidmaa_gui.utils import (
    clamp,
    get_circumference_of_circle,
    get_height_of_equilateral_triangle,
    get_radius_of_circle_described_on_equilateral,
    get_radius_of_circle_described_on_square,
    interpolate_pixels_along_line,
    translate_coordinates_to_other_image,
)


class TestTranslateCoordinates:
    def test_same_size(self):
        result = translate_coordinates_to_other_image((100, 200), (480, 640), (480, 640))
        assert result == (100, 200)

    def test_double_size(self):
        result = translate_coordinates_to_other_image((100, 200), (480, 640), (960, 1280))
        assert result == (200, 400)

    def test_half_size(self):
        result = translate_coordinates_to_other_image((100, 200), (480, 640), (240, 320))
        assert result == (50, 100)

    def test_origin(self):
        result = translate_coordinates_to_other_image((0, 0), (480, 640), (100, 100))
        assert result == (0, 0)

    def test_different_aspect_ratio(self):
        result = translate_coordinates_to_other_image((240, 320), (480, 640), (100, 200))
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(100.0)


class TestInterpolatePixelsAlongLine:
    def test_yields_2d_points(self):
        # Guards against the 3D signature (x1, y1, z1, x2, y2, z2) creeping
        # back in: app.py unpacks these as `for x, y in ...`.
        pixels = list(interpolate_pixels_along_line(0, 0, 4, 0))
        assert all(len(p) == 2 for p in pixels)

    def test_horizontal_line(self):
        pixels = list(interpolate_pixels_along_line(0, 0, 10, 0))
        assert len(pixels) == 11  # 0 to 10 inclusive
        assert pixels[0] == (0, 0)
        assert pixels[-1] == pytest.approx((10, 0))

    def test_vertical_line(self):
        pixels = list(interpolate_pixels_along_line(0, 0, 0, 10))
        assert len(pixels) == 11
        assert pixels[0] == (0, 0)
        assert pixels[-1] == pytest.approx((0, 10))

    def test_diagonal_line(self):
        pixels = list(interpolate_pixels_along_line(0, 0, 10, 10))
        assert len(pixels) == 11
        assert pixels[0] == (0, 0)
        assert pixels[-1] == pytest.approx((10, 10))

    def test_single_point(self):
        pixels = list(interpolate_pixels_along_line(5, 5, 5, 5))
        assert len(pixels) == 0  # no_steps == 0

    def test_negative_direction(self):
        pixels = list(interpolate_pixels_along_line(10, 0, 0, 0))
        assert len(pixels) == 11
        assert pixels[0] == (10, 0)
        assert pixels[-1] == pytest.approx((0, 0))

    def test_step_count_follows_longest_axis(self):
        # dist_x is 10, dist_y is 5 -> 10 steps, so 11 points.
        pixels = list(interpolate_pixels_along_line(0, 0, 10, 5))
        assert len(pixels) == 11
        assert pixels[0] == (0, 0)
        assert pixels[-1] == pytest.approx((10, 5))

    def test_intermediate_points_are_evenly_spaced(self):
        pixels = list(interpolate_pixels_along_line(0, 0, 10, 5))
        assert pixels[5] == pytest.approx((5, 2.5))


class TestClamp:
    def test_within_range(self):
        assert clamp(5, 0, 10) == 5

    def test_below_minimum(self):
        assert clamp(-5, 0, 10) == 0

    def test_above_maximum(self):
        # clamp uses maxn-1
        assert clamp(15, 0, 10) == 9

    def test_at_minimum(self):
        assert clamp(0, 0, 10) == 0

    def test_at_maximum_minus_1(self):
        assert clamp(9, 0, 10) == 9

    def test_at_maximum(self):
        # At maxn, should return maxn-1
        assert clamp(10, 0, 10) == 9


class TestGetHeightOfEquilateralTriangle:
    def test_unit_triangle(self):
        result = get_height_of_equilateral_triangle(1)
        assert result == pytest.approx(math.sqrt(3) / 2)

    def test_known_value(self):
        result = get_height_of_equilateral_triangle(10)
        assert result == pytest.approx(math.sqrt(75))

    def test_zero(self):
        assert get_height_of_equilateral_triangle(0) == 0


class TestGetRadiusOfCircleDescribedOnEquilateral:
    def test_unit_triangle(self):
        result = get_radius_of_circle_described_on_equilateral(1)
        expected = 2 / 3 * math.sqrt(3) / 2  # 2/3 * height
        assert result == pytest.approx(expected)

    def test_known_value(self):
        # For equilateral triangle with side a, circumscribed radius = a / sqrt(3)
        result = get_radius_of_circle_described_on_equilateral(10)
        expected = 10 / math.sqrt(3)
        assert result == pytest.approx(expected)


class TestGetRadiusOfCircleDescribedOnSquare:
    def test_unit_square(self):
        result = get_radius_of_circle_described_on_square(1)
        assert result == pytest.approx(math.sqrt(2) / 2)

    def test_known_value(self):
        result = get_radius_of_circle_described_on_square(10)
        assert result == pytest.approx(5 * math.sqrt(2))


class TestGetCircumferenceOfCircle:
    def test_unit_radius(self):
        result = get_circumference_of_circle(1)
        assert result == pytest.approx(2 * math.pi)

    def test_known_value(self):
        result = get_circumference_of_circle(5)
        assert result == pytest.approx(10 * math.pi)

    def test_zero_radius(self):
        assert get_circumference_of_circle(0) == 0
