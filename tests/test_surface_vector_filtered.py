import math
from types import SimpleNamespace

import pytest
from PIL import Image

from fidmaa_gui.app import MainWindow
from fidmaa_gui.utils import (
    bilinear_sample,
    median_filter_depthmap,
    sample_points_along_line,
)


class TestSamplePointsAlongLine:
    def test_uses_approximately_requested_step_and_keeps_endpoint(self):
        points = list(sample_points_along_line(0, 0, 10, 0, step=3))

        expected = [(0, 0), (10 / 3, 0), (20 / 3, 0), (10, 0)]
        assert len(points) == len(expected)
        for point, expected_point in zip(points, expected, strict=True):
            assert point == pytest.approx(expected_point)

    def test_diagonal_spacing_is_measured_along_line(self):
        points = list(sample_points_along_line(0, 0, 6, 8, step=5))

        assert points == [(0, 0), (3, 4), (6, 8)]

    def test_reversing_line_returns_same_points_in_reverse(self):
        forward = list(sample_points_along_line(2, 3, 15, 9, step=4))
        backward = list(sample_points_along_line(15, 9, 2, 3, step=4))

        assert len(forward) == len(backward)
        for point, reverse_point in zip(forward, reversed(backward), strict=True):
            assert point == pytest.approx(reverse_point)

    def test_rejects_non_positive_step(self):
        with pytest.raises(ValueError):
            list(sample_points_along_line(0, 0, 10, 0, step=0))


class TestFilteredDepthSampling:
    def test_median_filter_preserves_size_and_removes_impulse(self):
        depthmap = Image.new("L", (5, 5), 100)
        depthmap.putpixel((2, 2), 255)

        filtered = median_filter_depthmap(depthmap, size=3)

        assert filtered.size == depthmap.size
        assert filtered.getpixel((2, 2)) == 100

    def test_median_filter_uses_raw_first_channel(self):
        depthmap = Image.new("RGB", (3, 3), (100, 10, 200))

        filtered = median_filter_depthmap(depthmap, size=3)

        assert filtered.mode == "L"
        assert filtered.getpixel((1, 1)) == 100

    def test_bilinear_sample_interpolates_fractional_coordinate(self):
        image = Image.new("L", (2, 2))
        image.putdata([0, 10, 20, 30])

        assert bilinear_sample(image, 0.5, 0.5) == pytest.approx(15.0)

    def test_bilinear_sample_rejects_contributing_invalid_depth(self):
        image = Image.new("L", (2, 2))
        image.putdata([0, 10, 20, 30])

        assert bilinear_sample(image, 0.5, 0.5, invalid_value=0) is None


def test_surface_vector_filtered_is_stable_on_flat_surface():
    depthmap = Image.new("L", (480, 640), 100)
    window = SimpleNamespace(
        depthmap=depthmap,
        filtered_depthmap=median_filter_depthmap(depthmap),
        get_depthmap_distance=lambda value: value / 10.0,
        translate_click_to_mm=lambda _distance, x, y: (x, y),
        vector_length_simple=lambda x1, y1, z1, x2, y2, z2: math.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
        ),
    )

    lengths = [
        MainWindow.surface_vector_filtered(window, 10, 20, 31, 20, step)
        for step in range(2, 8)
    ]

    assert lengths == pytest.approx([21.0] * 6)
