"""Tests for MainWindow.surface_vector_filtered.

The underlying filtered/bilinear sampling primitives (median_filter_depthmap,
bilinear_sample, sample_points_along_line, measure_filtered_surface_length)
now live in portrait_analyser.depth_sampling and are covered by that
library's own test suite. This file only covers fidmaa-gui's glue: mapping
a display-space (480x640) click to photo-space and delegating to the
library, with self.filtered_depthmap lazily cached on first use.
"""

from types import SimpleNamespace

import pytest
from PIL import Image

from fidmaa_gui.app import MainWindow


def make_flat_surface_window(photo_width=480, photo_height=640, fill_value=100):
    depthmap = Image.new("L", (photo_width, photo_height), fill_value)
    return SimpleNamespace(
        depthmap=depthmap,
        filtered_depthmap=None,
        image=SimpleNamespace(size=(photo_width, photo_height)),
        float_min_value=0.5,
        float_max_value=2.0,
    )


def test_surface_vector_filtered_is_stable_on_flat_surface():
    """A flat depth map measured at increasing sample density should give a
    consistent length, not inflate with sensor noise."""
    window = make_flat_surface_window()

    lengths = [
        MainWindow.surface_vector_filtered(window, 10, 20, 41, 20, step)
        for step in range(2, 8)
    ]

    assert all(length is not None for length in lengths)
    assert lengths == pytest.approx([lengths[0]] * len(lengths), rel=0.05)


def test_surface_vector_filtered_caches_filtered_depthmap():
    window = make_flat_surface_window()
    assert window.filtered_depthmap is None

    MainWindow.surface_vector_filtered(window, 10, 20, 41, 20, step=5)

    assert window.filtered_depthmap is not None
    cached = window.filtered_depthmap
    MainWindow.surface_vector_filtered(window, 10, 20, 41, 20, step=5)
    assert window.filtered_depthmap is cached


def test_surface_vector_filtered_rejects_non_positive_step():
    window = make_flat_surface_window()
    with pytest.raises(ValueError):
        MainWindow.surface_vector_filtered(window, 10, 20, 41, 20, step=0)


def test_surface_vector_filtered_none_on_invalid_depth():
    window = make_flat_surface_window(photo_width=60, photo_height=60, fill_value=0)
    result = MainWindow.surface_vector_filtered(window, 10, 20, 41, 20, step=5)
    assert result is None
