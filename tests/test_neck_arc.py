"""Tests for the geometric neck arc (neck_arc.py).

Synthetic profiles and depth images throughout -- no HEIC files needed.
Depth here is already in millimetres of distance from the camera (that is
what neck_arc works in), so LARGER means FARTHER, the opposite of the raw
inverse-disparity convention used at the file format level.
"""

import numpy as np
import pytest
from PIL import Image

from fidmaa_gui import const
from fidmaa_gui.neck_arc import (
    _interpolate_gaps,
    _quadratic_fit_weights,
    compute_neck_arc,
    depth_band_mm,
    find_silhouette_edges,
    find_stabilisation_index,
)


class TestQuadraticFitWeights:
    def test_reads_back_a_known_slope(self):
        first, _second = _quadratic_fit_weights(5)
        x = np.arange(-5, 6, dtype=float)
        y = 3.0 * x + 7.0
        assert float(np.dot(first[::-1], y[::-1])) == pytest.approx(3.0, abs=1e-9)

    def test_reads_back_a_known_curvature(self):
        _first, second = _quadratic_fit_weights(5)
        x = np.arange(-5, 6, dtype=float)
        y = 2.0 * x**2
        # d2/dx2 of 2x^2 is 4
        assert float(np.dot(second[::-1], y[::-1])) == pytest.approx(4.0, abs=1e-9)


class TestInterpolateGaps:
    def test_fills_interior_holes_linearly(self):
        values = np.array([0.0, np.nan, np.nan, 3.0])
        filled = _interpolate_gaps(values)
        assert filled == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_returns_none_when_almost_everything_is_missing(self):
        assert _interpolate_gaps(np.array([np.nan, 1.0, np.nan])) is None


def _chin_then_neck_descent(fall_len=40, tail=200, fall_height=60.0):
    """Distance profile going down the midline: a short stretch on the chin,
    a sharp recession as the scan leaves it, then the neck holding steady."""
    chin = np.full(30, 400.0)
    fall = 400.0 + fall_height * np.linspace(0, 1, fall_len) ** 0.7
    neck = np.full(tail, 400.0 + fall_height)
    return np.concatenate([chin, fall, neck])


class TestFindStabilisationIndex:
    def test_finds_the_knee_after_the_fall(self):
        profile = _chin_then_neck_descent()
        first, _ = _quadratic_fit_weights(8)
        found = find_stabilisation_index(profile, first, ppmm=1.0, fraction=0.55)
        assert found is not None
        knee, steepest = found
        # The fall spans samples 30..70; the knee must land inside or just
        # after it, never back up on the chin and never far down the neck.
        assert 30 <= steepest <= 75
        assert steepest <= knee <= 110

    def test_a_deeper_chin_pushes_the_knee_further_down(self):
        first, _ = _quadratic_fit_weights(8)
        shallow = find_stabilisation_index(_chin_then_neck_descent(fall_len=25), first, 1.0, 0.55)
        deep = find_stabilisation_index(_chin_then_neck_descent(fall_len=70), first, 1.0, 0.55)
        assert shallow is not None and deep is not None
        assert deep[0] > shallow[0]

    def test_flat_profile_has_no_knee(self):
        first, _ = _quadratic_fit_weights(8)
        assert find_stabilisation_index(np.full(200, 400.0), first, 1.0, 0.55) is None

    def test_too_short_a_profile_is_rejected(self):
        first, _ = _quadratic_fit_weights(8)
        assert find_stabilisation_index(np.full(6, 400.0), first, 1.0, 0.55) is None


def _hill_cross_section(width=600, half=110, depth=55.0, centre=300, tilt=0.0):
    """A neck-like cross-section: a circular hill standing out of a flat,
    far background, optionally with a background tilt added."""
    x = np.arange(width, dtype=float)
    background = 500.0
    z = np.full(width, background)
    inside = np.abs(x - centre) < half
    dx = (x[inside] - centre) / half
    z[inside] = background - depth * np.sqrt(np.clip(1 - dx**2, 0, None))
    return z + tilt * (x - centre)


class TestFindSilhouetteEdges:
    def test_finds_both_flanks_of_a_hill(self):
        section = _hill_cross_section()
        first, _ = _quadratic_fit_weights(4)
        found = find_silhouette_edges(section, 300, first, 1.0, 200, 60)
        assert found is not None
        left, right = found
        assert 175 <= left <= 200
        assert 400 <= right <= 425

    def test_survives_a_background_tilt(self):
        """The detrending step exists exactly for this: without it the tilt
        dominates the derivative and the edges cannot be located."""
        plain = find_silhouette_edges(
            _hill_cross_section(), 300, _quadratic_fit_weights(4)[0], 1.0, 200, 60
        )
        tilted = find_silhouette_edges(
            _hill_cross_section(tilt=0.25),
            300,
            _quadratic_fit_weights(4)[0],
            1.0,
            200,
            60,
        )
        assert plain is not None and tilted is not None
        assert abs(tilted[0] - plain[0]) <= 12
        assert abs(tilted[1] - plain[1]) <= 12

    def test_a_wider_neck_gives_wider_edges(self):
        first, _ = _quadratic_fit_weights(4)
        narrow = find_silhouette_edges(_hill_cross_section(half=80), 300, first, 1.0, 200, 60)
        wide = find_silhouette_edges(_hill_cross_section(half=130), 300, first, 1.0, 250, 60)
        assert narrow is not None and wide is not None
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_featureless_section_returns_none(self):
        first, _ = _quadratic_fit_weights(4)
        assert find_silhouette_edges(np.full(600, 500.0), 300, first, 1.0, 200, 60) is None


class TestDepthBandMm:
    def test_converts_and_cuts_the_background(self):
        img = Image.new("L", (40, 40), 200)
        for y in range(40):
            for x in range(20, 40):
                img.putpixel((x, y), 100)
        band = depth_band_mm(img, 40, 40, 0.5, 4.0, 0, 40)
        assert band is not None
        near = band[0, 5]
        far = band[0, 30]
        assert far > near  # the lower raw value is farther away

        cut = depth_band_mm(img, 40, 40, 0.5, 4.0, 0, 40, background_cutoff_mm=near + 1.0)
        assert np.isfinite(cut[0, 5])
        assert not np.isfinite(cut[0, 30])

    def test_rejects_a_degenerate_band(self):
        img = Image.new("L", (40, 40), 200)
        assert depth_band_mm(img, 40, 40, 0.5, 4.0, 10, 12) is None


def _synthetic_portrait(width=900, height=1200, neck_half_px=150):
    """A crude but geometrically honest head-and-neck depth image: a chin
    that protrudes, a neck cylinder below it, and a far background.

    Raw values follow the file-format convention -- HIGHER means CLOSER --
    so the background gets a LOW value and the subject high ones.
    """
    far = 90
    img = Image.new("L", (width, height), far)
    pixels = img.load()
    centre = width // 2
    chin_bottom = 500

    for y in range(height):
        for x in range(width):
            if y < chin_bottom:  # the head: a wide, close, rounded mass
                dx = (x - centre) / 260.0
                dy = (y - 300) / 260.0
                r2 = dx * dx + dy * dy
                if r2 < 1.0:
                    pixels[x, y] = int(215 - 18 * r2)
            else:  # the neck: a narrower cylinder, set slightly farther back
                dx = (x - centre) / float(neck_half_px)
                if abs(dx) < 1.0:
                    pixels[x, y] = int(196 - 20 * dx * dx)
    return img


class TestComputeNeckArc:
    def _run(self, **kw):
        width, height = 900, 1200
        img = _synthetic_portrait(width, height, **kw)
        centre = width // 2
        # A U-shaped jaw: highest at the sides, dipping to the chin at the
        # centre, which is what "parallel to the lower face oval" follows.
        jaw = [(centre - 200 + i * 40, 430 + int(70 * np.sin(np.pi * i / 10))) for i in range(11)]
        return compute_neck_arc(
            img,
            jaw_points=jaw,
            chin_xy=(centre, 500),
            nose_xy=(centre, 330),
            lower_lip_xy=(centre, 440),
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=4.0,
        )

    def test_measures_a_synthetic_neck(self):
        result = self._run()
        assert result is not None
        assert result.left_x < result.apex_x < result.right_x
        assert result.raw_width_mm > 0
        assert result.circumference_mm > 0
        # The correction must be applied, and in the widening direction.
        assert result.corrected_width_mm > result.raw_width_mm
        assert result.corrected_width_mm == pytest.approx(
            result.raw_width_mm * const.NECK_SILHOUETTE_EROSION_CORRECTION
        )
        assert result.circumference_mm == pytest.approx(np.pi * result.corrected_width_mm)

    def test_measuring_line_sits_below_the_jaw(self):
        result = self._run()
        assert result is not None
        assert result.offset_px > 0
        assert result.measuring_y > result.debug["jaw_y_at_midline"]

    def test_arc_points_are_ordered_and_paired(self):
        result = self._run()
        assert result is not None
        assert len(result.arc_points_photo) == len(result.arc_points_3d) >= 2
        xs = [p[0] for p in result.arc_points_photo]
        assert xs == sorted(xs)
        assert result.front_arc_length_mm > 0

    def test_a_wider_neck_measures_wider(self):
        narrow = self._run(neck_half_px=110)
        wide = self._run(neck_half_px=190)
        assert narrow is not None and wide is not None
        assert wide.raw_width_mm > narrow.raw_width_mm
        assert wide.circumference_mm > narrow.circumference_mm

    def test_missing_calibration_returns_none(self):
        width, height = 900, 1200
        img = _synthetic_portrait(width, height)
        jaw = [(300 + i * 40, 500) for i in range(11)]
        assert (
            compute_neck_arc(
                img,
                jaw_points=jaw,
                chin_xy=(450, 500),
                nose_xy=(450, 330),
                lower_lip_xy=(450, 440),
                photo_width=width,
                photo_height=height,
                float_min=None,
                float_max=None,
            )
            is None
        )

    def test_no_jaw_points_returns_none(self):
        img = _synthetic_portrait()
        assert (
            compute_neck_arc(
                img,
                jaw_points=[],
                chin_xy=(450, 500),
                nose_xy=(450, 330),
                lower_lip_xy=(450, 440),
                photo_width=900,
                photo_height=1200,
                float_min=0.5,
                float_max=4.0,
            )
            is None
        )

    def test_blank_depth_returns_none(self):
        blank = Image.new("L", (900, 1200), 0)
        jaw = [(250 + i * 40, 500) for i in range(11)]
        assert (
            compute_neck_arc(
                blank,
                jaw_points=jaw,
                chin_xy=(450, 500),
                nose_xy=(450, 330),
                lower_lip_xy=(450, 440),
                photo_width=900,
                photo_height=1200,
                float_min=0.5,
                float_max=4.0,
            )
            is None
        )
