"""Circle through a 3D chord and the surface arc over it (pure maths)."""

import math

import pytest

from fidmaa_gui.arc_circle import (
    ARC_CIRCLE_FLAT,
    ARC_CIRCLE_HALF_CIRCLE_WARNING,
    ARC_CIRCLE_INVALID,
    ARC_CIRCLE_MIN_RATIO,
    ARC_CIRCLE_OK,
    arc_chord_ratio,
    arc_circle,
    solve_central_angle,
)


def _sampled_circle(radius_mm, theta_deg):
    """Chord and arc length of an exact circle."""
    theta = math.radians(theta_deg)
    chord = 2 * radius_mm * math.sin(theta / 2)
    return chord, radius_mm * theta


@pytest.mark.parametrize("theta_deg", [90.0, 136.0, 170.0, 200.0, 300.0])
def test_exact_circle_is_recovered(theta_deg):
    radius = 65.0
    chord, arc = _sampled_circle(radius, theta_deg)
    fit = arc_circle(chord, arc)
    assert fit.status == ARC_CIRCLE_OK
    assert fit.theta_deg == pytest.approx(theta_deg, abs=1e-6)
    assert fit.radius_mm == pytest.approx(radius, abs=1e-6)
    assert fit.circumference_mm == pytest.approx(2 * math.pi * radius, abs=1e-6)


def test_polyline_sampled_circle_converges_to_the_circle():
    # A real arc is a sum of chords; with many samples it approaches R.
    radius, theta = 65.0, math.radians(136.0)
    n = 400
    points = [
        (radius * math.cos(theta / 2 - i * theta / n), radius * math.sin(theta / 2 - i * theta / n))
        for i in range(n + 1)
    ]
    arc = sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False))
    fit = arc_circle(math.dist(points[0], points[-1]), arc)
    assert fit.radius_mm == pytest.approx(radius, rel=1e-4)


def test_user_validation_numbers():
    # IMG_2363: chord 12.07 cm, arc 15.45 cm.
    fit = arc_circle(120.7, 154.5)
    assert fit.ratio == pytest.approx(1.280, abs=5e-4)
    assert fit.theta_deg == pytest.approx(136, abs=1)
    assert fit.radius_mm == pytest.approx(65.1, abs=0.5)
    assert fit.circumference_mm == pytest.approx(409, abs=1)
    assert fit.warnings == ()
    # IMG_2386: chord 13.03 cm, arc 17.99 cm.
    fit = arc_circle(130.3, 179.9)
    assert fit.ratio == pytest.approx(1.381, abs=5e-4)
    assert fit.theta_deg == pytest.approx(154, abs=1)
    assert fit.radius_mm == pytest.approx(66.8, abs=0.5)
    assert fit.circumference_mm == pytest.approx(420, abs=1)


@pytest.mark.parametrize("ratio", [1.0, 1.0001, ARC_CIRCLE_MIN_RATIO, 0.98])
def test_flat_front_is_not_fitted(ratio):
    fit = arc_circle(100.0, 100.0 * ratio)
    assert fit.status == ARC_CIRCLE_FLAT
    assert fit.circumference_mm is None and fit.radius_mm is None
    assert fit.ratio == pytest.approx(ratio)


def test_just_above_the_flat_threshold_is_fitted():
    fit = arc_circle(100.0, 100.0 * 1.001)
    assert fit.status == ARC_CIRCLE_OK
    # theta ~ sqrt(24 (k - 1)) for small angles.
    assert fit.theta_rad == pytest.approx(math.sqrt(24 * 0.001), rel=1e-3)


def test_more_than_a_half_circle_is_computed_with_a_warning():
    chord, arc = _sampled_circle(60.0, 200.0)
    fit = arc_circle(chord, arc)
    assert fit.status == ARC_CIRCLE_OK
    assert fit.circumference_mm == pytest.approx(2 * math.pi * 60.0, abs=1e-6)
    assert fit.warnings == (ARC_CIRCLE_HALF_CIRCLE_WARNING,)
    # Exactly a half circle is not "more than" one.
    chord, arc = _sampled_circle(60.0, 179.0)
    assert arc_circle(chord, arc).warnings == ()


@pytest.mark.parametrize("chord, arc", [(None, 100.0), (100.0, None), (0.0, 10.0), (None, None)])
def test_invalid_depth_is_not_fitted(chord, arc):
    fit = arc_circle(chord, arc)
    assert fit.status == ARC_CIRCLE_INVALID
    assert fit.circumference_mm is None and fit.theta_rad is None


def test_ratio_is_monotonic_and_the_solver_stays_in_bounds():
    thetas = [i * 2 * math.pi / 1000 for i in range(1, 1000)]
    ratios = [arc_chord_ratio(theta) for theta in thetas]
    assert all(b > a for a, b in zip(ratios, ratios[1:], strict=False))
    for ratio in (1.000001, 1.1, 1.5707963, 3.0, 50.0, 1e6):
        theta = solve_central_angle(ratio)
        assert 0.0 < theta < 2 * math.pi
        # Near 2*pi the ratio is ill-conditioned (k' ~ 1/sin^2), hence rel 1e-8.
        assert arc_chord_ratio(theta) == pytest.approx(ratio, rel=1e-8)
    # A half circle: arc/chord = pi/2.
    assert solve_central_angle(math.pi / 2) == pytest.approx(math.pi, abs=1e-12)


@pytest.mark.parametrize("ratio", [1.0, 0.5, float("nan"), float("inf")])
def test_solver_rejects_ratios_without_a_circle(ratio):
    with pytest.raises(ValueError):
        solve_central_angle(ratio)
