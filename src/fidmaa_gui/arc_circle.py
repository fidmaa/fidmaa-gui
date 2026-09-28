"""Circle through a chord and the surface arc over it (Neck tool, >= 3 points).

The user clicks a polyline across the front of the neck from the left edge to
the right edge. Its straight 3D chord ``c`` and its length along the depth
surface ``s`` define one circle, if the front of the neck is taken to be a
circular arc:

    s = R * theta,   c = 2 * R * sin(theta / 2)

so the ratio ``k = s / c`` depends on the central angle ``theta`` only:

    k = theta / (2 * sin(theta / 2))

which increases monotonically from 1 (theta -> 0) to infinity (theta -> 2*pi).
Solving it for theta gives ``R = c / (2 * sin(theta / 2))`` and the
circumference ``2 * pi * R``.

Qt-free so the maths is unit-tested directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# k at or below this is treated as a flat front: theta would be below ~6.3
# degrees and R explodes with the smallest depth error.
ARC_CIRCLE_MIN_RATIO = 1.0005
# Bisection stops once the theta bracket is narrower than this (radians);
# ~60 halvings of the (0, 2*pi) bracket.
ARC_CIRCLE_THETA_TOLERANCE = 1e-13
ARC_CIRCLE_MAX_ITERATIONS = 200

ARC_CIRCLE_OK = "ok"
ARC_CIRCLE_FLAT = "flat"
ARC_CIRCLE_INVALID = "invalid"

ARC_CIRCLE_FLAT_TEXT = "front too flat to fit a circle"
ARC_CIRCLE_HALF_CIRCLE_WARNING = "arc spans more than a half circle — check the edge points"


def arc_chord_ratio(theta: float) -> float:
    """k(theta) = theta / (2 sin(theta/2)), the arc/chord ratio of a circle."""
    if theta <= 0.0:
        return 1.0
    return theta / (2.0 * math.sin(theta / 2.0))


def solve_central_angle(ratio: float) -> float:
    """Central angle theta in (0, 2*pi) with arc/chord ratio ``ratio`` (> 1).

    k(theta) is monotonic on (0, 2*pi), so bisection always converges; the
    result is within ``ARC_CIRCLE_THETA_TOLERANCE`` rad of the exact root.
    """
    if not ratio > 1.0 or not math.isfinite(ratio):
        raise ValueError(f"arc/chord ratio must be finite and > 1, got {ratio!r}")
    low, high = 0.0, 2.0 * math.pi
    for _ in range(ARC_CIRCLE_MAX_ITERATIONS):
        middle = 0.5 * (low + high)
        if arc_chord_ratio(middle) < ratio:
            low = middle
        else:
            high = middle
        if high - low < ARC_CIRCLE_THETA_TOLERANCE:
            break
    return 0.5 * (low + high)


@dataclass(frozen=True)
class ArcCircleResult:
    status: str
    chord_mm: float | None
    arc_mm: float | None
    ratio: float | None = None
    theta_rad: float | None = None
    radius_mm: float | None = None
    circumference_mm: float | None = None
    warnings: tuple[str, ...] = ()

    @property
    def theta_deg(self) -> float | None:
        return None if self.theta_rad is None else math.degrees(self.theta_rad)


def arc_circle(chord_mm: float | None, arc_mm: float | None) -> ArcCircleResult:
    """Fit the circle to a 3D chord and the surface arc over it (both in mm).

    ``None`` for either length (invalid depth) gives status "invalid"; an
    arc/chord ratio at or below ``ARC_CIRCLE_MIN_RATIO`` gives "flat". An arc
    of more than a half circle (theta > pi) is still computed, with a warning.
    """
    if chord_mm is None or arc_mm is None or not chord_mm > 0.0 or not arc_mm > 0.0:
        return ArcCircleResult(ARC_CIRCLE_INVALID, chord_mm, arc_mm)
    ratio = arc_mm / chord_mm
    if ratio <= ARC_CIRCLE_MIN_RATIO:
        return ArcCircleResult(ARC_CIRCLE_FLAT, chord_mm, arc_mm, ratio=ratio)
    theta = solve_central_angle(ratio)
    radius = chord_mm / (2.0 * math.sin(theta / 2.0))
    warnings = (ARC_CIRCLE_HALF_CIRCLE_WARNING,) if theta > math.pi else ()
    return ArcCircleResult(
        ARC_CIRCLE_OK,
        chord_mm,
        arc_mm,
        ratio=ratio,
        theta_rad=theta,
        radius_mm=radius,
        circumference_mm=2.0 * math.pi * radius,
        warnings=warnings,
    )
