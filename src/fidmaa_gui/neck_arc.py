"""Neck circumference from a curve parallel to the lower face oval.

The measurement follows the anatomy rather than the skin matte:

1.  **Height.**  Walk straight down the facial midline, starting ON THE CHIN
    (between the lower lip and the chin landmark -- not at the jaw line,
    which is already past the chin and hides the very fall we are looking
    for).  The depth first drops away sharply as the scan leaves the chin,
    then stabilises.  Where it stabilises, that is the neck.

2.  **Shape.**  The measuring curve is the FaceMesh lower-jaw polyline
    pushed straight down to that height -- literally parallel to the lower
    face oval, sitting just below it.  Its sag therefore comes from the
    subject's own jaw, not from a per-column depth search; such searches
    produce runaway V shapes as soon as one column finds a spurious "clear"
    point far below (a necklace, a fold, a shadow).

3.  **Edges.**  On that curve the neck reads as a hill in depth with a sharp
    fall on each side.  The background tilt is removed, the slope is taken
    at a small scale (a silhouette is only ~30 px wide and a wide window
    erases it), and the left/right edges are the peaks of those two falls.

4.  **Circumference.**  pi * width, with the width corrected for the depth
    sensor's silhouette erosion -- see NECK_SILHOUETTE_EROSION_CORRECTION.

Raw depth values are inverse to physical distance: a HIGHER raw value means
CLOSER to the camera.  Everything below works in millimetres instead, via
the EXIF calibration, so the geometry is in real units.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from portrait_analyser.incisor import (
    pixels_per_mm_at_distance,
    point_to_mm,
    vector_length_3d,
)

from . import const


@dataclass
class NeckArcResult:
    """A neck measurement plus everything needed to draw and judge it."""

    # Photo-space polyline of the measured arc, left to right.
    arc_points_photo: list[tuple[int, int]]
    # The same points as (x_mm, y_mm, z_mm) physical coordinates.
    arc_points_3d: list[tuple[float, float, float]]

    left_x: int
    right_x: int
    # Y of the measuring curve at the midline.
    measuring_y: int
    # How far the curve sits below the jaw line, at the midline.
    offset_px: int
    # Closest-to-camera point of the cross-section: the top of the "hill".
    apex_x: int
    midline_x: int

    # Straight lateral width of the neck, before the erosion correction.
    raw_width_mm: float
    # ... and after it; this is what the circumference is derived from.
    corrected_width_mm: float
    circumference_mm: float

    # 3D length of the visible surface between the endpoints. Diagnostic
    # only -- the circumference does NOT come from it (2 * arc overestimates
    # the tape by 37-53%, since the curve is not a planar section).
    front_arc_length_mm: float

    debug: dict = field(default_factory=dict)


def _quadratic_fit_weights(half_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay weights reading the 1st and 2nd derivative off a window
    of evenly spaced samples, as (first, second)."""
    x = np.arange(-half_width, half_width + 1, dtype=np.float64)
    pinv = np.linalg.pinv(np.stack([x**2, x, np.ones_like(x)], axis=1))
    return pinv[1], pinv[0] * 2.0


def _interpolate_gaps(values: np.ndarray) -> np.ndarray | None:
    """Fill NaNs by linear interpolation so a convolution has no holes."""
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return None
    filled = values.copy()
    index = np.arange(values.size)
    filled[~finite] = np.interp(index[~finite], index[finite], values[finite])
    return filled


def _derivative(filled: np.ndarray, weights: np.ndarray, ppmm: float) -> np.ndarray:
    """Slope in mm-of-depth per mm-of-travel, valid over the WHOLE profile.

    The signal is edge-padded before convolving, so the first and last
    half-windows stay usable.  Without this the steepest fall off the chin
    can land inside the dead zone at the start of the descent and simply
    never be seen -- which is exactly what happened when the scan began
    close to the chin.
    """
    half = (len(weights) - 1) // 2
    padded = np.pad(filled, half, mode="edge")
    return np.convolve(padded, weights[::-1], mode="valid") * ppmm


def depth_band_mm(
    filtered_depthmap,
    photo_width,
    photo_height,
    float_min,
    float_max,
    y_start,
    y_end,
    background_cutoff_mm=None,
):
    """Distance-from-camera in mm for photo rows [y_start, y_end).

    Returns NaN where depth is invalid, or beyond `background_cutoff_mm`
    (the wall behind the subject, which would otherwise dominate every
    derivative taken near the silhouette).
    """
    y_start = max(0, int(y_start))
    y_end = min(int(photo_height), int(y_end))
    if y_end - y_start < 4:
        return None

    full = filtered_depthmap.resize((photo_width, photo_height), resample=Image.Resampling.BILINEAR)
    raw = np.asarray(full, dtype=np.float64)[y_start:y_end]

    disparity = float_max * raw / 255.0 + float_min * (1.0 - raw / 255.0)
    valid = (raw > 0) & (disparity != 0)
    z_mm = np.full(raw.shape, np.nan)
    z_mm[valid] = (100.0 / disparity[valid]) * 10.0
    if background_cutoff_mm is not None:
        z_mm[z_mm > background_cutoff_mm] = np.nan
    return z_mm


def find_stabilisation_index(profile_mm, first_derivative_weights, ppmm, fraction):
    """Index where a descent has stopped falling steeply.

    `profile_mm` is distance-from-camera sampled downward.  The steepest
    point is the chin's own edge; the answer is the first sample after it
    whose slope has decayed to `fraction` of that peak.  Relative rather
    than absolute, so a prominent chin and a shallow one both work.

    Returns (stabilisation_index, steepest_index), or None.
    """
    filled = _interpolate_gaps(profile_mm)
    if filled is None or filled.size < 24:
        return None
    slope = _derivative(filled, first_derivative_weights, ppmm)

    low, high = 2, filled.size - 2
    if high <= low:
        return None
    steepest = low + int(np.argmax(slope[low:high]))
    if slope[steepest] <= 0:
        return None
    threshold = fraction * slope[steepest]
    stabilised = next((i for i in range(steepest, high) if slope[i] < threshold), None)
    if stabilised is None:
        return None
    return stabilised, steepest


def find_silhouette_edges(
    cross_section_mm,
    apex_index,
    first_derivative_weights,
    ppmm,
    max_half_width_px,
    detrend_half_px,
):
    """Left and right edge indices of the neck's hill in a cross-section.

    The background tilt is removed first, then the slope is read at the
    weights' (deliberately small) scale, and each edge is the peak of the
    first fall that clears a noise bar derived from the hill itself.
    """
    filled = _interpolate_gaps(cross_section_mm)
    if filled is None:
        return None

    n = filled.size
    lo = max(0, apex_index - detrend_half_px)
    hi = min(n, apex_index + detrend_half_px)
    if hi - lo < 8:
        return None
    x = np.arange(n, dtype=np.float64)
    tilt = np.polyval(np.polyfit(x[lo:hi], filled[lo:hi], 1), x)
    slope = _derivative(filled - tilt, first_derivative_weights, ppmm)

    # "An edge" means clearly steeper than the hill's own gentle curvature.
    bar = max(3.0 * float(np.median(np.abs(slope[lo:hi]))), 0.8)

    def peak_of_first_fall(direction):
        i = apex_index
        started = False
        best_value, best_index = 0.0, None
        while 0 < i < n - 1 and abs(i - apex_index) < max_half_width_px:
            i += direction
            outward = slope[i] * direction  # positive when receding outward
            if outward >= bar:
                started = True
                if outward > best_value:
                    best_value, best_index = outward, i
            elif started:
                break
        return best_index

    left = peak_of_first_fall(-1)
    right = peak_of_first_fall(+1)
    if left is None or right is None or right <= left:
        return None
    return left, right


def compute_neck_arc(
    filtered_depthmap,
    *,
    jaw_points,
    chin_xy,
    nose_xy,
    lower_lip_xy,
    photo_width,
    photo_height,
    float_min,
    float_max,
    n_samples=None,
    camera=None,
):
    """Measure the neck along a curve parallel to the lower face oval.

    `jaw_points` are the FaceMesh lower-jaw landmarks in photo pixels.
    Returns a NeckArcResult, or None if any stage cannot be resolved --
    never a fabricated fallback.

    `camera` is the photo's portrait_analyser CameraModel (TrueDepth
    capture-app intrinsics) or None for the legacy calibration polynomial.
    The horizontal focal length sets the pixels-per-mm scale.
    """
    if not jaw_points or float_min is None or float_max is None:
        return None
    n_samples = n_samples or const.NECK_ARC_SAMPLE_COUNT

    jaw = sorted(((float(x), float(y)) for x, y in jaw_points), key=lambda p: p[0])
    jaw_x = np.array([p[0] for p in jaw])
    jaw_y = np.array([p[1] for p in jaw])
    chin_x, chin_y = float(chin_xy[0]), float(chin_xy[1])
    nose_x = float(nose_xy[0])
    lip_y = float(lower_lip_xy[1])

    midline_x = int(round((chin_x + nose_x) / 2.0))
    if not (0 <= midline_x < photo_width):
        return None
    jaw_y_at_midline = float(np.interp(midline_x, jaw_x, jaw_y))

    # The descent starts on the chin itself, so the fall off the chin is
    # inside the profile rather than before it.
    descent_start_y = int(round((lip_y + chin_y) / 2.0))
    band_start = max(0, descent_start_y - 4)
    band_end = min(photo_height, descent_start_y + const.NECK_ARC_DESCENT_SEARCH_PX + 40)

    # A first, uncut band is needed to learn the chin's own distance before
    # the background can be defined relative to it.
    band = depth_band_mm(
        filtered_depthmap,
        photo_width,
        photo_height,
        float_min,
        float_max,
        band_start,
        band_end,
    )
    if band is None:
        return None
    chin_row = int(round(chin_y)) - band_start
    if not (0 <= chin_row < band.shape[0]):
        return None
    patch = band[
        max(0, chin_row - 5) : chin_row + 6,
        max(0, int(chin_x) - 5) : int(chin_x) + 6,
    ]
    if not np.isfinite(patch).any():
        return None
    chin_z_mm = float(np.nanmedian(patch))
    band[band > chin_z_mm + const.NECK_ARC_BACKGROUND_MARGIN_MM] = np.nan

    ppmm = pixels_per_mm_at_distance(
        chin_z_mm / 10.0, focal_px=None if camera is None else camera.fx
    )
    if ppmm is None or ppmm <= 0:
        return None

    wide_half = max(3, int(round(const.NECK_ARC_WIDE_SMOOTH_MM * ppmm)))
    edge_half = max(2, int(round(const.NECK_ARC_EDGE_SMOOTH_MM * ppmm)))
    wide_first, _ = _quadratic_fit_weights(wide_half)
    edge_first, _ = _quadratic_fit_weights(edge_half)

    # ---- 1. height: descend the midline until the surface stabilises ----
    descent_rows = np.arange(
        descent_start_y,
        min(photo_height, descent_start_y + const.NECK_ARC_DESCENT_SEARCH_PX),
    )
    rows_in_band = descent_rows - band_start
    keep = (rows_in_band >= 0) & (rows_in_band < band.shape[0])
    descent_rows, rows_in_band = descent_rows[keep], rows_in_band[keep]
    if descent_rows.size < 40:
        return None
    descent = band[rows_in_band, midline_x]

    found = find_stabilisation_index(descent, wide_first, ppmm, const.NECK_ARC_KNEE_SLOPE_FRACTION)
    if found is None:
        return None
    knee_i, steepest_i = found
    measuring_y = int(descent_rows[knee_i])
    offset_px = measuring_y - int(round(jaw_y_at_midline))

    # ---- 2. shape: the jaw polyline, pushed down by that offset ----
    x_from = max(0, int(jaw_x.min()) - 380)
    x_to = min(photo_width, int(jaw_x.max()) + 380)
    xs = np.arange(x_from, x_to)
    curve_y = np.interp(xs, jaw_x, jaw_y) + offset_px
    curve_rows = np.round(curve_y).astype(int) - band_start
    inside = (curve_rows >= 0) & (curve_rows < band.shape[0])
    if inside.sum() < 100:
        return None
    xs, curve_y, curve_rows = xs[inside], curve_y[inside], curve_rows[inside]
    cross_section = band[curve_rows, xs]

    # ---- 3. edges: the two sharp flanks of the hill ----
    filled_section = _interpolate_gaps(cross_section)
    if filled_section is None:
        return None
    near_midline = np.abs(xs - midline_x) < 60 * ppmm
    if not (near_midline & np.isfinite(cross_section)).any():
        return None
    apex_i = int(
        np.argmin(np.where(near_midline & np.isfinite(cross_section), filled_section, np.inf))
    )

    edges = find_silhouette_edges(
        cross_section,
        apex_i,
        edge_first,
        ppmm,
        int(const.NECK_ARC_MAX_HALF_WIDTH_MM * ppmm),
        int(const.NECK_ARC_DETREND_HALF_MM * ppmm),
    )
    if edges is None:
        return None
    left_i, right_i = edges
    left_x, right_x = int(xs[left_i]), int(xs[right_i])

    # ---- 4. circumference ----
    raw_width_mm = (right_x - left_x) / ppmm
    corrected_width_mm = raw_width_mm * const.NECK_SILHOUETTE_EROSION_CORRECTION
    circumference_mm = float(np.pi * corrected_width_mm)

    # ---- arc points, for drawing and as a diagnostic length ----
    sample_positions = np.linspace(left_i, right_i, n_samples).round().astype(int)
    arc_points_photo, arc_points_3d = [], []
    for i in sample_positions:
        px, py = int(xs[i]), int(round(curve_y[i]))
        z_mm = filled_section[i]
        if not np.isfinite(z_mm):
            continue
        z_cm = z_mm / 10.0
        point_mm = point_to_mm(px, py, z_cm, photo_width, photo_height, camera)
        if point_mm is None:
            continue
        x_mm, y_mm = point_mm
        arc_points_photo.append((px, py))
        arc_points_3d.append((x_mm, y_mm, z_mm))
    if len(arc_points_3d) < 2:
        return None

    front_arc_length_mm = sum(
        vector_length_3d(*arc_points_3d[i - 1], *arc_points_3d[i])
        for i in range(1, len(arc_points_3d))
    )

    return NeckArcResult(
        arc_points_photo=arc_points_photo,
        arc_points_3d=arc_points_3d,
        left_x=left_x,
        right_x=right_x,
        measuring_y=measuring_y,
        offset_px=offset_px,
        apex_x=int(xs[apex_i]),
        midline_x=midline_x,
        raw_width_mm=raw_width_mm,
        corrected_width_mm=corrected_width_mm,
        circumference_mm=circumference_mm,
        front_arc_length_mm=front_arc_length_mm,
        debug={
            "chin_z_mm": chin_z_mm,
            "ppmm": ppmm,
            "steepest_fall_y": int(descent_rows[steepest_i]),
            "jaw_y_at_midline": jaw_y_at_midline,
        },
    )
