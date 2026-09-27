# Chin/skin-fold-clearing arc sag — design

**Date:** 2026-08-12
**Status:** Approved, not yet implemented
**Related:** `NECK_DETECTOR_STATUS.md` (root-cause investigation and the lateral
confidence-gate work that preceded this)

## Problem

`compute_neck_circumference()` (in `portrait_analyser.neck`) samples a
sagging arc between two already-detected neck-side points to measure the
front surface of the neck. The sag amount (`amplitude`) is currently:

```python
full_gap = max(0, round(neck_midpoint_y) - neck_y)
edge_offset = round(full_gap / 3)
neck_y = neck_y + edge_offset
amplitude = full_gap - edge_offset
```

`neck_midpoint_y` is a Pose-derived estimate (interpolated between the
shoulder midpoint and the nose). This ties the sag entirely to a rough 2D
landmark distance that has no relationship to how far the chin, or a
submental skin fold, actually protrudes toward the camera in this specific
photo. **Confirmed unreliable for this dataset** — the sag either:

- looks fine by coincidence (`IMG_6552`, `IMG_6572`: clears the chin by
  ~180px, traces the visible neck surface plausibly), or
- barely dips at all despite a visible skin fold (`IMG_6659`: 44px
  clearance, arc visibly rides the fold instead of clearing below it), or
- collapses to a dead-straight line far from the neck entirely
  (`IMG_6598`: Pose's neck-midpoint estimate was anatomically *above* the
  chin — an impossible value — which zeroes `full_gap`).

The library also already contains an unused, more principled mechanism
(`_find_best_sag`: picks the sag that minimizes 3D depth variation across
the sampled arc) that is bypassed whenever Pose succeeds, which is nearly
always.

## Alternatives considered

- **Always use `_find_best_sag`.** Already exists, doesn't depend on Pose.
  Rejected as the primary mechanism because it's an indirect proxy —
  "flattest 3D variation" is a reasonable heuristic for *a* plausible
  surface, but doesn't specifically verify the arc is behind the chin's
  own depth, and has no explicit handling for a second, stacked fold
  (nothing in its search stops it from settling on a locally-flat
  patch that is itself sitting on top of a fold). The chosen design
  reuses its general depth-sampling approach but adds the two checks
  that make "clears the chin/folds" an explicit, verified condition
  rather than an inferred one.
- **Smaller patch: sanity-guard the Pose value + clamp min/max sag.**
  Lower risk, fixes the `IMG_6598`-style zero-sag collapse. Rejected as
  the primary fix because it doesn't address `IMG_6659`-style
  under-sag — a clamp raises the floor but still isn't *verifying*
  clearance against this specific photo's anatomy, just picking a less
  extreme guess.
- **New explicit clearance check (chosen).** More work, but directly
  verifies the stated requirement — behind the chin, behind any fold —
  against this photo's actual depth data, rather than inferring it from
  a landmark distance or a flatness proxy.

## Goal

Given the neck-side endpoints as already detected (unchanged — see
Non-goals), compute an arc sag that is **verified against depth data** to
clear the chin and any skin fold between the endpoints and the true neck
surface, without relying on Pose neck-midpoint at all.

## Non-goals

- **Lateral point detection** (`neck_y`, `x_left`/`x_right`, the FaceMesh
  jaw guard, the confidence-ratio gate) is out of scope. Treated as given
  input to this work.
- **The search-band sizing bug** found during investigation of `IMG_6598`
  (Pose neck-midpoint above the chin also lets `scan_end_y` balloon to the
  full image height in `neck_search_bounds_from_face_landmarks`) is a
  separate, real bug in point-detection code. Flagged for a future task,
  not fixed here.
- Not porting to `portrait_analyser` yet. This is a fidmaa-gui-local
  prototype; porting is a follow-up once validated against real subjects.

## Architecture

New module: `src/fidmaa_gui/neck_arc.py`. It does **not** vendor
`compute_neck_circumference`. It consumes the endpoints that function
already produces (`neck_y`, `left_x`, `right_x` on the returned
`NeckMeasurement`) and builds a replacement arc from them, reusing only
the library's already-public depth-sampling primitives
(`sample_filtered_depth`, `depth_raw_to_distance_cm`, `pixel_to_mm`,
`vector_length_3d` — all already imported in `app.py` today). No
`portrait_analyser` private (`_`-prefixed) symbols are imported.

Once validated, this module's logic is the basis for a PR against
`portrait-analyser` (`~/Programowanie/portrait-analyser`) that replaces
the inline Pose-fraction formula in `compute_neck_circumference`, released
to PyPI, then the version pin in `fidmaa-gui/pyproject.toml` is bumped.
That migration is a separate follow-up task, not part of this spec.

## Algorithm

Inputs: `filtered_depthmap`, `neck_y`, `x_left`, `x_right`, `chin_photo_xy`,
`photo_width`, `photo_height`, `float_min`, `float_max`, `n_samples=7`.

All "distance-from-camera" comparisons below (stability test, floor check,
no-reversion check) operate on **raw depth-map values**, not calibrated
cm/mm — same convention `_find_stable_depth_x_from_edge` already uses, and
why `find_clearing_amplitude` takes no `float_min`/`float_max`. Calibrated
conversion happens once, later, only in `compute_clearing_arc`'s final 3D
point sampling — mirroring `compute_neck_circumference`'s own step 3-4.

For each of `n_samples` evenly-spaced X columns between `x_left` and
`x_right`:

1. **Vertical walk.** Starting at `neck_y`, sample filtered depth
   downward in Y at native depthmap resolution (mirrors the step-size
   convention `_find_stable_depth_x_from_edge` already uses:
   `max(1.0, (photo_width - 1) / (depthmap.width - 1))`), up to a capped
   maximum extension (see "Safety bound" below).
2. **Stability test.** Reuse the existing statistical shape (robust
   quiet-threshold from the median/MAD of consecutive sample differences,
   then the first run of `stability_run` consecutive samples under that
   threshold) — the same approach `_find_stable_depth_x_from_edge` uses
   horizontally, applied vertically here. This is a new, small
   implementation local to `neck_arc.py`, not an import of the private
   horizontal version (different direction, different exit conditions).
3. **Floor check.** Step 2's quiet-threshold statistic (the robust
   median/MAD-derived "stable_change" value, already computed per column
   to run the stability test) doubles as the margin here: a candidate
   stable point is only accepted if its distance-from-camera exceeds the
   chin's own distance-from-camera (sampled the same way the existing
   thyromental distance calculation already samples chin depth) by at
   least that column's own `stable_change`. One adaptive threshold
   concept for the whole algorithm, not a second, separately-tuned
   number. This is the literal "clear the chin" test.
4. **No-reversion check.** After a candidate passes 2–3, keep sampling a
   short distance further. If depth decreases again by more than that
   same `stable_change` margin (a second, lower fold), treat the first
   candidate as not yet clear and keep walking from there. This is the
   "eventual skin flaps" case — plural, possibly stacked.
5. Each column's resulting clear-Y converts to the half-sine amplitude
   required at that column's own `t` position:
   `(clear_y - neck_y) / sin(pi * t)`, guarding samples near `t≈0`/`t≈1`
   (where `sin` is near zero) from producing runaway values.

**Final amplitude** = max of the per-column required amplitudes across all
columns that produced a valid reading. Using the max (not an average or
the center-only value) guarantees every sample point clears its own
column, not just the middle of the arc.

**Safety bound.** The vertical walk is capped at `neck_y +
const.NECK_ARC_MAX_SEARCH_EXTENSION_PX` (proposed default 400px, tunable)
— a fixed constant added directly to the input `neck_y`, with **no
dependency on `scan_end_y` or Pose neck-midpoint**. This is a deliberate
choice, not an oversight: `scan_end_y` itself can balloon in the same
degenerate-Pose scenario this design exists to be immune to (confirmed in
`IMG_6598`), so bounding against it would silently reinherit the very
failure mode being fixed. The same constant also caps how far a single
noisy column can push the final amplitude.

## New code surface

```python
# src/fidmaa_gui/neck_arc.py

@dataclass
class ClearingArcResult:
    amplitude_px: int
    arc_points_3d: list[tuple[float, float, float]]
    arc_points_photo: list[tuple[int, int]]
    front_arc_length_mm: float
    front_chord_length_mm: float
    # Diagnostics: raw (clear_y - neck_y) vertical offset found for each
    # sample column, i.e. how far *that column itself* had to walk down
    # to clear its own protrusion -- before projection through the
    # half-sine. Not the derived amplitude. None where a column had no
    # valid reading. Same length and column order as arc_points_photo.
    column_clearances_px: list[int | None]

def find_clearing_amplitude(
    filtered_depthmap, neck_y, x_left, x_right,
    chin_photo_xy, photo_width, photo_height,
    max_search_extension_px, n_samples=7, stability_run=4,
) -> tuple[int, list[int | None]]:
    """Returns (amplitude_px, column_clearances_px) -- see
    ClearingArcResult for the latter's meaning. Same n_samples column
    positions this result's amplitude will later be sampled at in
    compute_clearing_arc, so the diagnostics line up 1:1 with the final
    arc points."""

def compute_clearing_arc(
    filtered_depthmap, neck_y, x_left, x_right,
    chin_photo_xy, photo_width, photo_height,
    float_min, float_max,
    max_search_extension_px, n_samples=7,
) -> ClearingArcResult:
    """Calls find_clearing_amplitude for the amplitude and diagnostics,
    then samples the arc exactly like compute_neck_circumference's step
    3-4, reusing the same public sample_filtered_depth /
    depth_raw_to_distance_cm / pixel_to_mm / vector_length_3d
    primitives."""
```

## App integration (validation phase)

In `app.py::_run_detection_and_update()`, after the existing
`self.neck_measurement_auto = compute_neck_circumference(...)` call:

```python
self.neck_clearing_arc = None
if self.neck_measurement_auto is not None and self.filtered_depthmap is not None:
    self.neck_clearing_arc = compute_clearing_arc(
        self.filtered_depthmap,
        neck_y=self.neck_measurement_auto.neck_y,
        x_left=self.neck_measurement_auto.left_x,
        x_right=self.neck_measurement_auto.right_x,
        chin_photo_xy=self.neck_midpoint.chin,
        photo_width=img_w,
        photo_height=self.image.size[1],
        float_min=self.portrait.floatValueMin,
        float_max=self.portrait.floatValueMax,
        max_search_extension_px=const.NECK_ARC_MAX_SEARCH_EXTENSION_PX,
    )
```

The diagnostics panel shows **both** results during validation: the
existing "Automatic neck circumference (3D arc)" line (unchanged formula)
and a new "Chin-clearing arc (prototype)" block with its own amplitude,
arc length, and circumference — clearly labeled as a parallel measurement,
not yet the primary reported number. Nothing about today's headline value
changes as part of this work.

## Error handling

- A column with no valid depth reading anywhere along its walk is excluded
  from the max-amplitude combination (never forces a wrong deep sag from
  missing data).
- If every column fails, `find_clearing_amplitude` returns amplitude 0
  (straight line) rather than guessing, and this is visible in the
  diagnostics panel — never silently wrong.
- Chin depth itself unreadable → the floor check (step 3) is skipped for
  that run; the stability test (step 2) alone still applies. Logged, not
  raised.

## Testing

- **Unit tests, synthetic depth data** (mirrors `portrait-analyser`'s own
  `test_auto_sag_detects_chin` style): construct small PIL `"L"`-mode
  depth images with a synthetic near-camera bump standing in for a
  chin/fold, verify the walk finds clearance past it; a second bump
  further down verifies the no-reversion check keeps walking past
  *stacked* folds; an all-flat depth image verifies amplitude comes back
  as (near-)zero; an all-invalid-depth image verifies the graceful
  fallback to amplitude 0.
- **Real-photo validation**, reusing the sweep tooling already built this
  session: re-check `IMG_6659` (should now dip meaningfully more than
  today's 44px) and `IMG_6598` (should no longer collapse to a flat line
  near the collar) without regressing `IMG_6552`/`IMG_6572`'s already-good
  behavior. Run via the existing scratchpad probe scripts before calling
  this done, not just unit tests in isolation.

## Open questions for the follow-up port to portrait-analyser

Not blocking this spec, but worth carrying forward once this is validated
and ported: should `_find_best_sag` (the existing unused depth-flatness
search) be removed at that point, or kept as a fallback path for callers
without a chin landmark available at all (this new algorithm requires
`chin_photo_xy`; `_find_best_sag` does not)?
