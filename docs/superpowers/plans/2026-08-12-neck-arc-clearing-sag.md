# Chin/Skin-Fold-Clearing Arc Sag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Pose-midpoint-fraction arc sag (which can collapse to zero or under-sag on real subjects) with a depth-verified sag that explicitly clears the chin and any skin fold, computed entirely in a new fidmaa-gui-local module that consumes the existing library's lateral endpoint detection unchanged.

**Architecture:** New module `src/fidmaa_gui/neck_arc.py` walks downward from the already-detected neck row at each of several X columns, using the same stability-detection statistics `portrait_analyser` already uses horizontally (robust median/MAD quiet-threshold), plus two new checks: the candidate point must be farther from the camera than the chin (floor check), and must not get closer again shortly after (no-reversion check, for stacked folds). The per-column results combine into one half-sine amplitude. Shown in the app as a parallel "prototype" measurement alongside the existing one, not yet replacing it.

**Tech Stack:** Python 3.12, PySide6, PIL, numpy, `portrait_analyser` (public API only — `sample_filtered_depth`, `depth_raw_to_distance_cm`, `pixel_to_mm`, `vector_length_3d`), pytest, ruff.

## Global Constraints

- Raw depth values in this codebase are **inverse** to physical distance: a HIGHER raw value means CLOSER to the camera (confirmed against `portrait_analyser`'s own `_make_curved_depth_image` test fixture, which uses a higher `center_value` for the near/center of a simulated neck cylinder). Every comparison in this plan respects this sign — get it backwards and the algorithm accepts points *in front of* the chin as "clear."
- No `portrait_analyser` private (`_`-prefixed) symbols are imported anywhere in `neck_arc.py`. Only its public API.
- Lateral point detection (`neck_y`, `x_left`/`x_right`, jaw guard, confidence ratio) is unchanged — out of scope per `docs/superpowers/specs/2026-08-12-neck-arc-clearing-sag-design.md`.
- Line length 100, ruff-formatted (`ruff check --fix` + `ruff format`), per `CLAUDE.md`.
- Per this session's standing convention, **do not run `git commit` for any step in this plan** — implement and verify only; commits happen later, once the user decides how to batch them together with other pending work already on this branch. Where the step template below shows a commit step, skip it and move to the next task instead.

## Amendments (post-review, before implementation)

Tasks 1-3 were built and run against real photos before this amendment was
written (real-photo validation script, not just the synthetic fixtures
below). Two findings came out of that, both resolved with the user before
implementation proceeded:

1. **Jewelry is out of scope, by product decision.** `IMG_6572` regressed
   circumference by +25% (35.35cm -> 44.32cm) because a thin necklace chain
   was read as "a fold to clear," dragging the arc into an anatomically
   wrong deep V. Confirmed empirically (not just visually) that this is
   the necklace and not hair: sampling `portrait.hairmap` along both
   outlier columns returned 0 (no hair) at every row checked. **Subjects
   are asked to remove jewelry before photography**, so this failure mode
   is accepted as out of scope and no code change addresses it. If it
   resurfaces on a photo where jewelry was supposed to be removed, that is
   a photography-protocol issue, not a bug in this module.
2. **Hair is a related, untested, but plausible risk, and IS in scope**,
   since hair (unlike jewelry) is not something subjects remove. Nothing
   in `_walk_to_clear_y` originally consulted `portrait.hairmap`, so a
   strand crossing a sample column could in principle trigger the same
   "isolated column spike" pattern the necklace did. Chosen fix, after
   discussion: **exclude** hair-covered samples from the walk (same code
   path as already-invalid/zero depth — `continue` past a window
   containing one), not attempt to interpolate/flatten a substitute depth
   value from nearby skin. Interpolation was considered and rejected as
   higher-risk for this session: it needs a source to interpolate from
   (may not exist within a single column's vertical walk if hair covers
   many consecutive rows), and a wrong synthesized value can be just as
   misleading as a hair-biased real one, only more confident-looking.
   This is folded into Task 1 below (originally written without hair
   awareness) rather than kept as a separate task, since it changes that
   task's core interface.

---

### Task 1: Per-column vertical clearance walk

**Files:**
- Create: `src/fidmaa_gui/neck_arc.py`
- Test: `tests/test_neck_arc.py`

**Interfaces:**
- Produces: `_walk_to_clear_y(filtered_depthmap, x, start_y, chin_depth_raw, photo_width, photo_height, max_y, stability_run=4, hairmap=None, hair_threshold=30) -> tuple[int, int] | None` — returns `(clear_y, offset_px)` where `offset_px = clear_y - start_y`, or `None` if nothing valid is found by `max_y`. `chin_depth_raw` may be `None` (chin depth unreadable), in which case the floor check is skipped and only stability applies. `hairmap` may be `None` (no hair-exclusion, e.g. in tests that don't care about it); when given, any sample whose point falls on detected hair (matte value >= `hair_threshold`, same default as `portrait_analyser`'s own `hair_threshold=30`) is treated exactly like an already-invalid depth reading (`None` in the `values` list) — excluded from stability/floor/no-reversion, never fabricated.
- Produces: `_is_hair_at(hairmap, x, y, photo_width, photo_height, hair_threshold) -> bool` — small helper, nearest-neighbor sample into `hairmap`'s own resolution (mirrors `portrait_analyser._prepare_neck_skinmap`'s own resize-then-threshold pattern, but a single-point lookup instead of a resized mask), scaling `(x, y)` from photo-space if `hairmap.size != (photo_width, photo_height)`.

- [x] **Step 1: Write the failing tests**

Create `tests/test_neck_arc.py`:

```python
"""Tests for the depth-verified arc-clearing sag (neck_arc.py).

Uses synthetic PIL "L"-mode depth images -- no HEIC files needed. Mirrors
the synthetic-fixture style of portrait_analyser's own tests/test_neck.py.

Raw depth values here are inverse to distance: HIGHER raw value = CLOSER
to the camera (matches portrait_analyser's own _make_curved_depth_image
convention, where the near/center of a simulated neck cylinder gets a
higher raw value than its receding edges).
"""

from PIL import Image

from fidmaa_gui.neck_arc import _walk_to_clear_y


def _flat_depthmap(width, height, value):
    return Image.new("L", (width, height), value)


def _bump_then_flat_depthmap(width, height, bump_value, flat_value, bump_rows, start_y=0):
    """A near-camera bump (e.g. a chin/fold) for `bump_rows` rows starting
    at start_y, then flat (farther) depth for the rest of the image."""
    img = Image.new("L", (width, height), flat_value)
    for y in range(start_y, min(start_y + bump_rows, height)):
        for x in range(width):
            img.putpixel((x, y), bump_value)
    return img


def _stacked_bumps_depthmap(width, height, start_y):
    """chin bump (rows 0-10) -> a plateau that is stable but NOT clear
    enough (rows 10-25, still closer than the true surface) -> a second,
    closer bump at the SAME depth as the chin (rows 25-35, simulating a
    lower skin fold that is just as much "in front of" the true surface
    as the chin itself) -> the true, clearly-farther neck surface
    (rows 35+)."""
    img = Image.new("L", (width, height), 100)
    bands = [
        (start_y, start_y + 10, 150),  # chin
        (start_y + 10, start_y + 25, 130),  # in-between, not clear yet
        (start_y + 25, start_y + 35, 150),  # second, lower fold (== chin depth)
        # rows >= start_y + 35 stay at the fill value, 100 (true surface)
    ]
    for y0, y1, value in bands:
        for y in range(max(0, y0), min(height, y1)):
            for x in range(width):
                img.putpixel((x, y), value)
    return img


class TestWalkToClearY:
    def test_flat_area_already_farther_than_chin_returns_start_immediately(self):
        # Chin (150) is closer than the uniformly-100 area below it -- the
        # very first stability window already clears it. The reported
        # point is the MIDDLE of that window (stability_run // 2), not
        # its start -- same convention portrait_analyser's own
        # _find_stable_depth_x_from_edge uses -- so offset is small but
        # not necessarily exactly 0.
        depthmap = _flat_depthmap(200, 300, value=100)
        result = _walk_to_clear_y(
            depthmap, x=100, start_y=50, chin_depth_raw=150,
            photo_width=200, photo_height=300, max_y=250,
        )
        assert result is not None
        clear_y, offset = result
        assert 0 <= offset <= 4
        assert clear_y == 50 + offset

    def test_single_bump_is_cleared(self):
        # Chin/fold at 150 for 20 rows, then flat 100 (clearly farther).
        depthmap = _bump_then_flat_depthmap(
            200, 300, bump_value=150, flat_value=100, bump_rows=20, start_y=50,
        )
        result = _walk_to_clear_y(
            depthmap, x=100, start_y=50, chin_depth_raw=150,
            photo_width=200, photo_height=300, max_y=250,
        )
        assert result is not None
        clear_y, offset = result
        # Must have walked past the 20-row bump, not stopped inside it,
        # and not overshot wildly into the flat region either.
        assert 18 <= offset <= 30
        assert clear_y == 50 + offset

    def test_stays_inside_bump_is_rejected_by_floor_check(self):
        # A bump exactly AT chin depth must never be reported as "clear",
        # even though it's perfectly stable on its own.
        depthmap = _flat_depthmap(200, 300, value=150)
        result = _walk_to_clear_y(
            depthmap, x=100, start_y=50, chin_depth_raw=150,
            photo_width=200, photo_height=300, max_y=100,
        )
        # Never becomes farther than the chin within the search window.
        assert result is None

    def test_stacked_folds_walk_past_both(self):
        depthmap = _stacked_bumps_depthmap(200, 300, start_y=50)
        result = _walk_to_clear_y(
            depthmap, x=100, start_y=50, chin_depth_raw=150,
            photo_width=200, photo_height=300, max_y=280,
        )
        assert result is not None
        clear_y, offset = result
        # Must NOT settle inside the first "in-between" plateau (rows
        # 10-25 after start_y, i.e. offset in [10, 25)) -- that's the
        # no-reversion check's job.
        assert not (10 <= offset < 25)
        # Must land at or after the true surface begins (offset >= 35).
        assert offset >= 35

    def test_missing_depth_returns_none(self):
        # An empty (all-zero / invalid) depthmap: sample_filtered_depth
        # returns None for zero-value pixels in portrait_analyser's own
        # convention (0 = invalid disparity).
        depthmap = _flat_depthmap(200, 300, value=0)
        result = _walk_to_clear_y(
            depthmap, x=100, start_y=50, chin_depth_raw=150,
            photo_width=200, photo_height=300, max_y=250,
        )
        assert result is None

    def test_chin_depth_none_skips_floor_check(self):
        # With chin depth unreadable, stability alone is enough -- even a
        # value at/above what would normally fail the floor check. Same
        # window-midpoint convention as the test above -- small non-zero
        # offset is expected, not exactly 0.
        depthmap = _flat_depthmap(200, 300, value=150)
        result = _walk_to_clear_y(
            depthmap, x=100, start_y=50, chin_depth_raw=None,
            photo_width=200, photo_height=300, max_y=250,
        )
        assert result is not None
        clear_y, offset = result
        assert 0 <= offset <= 4

    def test_never_clears_within_max_y_returns_none(self):
        # Bump extends past max_y -- no valid clearance is reachable.
        depthmap = _bump_then_flat_depthmap(
            200, 300, bump_value=150, flat_value=100, bump_rows=100, start_y=50,
        )
        result = _walk_to_clear_y(
            depthmap, x=100, start_y=50, chin_depth_raw=150,
            photo_width=200, photo_height=300, max_y=80,
        )
        assert result is None


def _hair_band_image(width, height, y0, y1, hair_value=200):
    """A hairmap with detected hair (value >= threshold) for rows [y0, y1)
    and none (0) elsewhere -- same "L" mode convention as the depth
    fixtures above."""
    img = Image.new("L", (width, height), 0)
    for y in range(max(0, y0), min(height, y1)):
        for x in range(width):
            img.putpixel((x, y), hair_value)
    return img


class TestWalkToClearYHairExclusion:
    def test_hair_covered_region_is_excluded_not_treated_as_clear(self):
        # Depth is uniformly 100 (already farther than chin=150)
        # everywhere from start_y onward -- WITHOUT hair-awareness this
        # clears almost immediately (see
        # test_flat_area_already_farther_than_chin_returns_start_immediately
        # above, same fixture shape). A hair band sits right where that
        # early clearance would otherwise be found (rows 50-65), so the
        # walk must skip past it and only report clearance once a fully
        # hair-free stable window exists (Y >= 65).
        depthmap = _flat_depthmap(200, 300, value=100)
        hairmap = _hair_band_image(200, 300, y0=50, y1=65)
        result = _walk_to_clear_y(
            depthmap, x=100, start_y=50, chin_depth_raw=150,
            photo_width=200, photo_height=300, max_y=250,
            hairmap=hairmap,
        )
        assert result is not None
        clear_y, offset = result
        assert clear_y >= 65

    def test_hair_covers_entire_window_returns_none(self):
        # Hair-flagged for the whole search range -- even though the
        # underlying depth would clear immediately, nothing hair-free is
        # ever found, so this must fail closed (None), not fall back to
        # trusting the hair-covered reading.
        depthmap = _flat_depthmap(200, 300, value=100)
        hairmap = _hair_band_image(200, 300, y0=50, y1=300)
        result = _walk_to_clear_y(
            depthmap, x=100, start_y=50, chin_depth_raw=150,
            photo_width=200, photo_height=300, max_y=250,
            hairmap=hairmap,
        )
        assert result is None

    def test_no_hairmap_is_unaffected(self):
        # hairmap=None (the default) must reproduce
        # test_single_bump_is_cleared exactly -- this is the
        # backward-compatibility guarantee for every existing caller/test
        # that never passes hairmap.
        depthmap = _bump_then_flat_depthmap(
            200, 300, bump_value=150, flat_value=100, bump_rows=20, start_y=50,
        )
        result = _walk_to_clear_y(
            depthmap, x=100, start_y=50, chin_depth_raw=150,
            photo_width=200, photo_height=300, max_y=250,
        )
        assert result is not None
        _clear_y, offset = result
        assert 18 <= offset <= 30
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_neck_arc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fidmaa_gui.neck_arc'`

- [x] **Step 3: Write the implementation**

Create `src/fidmaa_gui/neck_arc.py`:

```python
"""Depth-verified arc sag: clears the chin and any skin fold between the
already-detected neck-side points, instead of guessing a sag amount from
an unreliable Pose neck-midpoint distance.

See docs/superpowers/specs/2026-08-12-neck-arc-clearing-sag-design.md.

Raw depth values are inverse to physical distance throughout this module:
a HIGHER raw value means CLOSER to the camera.
"""

import numpy as np
from portrait_analyser.depth_sampling import sample_filtered_depth


def _is_hair_at(hairmap, x, y, photo_width, photo_height, hair_threshold):
    """Nearest-neighbor sample into hairmap's own resolution -- mirrors
    portrait_analyser._prepare_neck_skinmap's resize-then-threshold
    pattern, but for a single point instead of a resized mask."""
    hair_w, hair_h = hairmap.size
    hx = round(x * (hair_w - 1) / (photo_width - 1)) if photo_width > 1 else 0
    hy = round(y * (hair_h - 1) / (photo_height - 1)) if photo_height > 1 else 0
    hx = max(0, min(hair_w - 1, hx))
    hy = max(0, min(hair_h - 1, hy))
    return hairmap.getpixel((hx, hy)) >= hair_threshold


def _walk_to_clear_y(
    filtered_depthmap,
    x,
    start_y,
    chin_depth_raw,
    photo_width,
    photo_height,
    max_y,
    stability_run=4,
    hairmap=None,
    hair_threshold=30,
):
    """Walk downward from start_y at column x, looking for the first Y
    that is both locally stable and (raw-depth-wise) farther from the
    camera than the chin, with no reversion shortly after.

    hair-covered samples (per hairmap, when given) are excluded exactly
    like an already-invalid depth reading -- never fabricated, never
    trusted as a "clear" point on their own.

    Returns (clear_y, offset_px) where offset_px = clear_y - start_y, or
    None if nothing valid is found by max_y.
    """
    if max_y <= start_y:
        return None

    native_step = max(1.0, (photo_height - 1) / (filtered_depthmap.height - 1))
    positions = list(np.arange(start_y, max_y + native_step * 0.25, native_step))
    if positions[-1] < max_y:
        positions.append(float(max_y))

    hair_l = hairmap.convert("L") if hairmap is not None else None
    values = []
    for y in positions:
        if hair_l is not None and _is_hair_at(hair_l, x, y, photo_width, photo_height, hair_threshold):
            values.append(None)
        else:
            values.append(sample_filtered_depth(filtered_depthmap, x, y, photo_width, photo_height))

    differences = [
        abs(b - a) for a, b in zip(values, values[1:]) if a is not None and b is not None
    ]
    if len(differences) < stability_run:
        return None

    sorted_differences = sorted(differences)
    quiet = sorted_differences[: max(2, len(sorted_differences) // 2)]
    quiet_median = float(np.median(quiet))
    quiet_mad = float(np.median(np.abs(np.asarray(quiet) - quiet_median)))
    stable_change = max(0.75, quiet_median + 3.0 * 1.4826 * quiet_mad)

    for start in range(0, len(values) - stability_run + 1):
        window = values[start : start + stability_run]
        if any(v is None for v in window):
            continue
        window_diffs = [abs(b - a) for a, b in zip(window, window[1:])]
        if max(window_diffs, default=0.0) > stable_change:
            continue
        if max(window) - min(window) > stable_change * (stability_run - 1):
            continue

        candidate_index = start + stability_run // 2
        candidate_raw = values[candidate_index]

        # Floor check: candidate must be farther from the camera (lower
        # raw value) than the chin, by at least this column's own
        # stability margin.
        if chin_depth_raw is not None and chin_depth_raw - candidate_raw < stable_change:
            continue

        # No-reversion check: a later sample coming back meaningfully
        # closer means we're still above a second, lower fold.
        look_ahead_end = min(len(values), candidate_index + stability_run * 4)
        reverted = False
        for later in values[candidate_index + 1 : look_ahead_end]:
            if later is None:
                continue
            if later - candidate_raw > stable_change:
                reverted = True
                break
        if reverted:
            continue

        clear_y = int(round(positions[candidate_index]))
        return clear_y, clear_y - int(round(start_y))

    return None
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_neck_arc.py -v`
Expected: All 10 tests PASS (7 original + 3 hair-exclusion). If
`test_single_bump_is_cleared` or `test_stacked_folds_walk_past_both` fail
on the exact offset ranges, inspect the actual returned offset (add a
temporary `print(offset)`), confirm it's still behaviorally correct (past
the bump, not inside an "in-between" plateau), and adjust the range
bounds in the test to match reality rather than an invented exact value
-- these two tests intentionally assert ranges, not exact pixels, for
this reason.

---

### Task 2: Combine per-column clearance into one amplitude

**Files:**
- Modify: `src/fidmaa_gui/neck_arc.py`
- Test: `tests/test_neck_arc.py`

**Interfaces:**
- Consumes: `_walk_to_clear_y` from Task 1 (same signature, now including `hairmap`/`hair_threshold`).
- Produces: `find_clearing_amplitude(filtered_depthmap, neck_y, x_left, x_right, chin_photo_xy, photo_width, photo_height, max_search_extension_px, n_samples=7, stability_run=4, hairmap=None, hair_threshold=30) -> tuple[int, list[int | None]]` — returns `(amplitude_px, column_clearances_px)`. `column_clearances_px` has exactly `n_samples` entries, same order as the columns `compute_clearing_arc` (Task 3) will sample. `hairmap`/`hair_threshold` are forwarded to every `_walk_to_clear_y` call unchanged.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_neck_arc.py`:

```python
from fidmaa_gui.neck_arc import find_clearing_amplitude


class TestFindClearingAmplitude:
    def test_symmetric_bump_drives_center_amplitude(self):
        # A chin bump centered under x=100 (columns near x=100 need the
        # most clearance; edge columns near x_left/x_right need little).
        # The painted band starts at y=30 -- above neck_y=50 -- so that
        # chin_photo_xy=(100, 40) (below) actually samples the elevated
        # value too, not the flat background; otherwise chin_depth_raw
        # would equal the background depth and nothing could ever be
        # "farther than the chin".
        width, height = 300, 300
        img = Image.new("L", (width, height), 100)
        for y in range(30, 70):
            for x in range(70, 131):
                img.putpixel((x, y), 150)

        amplitude, clearances = find_clearing_amplitude(
            img,
            neck_y=50,
            x_left=20,
            x_right=180,
            chin_photo_xy=(100, 40),
            photo_width=width,
            photo_height=height,
            max_search_extension_px=150,
        )
        assert amplitude > 0
        assert len(clearances) == 7
        # Endpoint columns (index 0 and 6, at x_left/x_right) sit outside
        # the bump entirely -- should need ~0 clearance.
        assert clearances[0] is not None and clearances[0] <= 5
        assert clearances[-1] is not None and clearances[-1] <= 5
        # The center column sits inside the bump -- should need real
        # clearance, more than the edges.
        center = clearances[len(clearances) // 2]
        assert center is not None and center >= 15

    def test_missing_column_excluded_but_reported(self):
        # Depth entirely invalid (0) -- every column fails to clear, but
        # the function must not crash and must report None for all of
        # them plus fall back to amplitude 0.
        width, height = 300, 300
        img = Image.new("L", (width, height), 0)
        amplitude, clearances = find_clearing_amplitude(
            img,
            neck_y=50,
            x_left=20,
            x_right=180,
            chin_photo_xy=(100, 40),
            photo_width=width,
            photo_height=height,
            max_search_extension_px=100,
        )
        assert amplitude == 0
        assert len(clearances) == 7
        assert all(c is None for c in clearances)

    def test_degenerate_x_range_returns_zero(self):
        width, height = 300, 300
        img = Image.new("L", (width, height), 100)
        amplitude, clearances = find_clearing_amplitude(
            img,
            neck_y=50,
            x_left=100,
            x_right=100,
            chin_photo_xy=(100, 40),
            photo_width=width,
            photo_height=height,
            max_search_extension_px=100,
        )
        assert amplitude == 0
        assert clearances == [None] * 7

    def test_hairmap_forces_more_clearance_at_covered_column(self):
        # Background depth is flat 100 everywhere in the walk region
        # (y>=50); the chin's OWN location (y=35-45, not overlapping the
        # walk region) is separately elevated to 150, matching a real
        # chin protrusion -- same pattern as
        # test_symmetric_bump_drives_center_amplitude above. WITHOUT
        # hair, every column clears almost immediately at y~50 (100 is
        # already farther than the 150 chin). A hair band sits over the
        # center column's early walk rows only (50-65, well below the
        # chin band), forcing just that column to walk further past it
        # -- proving hairmap/hair_threshold actually reach
        # _walk_to_clear_y through this layer, not just
        # accepted-and-ignored.
        width, height = 300, 300
        img = Image.new("L", (width, height), 100)
        for y in range(35, 46):
            for x in range(90, 111):
                img.putpixel((x, y), 150)
        hairmap = Image.new("L", (width, height), 0)
        for y in range(50, 65):
            for x in range(95, 106):  # only around the center column (x=100)
                hairmap.putpixel((x, y), 200)

        amplitude, clearances = find_clearing_amplitude(
            img,
            neck_y=50,
            x_left=20,
            x_right=180,
            chin_photo_xy=(100, 40),
            photo_width=width,
            photo_height=height,
            max_search_extension_px=150,
            hairmap=hairmap,
        )
        center = clearances[len(clearances) // 2]
        assert center is not None and center >= 15
        # Edge columns were never under the hair band -- unaffected.
        assert clearances[0] is not None and clearances[0] <= 5
        assert clearances[-1] is not None and clearances[-1] <= 5
        assert amplitude > 0
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_neck_arc.py -v -k FindClearingAmplitude`
Expected: FAIL with `ImportError: cannot import name 'find_clearing_amplitude'`

- [x] **Step 3: Write the implementation**

Add to `src/fidmaa_gui/neck_arc.py` (below `_walk_to_clear_y`, add `import math` to the top-of-file imports):

```python
def find_clearing_amplitude(
    filtered_depthmap,
    neck_y,
    x_left,
    x_right,
    chin_photo_xy,
    photo_width,
    photo_height,
    max_search_extension_px,
    n_samples=7,
    stability_run=4,
    hairmap=None,
    hair_threshold=30,
):
    """Returns (amplitude_px, column_clearances_px). See _walk_to_clear_y
    for the per-column algorithm; this combines n_samples columns evenly
    spaced between x_left and x_right into one half-sine amplitude, using
    the max of each column's own required amplitude so every sample point
    clears its own column, not just the center. hairmap/hair_threshold
    are forwarded unchanged to every _walk_to_clear_y call."""
    if x_right <= x_left:
        return 0, [None] * n_samples

    chin_x, chin_y = chin_photo_xy
    chin_depth_raw = sample_filtered_depth(
        filtered_depthmap, chin_x, chin_y, photo_width, photo_height
    )

    max_y = neck_y + max_search_extension_px
    step = (x_right - x_left) / max(n_samples - 1, 1)

    column_clearances = []
    required_amplitudes = []
    for i in range(n_samples):
        sx = round(x_left + i * step)
        t = i / max(n_samples - 1, 1)
        result = _walk_to_clear_y(
            filtered_depthmap,
            sx,
            neck_y,
            chin_depth_raw,
            photo_width,
            photo_height,
            max_y,
            stability_run=stability_run,
            hairmap=hairmap,
            hair_threshold=hair_threshold,
        )
        if result is None:
            column_clearances.append(None)
            continue
        _clear_y, offset = result
        column_clearances.append(offset)

        sin_t = math.sin(math.pi * t)
        if sin_t > 0.05:  # guard t near 0/1, where sin is near zero
            required_amplitudes.append(offset / sin_t)

    if not required_amplitudes:
        return 0, column_clearances

    return round(max(required_amplitudes)), column_clearances
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_neck_arc.py -v`
Expected: All tests (Task 1 + Task 2, 14 total) PASS.

---

### Task 3: Full arc sampling and result

**Files:**
- Modify: `src/fidmaa_gui/neck_arc.py`
- Test: `tests/test_neck_arc.py`

**Interfaces:**
- Consumes: `find_clearing_amplitude` from Task 2 (same signature, now including `hairmap`/`hair_threshold`).
- Produces: `ClearingArcResult` dataclass (`amplitude_px: int`, `arc_points_3d: list[tuple[float, float, float]]`, `arc_points_photo: list[tuple[int, int]]`, `front_arc_length_mm: float`, `front_chord_length_mm: float`, `column_clearances_px: list[int | None]`); `compute_clearing_arc(filtered_depthmap, neck_y, x_left, x_right, chin_photo_xy, photo_width, photo_height, float_min, float_max, max_search_extension_px, n_samples=7, hairmap=None, hair_threshold=30) -> ClearingArcResult | None` (`None` when fewer than 2 valid 3D points can be sampled, mirroring `compute_neck_circumference`'s own convention for "not computable"). The final arc-point sampling loop applies the *same* hair exclusion as the walk: a sample point that lands on detected hair is dropped from `arc_points_3d`/`arc_points_photo`, same as an already-invalid depth reading — the walk clearing hair in aggregate doesn't guarantee the exact final `(sx, sample_y)` point isn't itself on a hair pixel, and a hair-biased reading must never feed `front_arc_length_mm` either.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_neck_arc.py`:

```python
from fidmaa_gui.neck_arc import ClearingArcResult, compute_clearing_arc


def _curved_depthmap(width, height, center_value=200, edge_value=150, center_x=None):
    """Mirrors portrait_analyser's own _make_curved_depth_image: center
    (near camera, higher raw value) fading to edges (farther, lower
    value) -- simulates the front-facing curvature of a neck."""
    if center_x is None:
        center_x = width // 2
    img = Image.new("L", (width, height), edge_value)
    for y in range(height):
        for x in range(width):
            dist = abs(x - center_x)
            max_dist = max(center_x, width - center_x)
            ratio = 0 if max_dist == 0 else dist / max_dist
            val = int(center_value * (1 - ratio) + edge_value * ratio)
            img.putpixel((x, y), val)
    return img


class TestComputeClearingArc:
    def test_basic_construction(self):
        width, height = 300, 300
        img = _curved_depthmap(width, height)
        result = compute_clearing_arc(
            img,
            neck_y=50,
            x_left=20,
            x_right=180,
            chin_photo_xy=(100, 10),
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=4.0,
            max_search_extension_px=150,
        )
        assert isinstance(result, ClearingArcResult)
        assert len(result.arc_points_photo) == len(result.arc_points_3d) == 7
        assert len(result.column_clearances_px) == 7
        assert result.front_arc_length_mm >= result.front_chord_length_mm

    def test_bigger_amplitude_gives_longer_arc(self):
        # Same lateral geometry and background in both cases; "bumpy"
        # additionally has the chin's own elevation continue below
        # neck_y (a fold the arc must dip below) -- must force more
        # amplitude and a longer 3D arc than the flat case.
        width, height = 300, 300
        flat = Image.new("L", (width, height), 100)
        bumpy = Image.new("L", (width, height), 100)
        # Chin itself (sampled at y=40, above neck_y=50) reads elevated
        # in both cases, matching a real chin landmark.
        for img in (flat, bumpy):
            for y in range(30, 50):
                for x in range(70, 131):
                    img.putpixel((x, y), 150)
        # Only "bumpy" continues that elevation below neck_y=50.
        for y in range(50, 70):
            for x in range(70, 131):
                bumpy.putpixel((x, y), 150)

        flat_result = compute_clearing_arc(
            flat, neck_y=50, x_left=20, x_right=180, chin_photo_xy=(100, 40),
            photo_width=width, photo_height=height,
            float_min=0.5, float_max=4.0, max_search_extension_px=150,
        )
        bumpy_result = compute_clearing_arc(
            bumpy, neck_y=50, x_left=20, x_right=180, chin_photo_xy=(100, 40),
            photo_width=width, photo_height=height,
            float_min=0.5, float_max=4.0, max_search_extension_px=150,
        )
        assert flat_result is not None and bumpy_result is not None
        assert bumpy_result.amplitude_px > flat_result.amplitude_px
        assert bumpy_result.front_arc_length_mm > flat_result.front_arc_length_mm

    def test_insufficient_depth_returns_none(self):
        width, height = 300, 300
        img = Image.new("L", (width, height), 0)  # all invalid
        result = compute_clearing_arc(
            img, neck_y=50, x_left=20, x_right=180, chin_photo_xy=(100, 10),
            photo_width=width, photo_height=height,
            float_min=0.5, float_max=4.0, max_search_extension_px=150,
        )
        assert result is None

    def test_fully_hair_covered_column_is_dropped_others_unaffected(self):
        # Curved depthmap (all valid depth, no zeros). One interior sample
        # column (x=100) is hair-covered for its ENTIRE vertical extent --
        # guaranteed to still be hair-covered at whatever Y the final
        # half-sine sampling lands on for that column, without needing to
        # predict the exact amplitude in advance. chin_photo_xy=(140, 10)
        # deliberately avoids every one of the 7 sample columns (20, 50,
        # 75, 100, 125, 150, 175 for this x_left/x_right/n_samples), so
        # this isn't entangled with the "chin sampled exactly on a column"
        # edge case test_basic_construction already exercises.
        width, height = 300, 300
        img = _curved_depthmap(width, height)
        hairmap = Image.new("L", (width, height), 0)
        for x in range(95, 106):
            for y in range(height):
                hairmap.putpixel((x, y), 200)

        result = compute_clearing_arc(
            img, neck_y=50, x_left=25, x_right=175, chin_photo_xy=(140, 10),
            photo_width=width, photo_height=height,
            float_min=0.5, float_max=4.0, max_search_extension_px=150,
            hairmap=hairmap,
        )
        assert result is not None
        # The x=100 column (fully hair-covered) is dropped; the other 6
        # come through untouched.
        assert len(result.arc_points_photo) == 6
        assert all(x != 100 for x, _y in result.arc_points_photo)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_neck_arc.py -v -k ComputeClearingArc`
Expected: FAIL with `ImportError: cannot import name 'ClearingArcResult'`

- [x] **Step 3: Write the implementation**

Add to `src/fidmaa_gui/neck_arc.py` (add `math` already imported in Task 2; add these imports at the top: `from dataclasses import dataclass` and `from portrait_analyser.incisor import depth_raw_to_distance_cm, pixel_to_mm, vector_length_3d`):

```python
@dataclass
class ClearingArcResult:
    amplitude_px: int
    arc_points_3d: list[tuple[float, float, float]]
    arc_points_photo: list[tuple[int, int]]
    front_arc_length_mm: float
    front_chord_length_mm: float
    # Diagnostics: raw (clear_y - neck_y) vertical offset found for each
    # sample column -- how far that column itself had to walk down to
    # clear its own protrusion, before projection through the half-sine.
    # Not the derived amplitude. None where a column had no valid
    # reading. Same length and column order as arc_points_photo.
    column_clearances_px: list[int | None]


def compute_clearing_arc(
    filtered_depthmap,
    neck_y,
    x_left,
    x_right,
    chin_photo_xy,
    photo_width,
    photo_height,
    float_min,
    float_max,
    max_search_extension_px,
    n_samples=7,
    hairmap=None,
    hair_threshold=30,
):
    """Finds the clearing amplitude, then samples the arc exactly like
    compute_neck_circumference's own step 3-4. Returns None when fewer
    than 2 valid 3D points can be sampled, matching that function's
    convention for "not computable" rather than returning a fake zero
    measurement. hairmap/hair_threshold are forwarded to the amplitude
    search AND applied again here at final sampling -- the walk clearing
    hair in aggregate doesn't guarantee the exact final (sx, sample_y)
    point isn't itself on a hair pixel."""
    amplitude, column_clearances = find_clearing_amplitude(
        filtered_depthmap,
        neck_y,
        x_left,
        x_right,
        chin_photo_xy,
        photo_width,
        photo_height,
        max_search_extension_px,
        n_samples=n_samples,
        hairmap=hairmap,
        hair_threshold=hair_threshold,
    )

    hair_l = hairmap.convert("L") if hairmap is not None else None
    step = (x_right - x_left) / max(n_samples - 1, 1)
    arc_points_3d = []
    arc_points_photo = []
    for i in range(n_samples):
        sx = round(x_left + i * step)
        t = i / max(n_samples - 1, 1)
        sample_y = neck_y + round(amplitude * math.sin(math.pi * t))

        if hair_l is not None and _is_hair_at(
            hair_l, sx, sample_y, photo_width, photo_height, hair_threshold
        ):
            continue
        raw_depth = sample_filtered_depth(filtered_depthmap, sx, sample_y, photo_width, photo_height)
        if raw_depth is None:
            continue
        z_cm = depth_raw_to_distance_cm(raw_depth, float_min, float_max)
        if z_cm is None:
            continue
        x_mm = pixel_to_mm(sx, z_cm, photo_width)
        y_mm = pixel_to_mm(sample_y, z_cm, photo_height)
        if x_mm is None or y_mm is None:
            continue

        arc_points_3d.append((x_mm, y_mm, z_cm * 10.0))
        arc_points_photo.append((sx, sample_y))

    if len(arc_points_3d) < 2:
        return None

    front_arc_length_mm = 0.0
    for i in range(1, len(arc_points_3d)):
        front_arc_length_mm += vector_length_3d(*arc_points_3d[i - 1], *arc_points_3d[i])
    front_chord_length_mm = vector_length_3d(*arc_points_3d[0], *arc_points_3d[-1])

    return ClearingArcResult(
        amplitude_px=amplitude,
        arc_points_3d=arc_points_3d,
        arc_points_photo=arc_points_photo,
        front_arc_length_mm=front_arc_length_mm,
        front_chord_length_mm=front_chord_length_mm,
        column_clearances_px=column_clearances,
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_neck_arc.py -v`
Expected: All tests (Tasks 1-3, 18 total) PASS.

- [x] **Step 5: Run ruff on the new module and test file**

Run: `uv run ruff check --fix src/fidmaa_gui/neck_arc.py tests/test_neck_arc.py && uv run ruff format src/fidmaa_gui/neck_arc.py tests/test_neck_arc.py`
Expected: clean / "All checks passed!", reformats applied if needed. Re-run `uv run pytest tests/test_neck_arc.py -v` after formatting to confirm nothing broke.

---

### Task 4: Wire into the app as a parallel diagnostic

**Files:**
- Modify: `src/fidmaa_gui/const.py`
- Modify: `src/fidmaa_gui/app.py`

**Interfaces:**
- Consumes: `compute_clearing_arc` and `ClearingArcResult` from Task 3.

- [x] **Step 1: Add the safety-bound constant**

In `src/fidmaa_gui/const.py`, after the existing `NECK_LOW_CONFIDENCE_RATIO_THRESHOLD` block, add:

```python

# Caps how far below neck_y the chin/skin-fold clearing search (neck_arc.py)
# is willing to walk, and how far a single noisy column can push the final
# amplitude. Deliberately NOT derived from scan_end_y or Pose neck-midpoint
# -- both can be corrupted by the same degenerate-Pose scenario this search
# exists to be immune to (see NECK_DETECTOR_STATUS.md, IMG_6598). Provisional
# constant, tune against more subjects once validated.
NECK_ARC_MAX_SEARCH_EXTENSION_PX = 400
```

- [x] **Step 2: Add the import**

In `src/fidmaa_gui/app.py`, near the existing `from .calculations import findPoint, lower_jaw_x_bounds, neck_to_jaw_width_ratio` line, add a new import line directly below it:

```python
from .neck_arc import compute_clearing_arc
```

- [x] **Step 3: Reset the new attribute in `__init__`**

In `src/fidmaa_gui/app.py`, find:

```python
        self.neck_lateral_bounds = None
        self.neck_confidence_ratio = None
```

Replace with:

```python
        self.neck_lateral_bounds = None
        self.neck_confidence_ratio = None
        self.neck_clearing_arc = None
```

- [x] **Step 4: Reset the new attribute at the top of `_run_detection_and_update`**

In `src/fidmaa_gui/app.py`, find:

```python
        self.neck_lateral_bounds = None
        self.neck_confidence_ratio = None
        self.neck_midpoint = None
```

Replace with:

```python
        self.neck_lateral_bounds = None
        self.neck_confidence_ratio = None
        self.neck_clearing_arc = None
        self.neck_midpoint = None
```

- [x] **Step 5: Compute the clearing arc after the existing confidence ratio**

In `src/fidmaa_gui/app.py`, find:

```python
        if self.neck_measurement_auto is not None:
            self.neck_confidence_ratio = neck_to_jaw_width_ratio(
                self.neck_measurement_auto.mask_left_x,
                self.neck_measurement_auto.mask_right_x,
                self.neck_lateral_bounds,
            )
```

Replace with:

```python
        if self.neck_measurement_auto is not None:
            self.neck_confidence_ratio = neck_to_jaw_width_ratio(
                self.neck_measurement_auto.mask_left_x,
                self.neck_measurement_auto.mask_right_x,
                self.neck_lateral_bounds,
            )

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
                hairmap=self.portrait.hairmap,
            )
```

- [x] **Step 6: Show it in the diagnostics panel**

In `src/fidmaa_gui/app.py`, find (the block added for the confidence ratio, ending right before the diameter-approximation comment):

```python
                    if self.neck_confidence_ratio is not None:
                        txt += (
                            "\n  Neck/jaw width ratio:"
                            f" {self.neck_confidence_ratio:.2f}"
                            f" (warn at ≥ {const.NECK_LOW_CONFIDENCE_RATIO_THRESHOLD:.2f})"
                        )
                    if low_confidence:
                        txt += (
                            "\n  ⚠ No anatomical narrowing was found below the chin --"
                            " this measurement likely includes shoulder or chest skin,"
                            " not just the neck."
                        )

                    # Neck circumference approximation from diameter (left_x → right_x) * pi
```

Replace with:

```python
                    if self.neck_confidence_ratio is not None:
                        txt += (
                            "\n  Neck/jaw width ratio:"
                            f" {self.neck_confidence_ratio:.2f}"
                            f" (warn at ≥ {const.NECK_LOW_CONFIDENCE_RATIO_THRESHOLD:.2f})"
                        )
                    if low_confidence:
                        txt += (
                            "\n  ⚠ No anatomical narrowing was found below the chin --"
                            " this measurement likely includes shoulder or chest skin,"
                            " not just the neck."
                        )

                    if self.neck_clearing_arc is not None:
                        nca = self.neck_clearing_arc
                        prototype_circumference_cm = (
                            nca.front_arc_length_mm * nma.circumference_multiplier / 10.0
                        )
                        txt += "\n\nChin-clearing arc (prototype): "
                        txt += f"{prototype_circumference_cm:.2f} cm"
                        txt += f"\n  Amplitude used: {nca.amplitude_px} px"
                        txt += f"\n  Front surface arc (3D polyline): {nca.front_arc_length_mm:.2f} mm"
                        txt += (
                            "\n  Arc endpoint chord (3D Euclidean):"
                            f" {nca.front_chord_length_mm:.2f} mm"
                        )
                        txt += f"\n  Per-column clearance (px): {nca.column_clearances_px}"
                    elif self.neck_measurement_auto is not None:
                        txt += "\n\nChin-clearing arc (prototype): unavailable (insufficient depth data)"

                    # Neck circumference approximation from diameter (left_x → right_x) * pi
```

- [x] **Step 7: Run the full test suite and ruff across the repo**

Run: `uv run pytest -q`
Expected: all tests pass (existing suite + Task 1-3's new `tests/test_neck_arc.py`), none broken by the `app.py`/`const.py` edits (app.py's GUI logic isn't directly unit-tested, per the existing convention noted in `tests/test_app_calculations.py`'s own docstring, so this is a regression check, not new coverage for this step).

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: `All checks passed!` and no reformatting needed (both files were already formatted in Task 3, Step 5, but re-check the whole repo here to catch anything missed).

---

### Task 5: Real-photo validation

**Files:** None modified — this task runs the app's actual detection pipeline against real subjects to confirm the fix behaves as intended, reusing the scratchpad probe-script pattern already established this session.

**Interfaces:** None new — exercises `compute_clearing_arc` (Task 3) exactly as wired in Task 4.

- [x] **Step 1: Write a validation script** — already done during pre-implementation
  review (before this amendment) at
  `/private/tmp/claude-501/-Users-mpasternak-Programowanie-fidmaa-gui/29d4380f-5085-4e25-a4ba-ba0393ef1153/scratchpad/neck_faceparse/validate_clearing_arc2.py`
  (current session's scratchpad path — re-run from there, or recreate if
  that path is gone by the time this task executes). It: loads each
  image via `load_image`, runs `detect_neck_midpoint` →
  `neck_search_bounds_from_face_landmarks` → jaw guard →
  `compute_neck_circumference` to get `neck_measurement_auto` and
  `filtered_depthmap` (`median_filter_depthmap(portrait.depthmap,
  size=3)`), then calls `compute_clearing_arc` with those results
  (Task 4 Step 5's exact call — **update it to also pass
  `hairmap=portrait.hairmap`** for this task's re-run, since the script
  predates the hair-exclusion amendment), prints old vs new amplitude /
  `front_arc_length_mm` / circumference / `column_clearances_px`, and
  renders an annotated crop (old arc yellow, new arc cyan, chin magenta).

- [x] **Step 2: Re-run it (with hair-exclusion wired in) and check the
  expected outcomes**

Run: `uv run python <path-to-script>/validate_clearing_arc2.py`

Expected, based on the actual pre-implementation run's real numbers
(reproduce these first; hair-exclusion should not change them for these
four photos — none of the four have hair crossing a sample column):
- `IMG_6552`: front_arc_length_mm/circumference stay close to the old
  value (97.27mm/29.18cm old -> ~98.67mm/29.60cm new, ~1.4% difference)
  even though the raw amplitude number itself differs (107px "old"
  peak-offset vs 34px new "amplitude" -- these use different reference
  conventions, see the Amendments section above; the arc-length/
  circumference metric is what must stay stable, not the raw amplitude
  number). No regression.
- `IMG_6659`: new amplitude **meaningfully larger** than old (55px ->
  73px), arc length/circumference essentially unchanged (157.43mm/
  47.23cm -> 155.29mm/46.59cm) -- the new arc visually sits a modest,
  plausible amount below the old one, tracking a real skin-fold
  transition rather than under- or over-reacting to it.
- `IMG_6572`: **expected to still regress** (35.35cm old -> ~44.32cm
  new) -- this is the accepted, out-of-scope jewelry case from the
  Amendments section (necklace read as a fold to clear; confirmed via
  hairmap sampling that it is not hair). Do not treat this as a bug to
  fix in this task; re-verify the visual (necklace visible, arc dips
  well below it) matches the earlier finding, then move on.
- `IMG_6598`: new amplitude technically nonzero (0px -> ~20px) but
  circumference stays implausible (19.44cm -> ~19.51cm, essentially
  unchanged) -- `neck_y` itself is anchored near Y=3055, at/past the
  collar, nowhere near the actual neck. This is the separately-disclosed
  `scan_end_y`/row-selection bug from the linked design spec's
  Non-goals, not something this task fixes. Confirm the number stays in
  the same (still-wrong) ballpark, not a new failure mode.

- [x] **Step 3: If any subject doesn't match these expectations
  (excluding the two accepted cases above), diagnose before proceeding**

If a result is unexpected in a way *not* already explained by the
`IMG_6572` jewelry case or the `IMG_6598` row-selection bug, use
`column_clearances_px` from the printed output to see which columns the
search thought were already clear -- this is the same diagnostic the
app's own panel now shows (Task 4, Step 6), so what's seen in this
script is what the user will see in the running app. Do not adjust the
algorithm to force a specific number; if the diagnostics reveal a
genuine bug (e.g. a sign error, an off-by-one in the stability window,
or the hair exclusion not actually engaging), fix that and re-run this
task's Step 2 from the top.

---

## Self-Review Notes

(Completed during plan authoring, not a task for the implementer.)

- **Spec coverage:** Vertical walk + stability (spec Algorithm steps 1-2) → Task 1. Floor check + no-reversion (steps 3-4) → Task 1. Amplitude combination + safety bound (step 5, Safety bound) → Task 2. Arc sampling + `ClearingArcResult` → Task 3. App integration + parallel display → Task 4. Real-photo validation → Task 5. The spec's "Alternatives considered" and "Non-goals" sections don't produce code and have no corresponding task, correctly.
- **Placeholder scan:** No TBD/TODO; every step has real, complete code; test assertions use explicit ranges (documented as such) rather than vague "should work" language.
- **Type consistency:** `find_clearing_amplitude`'s `(amplitude_px, column_clearances_px)` return tuple matches `compute_clearing_arc`'s internal unpacking (`amplitude, column_clearances = find_clearing_amplitude(...)`) and `ClearingArcResult.column_clearances_px`'s field name/order. `_walk_to_clear_y`'s `(clear_y, offset_px)` tuple matches how Task 2 unpacks it (`_clear_y, offset = result`). Parameter name `max_search_extension_px` is identical across all three functions and the `const.py` constant name matches its usage in Task 4 Step 5.
- **Numerical verification:** All three functions' code was copy-pasted into a throwaway script and actually run against every test fixture in this plan (not just hand-traced) before finalizing. This caught and fixed three real bugs the first draft had: two tests asserting `offset == 0` where the algorithm's window-midpoint convention (mirroring `_find_stable_depth_x_from_edge`'s own `start + stability_run // 2`) actually produces a small non-zero value; and the stacked-folds fixture's second bump (148) being too close to the chin depth (150) to reliably fail the floor check against the near-zero `stable_change` a noise-free synthetic image produces, which would have let the walk falsely settle inside the second fold instead of clearing it. All 30 assertions across every task's tests pass against the final code in this plan.
- **Amendment verification (post-review):** Tasks 1-3 (pre-hair version) were materialized at their real file paths and run against real photos before this amendment was written — not just the synthetic fixtures. This is what surfaced the jewelry regression on `IMG_6572` (see Amendments section) and produced the real `IMG_6552`/`IMG_6659`/`IMG_6598` numbers now baked into Task 5's expected outcomes (replacing this plan's original, incorrect guess of "~180px clearance" for the two good cases — actual measured values were 107px/108px by the "peak minus neck_y" convention). The hair-exclusion addition (3 new tests in Task 1, 1 in Task 2, 1 in Task 3, 18 total) was then actually implemented and run, same rigor as Tasks 1-3: this caught one real fixture bug in the first draft of `test_hairmap_forces_more_clearance_at_covered_column` (`center` came back `None` instead of the required minimum) — the fixture's chin depth (100) exactly equaled the walk region's background depth (100, both flat), so the floor check could never pass at that column regardless of hair, the same trap `_curved_depthmap`-based tests avoid by giving the chin its own genuinely elevated depth. Fixed by adding a separate elevated chin patch (150, at the chin's own y=35-45, not overlapping the y>=50 walk region) — all 18 tests pass against the final code below.

---

## Implementation status: complete (2026-08-12)

All five tasks implemented directly in this session (not via subagent
dispatch — the mandated `superpowers:subagent-driven-development` skill's
worktree/per-task-commit mechanics conflict with this plan's own "no
commit" constraint and the fact that work was already in progress on
`main`; confirmed with the user before proceeding).

- `src/fidmaa_gui/neck_arc.py`, `tests/test_neck_arc.py`: 18/18 tests pass
  (`uv run pytest tests/test_neck_arc.py -v`).
- Full repo suite: **142/142 pass** (`uv run pytest -q`).
- `uv run ruff check .` / `ruff format --check .`: clean.
- Task 5 re-validation (with hair-exclusion wired in, via
  `validate_clearing_arc2.py` against the real pipeline): numbers for all
  four photos are **identical** to the pre-hair run — confirms hair
  exclusion is a correct no-op here (none of the four have hair crossing a
  sample column) rather than accidentally changing behavior. `IMG_6659`
  improves as intended (55px -> 73px clearance, arc length essentially
  unchanged). `IMG_6552` stays stable. `IMG_6572` and `IMG_6598` remain the
  two accepted, out-of-scope cases documented above and in Task 5.

**Not done, by design:** no `git commit` (Global Constraint #5 — pending
user decision on how to batch with other work already on `main`); no
worktree/branch (same reason); no port to `portrait-analyser` (explicit
Non-goal in the linked design spec).
