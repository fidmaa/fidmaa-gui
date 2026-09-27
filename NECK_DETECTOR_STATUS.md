# Neck detector: current state, verified failures, and consultation notes

**Date:** 2026-08-12  
**Application:** FIDMAA GUI 0.1.3  
**Installed analyser:** `portrait-analyser 0.6.1` (`portrait-analyser[pose]>=0.6.1,<0.7`)

## Executive summary

The current automatic neck detector is experimental and is **not a reliable anatomical neck detector**. It can find a useful vertical area below the chin, but it cannot reliably identify the left and right sides of the neck when skin is continuously visible from the neck onto the shoulders or upper chest.

`IMG_6640.heic` is the clearest failure. Its vertical measurement height is acceptable and does not cross the chin, but the lateral detector measures approximately shoulder-to-shoulder. A recently added FaceMesh lower-jaw guard moves the right endpoint inward, but it still does not identify the true neck boundary and the reported circumference remains implausible.

There are two separate unsolved problems:

1. **Detection:** locate the anatomical left and right neck points without selecting chin, hair, shoulder, or chest.
2. **Estimation:** convert the detected visible 3D curve into a complete neck circumference. A fixed multiplier is not geometrically reliable when the detected curve covers a different fraction of the neck on different subjects.

These problems should be solved and validated separately.

## Intended measurement

The intended visible curve should:

- lie immediately below the lowest chin/FaceMesh landmarks;
- not pass through the projected chin surface;
- begin at the left anatomical side of the neck;
- end at the right anatomical side of the neck;
- not continue onto either shoulder, hair, clothing, or upper chest;
- use the depth map to measure the 3D front surface between those endpoints;
- produce a circumference estimate only when endpoint detection is sufficiently confident.

The current implementation does not satisfy the endpoint requirements for all subjects.

## Current implementation

### 1. Input data

The HEIC loader supplies:

- full-resolution RGB photo;
- semantic skin matte;
- semantic hair matte;
- TrueDepth disparity map;
- EXIF depth calibration values.

The app currently uses seven arc samples and a fixed circumference multiplier of `3.0`.

### 2. Face and pose landmarks

MediaPipe supplies:

- 478 FaceMesh landmarks;
- FaceMesh chin landmark 152;
- nose, eyes, and mouth landmarks;
- Pose shoulders;
- a Pose-derived central neck point.

FaceMesh describes the face and jaw, not the sides of the neck. Pose provides shoulder locations, but no lateral neck landmarks.

### 3. Vertical search band

`portrait_analyser.neck.neck_search_bounds_from_face_landmarks()` creates a vertical search band below the face:

- face bottom is the lower of the detected chin and all FaceMesh landmarks;
- start is approximately face bottom plus 1.5% of face height;
- anatomical end is face bottom plus 60% of face height;
- the Pose neck midpoint may shorten the lower boundary;
- a minimum band of approximately 18% of face height is retained.

Within this band, `find_narrowest_skin_row()` searches for the first stable local minimum of skin width. If no stable local minimum exists, it uses a width score with a small penalty for lower rows.

This vertical selection is acceptable in `IMG_6640`: the line is below the chin. The main failure there is lateral, not vertical.

### 4. Skin and hair preparation

The analyser:

- thresholds the skin matte at 30;
- applies a 5×5 median filter;
- thresholds the hair matte at 30;
- expands the hair mask by one pixel;
- removes detected hair from the allowed skin mask.

Important clarification: the displayed **hairless depth image is not used as the source of distance measurements**. The computation samples the median-filtered original depth map. The hair matte is used as a validity mask so hair pixels should not be accepted as neck skin.

### 5. App-level FaceMesh jaw guard

The current working tree adds a lateral guard in `fidmaa_gui.app`:

- lower-jaw FaceMesh indices are taken from both sides of the face;
- their minimum and maximum X positions define a jaw envelope;
- a 2% width margin is added;
- skin pixels outside this envelope are masked only inside the neck search band.

Relevant code:

- `src/fidmaa_gui/app.py`, neck computation in `_run_detection_and_update()`;
- `src/fidmaa_gui/calculations.py`, `lower_jaw_x_bounds()`;
- `src/fidmaa_gui/const.py`, `FACE_MESH_LOWER_JAW_INDICES` and margin.

This is only a coarse safety guard. It is **not a neck-side detector**. A neck can be narrower or wider relative to the jaw, and the true neck-to-shoulder transition is not represented by FaceMesh.

### 6. Lateral skin endpoints

After selecting a row, `compute_neck_circumference()`:

1. finds the contiguous prepared-skin run nearest the estimated neck centre;
2. treats the run's first and last pixels as raw skin endpoints;
3. walks inward from each endpoint in the depth map;
4. accepts the first locally stable depth patch;
5. limits that inward search to approximately 12% of the skin-run width.

This removes many TrueDepth silhouette-wall artefacts, but it does not distinguish neck skin from contiguous shoulder/chest skin. A quiet depth patch on the shoulder can be accepted as a valid neck endpoint.

### 7. Sagging arc path

The initially selected row is adjusted toward the Pose neck midpoint:

- the endpoint row moves down by one third of the gap to the midpoint;
- the centre of the curve reaches the Pose neck midpoint;
- intermediate Y positions follow a half-sine curve;
- seven evenly spaced X positions are sampled;
- each sample is converted to calibrated 3D X/Y/Z coordinates;
- Euclidean distances between consecutive samples are summed.

The sag was restored because a straight line at the earlier selected row could pass through the projected chin surface. However, this sagging curve is not a true horizontal cross-section. Its vertical amplitude varies by subject and can therefore change the measured length independently of real circumference.

### 8. Reported estimates

The GUI currently reports:

- `front_arc_length_mm`: sagging 3D polyline length;
- `front_chord_length_mm`: direct 3D distance between arc endpoints;
- automatic circumference: `front_arc_length_mm × 3.0`;
- diagnostic circle estimate: `front_chord_length_mm × π`;
- straight-row surface-vector diagnostics.

No confidence score or rejection rule currently prevents an implausible result from being shown as an automatic circumference.

## Verified cases

All images are 2320×3087 and are located in:

`~/IPLweb Dropbox/Michał Pasternak/FIDMAA badanie #2 backup/Pacjenci FIDMA/`

The user initially referred to `IMG_6440.heic`; the available file and screenshot are `IMG_6640.heic`.

### Current results

| Image | Tape measurement | Search band | Final skin run | Stable endpoints | 3D arc | 3D chord | Arc × 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `IMG_6552` | 30.5 cm | Y 2117–2335 | X 704–1335 | X 738–1296 | 97.27 mm | 82.27 mm | 29.18 cm |
| `IMG_6572` | 33.3 cm | Y 2105–2312 | X 737–1407 | X 785–1363 | 117.83 mm | 98.77 mm | 35.35 cm |
| `IMG_6640` | not supplied | Y 2126–2316 | X 579–1317 | X 608–1302 | 189.52 mm | 124.40 mm | 56.86 cm |

`IMG_6552` and `IMG_6572` were not changed by the FaceMesh jaw guard because their central skin runs already fell within it.

### `IMG_6640` before and after the jaw guard

Detected landmarks:

- FaceMesh chin: approximately `(944.6, 2110.1)`;
- Pose neck midpoint: approximately `(1007.4, 2178.1)`;
- vertical search band: Y `2126–2316`;
- final endpoint row: Y `2143`.

Before the jaw guard:

- prepared skin run: X `579–1442`;
- stable depth endpoints: X `608–1379`;
- 3D arc: `206.76 mm`;
- chord: `137.71 mm`;
- circumference at multiplier 3.0: `62.03 cm`.

After the jaw guard:

- FaceMesh jaw guard: X `554–1317`;
- prepared skin run: X `579–1317`;
- stable depth endpoints: X `608–1302`;
- 3D arc: `189.52 mm`;
- chord: `124.40 mm`;
- circumference at multiplier 3.0: `56.86 cm`.

The right point moves inward, but the detector still measures approximately shoulder-to-shoulder. The guard reduces the error but does not solve the anatomical detection problem.

The outer points are also roughly 60–77 mm farther from the camera than the central neck surface. This makes the arc much more curved in depth than in the two usable examples.

### Useful failure indicator

The ratio `3D arc / endpoint chord` differs strongly:

- `IMG_6552`: approximately `1.182`;
- `IMG_6572`: approximately `1.193`;
- `IMG_6640` after guard: approximately `1.524`.

The first two curves cover a much smaller angular portion of the apparent neck surface. `IMG_6640` behaves almost like a half-circle/shoulder contour. This ratio could be used as a quality warning, but it does not by itself locate the correct endpoints.

## Confirmed problems

### A. Skin segmentation is not anatomical-part segmentation

The semantic matte answers “is this skin?” It does not answer “is this neck?” Bare neck, shoulder, and chest are one connected foreground component. Selecting the first and last skin pixel therefore becomes shoulder-to-shoulder in some poses and clothing arrangements.

This is the primary `IMG_6640` failure.

### B. FaceMesh cannot directly supply neck endpoints

FaceMesh ends at the jaw/chin. Its jaw envelope can reject an extreme lateral spill, but it cannot identify the neck-to-shoulder junction. A hard jaw-width clamp may also fail on subjects whose neck/jaw proportions differ.

### C. Current depth stabilisation solves the wrong boundary problem

Walking inward until depth becomes locally quiet is useful at foreground/background sensor walls. It can still settle on a quiet shoulder surface. Depth stability is not evidence that the point belongs to the neck.

### D. The sagging path is not a planar circumference section

The half-sine path mixes transverse width, depth curvature, and vertical travel. Different chin-to-neck-midpoint gaps create different path geometries. Therefore a constant multiplier may compensate one subject and fail another.

### E. Seven points can under-resolve or overreact to local depth shape

Seven points are visually clear and reduce accumulated sensor noise, but a single bad point or steep transition can strongly affect the polyline. Computational sampling and displayed markers may need to be separated.

### F. A fixed multiplier is not supported by the current geometry

Observed required multipliers vary substantially. Earlier examples implied approximately `3.08` and `2.63`. Changing the configured multiplier from 3.1 to 3.0 does not fix incorrect endpoints.

### G. No quality rejection

The application currently presents values such as 56.86 cm as an automatic result even when diagnostic ratios are obvious outliers. It should refuse or prominently flag low-confidence measurements.

## Approaches tried and reverted

These experiments should not be repeated without a new rationale:

1. **Straight row with `arc_sag=0`:** avoided variable vertical travel but passed through the projected chin in some images because the selected endpoint row was too close to the chin.
2. **Additional 8% face-height clearance below the chin:** prevented chin intersection but skipped the narrow upper-neck area in `IMG_6572`, moving the measurement too low and inflating the result.
3. **Fixed `+3.84 cm` correction to `diameter × π`:** fitted `IMG_6552` only and failed on `IMG_6572`; it was removed.
4. **FaceMesh jaw guard:** still present as a safety bound, but verified insufficient on `IMG_6640`.

## Recommended direction for consultation

### 1. Define and annotate ground-truth endpoints

For a representative set of subjects, manually mark:

- lowest chin exclusion boundary;
- desired left neck point;
- desired right neck point;
- tape-measure plane;
- measured circumference.

Without endpoint annotations, it is impossible to distinguish detector error from circumference-model error.

### 2. Detect neck sides as contours, not as skin-run extrema

A candidate method should track the left and right body contours below the jaw and identify the transition from near-vertical neck sides to the more horizontal shoulder slope. Useful signals may include:

- contour slope and curvature across several rows;
- bilateral symmetry around a robust neck centre;
- continuity from jaw-adjacent neck contours;
- Pose shoulder locations as exclusion targets;
- local RGB edges where hair/clothing separate from skin;
- skin and hair probabilities, not only thresholded masks;
- depth gradients and surface normals;
- plausible neck-width change across Y.

The left/right points should be optimised jointly. Independent “walk inward from the outer skin edge” searches are vulnerable to asymmetric hair and exposed shoulders.

### 3. Separate endpoint detection from depth sampling

The detector should first return two confident photo-space neck endpoints and a measurement level. Only then should the depth algorithm sample a curve between them. A failure to find anatomical endpoints should return `None`, not a shoulder-based circumference.

### 4. Reconsider the measurement curve

The consultant should decide whether the desired measurement represents:

- a horizontal anatomical cross-section;
- a projected curve intentionally dipping below the chin;
- a fitted 3D plane through detected neck points;
- or a model-based cross-section reconstructed from visible depth.

If a horizontal cross-section is required, the correct fix is to choose a lower Y level that clears the chin, not to bend the path vertically.

### 5. Replace or justify the circumference model

Possible models to evaluate after reliable endpoint detection:

- calibrated regression using arc, chord, neck width, depth sag, and subject scale;
- circle-segment reconstruction from arc and chord;
- ellipse fitting in an X/Z cross-sectional plane;
- population-informed posterior-arc estimation;
- direct learned regression trained on tape measurements.

As an exploratory diagnostic only, treating the arc/chord pair as a circular segment gives approximately:

- `IMG_6552`: 31.01 cm;
- `IMG_6572`: 36.62 cm;
- `IMG_6640`: 39.13 cm.

This is **not implemented** and is not validated. It illustrates that the implied angular coverage changes by subject and that a fixed multiplier cannot represent all three geometries.

### 6. Add explicit confidence gates

Potential rejection/warning features:

- arc/chord ratio outside a validated range;
- detected width implausible relative to jaw width or interpupillary distance;
- endpoint depth far behind the central neck surface;
- endpoint too close to a Pose shoulder;
- asymmetric left/right depth retreat;
- endpoint on hair, clothing, invalid depth, or a low-confidence matte region;
- rapid width growth over adjacent rows;
- no stable bilateral contour.

## Open questions for the consultant

1. What was the tape-measured circumference of the subject in `IMG_6640`?
2. At what exact anatomical height was each tape measurement taken?
3. Should the visible curve be planar, or is a vertical sag intentionally part of the measurement definition?
4. Should a neck wider than the FaceMesh jaw envelope be allowed? If so, by how much?
5. Can the intended neck endpoints be manually annotated on at least 20–30 varied subjects?
6. Is a dedicated neck/body-part segmentation model acceptable, or must the solution use only existing HEIC masks and MediaPipe?
7. Is it acceptable for the application to return “measurement unavailable” for low-confidence cases?

## Addendum (same day): consultation response and confidence gate

This section responds to the open questions above and records what was
verified directly against the real HEIC files (not just re-derived from
this document's numbers).

### Root cause, sharpened

Reproducing the pipeline against `IMG_6552`, `IMG_6572`, and `IMG_6640`
confirmed problem A but located it more precisely. Plotting skin-mask width
against Y over a wide band (chin down through the shoulders) shows:

- `IMG_6552`: a genuine flat "pipe" -- width ~620-630 px for roughly 100
  vertical px right below the chin -- then a clean flare to 738+ px. This
  plateau is a real local-minimum basin the row-picker can lock onto.
- `IMG_6640`: **no plateau at all**. Width climbs monotonically from 717 px
  at the chin itself (`Y=2110`, before the search band even starts) through
  774, 810, 868... past the shoulders (1467 px) and beyond. There is no
  basin to find, because there isn't one.

Overlaying the skin matte on the actual photos explains why: `IMG_6552`
and `IMG_6572` both wear a collar/jacket that occludes the shoulders, so
the mask's edge is really "where clothing starts" -- which happens to sit
close to the anatomical neck base. `IMG_6640` has no garment in frame at
all; skin is visible continuously from jaw to chest, so there is no
boundary of any kind in the skin matte between "neck" and "not neck". This
matches the user's own framing during this consultation: shoulders being
in/out of frame is not the deciding factor -- a photo can show neck-and-chest
only and still give the detector nothing to find, because the failure is
"no narrowing exists in the visible data," not "shoulders were visible."

### Depth-based rescue: tested, did not pan out for this failure mode

Three independent depth-based signals were tried, hoping to recover a
boundary the skin matte cannot show:

1. **Lateral depth discontinuity per row** (walking left/right from centre
   looking for a step in distance-from-camera). Real and sharp in
   `IMG_6552` (~3 cm step almost exactly at the mask edge). In `IMG_6640`,
   the same kind of step exists but lands at essentially the same wide X
   as the mask edge -- no narrower, no better.
2. **Surface-curvature half-width** (how far left/right from the closest
   point before depth recedes by 1-2 cm -- a tight cylinder should give a
   small half-width, a flatter chest a large one). This was **noisy and
   gradually increasing in both the working and failing image** -- it did
   not reproduce the clean plateau-then-flare signature the mask shows for
   `IMG_6552`, so as tested it is not a usable drop-in replacement.
3. **Closest-point (min depth) trajectory vs Y**. In `IMG_6640` this
   recedes steadily from the chin and only flattens around `Y≈2280`
   (already past `scan_end_y`), by which point the visible skin is already
   full chest-width (1309 px) anyway -- so even this signal's "landmark"
   is well past where a useful neck-only estimate would need to sit.

None of these are proof depth-based detection can never work -- a proper
parabolic-curvature fit with heavier smoothing, run by someone iterating
specifically on this, might do better than these quick probes. But none of
the three rescued this photo, so depth curvature is not treated as a quick
fix here; it remains a legitimate but unproven direction for a future,
dedicated effort (see "Recommended direction," #2 and #5).

### What was implemented: a shoulder-independent confidence gate

Added `neck_to_jaw_width_ratio()` (`fidmaa_gui.calculations`): the detected
neck-mask width divided by the FaceMesh jaw width already computed for the
lateral guard (`lower_jaw_x_bounds()`). This needs no Pose/shoulder data at
all, directly addressing the concern that shoulders are sometimes absent
or unreliable. Computed from the real pipeline output (`mask_left_x` /
`mask_right_x` on the final `NeckMeasurement`, and the clamped jaw guard
bounds):

| Image | Neck width | Jaw width | Ratio |
|---|---:|---:|---:|
| `IMG_6552` (good) | 631 | 775 | 0.8142 |
| `IMG_6572` (good) | 670 | 748 | 0.8957 |
| `IMG_6640` (bad) | 738 | 763 | 0.9672 |

`NECK_LOW_CONFIDENCE_RATIO_THRESHOLD = 0.93` (`fidmaa_gui.const`) sits
between the good and bad clusters with margin on both sides. **This is a
provisional, 3-subject calibration**, not a validated anatomical constant --
revisit once more subjects with annotated ground truth are available (see
"Recommended direction," #1).

Per product decision: the app does **not** refuse to show a result. When
the ratio is at or above threshold, the automatic circumference is still
computed and displayed, but flagged inline (`⚠ LOW CONFIDENCE` next to the
headline number) plus an explanatory line and the raw ratio in the
diagnostics block. This does not fix detection for photos like `IMG_6640`
-- no signal tested here supports a correct measurement for that photo --
it only stops the app from presenting a shoulder-to-shoulder number as if
it were trustworthy.

Tests: `tests/test_app_calculations.py::TestNeckToJawWidthRatio`, including
regression anchors pinned to the exact ratios above.

## Current code and test status

Current working-tree behavior includes:

- seven arc samples;
- multiplier 3.0;
- restored sag toward Pose neck midpoint;
- FaceMesh-bounded vertical search;
- app-level FaceMesh lower-jaw lateral guard;
- raw `3D chord × π` diagnostic without additive correction;
- console `Ctrl+C` shutdown handling;
- shoulder-independent low-confidence flag (neck/jaw width ratio, see
  addendum above) shown inline next to the automatic circumference.

Automated checks at the time of this report:

- `124 passed` (118 + 6 new `TestNeckToJawWidthRatio` cases);
- Ruff checks passed;
- diff whitespace checks passed.

The automated tests cover helper calculations and UI behavior, but there is not yet an image-level regression suite with manually annotated neck endpoints. Such a suite is required before this detector can be considered reliable.

## Addendum 2 (same day): geometric detector — the approach that worked

Everything above (skin-matte width, the confidence gate, and the
chin-clearing half-sine sag) was superseded. The working detector is in
`src/fidmaa_gui/neck_arc.py` and shares no machinery with them.

### Why the earlier attempts could not work

The skin matte answers "is this skin?", never "is this neck?" — on a
bare-necked subject the mask runs unbroken from jaw to chest, so no width
minimum exists to find (Problem A). A pretrained BiSeNet face-parsing model
with an explicit `neck` class (CelebAMask-HQ, class 14) was tried and
**rejected**: run against all three subjects it reproduced exactly the same
failure, painting all bare skin down to the garment line as "neck" on
`IMG_6640`. Its apparent success on `IMG_6552`/`IMG_6572` came from the
collar doing the work, not from anatomy. Segmentation of any kind is the
wrong tool here.

### What the detector actually does

1. **Height.** Descend the facial midline **starting on the chin** (midway
   between the lower lip and landmark 152). Starting at the jaw line — as an
   earlier version did — puts the scan *past* the chin, so the fall being
   searched for has already happened and is invisible; the steepest fall in
   fact lands only +7…+14 px from the jaw line. The measuring height is
   where the descent's slope has decayed to 55% of its own peak.
2. **Shape.** The measuring curve is the FaceMesh lower-jaw polyline pushed
   straight down to that height: parallel to the lower face oval, just below
   it. The sag comes from the subject's own jaw, so it cannot run away —
   unlike the per-column search it replaces, which produced V shapes down to
   the sternal notch whenever one column found a spurious "clear" point.
3. **Edges.** On that curve the neck is a hill in depth. The background tilt
   is fitted and removed, the slope is taken at a **5 mm** scale (a wide
   window smears a ~30 px silhouette into nothing — this was a real bug),
   and each edge is the peak of the first sharp fall outward from the apex.
4. **Circumference.** `pi * width * 1.157`.

### The circumference model, and what was rejected

`pi * width` undershoots the tape by a near-constant 13–14% on every
subject. Alternatives were tested against all three tapes:

| model | 6640 | 6552 | 6572 |
|---|---:|---:|---:|
| `pi * width` | −13.0% | −13.3% | −14.4% |
| ellipse (Ramanujan, depth read off the map) | +6.1% | +6.3% | +17.9% |
| ellipse fitted to the visible arc | +4.1% | +9.6% | +16.8% |
| `2 * visible 3D arc` | +39% | +37% | +53% |
| **width × 1.157, then pi** | **+0.8%** | **−2.1%** | **−2.1%** |

The ellipse model is the theoretically attractive one and it **failed**: the
front-to-back depth it needs is contaminated by the silhouette cliff, giving
b/a of 1.42/1.43/1.71 — anatomically impossible, and inconsistent between
subjects. It should not be retried without a better way to measure that
depth.

The correction is applied to the **width**, not folded into the
circumference factor, because that is where the error originates: at grazing
incidence TrueDepth stops returning samples, so the last trustworthy depth
pixel sits inside the true silhouette and every neck reads too narrow by a
similar fraction regardless of shape. That is also why one constant survives
three quite different builds (individually required: 3.612 / 3.622 / 3.671
against pi, a 1.6% spread).

**This remains one parameter fitted to three subjects.** The ±2% residuals
are in-sample and will be optimistic on new people. More tape measurements
are the single most valuable thing that could be added.

### Current results

| Image | measuring line | edges | circumference | tape | error |
|---|---:|---:|---:|---:|---:|
| `IMG_6640` | +104 px | 632–1260 | 32.27 cm | 32.0 | +0.8% |
| `IMG_6552` | +88 px | 690–1324 | 29.85 cm | 30.5 | −2.1% |
| `IMG_6572` | +70 px | 749–1366 | 32.62 cm | 33.3 | −2.1% |
| `IMG_6659` | +94 px | 762–1362 | 33.11 cm | — | — |
| `IMG_6596` | +242 px | 490–1371 | 39.67 cm | — | — |

`IMG_6640`, the case that defeated every previous approach, is now the most
accurate of the three measured ones.

`IMG_6596` (a man, thick neck) was judged visually correct: the arc sits on
the neck and its endpoints meet the collar on both sides. Three of its
numbers sit outside the range seen on the other subjects and are worth
watching if it ever misbehaves: the measuring line is +242 px below the jaw
(others: 70–104), the apex is 138 px off the facial midline (others: 26–75),
and the width is 109 mm (others: 82–91). The first and third are consistent
with a long, heavy neck; the second is unexplained.

**Every tape measurement so far is from a woman.** `IMG_6596` is the first
male subject and has no tape, so `NECK_SILHOUETTE_EROSION_CORRECTION` has
never been checked against a male neck's depth-to-width proportions — the
single most likely way for that constant to break.

`IMG_6598` was excluded from the set: the photo itself is unusable.

The arc is drawn on the photo in magenta (`_paint_geometric_neck_arc`), so
the measurement can be judged by eye instead of trusted blindly.
