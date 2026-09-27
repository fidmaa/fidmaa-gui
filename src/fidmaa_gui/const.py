MINIMUM_FACE_WIDTH_PERCENT = 0.55
MINIMUM_FACE_HEIGHT_PERCENT = 0.4
TRUEDEPTH_EXIF_ID = "front TrueDepth"
LAST_DIRECTORY_USED = "last_directory_used"

# Display dimensions for the main image view
MAIN_IMAGE_WIDTH = 480
MAIN_IMAGE_HEIGHT = 640

# Depth chart dimensions
DEPTH_CHART_WIDTH = 255
DEPTH_CHART_HEIGHT = 640

# Reconstruction chart height
RECONSTRUCTION_HEIGHT = 256

# Maximum iteration count for findPoint() to prevent infinite loops
FIND_POINT_MAX_ITERATIONS = 1000

# Outer lower-face contour used to keep a neck skin run from spilling onto a
# bare shoulder. A small margin tolerates FaceMesh jitter and natural width
# differences between the jaw and neck.
FACE_MESH_LOWER_JAW_INDICES = (
    172,
    136,
    150,
    149,
    176,
    148,
    152,
    377,
    400,
    378,
    379,
    365,
    397,
)
NECK_JAW_ENVELOPE_MARGIN_FRACTION = 0.02

# Below this (detected neck width / jaw width) ratio, the detector found a
# plausible anatomical narrowing below the chin. At or above it, the "neck"
# is about as wide as the jaw itself -- no narrowing was found, so the
# reported endpoints likely include shoulder/chest skin rather than just the
# neck (see NECK_DETECTOR_STATUS.md). Provisional: calibrated on three known
# subjects (IMG_6552 ~0.81 and IMG_6572 ~0.90, both plausible; IMG_6640
# ~0.97, confirmed shoulder-to-shoulder) -- revisit once more subjects with
# annotated ground truth are available.
NECK_LOW_CONFIDENCE_RATIO_THRESHOLD = 0.93

# FaceMesh landmark indices used by the geometric neck arc.
FACE_MESH_CHIN_INDEX = 152
FACE_MESH_NOSE_TIP_INDEX = 1
FACE_MESH_LOWER_LIP_INDEX = 17

# --- Geometric neck arc (neck_arc.py) -------------------------------------
# The measuring curve runs PARALLEL TO THE FACEMESH LOWER-JAW LINE, pushed
# down to the height where the depth surface stops receding below the chin.
# See NECK_DETECTOR_STATUS.md for how each of these was arrived at.

# How far the vertical descent may run below its start, in px.
NECK_ARC_DESCENT_SEARCH_PX = 520
# The descent is "stabilised" (i.e. off the chin and onto the neck) once its
# downward slope has decayed to this share of its own peak. Relative, not
# absolute, so it adapts to how far a given chin protrudes.
NECK_ARC_KNEE_SLOPE_FRACTION = 0.55
# Smoothing half-window for the slow shape (the descent).
NECK_ARC_WIDE_SMOOTH_MM = 12.0
# Smoothing half-window for locating the sharp silhouette flanks. Must stay
# small: a silhouette edge is only ~30 px wide, and a wide window smears it
# into invisibility (this was a real bug -- the edges could not be found).
NECK_ARC_EDGE_SMOOTH_MM = 5.0
# The cross-section carries a background tilt because the measuring curve
# follows the jaw. It is removed by fitting a line over this half-span
# around the apex before hunting for edges.
NECK_ARC_DETREND_HALF_MM = 45.0
# Hard bound on how far from the apex an edge may be found.
NECK_ARC_MAX_HALF_WIDTH_MM = 120.0
# Anything this much farther than the chin is background, not the subject.
NECK_ARC_BACKGROUND_MARGIN_MM = 140.0

# Measured width -> circumference.
#
# pi * width underestimates the tape by a remarkably constant 13-14% on every
# subject measured so far, which is why a plain circle model never worked.
# The most likely cause is silhouette erosion in the depth sensor: at grazing
# incidence TrueDepth stops returning samples, so the last trustworthy depth
# pixel sits INSIDE the true silhouette, making every neck read too narrow by
# a similar fraction regardless of its shape.
#
# Modelling the neck as an ellipse using a front-to-back depth read off the
# depth map was tried and REJECTED: that depth is contaminated by the
# silhouette cliff (it implied b/a of 1.4-1.7, anatomically impossible) and
# gave errors of +6%, +6%, +18% -- far worse, and inconsistent.
#
# So the correction is applied to the WIDTH, where the error actually
# originates, rather than hidden inside an arbitrary circumference factor.
# Calibrated on three subjects with tape measurements (30.5, 33.3, 32.0 cm),
# whose individually required factors were 3.612 / 3.622 / 3.671 -- a 1.6%
# spread. PROVISIONAL: this is one parameter fitted to three points; the
# residuals below (+0.6%, +0.4%, -1.0%) are in-sample and will be optimistic.
NECK_SILHOUETTE_EROSION_CORRECTION = 1.157

# How many points are sampled along the arc for display and 3D length.
NECK_ARC_SAMPLE_COUNT = 9
