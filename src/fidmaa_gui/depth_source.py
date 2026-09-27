"""Where a photo's depth comes from, and whether and how it can be measured.

Photos from the TrueDepth capture app carry absolute depth plus the camera's
intrinsics, exposed by portrait_analyser as ``IOSPortrait.camera`` (a
``CameraModel``). Legacy Camera-app photos have ``camera is None`` and keep the
calibration polynomial, so every metric conversion stays exactly as before.

Capture-app depth maps also differ in two ways the GUI must respect:

* raw code 0 means "no depth" (``IOSPortrait.depth_code_zero_is_invalid``),
  not the farthest distance, so it must never be turned into Z_far;
* ``IOSPortrait.depth_plausible is False`` flags depth that cannot be trusted
  (e.g. inverted: face at ~3 m, background at ~0.6 m, or not alignable with
  the photo). Nothing is measured on such photos.

All helpers take duck-typed objects (``getattr`` with defaults) so they work
with the lightweight stand-ins used by the unit tests.
"""

import dataclasses
import logging

from portrait_analyser import check_depth_plausibility

logger = logging.getLogger(__name__)

# AVDepthData "relative" accuracy (iPhone 17 Pro Camera app) is known to
# underestimate distance, so every millimetre value comes out short.
RELATIVE_DEPTH_WARNING = (
    "Relative depth (Camera app): mm values may be 7–28 % short. "
    "Use the TrueDepth capture app for absolute depth."
)

IMPLAUSIBLE_DEPTH_NA = "n/a (implausible depth)"


def portrait_camera(window):
    """``CameraModel`` of the loaded photo, or None (legacy / nothing loaded).

    The intrinsics refer to the full-resolution photo (``camera.width`` x
    ``camera.height``). The GUI always converts display (480x640) points to
    ``window.image.size`` first; should the two sizes ever differ, the model
    is rescaled to ``window.image.size`` so the maths stays consistent.
    """
    portrait = getattr(window, "portrait", None)
    if portrait is None:
        return None
    camera = getattr(portrait, "camera", None)
    image = getattr(window, "image", None)
    if camera is None or image is None:
        return camera
    return camera_for_size(camera, image.size)


def camera_for_size(camera, size):
    """Return ``camera`` expressed in pixels of an image of ``size``."""
    cam_w = getattr(camera, "width", None)
    cam_h = getattr(camera, "height", None)
    if cam_w is None or cam_h is None:
        return camera
    width, height = size
    if (cam_w, cam_h) == (width, height):
        return camera
    logger.warning(
        "Camera intrinsics refer to a %sx%s image but the photo is %sx%s; rescaling",
        cam_w,
        cam_h,
        width,
        height,
    )
    scale_x = width / cam_w
    scale_y = height / cam_h
    return dataclasses.replace(
        camera,
        fx=camera.fx * scale_x,
        fy=camera.fy * scale_y,
        cx=camera.cx * scale_x,
        cy=camera.cy * scale_y,
        width=width,
        height=height,
    )


def depth_is_implausible(portrait):
    """True only when the library positively flagged the depth as implausible."""
    return getattr(portrait, "depth_plausible", None) is False


def depth_zero_is_invalid(portrait):
    """True when raw depth code 0 means "no depth" (capture-app files)."""
    return getattr(portrait, "depth_code_zero_is_invalid", False) is True


def implausible_depth_warning(portrait):
    """Warning text for an implausible depth map, with the depths that tripped it."""
    detail = ""
    depth_m = getattr(portrait, "depth_m", None)
    skinmap = getattr(portrait, "skinmap", None)
    if depth_m is not None and skinmap is not None:
        plausible, face_cm, background_cm = check_depth_plausibility(
            depth_m, skinmap, getattr(portrait, "hairmap", None)
        )
        if plausible is False and face_cm is not None:
            background = "n/a" if background_cm is None else f"~{background_cm:.0f} cm"
            detail = f" (face ~{face_cm:.0f} cm, background {background})"
        else:
            # Flagged by the loader for another reason: the depth map could
            # not be aligned with the photo (unknown orientation / aspect).
            detail = " (depth map could not be aligned with the photo)"
    return f"Depth looks implausible{detail} — measurements disabled."


def depth_source_status(portrait):
    """Return ``(line, warning)`` describing the depth source of ``portrait``.

    ``warning`` is None unless the depth is relative or implausible.
    """
    accuracy = getattr(portrait, "depth_accuracy", None) if portrait is not None else None
    if depth_is_implausible(portrait):
        return (
            f"Depth: implausible ({accuracy or 'unknown accuracy'}, not measured)",
            implausible_depth_warning(portrait),
        )
    if accuracy == "absolute":
        if getattr(portrait, "camera", None) is not None:
            return "Depth: TrueDepth absolute", None
        return "Depth: absolute", None
    if accuracy == "relative":
        return "Depth: relative (Camera app)", RELATIVE_DEPTH_WARNING
    if accuracy is None:
        return "Depth: unknown", None
    return f"Depth: {accuracy}", None


def combine_warnings(*warnings):
    """Join the non-empty warnings into one string (None if there are none).

    MeasurementPanel prefixes the whole text with one warning sign, so every
    following warning gets its own on a new line.
    """
    present = [w for w in warnings if w]
    if not present:
        return None
    return "\n⚠ ".join(present)
