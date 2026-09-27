"""Where a photo's depth comes from, and which camera model measures it.

Photos from the TrueDepth capture app carry absolute depth plus the camera's
intrinsics, exposed by portrait_analyser as ``IOSPortrait.camera`` (a
``CameraModel``). Legacy Camera-app photos have ``camera is None`` and keep the
calibration polynomial, so every metric conversion stays exactly as before.
"""

# AVDepthData "relative" accuracy (iPhone 17 Pro Camera app) is known to
# underestimate distance, so every millimetre value comes out short.
RELATIVE_DEPTH_WARNING = (
    "Relative depth (Camera app): mm values may be 7–28 % short. "
    "Use the TrueDepth capture app for absolute depth."
)


def portrait_camera(window):
    """``CameraModel`` of the loaded photo, or None (legacy / nothing loaded).

    Takes any object with an optional ``portrait`` attribute so it also works
    on the lightweight stand-ins used by the unit tests.
    """
    portrait = getattr(window, "portrait", None)
    if portrait is None:
        return None
    return getattr(portrait, "camera", None)


def depth_source_status(portrait):
    """Return ``(line, warning)`` describing the depth source of ``portrait``.

    ``warning`` is None unless the depth is known to be relative.
    """
    accuracy = getattr(portrait, "depth_accuracy", None) if portrait is not None else None
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
