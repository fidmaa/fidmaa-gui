import math

from PySide6.QtCore import QPoint

from . import const


def lower_jaw_x_bounds(face_mesh_landmarks):
    """Return the FaceMesh lower-jaw X envelope with a small safety margin."""
    if not face_mesh_landmarks or len(face_mesh_landmarks) <= max(
        const.FACE_MESH_LOWER_JAW_INDICES
    ):
        return None

    jaw_xs = [face_mesh_landmarks[index][0] for index in const.FACE_MESH_LOWER_JAW_INDICES]
    jaw_left = min(jaw_xs)
    jaw_right = max(jaw_xs)
    margin = (jaw_right - jaw_left) * const.NECK_JAW_ENVELOPE_MARGIN_FRACTION
    return round(jaw_left - margin), round(jaw_right + margin)


def neck_to_jaw_width_ratio(mask_left_x, mask_right_x, jaw_bounds):
    """Return (detected neck width) / (jaw width), or None if jaw bounds are
    unavailable or degenerate.

    A neck is anatomically narrower than the jaw. A ratio near or above 1.0
    means the detected "neck" row is about as wide as the jaw itself -- no
    anatomical narrowing was found below the chin, so the endpoints most
    likely spill onto the shoulder or chest rather than tracing the neck.
    See NECK_DETECTOR_STATUS.md for the investigation this is based on.
    """
    if jaw_bounds is None:
        return None
    jaw_left, jaw_right = jaw_bounds
    jaw_width = jaw_right - jaw_left
    if jaw_width <= 0:
        return None
    return (mask_right_x - mask_left_x) / jaw_width


def findPoint(
    startX,
    startY,
    direction=1,
    maxWidth=None,
    maxHeight=None,
    angle=None,
    linear_coefficient=None,
):
    """
    Starting on (startX, startY), find a point going to the right (direction=1) or left (direction=-1)
    which lies on the bounds of the image (minimum point (0, 0), maximum point (maxWidth, maxHeight)),
    so that the angle of the line to the x axis is equal to angle (in deg).

    You can pass either angle or linear_coefficient.
    """

    if maxWidth is None:
        maxWidth = const.MAIN_IMAGE_WIDTH
    if maxHeight is None:
        maxHeight = const.MAIN_IMAGE_HEIGHT

    if direction not in [1, -1]:
        raise ValueError

    if linear_coefficient is None and angle is None:
        raise ValueError("Pass either angle or linear_coefficient")

    if linear_coefficient is None:
        linear_coefficient = math.tan(math.radians(angle))

    # Direct calculation instead of pixel-by-pixel iteration.
    # The line equation is: y = linear_coefficient * (x - startX) + startY
    # We need to find where the line exits the image bounds in the given direction.

    # Calculate x where y hits the top (y=0) and bottom (y=maxHeight) boundaries
    candidates = []

    if linear_coefficient != 0:
        # x where y = 0
        x_at_top = -startY / linear_coefficient + startX
        if 0 <= x_at_top <= maxWidth:
            candidates.append((x_at_top, 0))

        # x where y = maxHeight
        x_at_bottom = (maxHeight - startY) / linear_coefficient + startX
        if 0 <= x_at_bottom <= maxWidth:
            candidates.append((x_at_bottom, maxHeight))

    # y where x hits left (x=0) and right (x=maxWidth) boundaries
    y_at_left = linear_coefficient * (0 - startX) + startY
    if 0 <= y_at_left <= maxHeight:
        candidates.append((0, y_at_left))

    y_at_right = linear_coefficient * (maxWidth - startX) + startY
    if 0 <= y_at_right <= maxHeight:
        candidates.append((maxWidth, y_at_right))

    # Filter candidates by direction: direction=1 means x >= startX, direction=-1 means x <= startX
    valid = []
    for cx, cy in candidates:
        if direction == 1 and cx >= startX:
            valid.append((cx, cy))
        elif direction == -1 and cx <= startX:
            valid.append((cx, cy))

    if not valid:
        # Fallback: if no valid intersection found (e.g. vertical line), return start point
        return QPoint(startX, startY)

    # Pick the candidate farthest from the start point
    best = max(valid, key=lambda p: abs(p[0] - startX) + abs(p[1] - startY))
    return QPoint(int(best[0]), int(best[1]))


def findParalellPoint(midpointX, midpointY, angle, distance, direction=1):
    # Get the midpoint of a parallel line to the left -- this point
    # can also be used to calculate a perpendicular line

    lpx1 = midpointX + direction * math.sin(math.radians(-angle)) * distance
    lpy1 = midpointY + direction * math.cos(math.radians(-angle)) * distance

    return (lpx1, lpy1)
