"""Snap the Neck tool's first/last click to the library's neck edge on that row.

Qt-free. Coordinates are full-resolution photo pixels.

The edge search is the library's own per-row rule,
``portrait_analyser.neck_width.skin_edges_at_row`` (skin matte averaged over
rows, gaps bridged, 50 % crossings of the joined span around the neck
centre), with the same mm-based parameters ``measure_neck_width`` passes it.
The edge is then validated with the library's per-edge depth checks by
evaluating the candidate with the public ``neck_width_from_edges`` and
reading its ``NeckWidthRow.left_ok`` / ``right_ok``: depth 3 mm inside
valid and on the neck, and the depth 10 mm outside not nearer by
``OCCLUDER_MARGIN_CM`` (a collar in front of the neck side). Not applied:
the automatic detector's row-level shape checks (edge slope, off-centre,
neck-to-jaw ratio), which need the face landmarks and a band of rows.

The only local re-implementation is the matte -> float array conversion
(the library's private ``_skin_array``: grey, resized to the photo size).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from portrait_analyser.neck_width import (
    DEPTH_WINDOW_RADIUS,
    SKIN_GAP_BRIDGE_MM,
    SKIN_MIN_SPAN_MM,
    SKIN_SEARCH_HALF_WIDTH_MM,
    STATUS_NO_DEPTH,
    STATUS_OK,
    neck_width_from_edges,
    skin_edges_at_row,
)

# An end snaps only to an edge within this distance of the click (mm at the
# click's depth): about the width of a fingertip on the screen, far less
# than the ~65 mm half-width of a neck, so the opposite edge is never taken.
NECK_SNAP_WINDOW_MM = 15.0

SNAP_OK = "snapped"
SNAP_MANUAL = "manual"

SNAP_NOT_FOUND_TEXT = "edge not found near the click — placed manually"


@dataclass(frozen=True)
class SnapResult:
    status: str  # SNAP_OK or SNAP_MANUAL
    point: tuple[float, float]  # the snapped (or unchanged) end, photo px
    reason: str | None = None  # why it was not snapped (diagnostics)


def skin_array(skinmap, photo_size) -> np.ndarray:
    """The skin matte as a float array at photo size (library's _skin_array)."""
    skin = skinmap.convert("L")
    if skin.size != tuple(photo_size):
        skin = skin.resize(tuple(photo_size))
    return np.asarray(skin, dtype=np.float32)


def _px_per_mm(camera, z_cm: float) -> float:
    return camera.fx / (z_cm * 10.0)


def snap_neck_end(portrait, skin, end, other, *, window_mm=NECK_SNAP_WINDOW_MM) -> SnapResult:
    """Snap ``end`` horizontally to the neck edge on its row.

    ``other`` is the opposite end of the measurement (or the latest click):
    the neck centre for the skin search is taken half way between them, and
    the edge on ``end``'s side of it is used. ``y`` stays as clicked.
    """
    camera = getattr(portrait, "camera", None)
    depth = getattr(portrait, "depth", None)
    if skin is None or camera is None or depth is None:
        return SnapResult(SNAP_MANUAL, end, "no skin matte / camera / depth")
    x, y = end
    mid_x = 0.5 * (x + other[0])
    if abs(other[0] - x) < 1.0:
        return SnapResult(SNAP_MANUAL, end, "ends coincide")
    # mm -> px at the neck's depth (the library scales with the chin depth):
    # a click just outside the neck reads the background, which would
    # shrink every mm-based parameter.
    z_cm = depth.distance_cm(mid_x, y, DEPTH_WINDOW_RADIUS)
    if z_cm is None:
        z_cm = depth.distance_cm(x, y, DEPTH_WINDOW_RADIUS)
    if z_cm is None:
        return SnapResult(SNAP_MANUAL, end, "no depth at the click")
    px_mm = _px_per_mm(camera, z_cm)
    edges = skin_edges_at_row(
        skin,
        y,
        mid_x,
        gap_px=SKIN_GAP_BRIDGE_MM * px_mm,
        search_px=SKIN_SEARCH_HALF_WIDTH_MM * px_mm,
        min_span_px=SKIN_MIN_SPAN_MM * px_mm,
    )
    if edges is None:
        return SnapResult(SNAP_MANUAL, end, "no skin edges on this row")
    is_left = x < mid_x
    edge_x = edges[0] if is_left else edges[1]
    if abs(edge_x - x) > window_mm * px_mm:
        return SnapResult(SNAP_MANUAL, end, "nearest skin edge outside the snap window")
    candidate = (float(edge_x), float(y))
    row_result = neck_width_from_edges(portrait, candidate, other)
    rows = getattr(row_result, "rows", None) or []
    status = getattr(row_result, "status", None)
    if status not in (STATUS_OK, STATUS_NO_DEPTH) or not rows:
        return SnapResult(SNAP_MANUAL, end, f"edge check unavailable ({status})")
    row = rows[0]
    edge_ok = row.left_ok if candidate[0] <= other[0] else row.right_ok
    if not edge_ok:
        return SnapResult(SNAP_MANUAL, end, "edge rejected (collar / no depth on the neck)")
    return SnapResult(SNAP_OK, candidate)
