"""Neck tool: the first/last click snaps to the library's neck edge on its row.

Synthetic portrait: the cylinder neck of test_neck_arc_tool_ui with a skin
matte covering exactly the neck columns; the real library functions
(skin_edges_at_row, neck_width_from_edges) do the edge search and checks.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PySide6 import QtGui
from PySide6.QtCore import QPoint, QPointF, Qt

from fidmaa_gui.measurement_tools import MeasurementTool
from fidmaa_gui.neck_snap import NECK_SNAP_WINDOW_MM, SNAP_NOT_FOUND_TEXT

from .test_float_depth import PHOTO_SIZE, _float_window
from .test_neck_arc_tool_ui import ROW_Y, _cylinder_depth

MIDDLE = [(215.0, ROW_Y), (238.0, ROW_Y), (262.0, ROW_Y)]


def _snap_window(depth_m=None, collar=False):
    cylinder, left, right = _cylinder_depth()
    depth_m = cylinder if depth_m is None else depth_m
    if collar:
        # A collar 3 cm in front of the neck side, just outside the left edge.
        depth_m = depth_m.copy()
        depth_m[:, int(left) - 14 : int(left) - 1] = 0.42
    skin = np.zeros((640, 480), dtype=np.uint8)
    skin[:, int(left) : int(right)] = 255
    window = _float_window(
        depth_m=depth_m,
        skinmap=Image.fromarray(skin, "L"),
        photo=SimpleNamespace(size=PHOTO_SIZE),
    )
    window.portrait.depth.subject_depth_m = 0.45
    window.set_tool(MeasurementTool.NECK)
    return window, left, right


def _measure(window, first, last):
    for point in (first, *MIDDLE, last):
        window._on_image_clicked(QPointF(*point))
    window._finish_polyline()
    return window.polyline_editor.points


@pytest.mark.parametrize("offset", [-4.0, 0.0, 5.0, 9.0])
def test_ends_within_the_window_snap_to_the_neck_edges(qapp, offset):
    window, left, right = _snap_window()
    points = _measure(window, (left + offset, ROW_Y), (right - offset, ROW_Y))
    assert points[0][0] == pytest.approx(left, abs=0.6)
    assert points[-1][0] == pytest.approx(right, abs=0.6)
    assert points[0][1] == ROW_Y and points[-1][1] == ROW_Y  # y as clicked
    assert points[1:-1] == MIDDLE  # middle points are never snapped
    details = window.measurementPanel.detailsLabel.text()
    assert "Ends snapped to neck edges (first and last)" in details
    assert SNAP_NOT_FOUND_TEXT not in details
    window.close()


def test_snapped_ends_give_the_same_width_and_circumference(qapp):
    window, left, right = _snap_window()
    _measure(window, (left, ROW_Y), (right, ROW_Y))
    reference = window.measurementPanel.history.entries[-1]
    window.polyline_editor.clear()
    _measure(window, (left + 7, ROW_Y), (right - 5, ROW_Y))
    entry = window.measurementPanel.history.entries[-1]
    assert entry.linear_mm == pytest.approx(reference.linear_mm, rel=1e-3)
    assert entry.surface_mm == pytest.approx(reference.surface_mm, rel=1e-3)
    window.close()


def test_click_outside_the_snap_window_stays_with_a_warning(qapp):
    window, left, right = _snap_window()
    # ~25 mm inside the neck: farther than the 15 mm window from the edge.
    assert NECK_SNAP_WINDOW_MM == 15.0
    points = _measure(window, (left + 25, ROW_Y), (right - 2, ROW_Y))
    assert points[0] == (left + 25, ROW_Y)
    assert points[-1][0] == pytest.approx(right, abs=0.6)
    details = window.measurementPanel.detailsLabel.text()
    assert f"⚠ First end: {SNAP_NOT_FOUND_TEXT}" in details
    assert "Ends snapped to neck edges (last)" in details
    window.close()


def test_collar_rejected_edge_is_not_snapped(qapp):
    window, left, right = _snap_window(collar=True)
    points = _measure(window, (left + 4, ROW_Y), (right - 4, ROW_Y))
    assert points[0] == (left + 4, ROW_Y)
    assert points[-1][0] == pytest.approx(right, abs=0.6)
    assert f"⚠ First end: {SNAP_NOT_FOUND_TEXT}" in window.measurementPanel.detailsLabel.text()
    assert window._neck_end_snaps["first"][2].reason.startswith("edge rejected")
    window.close()


def test_toggle_off_restores_and_keeps_the_raw_clicks(qapp):
    window, left, right = _snap_window()
    assert window.snapNeckEndsAction.isChecked()  # default on
    first, last = (left + 5, ROW_Y), (right - 5, ROW_Y)
    _measure(window, first, last)
    assert window.polyline_editor.points[0] != first

    window.snapNeckEndsAction.setChecked(False)
    points = window.polyline_editor.points
    assert points[0] == first and points[-1] == last
    assert "snapped" not in window.measurementPanel.detailsLabel.text()

    window.polyline_editor.clear()
    points = _measure(window, first, last)
    assert points[0] == first and points[-1] == last
    assert SNAP_NOT_FOUND_TEXT not in window.measurementPanel.detailsLabel.text()
    window.close()


def test_dragged_end_snaps_again_on_release(qapp):
    window, left, right = _snap_window()
    _measure(window, (left + 3, ROW_Y), (right - 3, ROW_Y))
    snapped_left = window.polyline_editor.points[0]
    window._begin_point_drag(QPointF(*snapped_left))
    window._drag_point(QPointF(left + 8, ROW_Y + 2))
    # While dragging the end follows the pointer ...
    assert window.polyline_editor.points[0] == (left + 8, ROW_Y + 2)
    window._finish_point_drag(QPointF(left + 8, ROW_Y + 2))
    # ... and snaps horizontally on release, keeping the new row.
    point = window.polyline_editor.points[0]
    assert point[0] == pytest.approx(left, abs=0.6) and point[1] == ROW_Y + 2
    assert len(window.measurementPanel.history.entries) == 1
    window.close()


def test_snapped_ends_are_drawn_hollow(qapp):
    window, left, right = _snap_window()
    _measure(window, (left + 5, ROW_Y), (right - 25, ROW_Y))  # last: not snapped
    canvas = QtGui.QImage(480, 640, QtGui.QImage.Format_RGB32)
    canvas.fill(Qt.black)
    painter = QtGui.QPainter(canvas)
    window._paint_polyline_overlay(painter, lambda p: QPoint(round(p[0]), round(p[1])))
    painter.end()

    def yellow(x, y):
        color = canvas.pixelColor(round(x), round(y))
        return color.red() > 150 and color.green() > 150 and color.blue() < 100

    first, last = window.polyline_editor.points[0], window.polyline_editor.points[-1]
    # Snapped first end: a ring (radius 6) with a dark centre off the line.
    assert yellow(first[0], first[1] - 6)
    assert not yellow(first[0], first[1] - 2)
    # Manual last end: the usual filled dot.
    assert yellow(last[0], last[1] - 2)
    window.close()
