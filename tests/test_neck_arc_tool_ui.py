"""Neck tool polyline: arc-circle circumference from the chord and the surface arc.

``neck_width_from_edges`` is stubbed (see test_neck_width_ui); the chord and
the surface segments are monkeypatched on the window so the maths gets the
user's IMG_2363 lengths (chord 12.07 cm, arc 15.45 cm -> C ~ 40.9 cm).
"""

from types import SimpleNamespace

import portrait_analyser
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from fidmaa_gui.app import (
    NECK_TOOL_LEGACY_DETAIL,
    NECK_TOOL_NEEDS_3_POINTS,
    NECK_TOOL_VALUE_LABELS,
)
from fidmaa_gui.arc_circle import arc_circle
from fidmaa_gui.measurement_panel import format_cm, format_neck_circumference
from fidmaa_gui.measurement_tools import MeasurementTool

from .test_float_depth import _float_window
from .test_neck_width_ui import _ok_result, _record_calls
from .test_tool_interaction_ui import _window_with_flat_depth

CHORD_MM = 120.7
ARC_MM = 154.5
POINTS = [(160, 356), (190, 368), (240, 374), (290, 368), (320, 360)]


def _stub_lengths(window, monkeypatch, chord_mm=CHORD_MM, arc_mm=ARC_MM):
    """Chord between the first and last point; the arc split over the segments."""
    first, last = POINTS[0], POINTS[-1]
    chord_calls = []
    surface_calls = []

    def linear(x1, y1, x2, y2):
        if {(x1, y1), (x2, y2)} == {first, last}:
            chord_calls.append((x1, y1, x2, y2))
            return chord_mm
        return 10.0  # per-segment straight length, not used by the circle

    def surface(x1, y1, x2, y2, step):
        surface_calls.append(step)
        return None if arc_mm is None else arc_mm / (len(POINTS) - 1)

    # The end inset is tested on a synthetic cylinder below; here the clicked
    # points go to the chord and arc unchanged.
    monkeypatch.setattr(window, "_neck_arc_points", lambda points: list(points))
    monkeypatch.setattr(window, "linear_length_between_two_clicks_or_none", linear)
    monkeypatch.setattr(window, "surface_vector_filtered", surface)
    return chord_calls, surface_calls


def _click_all(window, points=POINTS):
    for x, y in points:
        window._on_image_clicked(QPointF(x, y))


def _neck_window(monkeypatch, result=None, **stub):
    calls = _record_calls(monkeypatch, result or _ok_result())
    window = _float_window()
    window.set_tool(MeasurementTool.NECK)
    lengths = _stub_lengths(window, monkeypatch, **stub)
    return window, calls, lengths


def test_five_point_polyline_gives_the_arc_circle_circumference(qapp, monkeypatch):
    window, calls, (chord_calls, surface_calls) = _neck_window(monkeypatch)
    _click_all(window)
    assert window.measurementPanel.history.entries == ()  # still in progress
    QTest.keyClick(window, Qt.Key_Return)

    expected = arc_circle(CHORD_MM, ARC_MM)
    assert expected.circumference_mm == pytest.approx(409, abs=1)
    panel = window.measurementPanel
    assert panel.value_label_texts() == NECK_TOOL_VALUE_LABELS
    assert NECK_TOOL_VALUE_LABELS == ("Circumference (arc circle):", "Width (3D):")
    assert panel.linearValueLabel.text() == format_cm(expected.circumference_mm, 1)
    assert panel.surfaceValueLabel.text() == format_cm(134.5)

    details = panel.detailsLabel.text()
    assert "5 points (finished)" in details
    assert f"Chord (3D): {format_cm(CHORD_MM)} · arc: {format_cm(ARC_MM)}" in details
    assert "Arc/chord: 1" in details and "280" in details
    assert "θ: 136" in details
    assert f"R: {format_cm(expected.radius_mm)}" in details
    assert f"Width model: {format_neck_circumference((391.4, 402.3), 422.5)}" in details
    assert "Quality: good" in details
    assert "half circle" not in details

    # Width from the first and last point only, in photo pixels.
    (_, left, right) = calls[-1]
    assert left == pytest.approx((160 * 3024 / 480, 356 * 4032 / 640))
    assert right == pytest.approx((320 * 3024 / 480, 360 * 4032 / 640))
    # Arc: the Polyline tool's surface path, with the surface-step option.
    assert set(surface_calls) == {window.surfaceStepSpin.value()}
    assert chord_calls

    (entry,) = panel.history.entries
    assert entry.tool == MeasurementTool.NECK
    assert entry.points == 5
    assert entry.linear_mm == 134.5
    assert entry.surface_mm == pytest.approx(expected.circumference_mm)
    assert "arc-circle" in panel.historyTable.horizontalHeaderItem(4).toolTip()
    window.close()


def test_right_click_finishes_the_neck_polyline(qapp, monkeypatch):
    window, _, _ = _neck_window(monkeypatch)
    _click_all(window)
    window._on_image_right_clicked(QPointF(320, 360))
    assert window.polyline_editor.finished
    (entry,) = window.measurementPanel.history.entries
    assert entry.points == 5
    window.close()


def test_in_progress_polyline_shows_the_live_result_and_backspace_undoes(qapp, monkeypatch):
    window, _, _ = _neck_window(monkeypatch)
    _click_all(window, POINTS[:3])
    assert "3 points (in progress)" in window.measurementPanel.detailsLabel.text()
    QTest.keyClick(window, Qt.Key_Backspace)
    assert len(window.polyline_editor.points) == 2
    # Two points: same captions, the arc circle waits for a third point.
    assert window.measurementPanel.value_label_texts() == NECK_TOOL_VALUE_LABELS
    assert window.measurementPanel.linearValueLabel.text() == NECK_TOOL_NEEDS_3_POINTS
    assert window.measurementPanel.history.entries == ()
    window.close()


def test_two_point_finish_is_width_only(qapp, monkeypatch):
    window, _, (chord_calls, surface_calls) = _neck_window(monkeypatch)
    _click_all(window, [POINTS[0], POINTS[-1]])
    QTest.keyClick(window, Qt.Key_Enter)

    panel = window.measurementPanel
    assert panel.value_label_texts() == NECK_TOOL_VALUE_LABELS
    assert panel.linearValueLabel.text() == NECK_TOOL_NEEDS_3_POINTS
    assert panel.surfaceValueLabel.text() == format_cm(134.5)
    details = panel.detailsLabel.text()
    assert f"Width model: {format_neck_circumference((391.4, 402.3), 422.5)}" in details
    assert chord_calls == [] and surface_calls == []
    (entry,) = panel.history.entries
    assert entry.points == 2
    assert entry.surface_mm == pytest.approx(0.5 * (391.4 + 402.3))
    window.close()


def test_dragging_a_middle_point_does_not_rerun_the_width(qapp, monkeypatch):
    window, calls, _ = _neck_window(monkeypatch)
    _click_all(window)
    window._finish_polyline()
    before = len(calls)
    window._begin_point_drag(QPointF(*POINTS[2]))
    window._finish_point_drag(QPointF(POINTS[2][0], POINTS[2][1] + 6))
    assert len(calls) == before
    assert len(window.measurementPanel.history.entries) == 1
    window.close()


def test_flat_front_is_na_and_history_falls_back_to_the_ellipse(qapp, monkeypatch):
    window, _, _ = _neck_window(monkeypatch, arc_mm=CHORD_MM * 1.0002)
    _click_all(window)
    window._finish_polyline()
    panel = window.measurementPanel
    assert panel.linearValueLabel.text() == "n/a (front too flat to fit a circle)"
    assert panel.surfaceValueLabel.text() == format_cm(134.5)
    (entry,) = panel.history.entries
    assert entry.surface_mm == pytest.approx(0.5 * (391.4 + 402.3))
    window.close()


def test_invalid_depth_on_the_arc_is_na(qapp, monkeypatch):
    window, _, _ = _neck_window(monkeypatch, arc_mm=None)
    _click_all(window)
    window._finish_polyline()
    panel = window.measurementPanel
    assert panel.linearValueLabel.text() == "n/a (invalid depth)"
    assert "arc: n/a (invalid depth)" in panel.detailsLabel.text()
    (entry,) = panel.history.entries
    assert entry.linear_mm == 134.5
    assert entry.surface_mm == pytest.approx(0.5 * (391.4 + 402.3))
    window.close()


def test_invalid_chord_depth_is_na(qapp, monkeypatch):
    window, _, _ = _neck_window(monkeypatch, chord_mm=None)
    _click_all(window)
    window._finish_polyline()
    assert window.measurementPanel.linearValueLabel.text() == "n/a (invalid depth)"
    window.close()


def test_more_than_a_half_circle_warns(qapp, monkeypatch):
    window, _, _ = _neck_window(monkeypatch, arc_mm=CHORD_MM * 1.8)
    _click_all(window)
    window._finish_polyline()
    details = window.measurementPanel.detailsLabel.text()
    assert "⚠ Arc spans more than a half circle — check the edge points" in details
    fit = arc_circle(CHORD_MM, CHORD_MM * 1.8)
    assert window.measurementPanel.linearValueLabel.text() == format_cm(fit.circumference_mm, 1)
    window.close()


def test_history_switch_off_leaves_the_surface_empty(qapp, monkeypatch):
    from fidmaa_gui import measurement_tools

    monkeypatch.setattr(measurement_tools, "NECK_HISTORY_SURFACE_IS_CIRCUMFERENCE", False)
    window, _, _ = _neck_window(monkeypatch)
    _click_all(window)
    window._finish_polyline()
    (entry,) = window.measurementPanel.history.entries
    assert entry.surface_mm is None
    window.close()


def test_no_camera_is_na_for_the_whole_polyline(qapp, monkeypatch):
    from .test_neck_width_ui import _failed_result

    window, _, _ = _neck_window(
        monkeypatch, result=_failed_result("no-camera", "no camera intrinsics")
    )
    _click_all(window)
    window._finish_polyline()
    panel = window.measurementPanel
    assert panel.linearValueLabel.text() == "n/a"
    assert panel.surfaceValueLabel.text() == "n/a"
    assert panel.history.entries == ()
    window.close()


def test_legacy_photo_polyline_is_na(qapp, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("legacy photos must not call neck_width_from_edges")

    monkeypatch.setattr(portrait_analyser, "neck_width_from_edges", fail, raising=False)
    window = _window_with_flat_depth()
    window.portrait = SimpleNamespace(
        camera=None,
        depth=None,
        teeth_bbox=None,
        teethmap=None,
        incisor_measurement=None,
        skinmap=None,
        hairmap=None,
    )
    window.set_tool(MeasurementTool.NECK)
    _click_all(window)
    window._finish_polyline()
    panel = window.measurementPanel
    assert panel.linearValueLabel.text() == "n/a (needs capture-app photo)"
    assert NECK_TOOL_LEGACY_DETAIL in panel.detailsLabel.text()
    assert panel.history.entries == ()
    window.close()


def test_real_flat_float_depth_goes_through_the_library_paths(qapp, monkeypatch):
    # No length stubs: the flat synthetic face plane gives arc == chord.
    _record_calls(monkeypatch, _ok_result())
    window = _float_window()
    window.set_tool(MeasurementTool.NECK)
    _click_all(window, [(40, 500), (60, 520), (80, 540)])
    window._finish_polyline()
    panel = window.measurementPanel
    assert panel.linearValueLabel.text() == "n/a (front too flat to fit a circle)"
    assert "Chord (3D): n/a" not in panel.detailsLabel.text()
    window.close()


# ------------------------------------------------ end inset on a cylinder

CYLINDER_R_MM = 65.0
CYLINDER_FRONT_MM = 450.0
BACKGROUND_MM = 1000.0
ROW_Y = 320.0


def _cylinder_depth():
    """Vertical circular cylinder (R 65 mm, front at 45 cm) in front of a 1 m
    background, one depth pixel per display pixel. The silhouette mixes with
    the background like real TrueDepth maps: the last neck column is half
    background, the first background column a quarter neck."""
    import numpy as np

    from .test_float_depth import CAMERA, PHOTO_SIZE

    centre_z = CYLINDER_FRONT_MM + CYLINDER_R_MM
    u = (np.arange(480) + 0.5) * PHOTO_SIZE[0] / 480
    a = (u - CAMERA.cx) / CAMERA.fx
    q = a * a + 1
    disc = centre_z**2 - q * (centre_z**2 - CYLINDER_R_MM**2)
    row = np.where(disc >= 0, (centre_z - np.sqrt(np.clip(disc, 0, None))) / q, BACKGROUND_MM)
    inside = np.flatnonzero(disc >= 0)
    left, right = int(inside[0]), int(inside[-1])
    mixed = row.copy()
    mixed[left] = 0.5 * (row[left] + BACKGROUND_MM)
    mixed[right] = 0.5 * (row[right] + BACKGROUND_MM)
    mixed[left - 1] = 0.25 * row[left] + 0.75 * BACKGROUND_MM
    mixed[right + 1] = 0.25 * row[right] + 0.75 * BACKGROUND_MM
    depth_m = np.tile(mixed / 1000.0, (640, 1))
    # Silhouette in display x: the outer boundaries of the edge columns.
    return depth_m, float(left), float(right + 1)


def _cylinder_fit(window, left, right, shift, *, inset=True):
    width = right - left
    a, b = left + shift * width, right - shift * width
    points = [(a + (b - a) * i / 4, ROW_Y) for i in range(5)]
    if inset:
        points = window._neck_arc_points(points)
    chord = window.linear_length_between_two_clicks_or_none(*points[0], *points[-1])
    return arc_circle(chord, window._measure_current_polyline(points).surface_mm)


@pytest.fixture
def cylinder_window(qapp):
    depth_m, left, right = _cylinder_depth()
    window = _float_window(depth_m=depth_m)
    window.portrait.depth.subject_depth_m = CYLINDER_FRONT_MM / 1000.0
    window.set_tool(MeasurementTool.NECK)
    yield window, left, right
    window.close()


def test_inset_ends_sit_3_mm_inside_the_clicked_edges(cylinder_window):
    from .test_float_depth import CAMERA

    window, left, right = cylinder_window
    points = [(left, ROW_Y), (238.0, ROW_Y), (right, ROW_Y)]
    moved = window._neck_arc_points(points)
    assert moved[1] == points[1]
    for end, inset_end in ((points[0], moved[0]), (points[-1], moved[-1])):
        z_cm = window.distance_for_click(*inset_end)
        photo_shift = abs(inset_end[0] - end[0]) * 3024 / 480
        # 3 mm at the depth found there (two-pass, like the library).
        assert photo_shift == pytest.approx(3.0 * CAMERA.fx / (z_cm * 10.0), rel=0.02)
    assert moved[0][0] > left and moved[-1][0] < right


def test_inset_keeps_the_mixed_silhouette_out_of_the_arc(cylinder_window):
    window, left, right = cylinder_window
    true_c = 2 * 3.141592653589793 * CYLINDER_R_MM
    raw = _cylinder_fit(window, left, right, 0.0, inset=False)
    fit = _cylinder_fit(window, left, right, 0.0)
    # Clicked exactly on the silhouette: the raw ends read the mixed pixels
    # (C more than 2x the truth); the inset ends do not, and stay below 180°.
    assert raw.circumference_mm > 2 * true_c
    assert fit.circumference_mm == pytest.approx(true_c, rel=0.15)
    assert fit.theta_deg < 180


def test_inset_arc_circle_is_stable_for_inward_edges(cylinder_window):
    window, left, right = cylinder_window
    c3 = _cylinder_fit(window, left, right, 0.03).circumference_mm
    c6 = _cylinder_fit(window, left, right, 0.06).circumference_mm
    assert c3 == pytest.approx(c6, rel=0.05)
    assert c6 == pytest.approx(2 * 3.141592653589793 * CYLINDER_R_MM, rel=0.03)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Goal not met yet: within ~3 mm of the silhouette the library's "
        "integration smoothing flattens the steep flank, the arc reads short "
        "and C reads ~9 % high (462 vs 425 mm for edges on vs 3 % inside the "
        "silhouette); see scratchpad report"
    ),
)
def test_goal_arc_circle_stable_within_3_percent_when_edges_move_3_percent(cylinder_window):
    window, left, right = cylinder_window
    c0 = _cylinder_fit(window, left, right, 0.0).circumference_mm
    c3 = _cylinder_fit(window, left, right, 0.03).circumference_mm
    assert c0 == pytest.approx(c3, rel=0.03)
