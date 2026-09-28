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
    NECK_TOOL_ARC_VALUE_LABELS,
    NECK_TOOL_LEGACY_DETAIL,
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
    assert panel.value_label_texts() == NECK_TOOL_ARC_VALUE_LABELS
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
    # Two points: back to the width-only captions.
    assert window.measurementPanel.value_label_texts() == NECK_TOOL_VALUE_LABELS
    assert window.measurementPanel.history.entries == ()
    window.close()


def test_two_point_finish_is_width_only(qapp, monkeypatch):
    window, _, (chord_calls, surface_calls) = _neck_window(monkeypatch)
    _click_all(window, [POINTS[0], POINTS[-1]])
    QTest.keyClick(window, Qt.Key_Enter)

    panel = window.measurementPanel
    assert panel.value_label_texts() == NECK_TOOL_VALUE_LABELS
    assert panel.linearValueLabel.text() == format_cm(134.5)
    assert panel.surfaceValueLabel.text() == format_neck_circumference((391.4, 402.3), 422.5)
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
