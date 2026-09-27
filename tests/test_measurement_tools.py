import pytest

from fidmaa_gui.measurement_tools import (
    HistoryEntry,
    MeasurementHistory,
    MeasurementTool,
    PolylineEditor,
    format_history_tsv,
    measure_polyline,
)


def test_vector_editor_finishes_on_second_point_and_restarts_on_third():
    editor = PolylineEditor(max_points=2)

    assert editor.add_point((10.0, 10.0)) is True
    assert editor.add_point((20.0, 10.0)) is False
    assert editor.finished
    assert editor.add_point((30.0, 30.0)) is True
    assert editor.points == [(30.0, 30.0)]
    assert not editor.finished


def test_polyline_editor_needs_two_points_to_finish():
    editor = PolylineEditor(max_points=None)
    assert editor.finish() is False
    editor.add_point((1.0, 1.0))
    assert editor.finish() is False
    assert not editor.finished
    editor.add_point((5.0, 1.0))
    editor.add_point((9.0, 1.0))
    assert not editor.finished
    assert editor.finish() is True
    assert editor.finished


def test_undo_and_clear():
    editor = PolylineEditor(max_points=None)
    editor.add_point((1.0, 1.0))
    editor.add_point((2.0, 2.0))
    editor.undo()
    assert editor.points == [(1.0, 1.0)]
    editor.clear()
    assert editor.points == []
    assert not editor.finished


def test_undo_does_nothing_on_finished_measurement():
    editor = PolylineEditor(max_points=2)
    editor.add_point((1.0, 1.0))
    editor.add_point((2.0, 2.0))
    editor.undo()
    assert editor.points == [(1.0, 1.0), (2.0, 2.0)]


def test_point_near_picks_closest_within_tolerance():
    editor = PolylineEditor(max_points=None)
    editor.add_point((10.0, 10.0))
    editor.add_point((14.0, 10.0))
    assert editor.point_near((13.0, 10.0), tolerance=6) == 1
    assert editor.point_near((100.0, 100.0), tolerance=6) is None
    editor.move_point(1, (50.0, 50.0))
    assert editor.points[1] == (50.0, 50.0)


def test_measure_polyline_sums_segments_and_skips_zero_length():
    points = [(0.0, 0.0), (3.0, 4.0), (3.0, 4.0), (3.0, 8.0)]
    result = measure_polyline(
        points,
        linear_mm=lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5,
        surface_mm=lambda a, b: 2 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5,
    )
    assert len(result.segments) == 2
    assert result.linear_mm == pytest.approx(9.0)
    assert result.surface_mm == pytest.approx(18.0)


def test_measure_polyline_surface_is_none_when_any_segment_is_invalid():
    result = measure_polyline(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        linear_mm=lambda a, b: 1.0,
        surface_mm=lambda a, b: None if a[0] == 1.0 else 1.0,
    )
    assert result.linear_mm == pytest.approx(2.0)
    assert result.surface_mm is None
    assert result.segments[1].surface_mm is None


def test_measure_polyline_linear_is_none_when_any_segment_is_invalid():
    """F1: a segment crossing invalid/out-of-range depth invalidates the
    polyline's linear total, independently of the surface total."""
    result = measure_polyline(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        linear_mm=lambda a, b: None if a[0] == 1.0 else 1.0,
        surface_mm=lambda a, b: 1.0,
    )
    assert result.linear_mm is None
    assert result.segments[1].linear_mm is None
    assert result.surface_mm == pytest.approx(2.0)


def test_history_add_replace_and_tsv():
    history = MeasurementHistory()
    first = history.add(HistoryEntry("a.heic", MeasurementTool.VECTOR, 2, 123.4, 130.0))
    second = history.add(HistoryEntry("a.heic", MeasurementTool.POLYLINE, 5, 200.0, None))
    history.replace(first, HistoryEntry("a.heic", MeasurementTool.VECTOR, 2, 100.0, 110.0))

    assert (first, second) == (0, 1)
    tsv = format_history_tsv(history.entries, number_format=lambda v: f"{v:.2f}")
    assert tsv.splitlines() == [
        "File\tTool\tPoints\tLinear [cm]\tSurface [cm]",
        "a.heic\tVector\t2\t10.00\t11.00",
        "a.heic\tPolyline\t5\t20.00\t",
    ]
    history.clear()
    assert history.entries == ()
