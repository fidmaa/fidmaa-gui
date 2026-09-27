"""Width-based neck measurement in the GUI (capture-app photos).

The library side (``portrait.neck_width``, ``neck_width_from_edges``) is
stubbed: these tests check the panel text, the overlay, the Neck tool's
coordinate scaling / history, and that legacy photos are left alone.
"""

import logging
from types import SimpleNamespace

import portrait_analyser
import pytest
from portrait_analyser.pose import PortraitPose
from PySide6 import QtGui
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMenu

from fidmaa_gui import measurement_tools
from fidmaa_gui.app import (
    IMPLAUSIBLE_DEPTH_DETAIL,
    NECK_TOOL_LEGACY_DETAIL,
    MainWindow,
)
from fidmaa_gui.depth_source import IMPLAUSIBLE_DEPTH_NA
from fidmaa_gui.measurement_panel import (
    DEFAULT_VALUE_LABELS,
    format_cm,
    format_neck_circumference,
)
from fidmaa_gui.measurement_tools import MeasurementTool

from .test_float_depth import PHOTO_SIZE, _float_window
from .test_tool_interaction_ui import _window_with_flat_depth

EDGES_OCCLUDED = (
    "neck edges not visible -- take the photo from further away / include the neck below the chin"
)


def _ok_result(**overrides):
    values = dict(
        status="ok",
        row_y=2240.0,
        left_x=1008.0,
        right_x=2016.0,
        width_mm=134.5,
        circumference_circle_mm=422.5,
        circumference_ellipse_mm=(391.4, 402.3),
        left_depth_cm=38.2,
        right_depth_cm=38.6,
        warnings=[],
        message=None,
        rows_used=[2230.0, 2235.0, 2240.0],
        band=(2100.0, 2600.0),
        band_source="vision-neck",
        rows=[],
        quality="good",
        quality_reasons=[],
        support_mm=9.0,
        height_below_chin_mm=12.0,
        roll_deg=1.5,
        neck_roll_deg=-2.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _failed_result(status, message):
    return SimpleNamespace(
        status=status,
        message=message,
        warnings=[],
        band=None,
        band_source=None,
        rows=[],
    )


def _neutral_float_window(neck_width):
    window = _float_window(neck_width=neck_width)
    window.ui.showNeckArcCheckBox.setChecked(False)
    window.ui.showCentroidsCheckBox.setChecked(False)
    window.neck_midpoint = SimpleNamespace(pose=PortraitPose.NEUTRAL_NECK, x=None)
    # Old detectors' results exist but must not be reported for float photos.
    window.neck_arc = SimpleNamespace(circumference_mm=364.0)
    window.neck_measurement_auto = SimpleNamespace(circumference_mm=380.0)
    window.neck_confidence_ratio = 5.0  # would add the skin-matte warning
    window.mouth_measurement = None
    window.segmentation_debug = None
    window._compute_neck_width()
    return window


# ---------------------------------------------------------------- panel


def test_panel_shows_width_and_circumference_for_capture_app_photo(qapp):
    window = _neutral_float_window(_ok_result())
    window._update_automatic_results()

    text = window.measurementPanel.autoLabel.text()
    expected = (
        f"Neck width: {format_cm(134.5)} · circumference "
        f"{format_neck_circumference((391.4, 402.3), 422.5)}"
    )
    assert expected in text.splitlines()
    assert "(ellipse)" in expected and "(circle)" in expected
    assert "≤" not in text and "π·W" not in text
    assert f"measured {format_cm(12.0, 1)} below the chin · quality: good" in text
    assert "Neck circumference" not in text
    assert "skin-matte" not in text
    assert "Depth: TrueDepth absolute" in text
    # No old skin-matte low-confidence warning either.
    assert window.measurementPanel.autoWarningLabel.isHidden()
    window.close()


def test_format_neck_circumference_uses_one_decimal():
    text = format_neck_circumference((391.4, 402.3), 422.5)
    assert text.startswith("≈ 39")
    assert "40" in text and "42" in text
    assert text.count("cm") == 2


def test_low_quality_is_a_visible_warning_without_duplicates(qapp):
    collar = "collar close to the neck edge (outside nearer by up to 0.3 cm)"
    support = "clean rows span only 3.7 mm (< 5 mm)"
    window = _neutral_float_window(
        _ok_result(quality="low", quality_reasons=[collar, support], warnings=[collar])
    )
    window._update_automatic_results()

    label = window.measurementPanel.autoWarningLabel
    assert not label.isHidden()
    assert f"Neck width quality: low — {collar}; {support}" in label.text()
    assert label.text().count("collar close") == 1
    assert "quality: low" in window.measurementPanel.autoLabel.text()
    window.close()


@pytest.mark.parametrize(
    "status, message, focal, expected",
    [
        ("relative-depth", "depth accuracy is 'relative'", None, "needs absolute TrueDepth"),
        ("no-camera", "no intrinsics -- pass camera=", None, "No camera calibration data"),
        ("no-camera", "the file's intrinsics are not usable", 2770.0, "not usable here"),
        ("error", "neck width measurement failed: RuntimeError: x", None, "Neck width measurement"),
        ("no-depth", "the portrait has no depth map", None, "The portrait has no depth map"),
    ],
)
def test_panel_messages_for_new_statuses(qapp, status, message, focal, expected):
    window = _neutral_float_window(_failed_result(status, message))
    window.portrait.focal_length_px = focal
    window._update_automatic_results()

    text = window.measurementPanel.autoLabel.text()
    assert "Neck width: n/a — " in text
    assert expected in text
    assert "camera=" not in text
    assert "Neck tool" not in text
    window.close()


def test_panel_shows_neck_width_warnings(qapp):
    warning = "left edge: outside nearer than neck edge (collar?) by 0.8 cm"
    window = _neutral_float_window(_ok_result(warnings=[warning]))
    window._update_automatic_results()

    warn_text = window.measurementPanel.autoWarningLabel.text()
    assert not window.measurementPanel.autoWarningLabel.isHidden()
    assert "collar?" in warn_text
    window.close()


def test_panel_shows_library_message_and_tool_hint_when_edges_hidden(qapp):
    window = _neutral_float_window(_failed_result("edges-occluded", EDGES_OCCLUDED))
    window._update_automatic_results()

    text = window.measurementPanel.autoLabel.text()
    assert "Neck width: n/a — Neck edges not visible — take the photo" in text
    assert "Neck tool (N)" in text
    assert "Neck circumference" not in text
    window.close()


def test_panel_no_camera_is_explained_without_the_tool_hint(qapp):
    window = _neutral_float_window(
        _failed_result("no-camera", "the file carries no camera intrinsics -- pass camera=")
    )
    window._update_automatic_results()

    text = window.measurementPanel.autoLabel.text()
    assert "Neck width: n/a — No camera calibration data in this file" in text
    assert "camera=" not in text
    assert "Neck tool" not in text
    window.close()


def test_panel_reports_unexpected_library_error(qapp, caplog):
    class Portrait(SimpleNamespace):
        @property
        def neck_width(self):
            raise RuntimeError("boom")

    window = _neutral_float_window(None)
    window.portrait = Portrait(**vars(window.portrait))
    with caplog.at_level(logging.ERROR):
        window._compute_neck_width()
    window._update_automatic_results()

    assert "Neck width: error (RuntimeError: boom)" in window.measurementPanel.autoLabel.text()
    assert any("neck_width failed" in record.message for record in caplog.records)
    window.close()


def test_diagnostics_include_neck_width_for_capture_app_photo(qapp):
    window = _neutral_float_window(_ok_result())
    window.neck_measurement_auto = None
    text = window._diagnostics_text()
    assert "[Neck width] status: ok" in text
    assert "Width (3D): 13.45 cm" in text
    window.close()


class _LegacyPortrait(SimpleNamespace):
    @property
    def neck_width(self):
        raise AssertionError("legacy photos must not compute the width-based neck")


def _legacy_neutral_window():
    window = _window_with_flat_depth()
    window.ui.showNeckArcCheckBox.setChecked(False)
    window.ui.showCentroidsCheckBox.setChecked(False)
    window.portrait = _LegacyPortrait(
        teeth_bbox=None, teethmap=None, incisor_measurement=None, skinmap=None, camera=None
    )
    window.neck_midpoint = SimpleNamespace(pose=PortraitPose.NEUTRAL_NECK, x=None)
    window.neck_arc = SimpleNamespace(circumference_mm=364.0)
    window.neck_measurement_auto = SimpleNamespace(circumference_mm=380.0)
    window.neck_confidence_ratio = 0.5
    window.mouth_measurement = None
    window.segmentation_debug = None
    return window


def test_legacy_panel_keeps_the_old_neck_lines(qapp):
    window = _legacy_neutral_window()
    window._compute_neck_width()
    window._update_automatic_results()

    lines = window.measurementPanel.autoLabel.text().splitlines()
    assert lines[1] == f"Neck circumference: {format_cm(364.0)}"
    assert lines[2] == f"Neck (skin-matte arc): {format_cm(380.0)}"
    assert "Neck width" not in window.measurementPanel.autoLabel.text()
    assert window.neck_width_auto is None
    assert not window._neck_width_overlay_visible()
    window.close()


# -------------------------------------------------------------- overlay

GREEN = MainWindow.NECK_WIDTH_COLOR


def _is_neck_green(color):
    return color.green() > 180 and color.red() < 110 and color.blue() < 170


def _display(x, y):
    return x * 480 / PHOTO_SIZE[0], y * 640 / PHOTO_SIZE[1]


def test_overlay_paints_the_measuring_row(qapp):
    result = _ok_result()
    window = _neutral_float_window(result)
    canvas = QtGui.QImage(480, 640, QtGui.QImage.Format_RGB32)
    canvas.fill(Qt.black)
    painter = QtGui.QPainter(canvas)

    def mapper(point):
        x, y = window._photo_to_main_display(point)
        return QPoint(round(x), round(y))

    window._paint_neck_width_overlay(painter, mapper)
    painter.end()

    mid_x, row_y = _display(0.5 * (result.left_x + result.right_x), result.row_y)
    assert _is_neck_green(canvas.pixelColor(round(mid_x), round(row_y)))
    left_x, _ = _display(result.left_x, result.row_y)
    # Edge marker: a short vertical tick.
    assert _is_neck_green(canvas.pixelColor(round(left_x), round(row_y) - 5))
    # Nothing far from the row.
    assert canvas.pixelColor(round(mid_x), round(row_y) - 30) == QtGui.QColor(Qt.black)
    window.close()


def _main_canvas(window):
    window.redrawImage(force=True)
    return window.ui.imageLabel.pixmap().toImage()


def test_main_photo_overlay_follows_the_view_menu_toggle(qapp):
    result = _ok_result()
    window = _neutral_float_window(result)
    mid_x, row_y = _display(0.5 * (result.left_x + result.right_x), result.row_y)
    point = (round(mid_x), round(row_y))

    assert window.neckWidthOverlayAction.isChecked()
    assert window._neck_width_overlay_visible()
    assert _is_neck_green(_main_canvas(window).pixelColor(*point))

    window.neckWidthOverlayAction.setChecked(False)
    assert not window._neck_width_overlay_visible()
    assert not _is_neck_green(_main_canvas(window).pixelColor(*point))
    window.close()


def test_zoom_overlay_paints_the_measuring_row(qapp):
    result = _ok_result()
    window = _neutral_float_window(result)
    canvas = QtGui.QImage(480, 640, QtGui.QImage.Format_RGB32)
    canvas.fill(Qt.black)
    painter = QtGui.QPainter(canvas)
    # Identity zoom: a 480x640 source shown uncropped at 480x640.
    window._paint_zoom_region_overlay(painter, (480, 640), (0, 0, 480, 640), (0, 0, 480, 640))
    painter.end()

    mid_x, row_y = _display(0.5 * (result.left_x + result.right_x), result.row_y)
    assert _is_neck_green(canvas.pixelColor(round(mid_x), round(row_y)))
    window.close()


def test_no_overlay_when_the_width_was_not_measured(qapp):
    window = _neutral_float_window(_failed_result("edges-occluded", EDGES_OCCLUDED))
    assert not window._neck_width_overlay_visible()
    window.close()


# ------------------------------------------------------------ Neck tool


def _record_calls(monkeypatch, result):
    calls = []

    def fake(portrait, left, right, **kwargs):
        calls.append((portrait, left, right))
        return result

    monkeypatch.setattr(portrait_analyser, "neck_width_from_edges", fake, raising=False)
    return calls


def test_neck_tool_two_clicks_measure_in_photo_pixels_and_add_history(qapp, monkeypatch):
    result = _ok_result(warnings=["right edge: outside nearer than neck edge (collar?) by 0.6 cm"])
    calls = _record_calls(monkeypatch, result)
    window = _float_window()
    window.set_tool(MeasurementTool.NECK)
    assert window.polyline_editor.max_points == 2
    assert window.lineOptionsWidget.isHidden()

    window._on_image_clicked(QPointF(160, 356))
    assert calls == []  # one click: nothing to measure yet
    window._on_image_clicked(QPointF(320, 360))

    assert len(calls) == 1
    portrait, left, right = calls[0]
    assert portrait is window.portrait
    assert left == pytest.approx((160 * 3024 / 480, 356 * 4032 / 640))
    assert right == pytest.approx((320 * 3024 / 480, 360 * 4032 / 640))

    panel = window.measurementPanel
    assert panel.resultGroup.title() == "Neck"
    assert panel.value_label_texts() == ("Width (3D):", "Circumference:")
    assert panel.linearValueLabel.text() == format_cm(134.5)
    assert panel.surfaceValueLabel.text() == format_neck_circumference((391.4, 402.3), 422.5)
    assert "collar?" in panel.detailsLabel.text()
    assert "just below the jaw" in panel.hintLabel.text()

    (entry,) = panel.history.entries
    assert entry.tool == MeasurementTool.NECK
    assert entry.points == 2
    assert entry.linear_mm == 134.5
    # Surface column: midpoint of the ellipse circumference range.
    assert entry.surface_mm == pytest.approx(0.5 * (391.4 + 402.3))
    assert panel.historyTable.item(0, 1).text() == "Neck"
    assert panel.historyTable.item(0, 4).text() != ""
    assert "Neck: circumference" in panel.historyTable.horizontalHeaderItem(4).toolTip()
    window.close()


def test_neck_history_surface_switch_off_leaves_the_column_empty(qapp, monkeypatch):
    monkeypatch.setattr(measurement_tools, "NECK_HISTORY_SURFACE_IS_CIRCUMFERENCE", False)
    _record_calls(monkeypatch, _ok_result())
    window = _float_window()
    window.set_tool(MeasurementTool.NECK)
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))
    (entry,) = window.measurementPanel.history.entries
    assert entry.surface_mm is None
    window.close()


def test_neck_tool_shows_quality(qapp, monkeypatch):
    reason = "left/right edge depths differ by 3.5 cm (head rotated?)"
    _record_calls(
        monkeypatch,
        _ok_result(
            quality="low", quality_reasons=[reason], warnings=[reason], height_below_chin_mm=None
        ),
    )
    window = _float_window()
    window.set_tool(MeasurementTool.NECK)
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))
    details = window.measurementPanel.detailsLabel.text()
    assert f"⚠ Neck width quality: low — {reason}" in details
    assert details.count("head rotated?") == 1
    assert "Quality: low" in details
    assert "below the chin" not in details
    window.close()


def test_neck_tool_relative_depth_is_na(qapp, monkeypatch):
    _record_calls(monkeypatch, _failed_result("relative-depth", "depth accuracy is 'relative'"))
    window = _float_window()
    window.set_tool(MeasurementTool.NECK)
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))
    panel = window.measurementPanel
    assert panel.linearValueLabel.text() == "n/a"
    assert "Neck width needs absolute TrueDepth depth" in panel.detailsLabel.text()
    assert panel.history.entries == ()
    window.close()


def test_dragging_a_neck_edge_replaces_the_history_row(qapp, monkeypatch):
    calls = _record_calls(monkeypatch, _ok_result())
    window = _float_window()
    window.set_tool(MeasurementTool.NECK)
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))
    _record_calls(monkeypatch, _ok_result(width_mm=130.0))

    window._begin_point_drag(QPointF(320, 360))
    window._finish_point_drag(QPointF(330, 360))

    assert len(calls) == 1
    (entry,) = window.measurementPanel.history.entries
    assert entry.linear_mm == 130.0
    window.close()


def test_other_tools_get_their_captions_back(qapp, monkeypatch):
    _record_calls(monkeypatch, _ok_result())
    window = _float_window()
    window.set_tool(MeasurementTool.NECK)
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))
    window.set_tool(MeasurementTool.VECTOR)
    assert window.measurementPanel.value_label_texts() == DEFAULT_VALUE_LABELS
    assert not window.lineOptionsWidget.isHidden()
    window.close()


def test_neck_tool_no_depth_result_is_na_and_recorded(qapp, monkeypatch):
    _record_calls(
        monkeypatch,
        _failed_result("no-depth", "no depth inside one of the clicked edges"),
    )
    window = _float_window()
    window.set_tool(MeasurementTool.NECK)
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))

    panel = window.measurementPanel
    assert panel.linearValueLabel.text() == "n/a (invalid depth)"
    assert "No depth inside one of the clicked edges" in panel.detailsLabel.text()
    (entry,) = panel.history.entries
    assert entry.linear_mm is None
    window.close()


def test_neck_tool_without_camera_is_na(qapp, monkeypatch):
    _record_calls(monkeypatch, _failed_result("no-camera", "the file carries no camera intrinsics"))
    window = _float_window()
    window.set_tool(MeasurementTool.NECK)
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))

    panel = window.measurementPanel
    assert panel.linearValueLabel.text() == "n/a"
    assert "No camera calibration data" in panel.detailsLabel.text()
    assert panel.history.entries == ()
    window.close()


def test_neck_tool_on_legacy_photo_is_na(qapp, monkeypatch):
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
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))

    panel = window.measurementPanel
    assert panel.linearValueLabel.text() == "n/a (needs capture-app photo)"
    assert NECK_TOOL_LEGACY_DETAIL in panel.detailsLabel.text()
    assert panel.history.entries == ()
    window.close()


def test_neck_tool_on_implausible_depth_is_na(qapp, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("implausible depth must not be measured")

    monkeypatch.setattr(portrait_analyser, "neck_width_from_edges", fail, raising=False)
    window = _float_window(depth_plausible=False)
    window.set_tool(MeasurementTool.NECK)
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))

    panel = window.measurementPanel
    assert panel.linearValueLabel.text() == IMPLAUSIBLE_DEPTH_NA
    assert IMPLAUSIBLE_DEPTH_DETAIL in panel.detailsLabel.text()
    assert panel.history.entries == ()
    window.close()


def test_neck_tool_without_photo_does_not_crash(qapp):
    window = MainWindow()
    window.set_tool(MeasurementTool.NECK)
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))
    assert "Open a photo" in window.measurementPanel.hintLabel.text()
    window.close()


def test_n_shortcut_selects_the_neck_tool(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()

    QTest.keyClick(window, Qt.Key_N)
    assert window.current_tool == MeasurementTool.NECK
    assert window.toolActions[MeasurementTool.NECK].isChecked()
    assert window.toolActions[MeasurementTool.NECK].shortcut() == QtGui.QKeySequence(Qt.Key_N)
    QTest.keyClick(window, Qt.Key_V)
    assert window.current_tool == MeasurementTool.VECTOR
    window.close()


def test_neck_tool_is_in_toolbar_and_tools_menu(qapp):
    window = MainWindow()
    action = window.toolActions[MeasurementTool.NECK]
    assert action in window.toolBar.actions()
    menus = window.menuBar().findChildren(QMenu)
    tools_menu = next(m for m in menus if m.title() == "&Tools")
    assert action in tools_menu.actions()
    view_menu = next(m for m in menus if m.title() == "&View")
    assert window.neckWidthOverlayAction in view_menu.actions()
    window.close()
