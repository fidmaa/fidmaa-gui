"""Tape-path neck circumference in the GUI (portrait-analyser ``neck_sag``).

The library result is stubbed (``NeckWidthResult.sag`` is a NeckSagResult in
portrait-analyser > 0.9.0): these tests check the automatic panel headline and
its fallback, the overlay (tape path, side points, front marker), the
diagnostics block and the Neck tool's comparison line.
"""

from types import SimpleNamespace

from PySide6 import QtGui
from PySide6.QtCore import QPoint, QPointF, Qt

from fidmaa_gui.app import MainWindow
from fidmaa_gui.measurement_panel import format_cm, format_neck_circumference
from fidmaa_gui.measurement_tools import MeasurementTool

from .test_float_depth import PHOTO_SIZE, _float_window
from .test_neck_width_ui import (
    _is_neck_green,
    _neutral_float_window,
    _ok_result,
    _record_calls,
)

LEFT = (1008.0, 2240.0)
RIGHT = (2016.0, 2240.0)
FRONT = (1512.0, 2440.0)
SAG_DROP = FRONT[1] - LEFT[1]


def _path_y(x):
    u = (x - LEFT[0]) / (RIGHT[0] - LEFT[0])
    return LEFT[1] + 4.0 * SAG_DROP * u * (1.0 - u)


def _path():
    xs = [LEFT[0] + (RIGHT[0] - LEFT[0]) * i / 60 for i in range(61)]
    return [(x, _path_y(x)) for x in xs]


def _sag(**overrides):
    values = dict(
        status="ok",
        quality="good",
        quality_reasons=[],
        warnings=[],
        message=None,
        left_xy=LEFT,
        right_xy=RIGHT,
        front_xy=FRONT,
        front_source="chin-end",
        beard_border_y=2404.0,
        chin_end_y=2440.0,
        path=_path(),
        plane_tilt_deg=42.3,
        plane_tilt_3pt_deg=67.1,
        plane_normal=(0.0, 0.74, 0.67),
        plane_origin_mm=(0.0, 50.0, 420.0),
        front_drop_deg=38.0,
        width_mm=134.5,
        sagitta_mm=36.2,
        front_arc_mm=197.4,
        off_plane_rms_mm=1.9,
        circle_radius_mm=65.9,
        circle_centre_mm=(0.0, -30.0),
        circle_rms_mm=0.8,
        circle_span_deg=141.0,
        ellipse_b_mm=45.0,
        circumferences_mm={
            "circle": 422.5,
            "sagitta_ellipse": 332.0,
            "fitted_circle": 414.0,
            "fitted_ellipse": 318.0,
            "arc_circle": 425.0,
        },
        circumference_prior_ellipse_mm=(391.4, 402.3),
        headline_mm=414.0,
        headline_model="fitted_circle",
        sensitivity={
            "circle": 0.049,
            "sagitta_ellipse": 0.071,
            "fitted_circle": 0.025,
            "fitted_ellipse": 0.070,
            "arc_circle": 0.052,
        },
        sensitivity_detail={
            "side-width+3%": {
                "circle": 0.03,
                "sagitta_ellipse": 0.071,
                "fitted_circle": 0.025,
                "fitted_ellipse": 0.070,
                "arc_circle": 0.052,
            },
            "ends-10%": {
                "circle": 0.0,
                "sagitta_ellipse": 0.0,
                "fitted_circle": -0.012,
                "fitted_ellipse": None,
                "arc_circle": 0.0,
            },
        },
        profile_mm=[],
        path_3d_mm=[],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _sag_result(sag=None, **overrides):
    sag = _sag() if sag is None else sag
    ok = sag.status == "ok"
    values = dict(
        sag=sag,
        circumference_quality=sag.quality if ok else None,
        circumference_quality_reasons=list(sag.quality_reasons) if ok else [],
        circumference_mm=sag.headline_mm if sag.status == "ok" else None,
        circumference_model=sag.headline_model if sag.status == "ok" else None,
        search_top_y=2215.0,
    )
    values.update(overrides)
    return _ok_result(**values)


def _failed_sag(message="no depth at the chin"):
    return SimpleNamespace(
        status="failed",
        message=message,
        warnings=[],
        quality=None,
        quality_reasons=[],
        front_xy=None,
        front_source=None,
        beard_border_y=None,
        chin_end_y=None,
        path=[],
        headline_mm=None,
        headline_model=None,
    )


# ---------------------------------------------------------------- panel


def test_panel_headline_is_the_tape_path_circumference(qapp):
    window = _neutral_float_window(_sag_result())
    window._update_automatic_results()

    lines = window.measurementPanel.autoLabel.text().splitlines()
    assert lines[1] == (
        f"Neck circumference: {format_cm(414.0, 1)} (fitted circle) · quality: good"
    )
    assert lines[2] == (
        f"  width {format_cm(134.5)} · sides {format_cm(12.0, 1)} below the chin"
        " · width quality: good"
    )
    assert lines[3] == (
        f"  compare: {format_cm(422.5, 1)} (π·W) · {format_cm(425.0, 1)} (arc circle) · "
        f"{format_neck_circumference((391.4, 402.3), None)}"
    )
    text = "\n".join(lines)
    assert "Neck width:" not in text
    assert "skin-matte" not in text
    assert "Depth: TrueDepth absolute" in text
    assert window.measurementPanel.autoWarningLabel.isHidden()
    tooltip = window.measurementPanel.autoLabel.toolTip()
    assert "fitted circle is shape-dependent: ~±5 % on round necks" in tooltip
    assert "+22 % on front-flat and −17 % on deep necks (synthetic)" in tooltip
    window.close()


def test_headline_quality_is_the_circumference_quality_not_the_width_quality(qapp):
    tape = "the tape samples span only 80 deg of the fitted circle (< 100)"
    sag = _sag(quality="low", quality_reasons=[tape])
    window = _neutral_float_window(
        _sag_result(
            sag,
            circumference_quality="low",
            circumference_quality_reasons=[tape],
            height_below_chin_mm=-4.0,
        )
    )
    window._update_automatic_results()

    lines = window.measurementPanel.autoLabel.text().splitlines()
    assert lines[1].endswith("(fitted circle) · quality: low")
    assert lines[2] == (
        f"  width {format_cm(134.5)} · sides {format_cm(4.0, 1)} above the chin"
        " · width quality: good"
    )
    warning = window.measurementPanel.autoWarningLabel.text()
    assert f"Neck circumference quality: low — {tape}" in warning
    assert warning.count("span only") == 1
    window.close()


def test_width_fallback_says_above_the_chin_for_a_negative_height(qapp):
    window = _neutral_float_window(_ok_result(height_below_chin_mm=-3.0))
    window._update_automatic_results()
    text = window.measurementPanel.autoLabel.text()
    assert f"measured {format_cm(3.0, 1)} above the chin" in text
    assert "-0" not in text and "−0" not in text
    assert window.measurementPanel.autoLabel.toolTip() == ""
    window.close()


def test_panel_lists_the_tape_path_quality_reasons_once(qapp):
    width_reason = "clean rows span only 3.7 mm (< 5 mm)"
    inferred = "front point inferred (no beard/chin boundary found)"
    other = "left edge: outside nearer than neck edge (collar?) by 0.8 cm"
    # Tape-only reasons on the sag (portrait-analyser 315e40b); the library
    # merges them into circumference_quality_reasons.
    sag = _sag(quality="low", quality_reasons=[inferred], warnings=[inferred])
    window = _neutral_float_window(
        _sag_result(
            sag,
            quality="low",
            quality_reasons=[width_reason],
            warnings=[width_reason, other],
            circumference_quality="low",
            circumference_quality_reasons=[width_reason, inferred],
        )
    )
    window._update_automatic_results()

    label = window.measurementPanel.autoWarningLabel
    assert not label.isHidden()
    text = label.text()
    assert f"Neck circumference quality: low — {width_reason}; {inferred}" in text
    assert f"Neck: {other}" in text
    assert text.count("front point inferred") == 1
    assert text.count("clean rows span") == 1
    panel_text = window.measurementPanel.autoLabel.text()
    assert "(fitted circle) · quality: low" in panel_text
    assert "width quality: low" in panel_text
    window.close()


def test_older_tape_path_library_merges_width_and_tape_reasons(qapp):
    # neck_sag without circumference_quality: width + tape reasons, deduplicated.
    width_reason = "clean rows span only 3.7 mm (< 5 mm)"
    sag = _sag(quality="low", quality_reasons=[width_reason, "front point inferred"])
    result = _sag_result(sag, quality="low", quality_reasons=[width_reason])
    del result.circumference_quality
    del result.circumference_quality_reasons
    window = _neutral_float_window(result)
    window._update_automatic_results()
    warning = window.measurementPanel.autoWarningLabel.text()
    assert f"Neck circumference quality: low — {width_reason}; front point inferred" in warning
    assert warning.count("clean rows span") == 1
    window.close()


def test_panel_falls_back_to_the_width_when_the_tape_path_failed(qapp):
    result = _sag_result(
        _failed_sag(), warnings=["no tape-plane circumference: no depth at the chin"]
    )
    window = _neutral_float_window(result)
    window._update_automatic_results()

    lines = window.measurementPanel.autoLabel.text().splitlines()
    assert lines[1] == (
        f"Neck width: {format_cm(134.5)} · circumference "
        f"{format_neck_circumference((391.4, 402.3), 422.5)}"
    )
    assert "  tape path: n/a — No depth at the chin" in lines
    assert "Neck circumference" not in "\n".join(lines)
    # The library's warning says the same: not repeated.
    assert window.measurementPanel.autoWarningLabel.isHidden()
    window.close()


def test_panel_falls_back_when_the_tape_path_is_none(qapp):
    window = _neutral_float_window(_ok_result(sag=None, circumference_mm=None))
    window._update_automatic_results()

    text = window.measurementPanel.autoLabel.text()
    assert f"Neck width: {format_cm(134.5)} · circumference" in text
    assert "tape path: n/a — not computed" in text
    assert window._neck_width_overlay_visible()  # the width row is still drawn
    window.close()


def test_library_without_tape_path_keeps_todays_lines(qapp):
    window = _neutral_float_window(_ok_result())
    window._update_automatic_results()
    text = window.measurementPanel.autoLabel.text()
    assert "Neck width:" in text
    assert "tape path" not in text
    window.close()


# -------------------------------------------------------------- overlay

FRONT_COLOR = MainWindow.NECK_SAG_FRONT_COLOR


def _display(x, y):
    return x * 480 / PHOTO_SIZE[0], y * 640 / PHOTO_SIZE[1]


def _is_front_orange(color):
    return color.red() > 200 and 100 < color.green() < 200 and color.blue() < 90


def _green_near(canvas, x, y, radius=1):
    return any(
        _is_neck_green(canvas.pixelColor(round(x) + dx, round(y) + dy))
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
    )


def _paint(window):
    canvas = QtGui.QImage(480, 640, QtGui.QImage.Format_RGB32)
    canvas.fill(Qt.black)
    painter = QtGui.QPainter(canvas)

    def mapper(point):
        x, y = window._photo_to_main_display(point)
        return QPoint(round(x), round(y))

    window._paint_neck_width_overlay(painter, mapper)
    painter.end()
    return canvas


def test_overlay_draws_the_tape_path_and_front_marker(qapp):
    window = _neutral_float_window(_sag_result())
    canvas = _paint(window)

    for photo_x in (1134.0, 1260.0, 1386.0, 1764.0, 1890.0):
        assert _green_near(canvas, *_display(photo_x, _path_y(photo_x))), photo_x
    front_x, front_y = _display(*FRONT)
    assert _is_front_orange(canvas.pixelColor(round(front_x), round(front_y)))
    # Side points keep their ticks.
    left_x, left_y = _display(*LEFT)
    assert _is_neck_green(canvas.pixelColor(round(left_x), round(left_y) - 5))
    # The horizontal measuring line is replaced by the path.
    mid_x, row_y = _display(0.5 * (LEFT[0] + RIGHT[0]), LEFT[1])
    assert not _green_near(canvas, mid_x, row_y)
    window.close()


def test_overlay_without_a_tape_path_draws_the_row(qapp):
    window = _neutral_float_window(_sag_result(_failed_sag()))
    canvas = _paint(window)
    mid_x, row_y = _display(0.5 * (LEFT[0] + RIGHT[0]), LEFT[1])
    assert _is_neck_green(canvas.pixelColor(round(mid_x), round(row_y)))
    front_x, front_y = _display(*FRONT)
    assert canvas.pixelColor(round(front_x), round(front_y)) == QtGui.QColor(Qt.black)
    window.close()


def test_zoom_and_main_overlays_draw_the_tape_path(qapp):
    window = _neutral_float_window(_sag_result())
    canvas = QtGui.QImage(480, 640, QtGui.QImage.Format_RGB32)
    canvas.fill(Qt.black)
    painter = QtGui.QPainter(canvas)
    window._paint_zoom_region_overlay(painter, (480, 640), (0, 0, 480, 640), (0, 0, 480, 640))
    painter.end()
    front_x, front_y = _display(*FRONT)
    assert _is_front_orange(canvas.pixelColor(round(front_x), round(front_y)))
    assert _green_near(canvas, *_display(1260.0, _path_y(1260.0)))

    window.redrawImage(force=True)
    main = window.ui.imageLabel.pixmap().toImage()
    assert _is_front_orange(main.pixelColor(round(front_x), round(front_y)))
    window.neckWidthOverlayAction.setChecked(False)
    window.redrawImage(force=True)
    main = window.ui.imageLabel.pixmap().toImage()
    assert not _is_front_orange(main.pixelColor(round(front_x), round(front_y)))
    window.close()


# ---------------------------------------------------------- diagnostics


def test_diagnostics_list_the_plane_and_every_model(qapp):
    window = _neutral_float_window(_sag_result())
    window.neck_measurement_auto = None
    text = window._diagnostics_text()
    assert "[Neck width] status: ok" in text
    assert "Search from Y 2215 px" in text
    assert "[Neck tape path] status: ok" in text
    assert "Front point: X 1512, Y 2440 px (chin-end: below the chin)" in text
    assert "Circumference: 41.4 cm (fitted circle, headline)" in text
    assert "Plane tilt: 42.3 deg (least squares), 3-point plane 67.1 deg" in text
    assert "Sagitta: 36.2 mm" in text
    assert "Front arc (3D path): 197.4 mm" in text
    assert "fitted circle: 41.4 cm (±2.5 %) — headline" in text
    assert "π·W: 42.2 cm (±4.9 %)" in text
    assert "arc circle: 42.5 cm (±5.2 %)" in text
    assert "sagitta ellipse: 33.2 cm (±7.1 %)" in text
    assert "fitted ellipse: 31.8 cm (±7.0 %)" in text
    assert "prior ellipse (b/a 0.85–0.90): 39.1–40.2 cm" in text
    assert text.index("fitted circle: 41.4") < text.index("π·W: 42.2")
    assert "Circle RMS: 0.8 mm" in text
    assert "Circle span: 141.0 deg" in text
    assert "side-width+3%: fitted circle +2.5 %, π·W +3.0 %, arc circle +5.2 %" in text
    assert "ends-10%: fitted circle -1.2 %" in text
    assert "fitted ellipse n/a" in text
    assert "Accuracy: fitted circle is shape-dependent" in text
    assert "Circumference quality: good" in text
    window.close()


def test_diagnostics_handle_missing_sensitivities(qapp):
    sensitivity = dict(_sag().sensitivity, fitted_circle=None)
    window = _neutral_float_window(_sag_result(_sag(sensitivity=sensitivity)))
    window.neck_measurement_auto = None
    text = window._diagnostics_text()
    assert "fitted circle: 41.4 cm (n/a) — headline" in text
    window.close()


def test_diagnostics_explain_a_failed_tape_path(qapp):
    window = _neutral_float_window(_sag_result(_failed_sag()))
    window.neck_measurement_auto = None
    text = window._diagnostics_text()
    assert "[Neck tape path] status: failed" in text
    assert "No depth at the chin" in text
    assert "Models" not in text
    window.close()


def test_diagnostics_of_a_library_without_tape_path_have_no_block(qapp):
    window = _neutral_float_window(_ok_result())
    window.neck_measurement_auto = None
    assert "[Neck tape path]" not in window._diagnostics_text()
    window.close()


# ------------------------------------------------------------ Neck tool


def _tool_details(monkeypatch, auto):
    _record_calls(monkeypatch, _ok_result())
    window = _float_window()
    window.neck_width_auto = auto
    window.set_tool(MeasurementTool.NECK)
    window._on_image_clicked(QPointF(160, 356))
    window._on_image_clicked(QPointF(320, 360))
    window._finish_polyline()
    details = window.measurementPanel.detailsLabel.text()
    entries = window.measurementPanel.history.entries
    window.close()
    return details, entries


def test_neck_tool_shows_the_automatic_tape_path_for_comparison(qapp, monkeypatch):
    details, entries = _tool_details(monkeypatch, _sag_result())
    assert f"Auto (tape path): {format_cm(414.0, 1)}" in details
    # History unchanged: the tool's own width, the ellipse midpoint.
    (entry,) = entries
    assert entry.linear_mm == 134.5


def test_neck_tool_without_automatic_tape_path_has_no_comparison(qapp, monkeypatch):
    for auto in (None, _ok_result(), _sag_result(_failed_sag())):
        details, _entries = _tool_details(monkeypatch, auto)
        assert "Auto (tape path)" not in details
