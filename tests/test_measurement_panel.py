from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QApplication

from fidmaa_gui.measurement_panel import MeasurementPanel, format_cm
from fidmaa_gui.measurement_tools import HistoryEntry, MeasurementTool


def test_format_cm_uses_locale_and_handles_invalid_depth(qapp):
    QLocale.setDefault(QLocale(QLocale.Polish, QLocale.Poland))
    try:
        assert format_cm(123.4) == "12,34 cm"
        assert format_cm(None) == "n/a (invalid depth)"
    finally:
        QLocale.setDefault(QLocale.c())


def test_tool_result_shows_big_linear_and_surface_values(qapp):
    panel = MeasurementPanel()
    panel.set_tool_result("Vector", 123.4, None, ["Δ depth: 3.0 mm"], "Click two points.")

    assert panel.resultGroup.title() == "Vector"
    assert panel.linearValueLabel.text() == format_cm(123.4)
    assert panel.surfaceValueLabel.text() == "n/a (invalid depth)"
    assert "Δ depth" in panel.detailsLabel.text()
    assert panel.hintLabel.text() == "Click two points."
    assert panel.linearValueLabel.font().pointSize() > panel.hintLabel.font().pointSize()

    panel.set_tool_result("Vector", None, None, [], "Click two points.")
    assert panel.linearValueLabel.text() == "—"
    panel.close()


def test_tool_result_shows_na_not_dash_when_segments_exist_but_linear_is_none(qapp):
    """F1: once a measurement has segments, an invalid-depth linear total is
    "n/a (invalid depth)", distinct from the blank "—" shown before any
    points are placed."""
    panel = MeasurementPanel()
    panel.set_tool_result("Polyline", None, 45.6, ["2 segments (finished)"], "hint")
    assert panel.linearValueLabel.text() == "n/a (invalid depth)"
    assert panel.surfaceValueLabel.text() == format_cm(45.6)
    panel.close()


def test_history_rows_replace_and_copy_to_clipboard(qapp):
    previous_locale = QLocale()
    QLocale.setDefault(QLocale.c())
    try:
        panel = MeasurementPanel()
        first = panel.add_history(HistoryEntry("a.heic", MeasurementTool.VECTOR, 2, 100.0, 110.0))
        panel.add_history(HistoryEntry("a.heic", MeasurementTool.POLYLINE, 4, 200.0, None))
        panel.replace_history(first, HistoryEntry("a.heic", MeasurementTool.VECTOR, 2, 90.0, 95.0))

        assert panel.historyTable.rowCount() == 2
        assert panel.historyTable.item(0, 3).text() == "9.00"
        assert panel.historyTable.item(1, 4).text() == ""

        text = panel.copy_history()
        assert QApplication.clipboard().text() == text
        assert text.splitlines()[1] == "a.heic\tVector\t2\t9.00\t9.50"

        panel.clear_history()
        assert panel.historyTable.rowCount() == 0
        assert panel.history.entries == ()
        panel.close()
    finally:
        QLocale.setDefault(previous_locale)


def test_clear_history_emits_history_cleared_signal(qapp):
    """F2: MainWindow needs this signal to reset its stale history-row index."""
    panel = MeasurementPanel()
    panel.add_history(HistoryEntry("a.heic", MeasurementTool.VECTOR, 2, 100.0, 110.0))
    received = []
    panel.historyCleared.connect(lambda: received.append(True))

    panel.clear_history()

    assert received == [True]
    panel.close()


def test_automatic_results_and_warning(qapp):
    panel = MeasurementPanel()
    panel.set_automatic_results(["Neck circumference: 36.40 cm"], "Low confidence")
    assert "36.40" in panel.autoLabel.text()
    assert panel.autoWarningLabel.text() == "⚠ Low confidence"
    assert not panel.autoWarningLabel.isHidden()

    panel.set_automatic_results(["Pose: NEUTRAL_NECK"], None)
    assert panel.autoWarningLabel.isHidden()
    panel.close()
