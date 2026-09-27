from types import SimpleNamespace

from PIL import Image
from portrait_analyser.depth_sampling import median_filter_depthmap
from PySide6 import QtGui
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from fidmaa_gui.app import MainWindow
from fidmaa_gui.depth_visualization import DepthDisplayMode
from fidmaa_gui.measurement_tools import MeasurementTool, PolylineEditor
from fidmaa_gui.region_measurement import Region, SelectionMode


def test_default_tool_is_vector_and_tools_are_exclusive(qapp):
    window = MainWindow()
    assert window.current_tool == MeasurementTool.VECTOR
    assert window.toolActions[MeasurementTool.VECTOR].isChecked()
    assert window.regionOptionsWidget.isHidden()

    window.toolActions[MeasurementTool.REGION].trigger()
    assert window.current_tool == MeasurementTool.REGION
    assert not window.toolActions[MeasurementTool.VECTOR].isChecked()
    assert not window.regionOptionsWidget.isHidden()
    assert window.lineOptionsWidget.isHidden()
    assert window._region_target == "a"

    window.set_tool(MeasurementTool.POLYLINE)
    assert window._region_target is None
    assert window.polyline_editor.max_points is None
    window.close()


def test_single_key_shortcuts_switch_tools(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()

    QTest.keyClick(window, Qt.Key_M)
    assert window.current_tool == MeasurementTool.POLYLINE
    QTest.keyClick(window, Qt.Key_R)
    assert window.current_tool == MeasurementTool.REGION
    QTest.keyClick(window, Qt.Key_V)
    assert window.current_tool == MeasurementTool.VECTOR
    assert window.polyline_editor.max_points == 2

    QTest.keyClick(window, Qt.Key_H)
    assert window.current_tool == MeasurementTool.REGION
    assert window.regionModeACombo.currentText() == SelectionMode.HIGHEST.value
    assert not hasattr(window, "pixelToolAction")
    window.close()


def test_depth_display_lives_on_toolbar_and_view_menu(qapp):
    window = MainWindow()
    assert window.depthDisplayCombo.parent() is not None
    toolbar_widgets = {window.toolBar.widgetForAction(a) for a in window.toolBar.actions()}
    assert window.depthDisplayCombo in toolbar_widgets

    window.depthDisplayActions[DepthDisplayMode.COLOR_CONTOURS].trigger()
    assert window.depthDisplayCombo.currentData() == DepthDisplayMode.COLOR_CONTOURS
    assert window.depthContourStepSpin.isEnabled()

    window.depthDisplayCombo.setCurrentIndex(
        window.depthDisplayCombo.findData(DepthDisplayMode.RAW)
    )
    assert window.depthDisplayActions[DepthDisplayMode.RAW].isChecked()
    window.close()


def test_region_panel_result_is_refreshed_on_clear_and_failure(qapp):
    window = MainWindow()
    window.set_tool(MeasurementTool.REGION)

    depth = Image.new("L", (480, 640), 30)
    window.image = Image.new("RGB", (480, 640))
    window.depthmap = depth
    window.filtered_depthmap = depth
    window.portrait = SimpleNamespace(teethmap=None)
    window.float_min_value = 2.0
    window.float_max_value = 4.0
    window.region_a = Region(50, 50, 100, 100)
    window.region_b = Region(50, 150, 100, 200)
    window.regionModeBCombo.setCurrentIndex(window.regionModeBCombo.findData(SelectionMode.HIGHEST))

    window._calculate_region_measurement()
    assert window.region_measurement_result is not None
    assert window.measurementPanel.linearValueLabel.text() != "—"

    window.clear_region_measurement(repaint=False)
    assert window.measurementPanel.linearValueLabel.text() == "—"

    # Re-run a successful measurement, then force the "no depth data" early
    # return: the panel must not keep showing the stale successful result.
    window.region_a = Region(50, 50, 100, 100)
    window.region_b = Region(50, 150, 100, 200)
    window._calculate_region_measurement()
    assert window.measurementPanel.linearValueLabel.text() != "—"

    window.filtered_depthmap = None
    window._calculate_region_measurement()
    assert "Load an image" in window.regionResultEdit.toPlainText()
    assert window.region_measurement_result is None
    assert window.measurementPanel.linearValueLabel.text() == "—"
    window.close()


def _window_with_flat_depth():
    window = MainWindow()
    depth = Image.new("RGB", (480, 640), (30, 30, 30))
    window.image = Image.new("RGB", (480, 640))
    window.depthmap = depth
    window.filtered_depthmap = median_filter_depthmap(depth, size=3)
    window.float_min_value = 2.0
    window.float_max_value = 4.0
    window.filename = "/tmp/patient.heic"
    return window


def test_vector_two_clicks_show_linear_and_surface_and_add_history(qapp):
    window = _window_with_flat_depth()
    window._on_image_clicked(QPointF(100, 100))
    assert window.measurementPanel.linearValueLabel.text() == "—"
    window._on_image_clicked(QPointF(100, 300))

    entries = window.measurementPanel.history.entries
    assert len(entries) == 1
    assert entries[0].tool == MeasurementTool.VECTOR
    assert entries[0].filename == "patient.heic"
    assert entries[0].linear_mm > 0
    assert abs(entries[0].surface_mm - entries[0].linear_mm) / entries[0].linear_mm < 0.02
    assert "cm" in window.measurementPanel.linearValueLabel.text()

    window._on_image_clicked(QPointF(200, 200))
    assert window.polyline_editor.points == [(200.0, 200.0)]
    assert len(window.measurementPanel.history.entries) == 1
    window.close()


def test_dragging_a_finished_vector_point_updates_the_same_history_row(qapp):
    window = _window_with_flat_depth()
    window._on_image_clicked(QPointF(100, 100))
    window._on_image_clicked(QPointF(100, 300))
    before = window.measurementPanel.history.entries[0].linear_mm

    window._begin_point_drag(QPointF(101, 300))
    window._drag_point(QPointF(100, 400))
    window._finish_point_drag(QPointF(100, 400))
    window._on_image_clicked(QPointF(100, 400))  # click emitted after a short drag

    entries = window.measurementPanel.history.entries
    assert len(entries) == 1
    assert entries[0].linear_mm > before
    assert window.polyline_editor.points[1] == (100.0, 400.0)
    window.close()


def test_polyline_finishes_with_enter_or_right_click_only_after_two_points(qapp):
    window = _window_with_flat_depth()
    window.show()
    window.set_tool(MeasurementTool.POLYLINE)

    window._on_image_clicked(QPointF(50, 50))
    QTest.keyClick(window, Qt.Key_Return)
    window._on_image_right_clicked(QPointF(0, 0))
    assert window.measurementPanel.history.entries == ()

    for y in (150, 250, 350, 450):
        window._on_image_clicked(QPointF(50, y))
    assert window.measurementPanel.history.entries == ()
    QTest.keyClick(window, Qt.Key_Return)

    entries = window.measurementPanel.history.entries
    assert len(entries) == 1
    assert entries[0].tool == MeasurementTool.POLYLINE
    assert entries[0].points == 5
    assert "4 segments" in window.measurementPanel.detailsLabel.text()

    window._on_image_clicked(QPointF(300, 50))
    window._on_image_clicked(QPointF(300, 150))
    window._on_image_right_clicked(QPointF(300, 150))
    assert len(window.measurementPanel.history.entries) == 2
    window.close()


def test_backspace_and_escape_edit_polyline(qapp):
    window = _window_with_flat_depth()
    window.show()
    window.set_tool(MeasurementTool.POLYLINE)
    for y in (100, 200, 300):
        window._on_image_clicked(QPointF(60, y))
    QTest.keyClick(window, Qt.Key_Backspace)
    assert len(window.polyline_editor.points) == 2
    QTest.keyClick(window, Qt.Key_Escape)
    assert window.polyline_editor.points == []
    window.close()


def test_tools_do_not_crash_without_a_photo(qapp):
    window = MainWindow()
    window.show()
    window._on_image_clicked(QPointF(10, 10))
    window._on_image_clicked(QPointF(20, 20))
    window._on_image_right_clicked(QPointF(20, 20))
    for key in (Qt.Key_Return, Qt.Key_Backspace, Qt.Key_Escape):
        QTest.keyClick(window, key)
    assert window.measurementPanel.linearValueLabel.text() == "—"
    assert window.measurementPanel.history.entries == ()
    window.close()


def test_loading_new_photo_clears_points_but_keeps_history(qapp):
    window = _window_with_flat_depth()
    window._on_image_clicked(QPointF(100, 100))
    window._on_image_clicked(QPointF(100, 300))
    window._on_image_clicked(QPointF(150, 150))

    window.set_tool(window.current_tool)  # what _loadImage does before loading

    assert window.polyline_editor.points == []
    assert len(window.measurementPanel.history.entries) == 1
    assert isinstance(window.polyline_editor, PolylineEditor)
    window.close()


def test_polyline_points_are_painted_on_the_photo(qapp):
    window = _window_with_flat_depth()
    window.polyline_editor.add_point((100.0, 100.0))
    window.polyline_editor.add_point((200.0, 100.0))
    canvas = QtGui.QImage(480, 640, QtGui.QImage.Format_RGB32)
    canvas.fill(Qt.black)
    painter = QtGui.QPainter(canvas)
    window._paint_polyline_overlay(painter, lambda p: QPoint(round(p[0]), round(p[1])))
    painter.end()

    on_line = canvas.pixelColor(150, 100)
    assert on_line.red() > 200 and on_line.green() > 180 and on_line.blue() < 80
    window.close()
