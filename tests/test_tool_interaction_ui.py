from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from fidmaa_gui.app import MainWindow
from fidmaa_gui.depth_visualization import DepthDisplayMode
from fidmaa_gui.measurement_tools import MeasurementTool
from fidmaa_gui.region_measurement import SelectionMode


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
