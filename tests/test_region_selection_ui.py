from types import SimpleNamespace

from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from fidmaa_gui.app import MainWindow
from fidmaa_gui.depth_visualization import DepthDisplayMode
from fidmaa_gui.QClickableLabel import QClickableLabel
from fidmaa_gui.region_measurement import Region, SelectionMode


def test_clickable_label_separates_click_drag_and_pointer_motion(qapp):
    label = QClickableLabel()
    label.resize(100, 100)
    label.show()
    events = {"clicked": [], "pressed": [], "dragged": [], "released": [], "moved": []}
    label.clicked.connect(events["clicked"].append)
    label.pressed.connect(events["pressed"].append)
    label.dragged.connect(events["dragged"].append)
    label.released.connect(events["released"].append)
    label.pointerMoved.connect(events["moved"].append)

    QTest.mousePress(label, Qt.LeftButton, pos=QPointF(10, 10).toPoint())
    QTest.mouseRelease(label, Qt.LeftButton, pos=QPointF(10, 10).toPoint())

    assert len(events["pressed"]) == 1
    assert len(events["released"]) == 1
    assert len(events["clicked"]) == 1

    QTest.mousePress(label, Qt.LeftButton, pos=QPointF(20, 20).toPoint())
    QTest.mouseMove(label, QPointF(50, 50).toPoint())
    QTest.mouseRelease(label, Qt.LeftButton, pos=QPointF(50, 50).toPoint())

    assert events["moved"]
    assert events["dragged"]
    assert len(events["clicked"]) == 1
    label.close()


def test_region_drag_creates_a_then_automatically_selects_b(qapp):
    window = MainWindow()
    window._set_region_interaction_mode(1)

    window._begin_region_drag(QPointF(100, 100))
    window._drag_region(QPointF(130, 100))
    window._finish_region_drag(QPointF(130, 100))

    assert window.region_a == Region(70, 70, 130, 130)
    assert window._region_target == "b"
    assert window.regionBButton.isChecked()
    window.close()


def test_single_key_tool_shortcuts_and_tools_menu(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()

    QTest.keyClick(window, Qt.Key_H)
    assert window._region_target == "a"
    assert window.regionModeACombo.currentText() == SelectionMode.HIGHEST.value

    window.region_a = Region(10, 10, 30, 30)
    QTest.keyClick(window, Qt.Key_P)
    assert window._region_target is None
    assert window.pointModeButton.isChecked()

    QTest.keyClick(window, Qt.Key_L)
    assert window._region_target == "b"
    assert window.regionModeBCombo.currentText() == SelectionMode.LOWEST.value

    QTest.keyClick(window, Qt.Key_F)
    assert window.regionModeBCombo.currentText() == SelectionMode.FLATTEST.value
    assert window.pixelToolAction.shortcut().toString().lower() == "p"
    assert window.regionToolActions[SelectionMode.HIGHEST].shortcut().toString().lower() == "h"
    window.close()


def test_circle_can_be_moved_and_resized_from_perimeter(qapp):
    window = MainWindow()
    window.region_a = Region(70, 70, 130, 130)
    window._set_region_interaction_mode(1)

    window._begin_region_drag(QPointF(100, 100))
    window._drag_region(QPointF(110, 110))
    window._finish_region_drag(QPointF(110, 110))
    assert window.region_a == Region(80, 80, 140, 140)

    window._set_region_interaction_mode(1)
    window._begin_region_drag(QPointF(140, 110))
    window._drag_region(QPointF(160, 110))
    window._finish_region_drag(QPointF(160, 110))
    assert window.region_a == Region(60, 60, 160, 160)
    window.close()


def test_circle_hover_uses_move_resize_and_draw_cursors(qapp):
    window = MainWindow()
    window.region_a = Region(70, 70, 130, 130)
    window._set_region_interaction_mode(1)

    expected_cursors = [
        (QPointF(100, 100), Qt.CursorShape.SizeAllCursor),
        (QPointF(130, 100), Qt.CursorShape.SizeHorCursor),
        (QPointF(100, 70), Qt.CursorShape.SizeVerCursor),
        (QPointF(121, 121), Qt.CursorShape.SizeFDiagCursor),
        (QPointF(79, 121), Qt.CursorShape.SizeBDiagCursor),
        (QPointF(180, 180), Qt.CursorShape.CrossCursor),
    ]
    for point, expected_cursor in expected_cursors:
        window._update_region_cursor(point)
        assert window.ui.imageLabel.cursor().shape() == expected_cursor

    window._begin_region_drag(QPointF(180, 180))
    assert window.ui.imageLabel.cursor().shape() == Qt.CursorShape.CrossCursor
    window._finish_region_drag(QPointF(180, 180))

    window._set_region_interaction_mode(0)
    window._update_region_cursor(QPointF(100, 100))
    assert window.ui.imageLabel.cursor().shape() == Qt.CursorShape.CrossCursor
    window.close()


def test_new_region_is_always_a_circle_drawn_from_center_to_radius(qapp):
    window = MainWindow()
    window._set_region_interaction_mode(1)

    window._begin_region_drag(QPointF(100, 100))
    window._drag_region(QPointF(130, 140))
    window._finish_region_drag(QPointF(130, 140))

    assert window.region_a == Region(50, 50, 150, 150)
    assert window.region_a.radius == 50
    window.close()


def test_panel_calculates_linear_and_surface_summaries(qapp):
    window = MainWindow()
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
    assert "Linear 3D:" in window.regionResultEdit.toPlainText()
    assert "Surface 3D:" in window.regionResultEdit.toPlainText()
    assert "(n=10" in window.regionResultEdit.toPlainText()
    assert window.depthDisplayCombo.currentText() == DepthDisplayMode.COLOR.value
    assert {
        window.depthDisplayCombo.itemText(index)
        for index in range(window.depthDisplayCombo.count())
    } == {mode.value for mode in DepthDisplayMode}
    window.portrait = None
    window.close()
