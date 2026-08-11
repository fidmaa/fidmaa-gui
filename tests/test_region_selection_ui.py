from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from fidmaa_gui.app import MainWindow
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

    window._begin_region_drag(QPointF(10, 20))
    window._drag_region(QPointF(60, 80))
    window._finish_region_drag(QPointF(60, 80))

    assert window.region_a == Region(10, 20, 60, 80)
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


def test_region_can_be_moved_and_resized_from_corner(qapp):
    window = MainWindow()
    window.region_a = Region(10, 20, 60, 80)
    window._set_region_interaction_mode(1)

    window._begin_region_drag(QPointF(30, 40))
    window._drag_region(QPointF(40, 50))
    window._finish_region_drag(QPointF(40, 50))
    assert window.region_a == Region(20, 30, 70, 90)

    window._set_region_interaction_mode(1)
    window._begin_region_drag(QPointF(20, 30))
    window._drag_region(QPointF(10, 15))
    window._finish_region_drag(QPointF(10, 15))
    assert window.region_a.left == 10
    assert window.region_a.top == 15
    assert window.region_a.right == 70
    assert window.region_a.bottom == 90
    window.close()


def test_shift_constrains_new_region_to_square(qapp):
    window = MainWindow()
    window._set_region_interaction_mode(1)

    window._begin_region_drag(QPointF(10, 10))
    with patch(
        "fidmaa_gui.app.QApplication.keyboardModifiers",
        return_value=Qt.ShiftModifier,
    ):
        window._drag_region(QPointF(30, 70))
        window._finish_region_drag(QPointF(30, 70))

    assert window.region_a.width == window.region_a.height == 60
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
    window.portrait = None
    window.close()
