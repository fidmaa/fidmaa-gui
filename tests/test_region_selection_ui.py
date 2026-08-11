from types import SimpleNamespace

from PIL import Image
from PySide6 import QtGui
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from fidmaa_gui.app import MainWindow
from fidmaa_gui.depth_visualization import DepthDisplayMode
from fidmaa_gui.QClickableLabel import QClickableLabel
from fidmaa_gui.region_measurement import (
    CandidatePoint,
    Region,
    RegionMeasurementResult,
    SelectionMode,
    VectorSample,
)


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


def test_single_click_places_a_fixed_radius_patch(qapp):
    window = MainWindow()
    window.regionRadiusSpin.setValue(30)
    window.regionSelectionButton.setChecked(True)

    window._begin_region_drag(QPointF(100, 100))
    window._finish_region_drag(QPointF(100, 100))

    assert window.region_a == Region(70, 70, 130, 130)
    assert window._region_target == "a"
    assert window.regionSelectionButton.isChecked()
    assert window.measurementControlsAction.isChecked()
    window.close()


def test_single_key_tool_shortcuts_and_tools_menu(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()

    available_modes = {
        window.regionModeACombo.itemData(index) for index in range(window.regionModeACombo.count())
    }
    assert available_modes == set(SelectionMode)

    QTest.keyClick(window, Qt.Key_H)
    assert window._region_target == "a"
    assert window.regionModeACombo.currentText() == SelectionMode.HIGHEST.value
    assert window.regionSelectionButton.isChecked()

    window.region_a = Region(10, 10, 30, 30)
    QTest.keyClick(window, Qt.Key_P)
    assert window._region_target is None
    assert not window.regionSelectionButton.isChecked()
    assert not window.measurementControlsAction.isChecked()

    QTest.keyClick(window, Qt.Key_L)
    assert window._region_target == "b"
    assert window.regionModeBCombo.currentText() == SelectionMode.LOWEST.value

    QTest.keyClick(window, Qt.Key_F)
    assert window.regionModeBCombo.currentText() == SelectionMode.FLATTEST.value
    assert window.pixelToolAction.shortcut().toString().lower() == "p"
    assert window.regionToolActions[SelectionMode.HIGHEST].shortcut().toString().lower() == "h"
    assert window.regionToolActions[SelectionMode.LOCAL_PEAK].shortcut().isEmpty()
    window.close()


def test_neck_skin_to_depth_overlay_is_visible_without_region_controls(qapp):
    window = MainWindow()
    window.portrait = SimpleNamespace(skinmap=Image.new("L", (480, 640), 255))
    window.neck_measurement_auto = SimpleNamespace(
        mask_left_x=100,
        mask_right_x=380,
        left_x=120,
        right_x=360,
        neck_y=200,
        arc_points_photo=[(120, 200), (240, 230), (360, 200)],
    )
    window.ui.showNeckArcCheckBox.setEnabled(True)
    window.ui.showNeckArcCheckBox.setChecked(True)
    window._region_target = None

    canvas = QtGui.QImage(120, 160, QtGui.QImage.Format_RGB32)
    canvas.fill(Qt.black)
    painter = QtGui.QPainter(canvas)
    window._paint_zoom_region_overlay(
        painter,
        (480, 640),
        (0, 0, 480, 640),
        (0, 0, 120, 160),
    )
    painter.end()

    skin_edge = canvas.pixelColor(19, 50)
    stable_depth = canvas.pixelColor(30, 50)
    assert skin_edge.red() > 200 and skin_edge.green() > 120
    assert stable_depth.green() > 180 and stable_depth.blue() > 180
    assert "skin matte" in window.ui.showNeckArcCheckBox.text().lower()
    assert "stable depth" in window.ui.showNeckArcCheckBox.text().lower()
    window.close()


def test_zoomed_teeth_centroids_follow_keep_aspect_ratio_offsets(qapp):
    window = MainWindow()
    window.zoomedTeethMapLabel.setFixedSize(200, 100)
    centroids = SimpleNamespace(
        upper_centroid=(25, 50),
        lower_centroid=None,
    )

    window.paintZoomedTeethmap(
        Image.new("L", (100, 100), 255),
        centroids=centroids,
        source_size=(100, 100),
        crop_box=(0, 0, 100, 100),
    )

    rendered = window.zoomedTeethMapLabel.pixmap().toImage()
    correctly_offset = rendered.pixelColor(75, 50)
    old_unoffset_position = rendered.pixelColor(50, 50)
    assert correctly_offset.red() < 100
    assert correctly_offset.green() > 180 and correctly_offset.blue() > 180
    assert old_unoffset_position.red() > 200
    window.close()


def test_patch_can_be_moved_and_radius_control_resizes_both_patches(qapp):
    window = MainWindow()
    window.regionRadiusSpin.setValue(30)
    window.region_a = Region(70, 70, 130, 130)
    window.region_b = Region(170, 170, 230, 230)
    window.regionSelectionButton.setChecked(True)

    window._begin_region_drag(QPointF(100, 100))
    window._drag_region(QPointF(110, 110))
    window._finish_region_drag(QPointF(110, 110))
    assert window.region_a == Region(80, 80, 140, 140)

    window.regionRadiusSpin.setValue(50)
    assert window.region_a == Region(60, 60, 160, 160)
    assert window.region_b == Region(150, 150, 250, 250)
    window.close()


def test_patch_hover_uses_move_and_place_cursors(qapp):
    window = MainWindow()
    window.region_a = Region(70, 70, 130, 130)
    window.regionSelectionButton.setChecked(True)

    expected_cursors = [
        (QPointF(100, 100), Qt.CursorShape.SizeAllCursor),
        (QPointF(130, 100), Qt.CursorShape.SizeAllCursor),
        (QPointF(100, 70), Qt.CursorShape.SizeAllCursor),
        (QPointF(121, 121), Qt.CursorShape.SizeAllCursor),
        (QPointF(79, 121), Qt.CursorShape.SizeAllCursor),
        (QPointF(180, 180), Qt.CursorShape.CrossCursor),
    ]
    for point, expected_cursor in expected_cursors:
        window._update_region_cursor(point)
        assert window.ui.imageLabel.cursor().shape() == expected_cursor

    window._begin_region_drag(QPointF(180, 180))
    assert window.ui.imageLabel.cursor().shape() == Qt.CursorShape.CrossCursor
    window._finish_region_drag(QPointF(180, 180))

    window.regionSelectionButton.setChecked(False)
    window._update_region_cursor(QPointF(100, 100))
    assert window.ui.imageLabel.cursor().shape() == Qt.CursorShape.CrossCursor
    window.close()


def test_configured_radius_is_used_without_drawing_the_perimeter(qapp):
    window = MainWindow()
    window.regionRadiusSpin.setValue(50)
    window.regionSelectionButton.setChecked(True)

    window._begin_region_drag(QPointF(100, 100))
    window._finish_region_drag(QPointF(100, 100))

    assert window.region_a == Region(50, 50, 150, 150)
    assert window.region_a.radius == 50
    window.close()


def test_clicking_a_moves_a_even_when_b_was_the_previous_target(qapp):
    window = MainWindow()
    window.regionRadiusSpin.setValue(30)
    original_b = Region(170, 170, 230, 230)
    window.region_a = Region(70, 70, 130, 130)
    window.region_b = original_b
    window._last_region_target = "b"
    window.regionSelectionButton.setChecked(True)
    assert window._region_target == "b"

    window._begin_region_drag(QPointF(100, 100))
    window._drag_region(QPointF(110, 110))
    window._finish_region_drag(QPointF(110, 110))

    assert window.region_a == Region(80, 80, 140, 140)
    assert window.region_b == original_b
    assert window._region_target == "a"
    window.close()


def test_dragging_outside_does_not_replace_either_completed_circle(qapp):
    window = MainWindow()
    original_a = Region(70, 70, 130, 130)
    original_b = Region(170, 170, 230, 230)
    window.region_a = original_a
    window.region_b = original_b
    window.regionSelectionButton.setChecked(True)

    window._begin_region_drag(QPointF(350, 350))
    window._drag_region(QPointF(390, 390))
    window._finish_region_drag(QPointF(390, 390))

    assert window.region_a == original_a
    assert window.region_b == original_b
    assert window.ui.imageLabel.cursor().shape() == Qt.CursorShape.ArrowCursor
    window.close()


def test_zoom_map_shows_regions_candidates_vectors_and_bottom_info(qapp):
    window = MainWindow()
    window.zoomedDepthMapLabel.setFixedSize(200, 100)
    window.region_a = Region(80, 78, 100, 98)
    window.region_b = Region(110, 78, 130, 98)
    window.regionSelectionButton.setChecked(True)
    candidate_a = CandidatePoint(95, 88, (0.0, 0.0, 0.0), 0.0)
    candidate_b = CandidatePoint(115, 88, (0.0, 0.0, 0.0), 0.0)
    window.region_measurement_result = RegionMeasurementResult(
        candidates_a=(candidate_a,),
        candidates_b=(candidate_b,),
        samples=(VectorSample(candidate_a, candidate_b, 0.0, 0.0),),
    )
    window.depthDisplayCombo.setCurrentIndex(
        window.depthDisplayCombo.findData(DepthDisplayMode.RAW)
    )

    window.paintZoomedDepthmap(
        Image.new("L", (100, 50), 128),
        mouse_x=100,
        mouse_y=95,
        source_size=(480, 640),
        crop_box=(50, 70, 150, 120),
    )

    rendered = window.zoomedDepthMapLabel.pixmap().toImage()
    circle_a = rendered.pixelColor(60, 36)
    selected_a = rendered.pixelColor(90, 36)
    vector = rendered.pixelColor(110, 36)
    selected_b = rendered.pixelColor(130, 36)
    assert circle_a.blue() > circle_a.red()
    assert selected_a.blue() > selected_a.red()
    assert vector.red() > 180 and vector.green() > 180 and vector.blue() > 180
    assert selected_b.red() > selected_b.green()
    assert selected_b.blue() > selected_b.green()
    assert rendered.pixelColor(190, 90).red() < rendered.pixelColor(190, 10).red()
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
    assert window.depthContourStepSpin.value() == 3
    assert not window.depthContourStepSpin.isEnabled()
    assert window.depthFocusRadiusSpin.value() == 2
    assert not window.depthFocusRadiusSpin.isEnabled()
    window.depthDisplayCombo.setCurrentIndex(
        window.depthDisplayCombo.findData(DepthDisplayMode.COLOR_CONTOURS)
    )
    assert window.depthContourStepSpin.isEnabled()
    assert not window.depthFocusRadiusSpin.isEnabled()
    window.depthDisplayCombo.setCurrentIndex(
        window.depthDisplayCombo.findData(DepthDisplayMode.CURSOR_BANDS)
    )
    assert not window.depthContourStepSpin.isEnabled()
    assert window.depthFocusRadiusSpin.isEnabled()
    window.portrait = None
    window.close()
