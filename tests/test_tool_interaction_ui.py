from types import SimpleNamespace

from PIL import Image
from portrait_analyser.depth_sampling import median_filter_depthmap
from portrait_analyser.exceptions import NoDepthMapFound
from portrait_analyser.pose import PortraitPose
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


def test_escape_clears_region_measurement(qapp):
    """F7: Esc previously did nothing in Region mode; the global constraint
    is that Esc clears the current measurement in every tool."""
    window = MainWindow()
    window.show()
    window.set_tool(MeasurementTool.REGION)
    window.region_a = Region(50, 50, 100, 100)
    window.region_b = Region(50, 150, 100, 200)

    QTest.keyClick(window, Qt.Key_Escape)

    assert window.region_a is None
    assert window.region_b is None
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


def _window_with_half_invalid_depth():
    """Left half (x<240) is calibrated depth (raw 128, ~36cm); right half is
    raw 0, which depth_raw_to_distance_cm() maps to ~200cm -- outside the
    15-80cm calibrated range, so pixel_to_mm() returns None for it (F1)."""
    window = MainWindow()
    depth = Image.new("RGB", (480, 640), (128, 128, 128))
    pixels = depth.load()
    for y in range(640):
        for x in range(240, 480):
            pixels[x, y] = (0, 0, 0)
    window.image = Image.new("RGB", (480, 640))
    window.depthmap = depth
    window.filtered_depthmap = median_filter_depthmap(depth, size=3)
    window.float_min_value = 0.5
    window.float_max_value = 5.0
    window.filename = "/tmp/patient.heic"
    return window


def test_vector_across_invalid_depth_shows_na_and_does_not_raise(qapp):
    window = _window_with_half_invalid_depth()

    window._on_image_clicked(QPointF(100, 100))  # calibrated depth
    window._on_image_clicked(QPointF(400, 100))  # raw 0 -> uncalibrated; must not raise

    assert window.measurementPanel.linearValueLabel.text() == "n/a (invalid depth)"
    assert "Δ depth: n/a" in window.measurementPanel.detailsLabel.text()

    entries = window.measurementPanel.history.entries
    assert len(entries) == 1
    assert entries[0].linear_mm is None
    window.close()


def test_polyline_through_invalid_depth_recovers_and_records_history(qapp):
    window = _window_with_half_invalid_depth()
    window.show()
    window.set_tool(MeasurementTool.POLYLINE)

    window._on_image_clicked(QPointF(100, 100))  # calibrated
    window._on_image_clicked(QPointF(400, 100))  # uncalibrated point; used to raise here
    window._on_image_clicked(QPointF(100, 300))  # calibrated again; must not raise either
    window._begin_point_drag(QPointF(100, 300))
    window._drag_point(QPointF(120, 300))
    window._finish_point_drag(QPointF(120, 300))
    QTest.keyClick(window, Qt.Key_Return)  # finish; must not raise

    # Points are kept (and thus drawn), including the one over invalid depth.
    assert window.polyline_editor.points == [(100.0, 100.0), (400.0, 100.0), (120.0, 300.0)]

    entries = window.measurementPanel.history.entries
    assert len(entries) == 1
    assert entries[0].points == 3
    assert entries[0].linear_mm is None  # a segment crossed invalid depth
    # History row is recorded with empty cells for the invalid columns.
    assert window.measurementPanel.historyTable.item(0, 3).text() == ""
    window.close()


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


def test_clearing_history_resets_stale_index_so_next_change_adds_a_row(qapp):
    """F2: without resetting _tool_history_index on Clear, the next update
    (e.g. a Step spin box change) does measurementPanel.history.replace()
    against an empty history and raises IndexError."""
    window = _window_with_flat_depth()
    window._on_image_clicked(QPointF(100, 100))
    window._on_image_clicked(QPointF(100, 300))
    assert len(window.measurementPanel.history.entries) == 1

    window.measurementPanel.clear_history()
    assert window._tool_history_index is None

    window.surfaceStepSpin.setValue(window.surfaceStepSpin.value() + 1)  # must not raise

    entries = window.measurementPanel.history.entries
    assert len(entries) == 1
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


def test_backspace_reaches_polyline_after_focus_was_on_a_spin_box(qapp):
    """F5: a click must move focus to imageLabel, so Backspace/Enter/Esc sent
    right after editing a spin box still reach MainWindow.keyPressEvent
    instead of being consumed by the spin box."""
    window = _window_with_flat_depth()
    window.show()
    QTest.qWaitForWindowActive(window)
    window.set_tool(MeasurementTool.POLYLINE)
    window.surfaceStepSpin.setFocus()

    window._on_image_clicked(QPointF(60, 100))
    window._on_image_clicked(QPointF(60, 200))
    assert window.ui.imageLabel.hasFocus()
    assert len(window.polyline_editor.points) == 2

    QTest.keyClick(window.ui.imageLabel, Qt.Key_Backspace)
    assert len(window.polyline_editor.points) == 1
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


def _stub_portrait(float_min=2.0, float_max=4.0):
    return SimpleNamespace(
        photo=Image.new("RGB", (480, 640)),
        depthmap=Image.new("RGB", (480, 640), (30, 30, 30)),
        teethmap=None,
        skinmap=None,
        hairmap=None,
        teeth_bbox=None,
        incisor_distance=None,
        incisor_measurement=None,
        floatValueMin=float_min,
        floatValueMax=float_max,
    )


def test_hint_is_refreshed_after_a_photo_loads(qapp, monkeypatch):
    """F4: _update_tool_result() used to only run again on the next click, so
    the "Open a photo with depth data first." hint stayed stale right after
    a photo successfully loaded, through the real _run_detection_and_update()."""
    window = MainWindow()
    assert "Open a photo with depth data first." in window.measurementPanel.hintLabel.text()

    monkeypatch.setattr("fidmaa_gui.app.load_image", lambda *a, **k: _stub_portrait())
    # Keep the (real, unstubbed) detection pipeline fast/deterministic: no
    # face/neck is "detected" so _run_detection_and_update() takes its
    # no-face path, which pops a modal via critical_error() -- stub that too.
    monkeypatch.setattr("fidmaa_gui.app.detect_neck_midpoint", lambda image: (None, None, None))
    monkeypatch.setattr(
        "portrait_analyser.extended_neck.detect_neck_midpoint_from_segmentation",
        lambda image: (None, None),
    )
    monkeypatch.setattr(window, "critical_error", lambda err: None)

    window._loadImage("/tmp/photo.heic")

    assert "Open a photo with depth data first." not in window.measurementPanel.hintLabel.text()
    window.close()


def test_loading_new_photo_clears_points_but_keeps_history(qapp, monkeypatch):
    """F3: goes through the real _loadImage() (with load_image() and
    _run_detection_and_update() stubbed out) rather than just calling
    set_tool() directly, so the filename-only-set-after-success fix is
    exercised end-to-end."""
    window = _window_with_flat_depth()
    window._on_image_clicked(QPointF(100, 100))
    window._on_image_clicked(QPointF(100, 300))
    window._on_image_clicked(QPointF(150, 150))
    assert len(window.measurementPanel.history.entries) == 1

    monkeypatch.setattr("fidmaa_gui.app.load_image", lambda *a, **k: _stub_portrait())
    monkeypatch.setattr(window, "_run_detection_and_update", lambda *a, **k: None)

    window._loadImage("/tmp/second.heic")

    assert window.filename == "/tmp/second.heic"
    assert window.polyline_editor.points == []
    assert len(window.measurementPanel.history.entries) == 1
    assert isinstance(window.polyline_editor, PolylineEditor)
    window.close()


def test_failed_load_does_not_change_filename_or_reset_measurement(qapp, monkeypatch):
    """F3: _loadImage() used to set self.filename before attempting the load,
    so a failed load (old photo stays on screen) left self.filename pointing
    at a file that was never opened, mislabelling later history rows."""
    window = _window_with_flat_depth()
    window._on_image_clicked(QPointF(100, 100))  # in-progress measurement
    original_filename = window.filename
    original_points = list(window.polyline_editor.points)

    def raise_no_depth(*args, **kwargs):
        raise NoDepthMapFound("no depth data")

    monkeypatch.setattr("fidmaa_gui.app.load_image", raise_no_depth)
    monkeypatch.setattr(window, "critical_error", lambda err: None)  # avoid a modal

    window._loadImage("/tmp/other.heic")

    assert window.filename == original_filename
    assert window.polyline_editor.points == original_points
    window.close()


def test_open3d_action_disabled_until_photo_loaded(qapp, monkeypatch):
    """F6: opening the 3D view before a photo is loaded used to raise
    AttributeError (no depthmap yet); disable the action until load succeeds."""
    window = MainWindow()
    assert not window.open3DAction.isEnabled()

    monkeypatch.setattr("fidmaa_gui.app.load_image", lambda *a, **k: _stub_portrait())
    monkeypatch.setattr(window, "_run_detection_and_update", lambda *a, **k: None)

    window._loadImage("/tmp/photo.heic")

    assert window.open3DAction.isEnabled()
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


def _neutral_window():
    window = _window_with_flat_depth()
    # The neck-arc/centroid overlays are checked by default and paint from
    # full NeckMeasurement/IncisorMeasurement dataclasses; these tests only
    # exercise the automatic-results text, so keep redrawImage() (triggered
    # indirectly by toggling mouthOpeningAction) from touching that painting
    # code with these deliberately minimal SimpleNamespace stubs.
    window.ui.showNeckArcCheckBox.setChecked(False)
    window.ui.showCentroidsCheckBox.setChecked(False)
    window.portrait = SimpleNamespace(
        teeth_bbox=None, teethmap=None, incisor_measurement=None, skinmap=None
    )
    # Real NeckMidpoint objects always carry `x` (Optional[float], default
    # None); app code guards on `neck_midpoint.x is not None`, so the stub
    # must have the attribute even when unset.
    window.neck_midpoint = SimpleNamespace(pose=PortraitPose.NEUTRAL_NECK, x=None)
    window.neck_arc = SimpleNamespace(circumference_mm=364.0)
    window.neck_measurement_auto = SimpleNamespace(circumference_mm=380.0)
    window.neck_confidence_ratio = 0.5
    window.mouth_measurement = None
    window.segmentation_debug = None
    return window


def test_automatic_results_show_neck_and_no_tmd(qapp):
    window = _neutral_window()
    window._update_automatic_results()

    text = window.measurementPanel.autoLabel.text()
    assert "NEUTRAL" in text
    assert "Neck circumference" in text
    assert "36" in text
    assert "Thyromental" not in text
    assert window.measurementPanel.autoWarningLabel.isHidden()
    assert not hasattr(window, "tmd_mm")
    window.close()


def test_low_confidence_skin_matte_neck_shows_warning(qapp):
    window = _neutral_window()
    window.neck_confidence_ratio = 5.0
    window._update_automatic_results()
    assert not window.measurementPanel.autoWarningLabel.isHidden()
    window.close()


def test_no_face_shows_force_hint(qapp):
    window = _neutral_window()
    window.neck_midpoint = None
    window._update_automatic_results()
    assert "Force extended neck analysis" in window.measurementPanel.autoWarningLabel.text()
    window.close()


def test_facemesh_mouth_opening_is_opt_in(qapp):
    window = _neutral_window()
    assert not window.mouthOpeningAction.isChecked()
    window.neck_midpoint = SimpleNamespace(pose=PortraitPose.OPEN_MOUTH, x=None)
    window.mouth_measurement = SimpleNamespace(distance_3d_mm=42.0)
    # Checking the box re-runs _compute_mouth_measurement() from the FaceMesh
    # debug data of the last detection, so the stub needs enough of that data
    # (landmarks + depthmap + calibration) for a real (non-None) recompute.
    window.portrait.depthmap = window.depthmap
    window.portrait.floatValueMin = window.float_min_value
    window.portrait.floatValueMax = window.float_max_value
    landmarks = [(240.0, 300.0)] * 18
    landmarks[0] = (240.0, 295.0)  # upper lip
    landmarks[17] = (240.0, 305.0)  # lower lip
    window.face_mesh_debug = SimpleNamespace(landmarks=landmarks)

    window._update_automatic_results()
    assert "Mouth opening" not in window.measurementPanel.autoLabel.text()

    window.mouthOpeningAction.setChecked(True)
    window._update_automatic_results()
    assert "Mouth opening" in window.measurementPanel.autoLabel.text()
    window.close()


def test_mouth_toggle_refreshes_diagnostics_text(qapp):
    """F8: toggling the mouth-opening checkbox used to leave dataOutputEdit
    showing stale diagnostics text until the next photo load."""
    window = _neutral_window()
    window.neck_midpoint = SimpleNamespace(pose=PortraitPose.OPEN_MOUTH, x=None)
    window.mouth_measurement = SimpleNamespace(distance_3d_mm=42.0)
    window.portrait.depthmap = window.depthmap
    window.portrait.floatValueMin = window.float_min_value
    window.portrait.floatValueMax = window.float_max_value
    landmarks = [(240.0, 300.0)] * 18
    landmarks[0] = (240.0, 295.0)  # upper lip
    landmarks[17] = (240.0, 305.0)  # lower lip
    window.face_mesh_debug = SimpleNamespace(landmarks=landmarks)

    window.ui.dataOutputEdit.setPlainText("stale diagnostics")
    assert "Mouth opening" not in window.ui.dataOutputEdit.toPlainText()

    window.mouthOpeningAction.setChecked(True)

    assert "Mouth opening" in window.ui.dataOutputEdit.toPlainText()
    window.close()


def test_legacy_depth_profile_ui_is_hidden_by_default_and_toggleable(qapp):
    window = MainWindow()
    legacy = window._legacy_profile_widgets()
    assert window.ui.chartLabel in legacy
    assert window.reconstructionLabel in legacy
    assert all(widget.isHidden() for widget in legacy)
    assert window.ui.dataOutputEdit.isHidden()

    window.depthProfileAction.setChecked(True)
    assert not any(widget.isHidden() for widget in legacy)
    window.diagnosticsAction.setChecked(True)
    assert not window.ui.dataOutputEdit.isHidden()

    window.depthProfileAction.setChecked(False)
    assert all(widget.isHidden() for widget in legacy)
    window.close()


def _face_mesh_window():
    window = MainWindow()
    window.image = Image.new("RGB", (960, 1280))
    window.face_mesh_debug = SimpleNamespace(landmarks=[(480.0, 640.0)])  # display (240, 320)
    window.ui.showFaceMeshLandmarksCheckBox.setEnabled(True)
    return window


def _paint_zoom(window):
    canvas = QtGui.QImage(160, 160, QtGui.QImage.Format_RGB32)
    canvas.fill(Qt.black)
    painter = QtGui.QPainter(canvas)
    # 80x80 display-space crop around the landmark, source == display size
    window._paint_zoom_region_overlay(painter, (480, 640), (200, 280, 280, 360), (0, 0, 160, 160))
    painter.end()
    return canvas


def test_face_mesh_modes_sync_with_checkbox(qapp):
    window = _face_mesh_window()
    assert window.faceMeshModeActions["off"].isChecked()

    window.faceMeshModeActions["everywhere"].trigger()
    assert window.ui.showFaceMeshLandmarksCheckBox.isChecked()
    assert window._face_mesh_in_zoom()

    window.ui.showFaceMeshLandmarksCheckBox.setChecked(False)
    assert window.faceMeshModeActions["off"].isChecked()
    window.ui.showFaceMeshLandmarksCheckBox.setChecked(True)
    assert window.faceMeshModeActions["main"].isChecked()
    assert not window._face_mesh_in_zoom()
    window.close()


def test_face_mesh_is_painted_in_zoom_only_in_everywhere_mode(qapp):
    window = _face_mesh_window()
    window.faceMeshModeActions["main"].trigger()
    assert _paint_zoom(window).pixelColor(80, 80).red() < 50

    window.faceMeshModeActions["everywhere"].trigger()
    canvas = _paint_zoom(window)
    centre = canvas.pixelColor(80, 80)
    assert centre.red() > 200 and centre.green() > 120 and centre.blue() < 80
    # drawn at display resolution: a fixed-size dot, not an enlarged 2x2-pixel blob
    assert canvas.pixelColor(80 + 6, 80).red() < 50
    window.close()


def test_region_options_are_not_squeezed_in_a_short_window(qapp):
    window = MainWindow()
    window.resize(1500, 700)
    window.show()
    window.set_tool(MeasurementTool.REGION)
    qapp.processEvents()

    for widget in (
        window.regionModeACombo,
        window.regionMaskBCombo,
        window.regionRadiusSpin,
        window.regionVectorCountSpin,
    ):
        assert widget.height() >= widget.minimumSizeHint().height()
    assert window.regionResultEdit.height() >= 120
    window.close()


def test_teeth_zoom_is_boosted_for_display_but_readout_stays_raw(qapp):
    window = MainWindow()
    window.zoomedTeethMapLabel.setFixedSize(100, 100)
    teethmap = Image.new("L", (100, 100), 0)
    teethmap.paste(15, (0, 0, 100, 50))  # weak "upper teeth"
    teethmap.paste(192, (0, 50, 100, 100))
    window.portrait = SimpleNamespace(teethmap=teethmap)
    painted = []
    original = window._paintZoomedMap

    def spy(*args, **kwargs):
        painted.append(kwargs.get("display_map"))
        return original(*args, **kwargs)

    window._paintZoomedMap = spy

    window.paintZoomedTeethmap(teethmap, source_size=(100, 100), crop_box=(0, 0, 100, 100))
    boosted = painted[-1]
    assert boosted.getpixel((10, 10)) > 60
    assert teethmap.getpixel((10, 10)) == 15  # raw map untouched

    window.boostTeethMapAction.setChecked(False)
    window.paintZoomedTeethmap(teethmap, source_size=(100, 100), crop_box=(0, 0, 100, 100))
    assert painted[-1] is None
    window.portrait = None
    window.close()
