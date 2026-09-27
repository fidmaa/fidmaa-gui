"""TrueDepth capture-app photos: file intrinsics (CameraModel) and depth source.

The expected millimetres are computed by hand with the pinhole model
``X_mm = (u - cx) * Z_mm / fx`` so these tests do not merely echo the
library's own conversion. Legacy photos (``camera is None``) must keep the
calibration polynomial exactly.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from portrait_analyser import CameraModel
from portrait_analyser.depth_sampling import median_filter_depthmap
from portrait_analyser.exceptions import NoDepthMapFound
from portrait_analyser.incisor import pixel_to_mm
from portrait_analyser.pose import PortraitPose
from PySide6.QtCore import QPointF

from fidmaa_gui import errors
from fidmaa_gui.app import MainWindow
from fidmaa_gui.depth_source import (
    RELATIVE_DEPTH_WARNING,
    combine_warnings,
    depth_source_status,
    portrait_camera,
)
from fidmaa_gui.neck_arc import compute_neck_arc
from fidmaa_gui.region_measurement import RegionMeasurementEngine

# Intrinsics of IMG_2346 (capture app, 3024x4032 upright photo).
PHOTO_SIZE = (3024, 4032)
CAMERA = CameraModel(fx=2766.1, fy=2766.1, cx=1499.1, cy=2019.6)
FLOAT_MIN = 0.34
FLOAT_MAX = 3.40


def _distance_cm(raw, float_min=FLOAT_MIN, float_max=FLOAT_MAX):
    """Raw 8-bit disparity -> distance, written out by hand."""
    disparity = float_max * raw / 255.0 + float_min * (1.0 - raw / 255.0)
    return 100.0 / disparity


# ---------------------------------------------------------------- translate


def test_translate_click_uses_pinhole_model_with_camera():
    window = SimpleNamespace(
        image=SimpleNamespace(size=PHOTO_SIZE), portrait=SimpleNamespace(camera=CAMERA)
    )

    x_mm, y_mm = MainWindow.translate_click_to_mm(window, 35.0, 300, 280)

    photo_x = 300 * 3024 / 480  # 1890
    photo_y = 280 * 4032 / 640  # 1764
    assert x_mm == pytest.approx((photo_x - 1499.1) * 350.0 / 2766.1)
    assert y_mm == pytest.approx((photo_y - 2019.6) * 350.0 / 2766.1)


def test_translate_click_scales_each_axis_for_landscape_photos():
    """4032x3024 capture-app photos are squeezed into the 480x640 display;
    display -> photo scaling is per axis, so measurements stay correct even
    though the preview is distorted."""
    camera = CameraModel(fx=2763.7, fy=2763.7, cx=2012.3, cy=1499.1)
    window = SimpleNamespace(
        image=SimpleNamespace(size=(4032, 3024)), portrait=SimpleNamespace(camera=camera)
    )

    x_mm, y_mm = MainWindow.translate_click_to_mm(window, 36.0, 120, 480)

    assert x_mm == pytest.approx((120 * 4032 / 480 - 2012.3) * 360.0 / 2763.7)
    assert y_mm == pytest.approx((480 * 3024 / 640 - 1499.1) * 360.0 / 2763.7)


def test_translate_click_with_camera_has_no_calibrated_range_limit():
    window = SimpleNamespace(
        image=SimpleNamespace(size=PHOTO_SIZE), portrait=SimpleNamespace(camera=CAMERA)
    )
    # 100 cm is outside the polynomial's 15-80 cm range; the pinhole model
    # still converts it.
    assert None not in MainWindow.translate_click_to_mm(window, 100.0, 10, 10)


@pytest.mark.parametrize(
    "window",
    [
        SimpleNamespace(image=SimpleNamespace(size=(2320, 3087))),
        SimpleNamespace(image=SimpleNamespace(size=(2320, 3087)), portrait=None),
        SimpleNamespace(
            image=SimpleNamespace(size=(2320, 3087)), portrait=SimpleNamespace(camera=None)
        ),
    ],
    ids=["no-portrait", "portrait-none", "legacy-camera-none"],
)
def test_translate_click_legacy_path_is_the_polynomial(window):
    result = MainWindow.translate_click_to_mm(window, 30.0, 300, 280)

    assert result == (
        pixel_to_mm(300 * 2320 / 480, 30.0, 2320),
        pixel_to_mm(280 * 3087 / 640, 30.0, 3087),
    )


# ------------------------------------------------------- vector, L-mode depth


def _capture_app_window(depth):
    window = MainWindow()
    window.image = Image.new("RGB", PHOTO_SIZE)
    window.depthmap = depth
    window.filtered_depthmap = median_filter_depthmap(depth, size=3)
    window.float_min_value = FLOAT_MIN
    window.float_max_value = FLOAT_MAX
    window.filename = "/tmp/capture.heic"
    window.portrait = SimpleNamespace(
        camera=CAMERA,
        depth_accuracy="absolute",
        teeth_bbox=None,
        teethmap=None,
        skinmap=None,
        hairmap=None,
        incisor_measurement=None,
    )
    return window


def test_vector_tool_on_l_mode_depth_uses_pinhole_maths(qapp):
    """Capture-app depth maps are 8-bit "L" (legacy ones are RGB); reading a
    pixel used to raise TypeError ('int' object is not subscriptable)."""
    raw = 160
    window = _capture_app_window(Image.new("L", (480, 640), raw))

    window._on_image_clicked(QPointF(100, 100))
    window._on_image_clicked(QPointF(100, 300))

    z_mm = _distance_cm(raw) * 10.0
    dy_px = (300 - 100) * 4032 / 640  # 1260 photo pixels, same depth
    expected = dy_px * z_mm / 2766.1

    entry = window.measurementPanel.history.entries[0]
    assert entry.linear_mm == pytest.approx(expected, rel=1e-9)
    # The surface profile goes through the library with the same camera; on
    # a flat depth map it matches the straight line.
    assert entry.surface_mm == pytest.approx(expected, rel=1e-6)
    window.close()


def test_vector_between_different_depths_uses_pinhole_maths(qapp):
    depth = Image.new("L", (480, 640), 150)
    depth.paste(200, (240, 0, 480, 640))  # right half closer
    window = _capture_app_window(depth)

    length = window.linear_length_between_two_clicks_or_none(100, 200, 380, 400)

    za = _distance_cm(150) * 10.0
    zb = _distance_cm(200) * 10.0
    ax = (100 * 3024 / 480 - 1499.1) * za / 2766.1
    ay = (200 * 4032 / 640 - 2019.6) * za / 2766.1
    bx = (380 * 3024 / 480 - 1499.1) * zb / 2766.1
    by = (400 * 4032 / 640 - 2019.6) * zb / 2766.1
    assert length == pytest.approx(math.dist((ax, ay, za), (bx, by, zb)), rel=1e-9)
    assert window.get_depthmap_value(100, 200) == 150
    window.close()


def test_rgb_and_l_depth_give_the_same_legacy_measurement(qapp):
    rgb = MainWindow()
    lum = MainWindow()
    for window, depth in (
        (rgb, Image.new("RGB", (480, 640), (128, 128, 128))),
        (lum, Image.new("L", (480, 640), 128)),
    ):
        window.image = Image.new("RGB", (480, 640))
        window.depthmap = depth
        window.filtered_depthmap = median_filter_depthmap(depth, size=3)
        window.float_min_value = 0.5
        window.float_max_value = 5.0

    assert rgb.get_depthmap_value(10, 10) == lum.get_depthmap_value(10, 10) == 128
    assert rgb.linear_length_between_two_clicks_or_none(
        100, 100, 100, 300
    ) == lum.linear_length_between_two_clicks_or_none(100, 100, 100, 300)
    rgb.close()
    lum.close()


# ------------------------------------------------------------- region engine


def test_region_engine_uses_camera_for_3d_points():
    engine = RegionMeasurementEngine(
        filtered_depthmap=Image.new("L", (480, 640), 100),
        display_size=(480, 640),
        image_size=PHOTO_SIZE,
        depth_to_cm=lambda value: value,
        pixels_per_mm=lambda value: pytest.fail("polynomial must not be used"),
        surface_length=lambda start, end: None,
        camera=CAMERA,
    )

    x_mm, y_mm, z_mm = engine._to_3d(np.array([300.0]), np.array([280.0]), np.array([35.0]))

    assert z_mm[0] == pytest.approx(350.0)
    assert x_mm[0] == pytest.approx((1890 - 1499.1) * 350.0 / 2766.1)
    assert y_mm[0] == pytest.approx((1764 - 2019.6) * 350.0 / 2766.1)


# ---------------------------------------------------------------- neck arc


def _neck_arc_inputs():
    # Same synthetic head-and-neck as tests/test_neck_arc.py.
    width, height = 900, 1200
    img = Image.new("L", (width, height), 90)
    pixels = img.load()
    centre = width // 2
    for y in range(height):
        for x in range(width):
            if y < 500:
                dx = (x - centre) / 260.0
                dy = (y - 300) / 260.0
                r2 = dx * dx + dy * dy
                if r2 < 1.0:
                    pixels[x, y] = int(215 - 18 * r2)
            else:
                dx = (x - centre) / 150.0
                if abs(dx) < 1.0:
                    pixels[x, y] = int(196 - 20 * dx * dx)
    jaw = [(centre - 200 + i * 40, 430 + int(70 * np.sin(np.pi * i / 10))) for i in range(11)]
    return img, dict(
        jaw_points=jaw,
        chin_xy=(centre, 500),
        nose_xy=(centre, 330),
        lower_lip_xy=(centre, 440),
        photo_width=width,
        photo_height=height,
        float_min=0.5,
        float_max=4.0,
    )


def test_neck_arc_uses_camera_focal_length_and_principal_point():
    img, kwargs = _neck_arc_inputs()
    # Focal length of the same order as the polynomial gives at this distance
    # (~2780 px), so the synthetic neck stays within the anatomical limits.
    camera = CameraModel(fx=2700.0, fy=2720.0, cx=440.0, cy=610.0)

    result = compute_neck_arc(img, camera=camera, **kwargs)

    assert result is not None
    chin_z_mm = result.debug["chin_z_mm"]
    assert result.debug["ppmm"] == pytest.approx(2700.0 / chin_z_mm)
    assert result.raw_width_mm == pytest.approx(
        (result.right_x - result.left_x) * chin_z_mm / 2700.0
    )
    (px, py), (x_mm, y_mm, z_mm) = result.arc_points_photo[0], result.arc_points_3d[0]
    assert x_mm == pytest.approx((px - 440.0) * z_mm / 2700.0)
    assert y_mm == pytest.approx((py - 610.0) * z_mm / 2720.0)


def test_neck_arc_without_camera_is_unchanged():
    img, kwargs = _neck_arc_inputs()
    assert compute_neck_arc(img, camera=None, **kwargs) == compute_neck_arc(img, **kwargs)


# ---------------------------------------------- library calls get the camera


def test_detection_passes_camera_to_dual_mask_detector(qapp, monkeypatch):
    window = MainWindow()
    window.image = Image.new("RGB", PHOTO_SIZE)
    window.depthmap = Image.new("L", (480, 640), 100)
    window.filtered_depthmap = median_filter_depthmap(window.depthmap, size=3)
    window.float_min_value = FLOAT_MIN
    window.float_max_value = FLOAT_MAX
    window.portrait = SimpleNamespace(
        camera=CAMERA,
        depth_accuracy="absolute",
        skinmap=Image.new("L", PHOTO_SIZE),
        hairmap=None,
        depthmap=window.depthmap,
        teethmap=None,
        teeth_bbox=None,
        incisor_measurement=None,
        floatValueMin=FLOAT_MIN,
        floatValueMax=FLOAT_MAX,
    )
    seen = {}

    def fake_dual_mask(*args, **kwargs):
        seen.update(kwargs)
        return None, None

    monkeypatch.setattr(
        "portrait_analyser.extended_neck.detect_neck_midpoint_from_dual_mask", fake_dual_mask
    )
    monkeypatch.setattr(
        "portrait_analyser.extended_neck.detect_neck_midpoint_from_segmentation",
        lambda image: (None, None),
    )
    monkeypatch.setattr(window, "critical_error", lambda err: None)

    window._run_detection_and_update(force_extended=True)

    assert seen["camera"] is CAMERA
    assert "Depth: TrueDepth absolute" in window.measurementPanel.autoLabel.text()
    window.close()


def test_mouth_measurement_gets_the_camera(qapp, monkeypatch):
    window = MainWindow()
    window.image = Image.new("RGB", PHOTO_SIZE)
    window.portrait = SimpleNamespace(
        camera=CAMERA,
        incisor_measurement=None,
        depthmap=Image.new("L", (480, 640), 100),
        floatValueMin=FLOAT_MIN,
        floatValueMax=FLOAT_MAX,
    )
    window.neck_midpoint = SimpleNamespace(pose=PortraitPose.OPEN_MOUTH)
    window.face_mesh_debug = SimpleNamespace(landmarks=[(0, 0)])
    window.mouthOpeningAction.setChecked(True)
    seen = {}

    def fake_mouth(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr("fidmaa_gui.app.compute_mouth_measurement_from_facemesh", fake_mouth)

    window._compute_mouth_measurement()

    assert seen["camera"] is CAMERA
    window.close()


# ----------------------------------------------------------- depth source


def test_portrait_camera_tolerates_missing_attributes():
    assert portrait_camera(SimpleNamespace()) is None
    assert portrait_camera(SimpleNamespace(portrait=None)) is None
    assert portrait_camera(SimpleNamespace(portrait=SimpleNamespace())) is None
    assert portrait_camera(SimpleNamespace(portrait=SimpleNamespace(camera=CAMERA))) is CAMERA


@pytest.mark.parametrize(
    "portrait, line, warning",
    [
        (
            SimpleNamespace(depth_accuracy="absolute", camera=CAMERA),
            "Depth: TrueDepth absolute",
            None,
        ),
        (SimpleNamespace(depth_accuracy="absolute", camera=None), "Depth: absolute", None),
        (
            SimpleNamespace(depth_accuracy="relative", camera=None),
            "Depth: relative (Camera app)",
            RELATIVE_DEPTH_WARNING,
        ),
        (SimpleNamespace(depth_accuracy=None, camera=None), "Depth: unknown", None),
        (SimpleNamespace(), "Depth: unknown", None),
        (None, "Depth: unknown", None),
    ],
)
def test_depth_source_status(portrait, line, warning):
    assert depth_source_status(portrait) == (line, warning)


def test_relative_depth_warning_mentions_the_error_range():
    assert "7–28 %" in RELATIVE_DEPTH_WARNING


def test_combine_warnings():
    assert combine_warnings(None, "") is None
    assert combine_warnings("a", None) == "a"
    assert combine_warnings("a", "b") == "a\n⚠ b"


def _neutral_window(depth_accuracy):
    window = MainWindow()
    window.ui.showNeckArcCheckBox.setChecked(False)
    window.ui.showCentroidsCheckBox.setChecked(False)
    window.filename = "/tmp/patient.heic"
    window.portrait = SimpleNamespace(
        teeth_bbox=None,
        teethmap=None,
        incisor_measurement=None,
        skinmap=None,
        depth_accuracy=depth_accuracy,
        camera=None,
    )
    window.neck_midpoint = SimpleNamespace(pose=PortraitPose.NEUTRAL_NECK, x=None)
    window.neck_arc = SimpleNamespace(circumference_mm=364.0)
    window.neck_measurement_auto = SimpleNamespace(circumference_mm=380.0)
    window.neck_confidence_ratio = 0.5
    window.mouth_measurement = None
    window.segmentation_debug = None
    return window


def test_relative_depth_shows_indicator_and_warning(qapp):
    window = _neutral_window("relative")
    window._update_automatic_results()

    assert "Depth: relative (Camera app)" in window.measurementPanel.autoLabel.text()
    assert not window.measurementPanel.autoWarningLabel.isHidden()
    assert "7–28 %" in window.measurementPanel.autoWarningLabel.text()
    window.close()


def test_relative_depth_warning_is_combined_with_neck_warning(qapp):
    window = _neutral_window("relative")
    window.neck_confidence_ratio = 5.0
    window._update_automatic_results()

    warning = window.measurementPanel.autoWarningLabel.text()
    assert "low confidence" in warning
    assert "7–28 %" in warning
    window.close()


def test_unknown_depth_accuracy_shows_unknown_without_warning(qapp):
    window = _neutral_window(None)
    window._update_automatic_results()

    assert "Depth: unknown" in window.measurementPanel.autoLabel.text()
    assert window.measurementPanel.autoWarningLabel.isHidden()
    window.close()


def test_no_face_keeps_force_hint_and_adds_depth_warning(qapp):
    window = _neutral_window("relative")
    window.neck_midpoint = None
    window._update_automatic_results()

    assert "Depth: relative" in window.measurementPanel.autoLabel.text()
    warning = window.measurementPanel.autoWarningLabel.text()
    assert "Force extended neck analysis" in warning
    assert "7–28 %" in warning
    window.close()


# ----------------------------------------------------------- loading errors


def test_no_depth_error_dialog_shows_library_reason(qapp, monkeypatch):
    window = MainWindow()
    shown = []

    def raise_no_depth(*args, **kwargs):
        raise NoDepthMapFound("pyheif cannot decode <16-bit> depth; needs macOS")

    monkeypatch.setattr("fidmaa_gui.app.load_image", raise_no_depth)
    monkeypatch.setattr(window, "critical_error", shown.append)

    window._loadImage("/tmp/capture.heic")

    assert len(shown) == 1
    assert shown[0].startswith(errors.NO_DEPTH_DATA_ERROR)
    assert "pyheif cannot decode &lt;16-bit&gt; depth; needs macOS" in shown[0]
    assert window.filename is None
    window.close()


def test_no_depth_error_without_message_is_the_generic_text():
    assert errors.no_depth_data_error(NoDepthMapFound()) == errors.NO_DEPTH_DATA_ERROR
