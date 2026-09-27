"""Capture-app photos are measured on full-precision float depth (``portrait.depth``).

Legacy (8-bit Camera-app) photos must give exactly the numbers they always
did, whether or not the portrait carries a ``LegacyDepthMap``.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from portrait_analyser import CameraModel
from portrait_analyser.depth_sampling import median_filter_depthmap
from PySide6.QtCore import QPointF

from fidmaa_gui.app import MainWindow
from fidmaa_gui.depth_source import depth_source_status, float_depth
from fidmaa_gui.measurement_tools import MeasurementTool
from fidmaa_gui.neck_arc import compute_neck_arc, depth_band_mm
from fidmaa_gui.region_measurement import Region, RegionMeasurementEngine

depth_map = pytest.importorskip("portrait_analyser.depth_map")
FloatDepthMap = depth_map.FloatDepthMap
LegacyDepthMap = depth_map.LegacyDepthMap

PHOTO_SIZE = (3024, 4032)
CAMERA = CameraModel(fx=2766.1, fy=2766.1, cx=1499.1, cy=2019.6)
REPAIRED = "reciprocal (depth stored as disparity)"

# Not representable by an 8-bit disparity code: the float path must keep them.
NEAR_M = 0.35123
FAR_M = 0.36871
HOLE = (slice(300, 311), slice(100, 111))  # rows, cols of the native 480x640 map


def _float_depth_m():
    depth_m = np.full((640, 480), NEAR_M)
    depth_m[:, 240:] = FAR_M
    depth_m[HOLE] = np.nan
    return depth_m


def _photo(x, y):
    return x * PHOTO_SIZE[0] / 480, y * PHOTO_SIZE[1] / 640


def _point_mm(display_x, display_y, z_m):
    photo_x, photo_y = _photo(display_x, display_y)
    z_mm = z_m * 1000.0
    return (
        (photo_x - CAMERA.cx) * z_mm / CAMERA.fx,
        (photo_y - CAMERA.cy) * z_mm / CAMERA.fy,
        z_mm,
    )


def _float_window(depth_m=None, **portrait_attrs):
    depth = FloatDepthMap(_float_depth_m() if depth_m is None else depth_m, PHOTO_SIZE)
    display, float_min, float_max = depth.display_encoding()
    window = MainWindow()
    window.image = Image.new("RGB", PHOTO_SIZE)
    window.depthmap = display
    window.filtered_depthmap = median_filter_depthmap(display, size=3)
    window.float_min_value = float_min
    window.float_max_value = float_max
    window.filename = "/tmp/capture.heic"
    attrs = dict(
        depth=depth,
        camera=CAMERA,
        depth_accuracy="absolute",
        depth_plausible=True,
        depth_repaired=None,
        depth_code_zero_is_invalid=True,
        teeth_bbox=None,
        teethmap=None,
        skinmap=None,
        hairmap=None,
        incisor_measurement=None,
    )
    attrs.update(portrait_attrs)
    window.portrait = SimpleNamespace(**attrs)
    return window


# ------------------------------------------------------------ point reads


def test_float_depth_is_only_picked_for_float_maps():
    assert float_depth(None) is None
    assert float_depth(SimpleNamespace()) is None
    legacy = LegacyDepthMap(Image.new("L", (4, 4), 100), 0.5, 2.0, (4, 4))
    assert float_depth(SimpleNamespace(depth=legacy)) is None
    floaty = FloatDepthMap(np.full((4, 4), 0.4), (4, 4))
    assert float_depth(SimpleNamespace(depth=floaty)) is floaty


def test_click_distance_reads_the_float_map(qapp):
    window = _float_window()
    assert window.distance_for_click(50, 500) == pytest.approx(NEAR_M * 100, rel=1e-12)
    assert window.distance_for_click(400, 100) == pytest.approx(FAR_M * 100, rel=1e-12)
    # The hole: NaN in the float map -> no depth, never a made-up distance.
    assert window.distance_for_click(105 * 480 / 479, 305 * 640 / 639) is None
    window.close()


def test_vector_uses_full_precision_float_depth(qapp):
    window = _float_window()
    window._on_image_clicked(QPointF(100, 200))
    window._on_image_clicked(QPointF(380, 400))

    expected = math.dist(_point_mm(100, 200, NEAR_M), _point_mm(380, 400, FAR_M))
    entry = window.measurementPanel.history.entries[0]
    assert entry.linear_mm == pytest.approx(expected, rel=1e-12)
    details = window.measurementPanel.detailsLabel.text()
    assert f"Depth A: {NEAR_M * 100:.1f} cm · B: {FAR_M * 100:.1f} cm" in details
    assert f"Δ depth: {(FAR_M - NEAR_M) * 1000:+.1f} mm" in details
    window.close()


def test_float_measurements_ignore_the_display_image(qapp):
    window = _float_window()
    before = (
        window.linear_length_between_two_clicks_or_none(100, 200, 100, 500),
        window.surface_vector_filtered(100, 200, 100, 500, 3),
    )
    # Garbage display encoding: a measurement reading it would change.
    window.depthmap = Image.new("L", (480, 640), 7)
    window.filtered_depthmap = Image.new("L", (480, 640), 250)
    window.float_min_value, window.float_max_value = 0.01, 0.02
    after = (
        window.linear_length_between_two_clicks_or_none(100, 200, 100, 500),
        window.surface_vector_filtered(100, 200, 100, 500, 3),
    )
    assert before == after
    window.close()


def test_surface_on_flat_float_depth_matches_the_straight_line(qapp):
    window = _float_window()
    expected = math.dist(_point_mm(60, 100, NEAR_M), _point_mm(60, 600, NEAR_M))
    assert window.linear_length_between_two_clicks_or_none(60, 100, 60, 600) == pytest.approx(
        expected, rel=1e-12
    )
    assert window.surface_vector_filtered(60, 100, 60, 600, 3) == pytest.approx(expected, rel=1e-9)
    window.close()


def test_vector_touching_a_float_hole_is_na(qapp):
    # A single missing pixel: the median filter would fill it, the point read not.
    depth_m = np.full((640, 480), NEAR_M)
    depth_m[320, 200] = np.nan
    window = _float_window(depth_m)
    display_x, display_y = 200 * 480 / 479, 320 * 640 / 639
    assert window.distance_for_click(display_x, display_y) is None

    window._on_image_clicked(QPointF(100, 100))
    window._on_image_clicked(QPointF(display_x, display_y))

    assert window.measurementPanel.linearValueLabel.text().startswith("n/a")
    assert "Δ depth: n/a" in window.measurementPanel.detailsLabel.text()
    window.close()


def test_surface_across_a_float_hole_is_na(qapp):
    window = _float_window()
    # Straight through the 11x11 hole (the median filter keeps its middle NaN).
    assert window.surface_vector_filtered(60, 305, 150, 305, 1) is None
    window.close()


def test_polyline_uses_float_depth(qapp):
    window = _float_window()
    window.set_tool(MeasurementTool.POLYLINE)
    points = [(60, 100), (60, 300), (160, 450)]
    for point in points:
        window._on_image_clicked(QPointF(*point))
    window._on_image_right_clicked(QPointF(*points[-1]))

    expected = sum(
        math.dist(_point_mm(*a, NEAR_M), _point_mm(*b, NEAR_M))
        for a, b in zip(points, points[1:], strict=False)
    )
    entry = window.measurementPanel.history.entries[-1]
    assert entry.linear_mm == pytest.approx(expected, rel=1e-12)
    window.close()


def test_zoom_readout_shows_centimetres_for_float_depth(qapp):
    window = _float_window()
    assert window.depth_readout_at_native(50, 50) == f"{NEAR_M * 100:.1f} cm"
    assert window.depth_readout_at_native(105, 305) == "n/a"
    assert window.depth_readout_at_native(-1, 5) == "n/a"
    window.close()


def test_zoom_readout_keeps_raw_codes_for_legacy(qapp):
    window = MainWindow()
    window.portrait = SimpleNamespace(camera=None)
    assert window.depth_readout_at_native(5, 5) is None
    window.close()


# ------------------------------------------------------------- region engine


def test_region_engine_samples_float_centimetres():
    depth = FloatDepthMap(_float_depth_m(), PHOTO_SIZE).median_filtered()
    engine = RegionMeasurementEngine(
        filtered_depthmap=depth,
        display_size=(480, 640),
        image_size=PHOTO_SIZE,
        depth_to_cm=lambda value: pytest.fail("float depth is already in cm"),
        pixels_per_mm=lambda value: pytest.fail("camera given"),
        surface_length=lambda start, end: None,
        camera=CAMERA,
    )
    x, y, cm = engine._sample_region(Region(20, 20, 60, 60))
    assert np.allclose(cm, NEAR_M * 100, rtol=1e-12)
    _, _, holes = engine._sample_region(Region(95, 295, 115, 315))
    assert np.isnan(holes).any()
    assert np.isfinite(holes).any()


def test_region_tool_on_float_depth(qapp):
    window = _float_window()
    window.set_tool(MeasurementTool.REGION)
    window.region_a = Region(40, 80, 100, 140)
    window.region_b = Region(40, 480, 100, 540)
    window._calculate_region_measurement()

    result = window.region_measurement_result
    assert result is not None
    for sample in result.samples:
        start = _point_mm(sample.start.x, sample.start.y, NEAR_M)
        end = _point_mm(sample.end.x, sample.end.y, NEAR_M)
        assert sample.linear_mm == pytest.approx(math.dist(start, end), rel=1e-9)
    assert "Linear 3D:" in window.regionResultEdit.toPlainText()
    window.close()


# ------------------------------------------------------------------ neck arc


def test_neck_band_reads_float_depth_in_mm():
    depth_m = np.full((64, 48), 0.4)
    depth_m[30, 20] = np.nan
    depth = FloatDepthMap(depth_m, (480, 640))
    band = depth_band_mm(None, 480, 640, None, None, 0, 640, depth=depth)
    assert band.shape == (640, 480)
    assert np.nanmax(np.abs(band - 400.0)) < 1e-9
    # Photo pixels interpolating the hole are invalid, the rest are not.
    photo_x, photo_y = 20 * 479 / 47, 30 * 639 / 63
    assert np.isnan(band[round(photo_y), round(photo_x)])
    assert np.isfinite(band[0, 0])
    far = depth_band_mm(None, 480, 640, None, None, 0, 640, depth=depth, background_cutoff_mm=300)
    assert np.isnan(far).all()


def _synthetic_neck_codes():
    # Same synthetic head-and-neck as tests/test_neck_arc.py.
    width, height = 900, 1200
    codes = np.full((height, width), 90.0)
    yy, xx = np.mgrid[0:height, 0:width]
    centre = width // 2
    r2 = ((xx - centre) / 260.0) ** 2 + ((yy - 300) / 260.0) ** 2
    head = (yy < 500) & (r2 < 1.0)
    codes[head] = np.floor(215 - 18 * r2[head])
    dx = (xx - centre) / 150.0
    neck = (yy >= 500) & (np.abs(dx) < 1.0)
    codes[neck] = np.floor(196 - 20 * dx[neck] ** 2)
    jaw = [(centre - 200 + i * 40, 430 + int(70 * np.sin(np.pi * i / 10))) for i in range(11)]
    kwargs = dict(
        jaw_points=jaw,
        chin_xy=(centre, 500),
        nose_xy=(centre, 330),
        lower_lip_xy=(centre, 440),
        photo_width=width,
        photo_height=height,
    )
    return codes, kwargs


def test_neck_arc_on_float_depth_matches_the_8bit_result():
    codes, kwargs = _synthetic_neck_codes()
    float_min, float_max = 0.5, 4.0
    image = Image.fromarray(codes.astype(np.uint8), mode="L")
    depth_m = 1.0 / (float_max * codes / 255.0 + float_min * (1.0 - codes / 255.0))
    depth = FloatDepthMap(depth_m, (kwargs["photo_width"], kwargs["photo_height"]))
    camera = CameraModel(fx=2700.0, fy=2720.0, cx=440.0, cy=610.0)

    legacy = compute_neck_arc(
        image, float_min=float_min, float_max=float_max, camera=camera, **kwargs
    )
    result = compute_neck_arc(
        None, float_min=None, float_max=None, camera=camera, depth=depth, **kwargs
    )

    assert legacy is not None and result is not None
    assert result.circumference_mm == pytest.approx(legacy.circumference_mm, rel=0.02)
    assert result.measuring_y == pytest.approx(legacy.measuring_y, abs=3)


# ------------------------------------------------------------- depth label


def test_repaired_depth_status_line_and_warning():
    line, warning = depth_source_status(
        SimpleNamespace(
            depth_accuracy="absolute",
            camera=CAMERA,
            depth_plausible=True,
            depth_repaired=REPAIRED,
        )
    )
    assert line == "Depth: TrueDepth absolute (repaired: depth stored as disparity)"
    assert "repaired" in warning and "depth stored as disparity" in warning


def test_repaired_depth_is_shown_and_still_measured(qapp):
    window = _float_window(depth_repaired=REPAIRED)
    window.neck_midpoint = None
    window._update_automatic_results()

    assert (
        "Depth: TrueDepth absolute (repaired: depth stored as disparity)"
        in window.measurementPanel.autoLabel.text()
    )
    assert not window.measurementPanel.autoWarningLabel.isHidden()
    assert "repaired" in window.measurementPanel.autoWarningLabel.text()

    window._on_image_clicked(QPointF(100, 200))
    window._on_image_clicked(QPointF(100, 400))
    assert window.measurementPanel.history.entries[0].linear_mm is not None
    assert "repaired" in window._diagnostics_text()
    window.close()


# ----------------------------------------------- legacy stays byte-identical


def _legacy_depthmap():
    rng = np.random.default_rng(1234)
    codes = np.clip(
        150 + 40 * np.sin(np.mgrid[0:640, 0:480][1] / 37.0) + rng.integers(-6, 7, (640, 480)),
        1,
        255,
    ).astype(np.uint8)
    return Image.fromarray(np.stack([codes] * 3, axis=-1), mode="RGB")


def _legacy_window(with_depth):
    depthmap = _legacy_depthmap()
    window = MainWindow()
    window.image = Image.new("RGB", (2320, 3087))
    window.depthmap = depthmap
    window.filtered_depthmap = median_filter_depthmap(depthmap, size=3)
    window.float_min_value = 0.2734
    window.float_max_value = 3.1896
    window.filename = "/tmp/legacy.heic"
    attrs = dict(
        camera=None,
        depth_accuracy="relative",
        teeth_bbox=None,
        teethmap=None,
        skinmap=None,
        hairmap=None,
        incisor_measurement=None,
    )
    if with_depth:
        attrs["depth"] = LegacyDepthMap(
            depthmap, window.float_min_value, window.float_max_value, window.image.size
        )
    window.portrait = SimpleNamespace(**attrs)
    return window


def _legacy_scenario(window):
    pairs = [(200, 250, 280, 250), (240, 150, 240, 400), (10, 10, 470, 630)]
    values = [window.distance_for_click(x, y) for x in range(0, 480, 37) for y in range(0, 640, 41)]
    for pair in pairs:
        values.append(window.linear_length_between_two_clicks_or_none(*pair))
        values.extend(window.surface_vector_filtered(*pair, step) for step in (1, 3, 5))
    window._on_image_clicked(QPointF(200, 260))
    window._on_image_clicked(QPointF(280, 300))
    values.append(window.measurementPanel.detailsLabel.text())
    window.set_tool(MeasurementTool.REGION)
    window.region_a = Region(190, 230, 250, 290)
    window.region_b = Region(250, 300, 310, 360)
    window._calculate_region_measurement()
    values.append(window.regionResultEdit.toPlainText())
    values.append(repr(window.region_measurement_result))
    values.append(window.measurementPanel.history.entries)
    return values


def test_legacy_depth_map_gives_byte_identical_results(qapp):
    old = _legacy_window(with_depth=False)
    new = _legacy_window(with_depth=True)
    assert _legacy_scenario(new) == _legacy_scenario(old)
    old.close()
    new.close()


def test_legacy_neck_band_via_depth_map_is_identical():
    filtered = median_filter_depthmap(_legacy_depthmap(), size=3)
    depth = LegacyDepthMap(filtered, 0.2734, 3.1896, (2320, 3087))
    positional = depth_band_mm(filtered, 2320, 3087, 0.2734, 3.1896, 1000, 1400)
    via_depth = depth_band_mm(None, 2320, 3087, None, None, 1000, 1400, depth=depth)
    assert np.array_equal(positional, via_depth, equal_nan=True)


def test_3d_view_gets_the_float_maps_display_encoding(qapp, monkeypatch):
    window = _float_window()
    # A stale 8-bit image must not reach the viewer.
    window.depthmap = Image.new("L", (480, 640), 7)
    seen = {}

    class FakeProcess:
        def __init__(self, target, args):
            seen["args"] = args

        def start(self):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr("multiprocessing.Process", FakeProcess)
    window.open3DView()

    image, float_min, float_max = window.portrait.depth.display_encoding()
    _, depthmap, seen_min, seen_max = seen["args"]
    assert depthmap.tobytes() == image.convert("L").tobytes()
    assert (seen_min, seen_max) == (float_min, float_max)
    window.close()
