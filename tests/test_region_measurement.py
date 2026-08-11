import math

import numpy as np
import pytest
from PIL import Image

from fidmaa_gui.region_measurement import (
    CandidatePoint,
    MeasurementError,
    Region,
    RegionMask,
    RegionMeasurementEngine,
    RegionMeasurementResult,
    SelectionMode,
    VectorSample,
    pair_candidates,
    spatially_distributed_points,
)


def make_engine(depth, *, teethmap=None, surface_length=None):
    depth_image = Image.fromarray(np.asarray(depth, dtype=np.uint8), mode="L")
    if surface_length is None:

        def surface_length(start, end):
            return math.dist(start, end)

    return RegionMeasurementEngine(
        filtered_depthmap=depth_image,
        display_size=depth_image.size,
        image_size=depth_image.size,
        depth_to_cm=lambda value: value,
        pixels_per_mm=lambda value: np.ones_like(value, dtype=float),
        surface_length=surface_length,
        teethmap=teethmap,
    )


def test_local_peak_removes_pose_tilt_before_selecting_chin_bump():
    y, x = np.indices((41, 41), dtype=np.float64)
    depth = 110.0 + 0.65 * x + 0.15 * y
    peak_x, peak_y = 25, 18
    depth -= 13.0 * np.exp(-((x - peak_x) ** 2 + (y - peak_y) ** 2) / 18.0)
    engine = make_engine(np.rint(depth))

    candidates = engine.select_candidates(
        Region(0, 0, 41, 41),
        mode=SelectionMode.LOCAL_PEAK,
        mask=RegionMask.NONE,
        percentile=10,
        count=5,
    )

    assert np.mean([point.x for point in candidates]) == pytest.approx(peak_x, abs=4)
    assert np.mean([point.y for point in candidates]) == pytest.approx(peak_y, abs=4)
    assert min(point.x for point in candidates) > 10


def test_local_valley_ignores_tilt_and_protruding_bone_below_notch():
    y, x = np.indices((41, 41), dtype=np.float64)
    depth = 110.0 + 0.55 * x + 0.12 * y
    valley_x, valley_y = 17, 19
    depth += 12.0 * np.exp(-((x - valley_x) ** 2 + (y - valley_y) ** 2) / 16.0)
    depth[y > 32] -= 8.0
    engine = make_engine(np.rint(depth))

    candidates = engine.select_candidates(
        Region(0, 0, 41, 41),
        mode=SelectionMode.LOCAL_VALLEY,
        mask=RegionMask.NONE,
        percentile=10,
        count=5,
    )

    assert np.mean([point.x for point in candidates]) == pytest.approx(valley_x, abs=4)
    assert np.mean([point.y for point in candidates]) == pytest.approx(valley_y, abs=4)
    assert max(point.y for point in candidates) < 32


def test_local_peak_and_valley_return_requested_candidate_count():
    depth = np.full((20, 20), 50, dtype=np.uint8)
    engine = make_engine(depth)
    region = Region(0, 0, 20, 20)

    peak = engine.select_candidates(
        region,
        mode=SelectionMode.LOCAL_PEAK,
        mask=RegionMask.NONE,
        percentile=10,
        count=5,
    )
    valley = engine.select_candidates(
        region,
        mode=SelectionMode.LOCAL_VALLEY,
        mask=RegionMask.NONE,
        percentile=10,
        count=5,
    )

    assert len(peak) == 5
    assert len(valley) == 5


def test_highest_and_lowest_select_absolute_depth_extremes():
    depth = np.full((21, 21), 50, dtype=np.uint8)
    depth[:, :8] = 20
    depth[:, 13:] = 80
    engine = make_engine(depth)
    region = Region(0, 0, 21, 21)

    highest = engine.select_candidates(
        region,
        mode=SelectionMode.HIGHEST,
        mask=RegionMask.NONE,
        percentile=10,
        count=5,
    )
    lowest = engine.select_candidates(
        region,
        mode=SelectionMode.LOWEST,
        mask=RegionMask.NONE,
        percentile=10,
        count=5,
    )

    assert all(point.x < 8 for point in highest)
    assert all(point.x >= 13 for point in lowest)


def test_bounding_box_corners_are_excluded_by_circular_mask():
    depth = np.full((11, 11), 50, dtype=np.uint8)
    depth[0, 0] = 1
    engine = make_engine(depth)

    candidates = engine.select_candidates(
        Region(0, 0, 11, 11),
        mode=SelectionMode.HIGHEST,
        mask=RegionMask.NONE,
        percentile=10,
        count=5,
    )

    assert all((point.x, point.y) != (0, 0) for point in candidates)


def test_teeth_mask_restricts_candidates_before_ranking():
    depth = np.tile(np.arange(20, 30, dtype=np.uint8), (10, 1))
    teeth = Image.new("L", (10, 10), 0)
    for y in range(10):
        for x in range(5):
            teeth.putpixel((x, y), 255)
    engine = make_engine(depth, teethmap=teeth)

    candidates = engine.select_candidates(
        Region(0, 0, 10, 10),
        mode=SelectionMode.LOWEST,
        mask=RegionMask.TEETH,
        percentile=10,
        count=5,
    )

    assert all(point.x < 5 for point in candidates)


def test_missing_requested_teeth_mask_is_an_explicit_error():
    engine = make_engine(np.full((10, 10), 50))

    with pytest.raises(MeasurementError, match="no teeth mask"):
        engine.select_candidates(
            Region(0, 0, 10, 10),
            mode=SelectionMode.FLATTEST,
            mask=RegionMask.TEETH,
            percentile=10,
            count=5,
        )


def test_flattest_prefers_planar_surface_over_rough_surface():
    depth = np.full((20, 20), 50, dtype=np.uint8)
    checker = np.indices((20, 10)).sum(axis=0) % 2
    depth[:, 10:] = np.where(checker, 45, 55)
    engine = make_engine(depth)

    candidates = engine.select_candidates(
        Region(0, 0, 20, 20),
        mode=SelectionMode.FLATTEST,
        mask=RegionMask.NONE,
        percentile=10,
        count=5,
    )

    assert all(point.x < 10 for point in candidates)


def test_flattest_discards_extreme_depth_band_before_plane_fit():
    depth = np.full((20, 20), 50, dtype=np.uint8)
    depth[:5, :5] = 10  # perfectly flat, but outside the central depth band
    engine = make_engine(depth)

    candidates = engine.select_candidates(
        Region(0, 0, 20, 20),
        mode=SelectionMode.FLATTEST,
        mask=RegionMask.NONE,
        percentile=10,
        count=5,
    )

    assert all(not (point.x < 5 and point.y < 5) for point in candidates)


def make_point(x, y, z=100.0, score=0.0):
    return CandidatePoint(x, y, (float(x), float(y), z), score)


def test_spatial_selection_is_deterministic_and_dispersed():
    points = [make_point(x, 0, score=x) for x in range(20)]

    first = spatially_distributed_points(points, 5)
    second = spatially_distributed_points(points, 5)

    assert first == second
    assert first[0].x == 0
    assert first[1].x == 19
    assert len(first) == 5


def test_pairing_uses_cross_profile_order_without_crossing():
    region_a = Region(0, 0, 10, 10)
    region_b = Region(0, 20, 10, 30)
    points_a = [make_point(8, 2), make_point(2, 2), make_point(5, 2)]
    points_b = [make_point(5, 22), make_point(2, 22), make_point(8, 22)]

    pairs = pair_candidates(points_a, points_b, region_a, region_b)

    assert {(a.x, b.x) for a, b in pairs} == {(2, 2), (5, 5), (8, 8)}


def test_result_reports_separate_linear_and_surface_sample_statistics():
    starts = [make_point(index, 0) for index in range(5)]
    ends = [make_point(index, 10) for index in range(5)]
    samples = tuple(
        VectorSample(start, end, 10.0 + index, None if index == 0 else 20.0)
        for index, (start, end) in enumerate(zip(starts, ends, strict=True))
    )
    result = RegionMeasurementResult(tuple(starts), tuple(ends), samples)

    linear = result.linear_stats()

    assert linear.mean_mm == pytest.approx(12.0)
    assert linear.sample_sd_mm == pytest.approx(np.std([10, 11, 12, 13, 14], ddof=1))
    assert linear.count == 5
    assert result.surface_stats() is None


def test_measure_uses_the_same_pairs_for_linear_and_surface_lengths():
    engine = make_engine(
        np.full((30, 20), 50),
        surface_length=lambda start, end: math.dist(start, end) + 5,
    )

    result = engine.measure(
        Region(0, 0, 20, 10),
        Region(0, 20, 20, 30),
        mode_a=SelectionMode.HIGHEST,
        mode_b=SelectionMode.HIGHEST,
        percentile=10,
        vector_count=5,
    )

    assert len(result.samples) == 5
    for sample in result.samples:
        assert sample.surface_mm == pytest.approx(
            math.dist(
                (sample.start.x, sample.start.y),
                (sample.end.x, sample.end.y),
            )
            + 5
        )
