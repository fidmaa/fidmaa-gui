import numpy as np
from PIL import Image

from fidmaa_gui.depth_visualization import (
    render_depth_contour_overlay,
    render_depth_focus_visualization,
    render_depth_visualization,
)


def test_enhanced_color_separates_adjacent_high_depth_values():
    depth = Image.fromarray(
        np.array(
            [
                [241, 242, 243, 244],
                [241, 242, 243, 244],
                [241, 242, 243, 244],
            ],
            dtype=np.uint8,
        ),
        mode="L",
    )

    visualization = render_depth_visualization(depth)
    colors = np.asarray(visualization.image)[1]

    assert visualization.low_raw == 241
    assert visualization.high_raw == 244
    assert len({tuple(color) for color in colors}) == 4


def test_invalid_zero_depth_remains_black_and_is_excluded_from_range():
    depth = Image.fromarray(np.array([[0, 241, 244]], dtype=np.uint8), mode="L")

    visualization = render_depth_visualization(depth)

    assert tuple(np.asarray(visualization.image)[0, 0]) == (0, 0, 0)
    assert visualization.low_raw == 241
    assert visualization.high_raw == 244


def test_cursor_depth_bands_color_only_nearby_exact_levels():
    depth = Image.fromarray(
        np.array([[99, 100, 101, 102, 103, 104, 105]], dtype=np.uint8),
        mode="L",
    )

    visualization = render_depth_focus_visualization(depth, 102, radius=2)
    colors = np.asarray(visualization.image)[0]

    assert visualization.low_raw == 100
    assert visualization.high_raw == 104
    assert tuple(colors[3]) == (255, 255, 255)
    assert len({tuple(color) for color in colors[1:6]}) == 5
    assert colors[0, 0] == colors[0, 1] == colors[0, 2]
    assert colors[6, 0] == colors[6, 1] == colors[6, 2]


def test_cursor_depth_band_radius_must_be_positive():
    with np.testing.assert_raises_regex(ValueError, "at least 1"):
        render_depth_focus_visualization(Image.new("L", (3, 3), 100), 100, radius=0)


def test_contours_mark_boundaries_between_median_filtered_raw_levels():
    values = np.tile(np.array([241, 241, 242, 242, 243, 243], dtype=np.uint8), (5, 1))
    depth = Image.fromarray(values, mode="L")

    plain = np.asarray(render_depth_visualization(depth).image)
    contoured = np.asarray(render_depth_visualization(depth, contours=True).image)

    white_plain = np.all(plain == 255, axis=2).sum()
    white_contoured = np.all(contoured == 255, axis=2).sum()
    assert white_contoured > white_plain


def test_contours_are_recomputed_as_one_pixel_lines_at_display_resolution():
    values = np.tile(np.array([241] * 6 + [242] * 6, dtype=np.uint8), (8, 1))
    depth = Image.fromarray(values, mode="L")

    overlay = np.asarray(render_depth_contour_overlay(depth, (120, 80)))
    active_columns = np.flatnonzero(overlay[..., 3].any(axis=0))

    assert overlay.shape == (80, 120, 4)
    assert len(active_columns) == 1
    assert active_columns[0] in (59, 60)


def test_contour_step_draws_only_every_nth_raw_depth_level():
    row = np.repeat(np.arange(240, 246, dtype=np.uint8), 3)
    depth = Image.fromarray(np.tile(row, (8, 1)), mode="L")

    every_level = np.asarray(render_depth_contour_overlay(depth, depth.size, level_step=1))
    every_third = np.asarray(render_depth_contour_overlay(depth, depth.size, level_step=3))
    every_level_columns = np.flatnonzero(every_level[..., 3].any(axis=0))
    every_third_columns = np.flatnonzero(every_third[..., 3].any(axis=0))

    assert len(every_level_columns) == 5
    assert len(every_third_columns) == 1


def test_contour_step_must_be_positive():
    depth = Image.new("L", (5, 5), 128)

    with np.testing.assert_raises_regex(ValueError, "at least 1"):
        render_depth_contour_overlay(depth, depth.size, level_step=0)


def test_all_invalid_depth_produces_black_image_without_range():
    visualization = render_depth_visualization(Image.new("L", (4, 3), 0))

    assert visualization.low_raw is None
    assert visualization.high_raw is None
    assert not np.asarray(visualization.image).any()
