"""Display-only depth-map enhancement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image


class DepthDisplayMode(str, Enum):
    RAW = "Raw grayscale"
    COLOR = "Enhanced color"
    COLOR_CONTOURS = "Color + contours"
    CURSOR_BANDS = "Cursor depth bands"


@dataclass(frozen=True)
class DepthVisualization:
    image: Image.Image
    low_raw: int | None
    high_raw: int | None


def render_depth_visualization(
    depthmap: Image.Image,
    *,
    contours: bool = False,
    contour_step: int = 1,
) -> DepthVisualization:
    """Stretch valid local depth values over Viridis without changing source data."""
    depth = np.asarray(depthmap.convert("L"), dtype=np.uint8)
    valid = depth > 0
    if not np.any(valid):
        return DepthVisualization(Image.new("RGB", depthmap.size, "black"), None, None)

    low = int(np.floor(np.percentile(depth[valid], 2)))
    high = int(np.ceil(np.percentile(depth[valid], 98)))
    if high <= low:
        low = max(0, low - 1)
        high = min(255, high + 1)

    normalized = np.clip(
        (depth.astype(np.float32) - low) * 255.0 / max(1, high - low),
        0,
        255,
    ).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0

    if contours:
        overlay = np.asarray(
            render_depth_contour_overlay(
                depthmap,
                depthmap.size,
                level_step=contour_step,
            )
        )
        colored[overlay[..., 3] > 0] = (255, 255, 255)

    return DepthVisualization(Image.fromarray(colored, mode="RGB"), low, high)


def render_depth_focus_visualization(
    depthmap: Image.Image,
    center_raw: int,
    *,
    radius: int = 2,
) -> DepthVisualization:
    """Color only exact raw levels within ``center_raw ± radius``.

    Values outside the selected band remain as a deliberately dim grayscale
    context. Each in-band integer level receives a discrete Turbo color, while
    the exact cursor level is white so it remains immediately identifiable.
    """
    if radius < 1:
        raise ValueError("radius must be at least 1")

    depth = np.asarray(depthmap.convert("L"), dtype=np.uint8)
    valid = depth > 0
    gray = np.rint(depth.astype(np.float32) * 0.18).astype(np.uint8)
    colored = np.repeat(gray[..., None], 3, axis=2)
    colored[~valid] = 0

    delta = depth.astype(np.int16) - int(center_raw)
    in_band = valid & (np.abs(delta) <= radius)
    palette_position = np.clip(
        np.rint((delta.astype(np.float32) + radius) * 255 / (2 * radius)),
        0,
        255,
    ).astype(np.uint8)
    turbo_bgr = cv2.applyColorMap(palette_position, cv2.COLORMAP_TURBO)
    turbo_rgb = cv2.cvtColor(turbo_bgr, cv2.COLOR_BGR2RGB)
    colored[in_band] = turbo_rgb[in_band]
    colored[valid & (delta == 0)] = (255, 255, 255)

    return DepthVisualization(
        Image.fromarray(colored, mode="RGB"),
        max(1, int(center_raw) - radius),
        min(255, int(center_raw) + radius),
    )


def render_depth_contour_overlay(
    depthmap: Image.Image,
    display_size: tuple[int, int],
    *,
    level_step: int = 1,
) -> Image.Image:
    """Render crisp, optionally thinned raw-level boundaries as an RGBA layer."""
    if level_step < 1:
        raise ValueError("level_step must be at least 1")
    width, height = display_size
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    depth = np.asarray(depthmap.convert("L"), dtype=np.uint8)
    if width < 1 or height < 1 or min(depth.shape) < 3:
        return Image.fromarray(overlay, mode="RGBA")

    valid = depth > 0
    filtered = cv2.medianBlur(depth, 3)
    if (width, height) != depthmap.size:
        filtered = cv2.resize(filtered, (width, height), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(
            valid.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    contour_pixels = _raw_level_boundaries(filtered, valid, level_step=level_step)
    overlay[contour_pixels] = (255, 255, 255, 220)
    return Image.fromarray(overlay, mode="RGBA")


def _raw_level_boundaries(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    level_step: int = 1,
) -> np.ndarray:
    """Return one-sided boundaries so every rendered contour stays one pixel wide."""
    boundaries = np.zeros(depth.shape, dtype=bool)
    contour_level = depth if level_step == 1 else depth // level_step

    horizontal = (contour_level[:, 1:] != contour_level[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    boundaries[:, 1:] |= horizontal

    vertical = (contour_level[1:, :] != contour_level[:-1, :]) & valid[1:, :] & valid[:-1, :]
    boundaries[1:, :] |= vertical
    return boundaries
