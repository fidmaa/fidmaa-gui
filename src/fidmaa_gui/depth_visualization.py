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


@dataclass(frozen=True)
class DepthVisualization:
    image: Image.Image
    low_raw: int | None
    high_raw: int | None


def render_depth_visualization(
    depthmap: Image.Image,
    *,
    contours: bool = False,
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

    if contours and min(depth.shape) >= 3:
        filtered = cv2.medianBlur(depth, 3)
        contour_pixels = _raw_level_boundaries(filtered, valid)
        colored[contour_pixels] = (255, 255, 255)

    return DepthVisualization(Image.fromarray(colored, mode="RGB"), low, high)


def _raw_level_boundaries(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(depth.shape, dtype=bool)

    horizontal = (depth[:, 1:] != depth[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    boundaries[:, 1:] |= horizontal
    boundaries[:, :-1] |= horizontal

    vertical = (depth[1:, :] != depth[:-1, :]) & valid[1:, :] & valid[:-1, :]
    boundaries[1:, :] |= vertical
    boundaries[:-1, :] |= vertical
    return boundaries
