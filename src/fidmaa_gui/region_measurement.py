"""Robust measurements between two user-selected image regions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import cv2
import numpy as np
from PIL import Image


class MeasurementError(ValueError):
    """Raised when a region measurement cannot be calculated."""


class SelectionMode(str, Enum):
    HIGHEST = "Highest"
    LOWEST = "Lowest"
    FLATTEST = "Flattest"


class RegionMask(str, Enum):
    NONE = "None"
    TEETH = "Teeth"


@dataclass(frozen=True)
class Region:
    """Circular ROI represented by its square bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def left(self) -> float:
        return min(self.x1, self.x2)

    @property
    def right(self) -> float:
        return max(self.x1, self.x2)

    @property
    def top(self) -> float:
        return min(self.y1, self.y2)

    @property
    def bottom(self) -> float:
        return max(self.y1, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def radius(self) -> float:
        return min(self.width, self.height) / 2.0

    def contains(self, x: float, y: float, tolerance: float = 0.0) -> bool:
        center_x, center_y = self.center
        return math.hypot(x - center_x, y - center_y) <= self.radius + tolerance

    def pixel_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        center_x, center_y = self.center
        radius = self.radius
        return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

    def clipped_bounds(self, display_size: tuple[int, int]) -> tuple[int, int, int, int]:
        width, height = display_size
        left = max(0, min(width - 1, math.floor(self.left)))
        top = max(0, min(height - 1, math.floor(self.top)))
        right = max(left + 1, min(width, math.ceil(self.right)))
        bottom = max(top + 1, min(height, math.ceil(self.bottom)))
        return left, top, right, bottom


@dataclass(frozen=True)
class CandidatePoint:
    x: int
    y: int
    point_3d_mm: tuple[float, float, float]
    score: float


@dataclass(frozen=True)
class VectorSample:
    start: CandidatePoint
    end: CandidatePoint
    linear_mm: float
    surface_mm: float | None


@dataclass(frozen=True)
class MeasurementStats:
    mean_mm: float
    sample_sd_mm: float
    count: int


@dataclass(frozen=True)
class RegionMeasurementResult:
    candidates_a: tuple[CandidatePoint, ...]
    candidates_b: tuple[CandidatePoint, ...]
    samples: tuple[VectorSample, ...]

    def linear_stats(self, minimum_count: int = 5) -> MeasurementStats | None:
        return _measurement_stats([sample.linear_mm for sample in self.samples], minimum_count)

    def surface_stats(self, minimum_count: int = 5) -> MeasurementStats | None:
        return _measurement_stats(
            [sample.surface_mm for sample in self.samples if sample.surface_mm is not None],
            minimum_count,
        )


def _measurement_stats(values: list[float], minimum_count: int) -> MeasurementStats | None:
    if len(values) < minimum_count:
        return None
    return MeasurementStats(
        mean_mm=float(np.mean(values)),
        sample_sd_mm=float(np.std(values, ddof=1)),
        count=len(values),
    )


class RegionMeasurementEngine:
    """Select endpoint pools and measure matched profiles between two ROIs."""

    def __init__(
        self,
        *,
        filtered_depthmap: Image.Image,
        display_size: tuple[int, int],
        image_size: tuple[int, int],
        depth_to_cm: Callable[[np.ndarray | float], np.ndarray | float],
        pixels_per_mm: Callable[[np.ndarray | float], np.ndarray | float],
        surface_length: Callable[[tuple[int, int], tuple[int, int]], float | None],
        teethmap: Image.Image | None = None,
    ) -> None:
        self.depth = np.asarray(filtered_depthmap.convert("L"), dtype=np.float64)
        self.display_size = display_size
        self.image_size = image_size
        self.depth_to_cm = depth_to_cm
        self.pixels_per_mm = pixels_per_mm
        self.surface_length = surface_length
        self.teethmap = teethmap

    def measure(
        self,
        region_a: Region,
        region_b: Region,
        *,
        mode_a: SelectionMode,
        mode_b: SelectionMode,
        mask_a: RegionMask = RegionMask.NONE,
        mask_b: RegionMask = RegionMask.NONE,
        percentile: float = 10.0,
        vector_count: int = 10,
    ) -> RegionMeasurementResult:
        if not 5 <= vector_count <= 10:
            raise MeasurementError("Vector count must be between 5 and 10")
        if not 5 <= percentile <= 20:
            raise MeasurementError("Candidate percentile must be between 5 and 20")

        candidates_a = self.select_candidates(
            region_a,
            mode=mode_a,
            mask=mask_a,
            percentile=percentile,
            count=vector_count,
        )
        candidates_b = self.select_candidates(
            region_b,
            mode=mode_b,
            mask=mask_b,
            percentile=percentile,
            count=vector_count,
        )
        pairs = pair_candidates(candidates_a, candidates_b, region_a, region_b)
        samples = []
        for start, end in pairs:
            linear_mm = math.dist(start.point_3d_mm, end.point_3d_mm)
            surface_mm = self.surface_length((start.x, start.y), (end.x, end.y))
            samples.append(VectorSample(start, end, linear_mm, surface_mm))

        return RegionMeasurementResult(tuple(candidates_a), tuple(candidates_b), tuple(samples))

    def select_candidates(
        self,
        region: Region,
        *,
        mode: SelectionMode,
        mask: RegionMask,
        percentile: float,
        count: int,
    ) -> list[CandidatePoint]:
        x_coords, y_coords, raw_depth = self._sample_region(region)
        valid = region.pixel_mask(x_coords, y_coords)
        valid &= np.isfinite(raw_depth) & (raw_depth > 0)
        if mask == RegionMask.TEETH:
            valid &= self._sample_teeth_mask(x_coords, y_coords)

        if np.count_nonzero(valid) < count:
            raise MeasurementError(f"Region contains fewer than {count} valid candidate pixels")

        z_cm = self._depth_array_to_cm(raw_depth)
        valid &= np.isfinite(z_cm) & (z_cm > 0)
        x_mm, y_mm, z_mm = self._to_3d(x_coords, y_coords, z_cm)
        valid &= np.isfinite(x_mm) & np.isfinite(y_mm) & np.isfinite(z_mm)

        if mode == SelectionMode.HIGHEST:
            score = z_cm
        elif mode == SelectionMode.LOWEST:
            score = -z_cm
        elif mode == SelectionMode.FLATTEST:
            score, flat_valid = self._flatness_score(x_mm, y_mm, z_mm, valid)
            valid &= flat_valid
        else:  # pragma: no cover - defensive guard for callers outside Qt
            raise MeasurementError(f"Unsupported selection mode: {mode}")

        valid_indexes = np.flatnonzero(valid.ravel())
        if len(valid_indexes) < count:
            raise MeasurementError(f"Region contains fewer than {count} valid candidate pixels")

        flat_scores = score.ravel()[valid_indexes]
        order = np.lexsort(
            (
                x_coords.ravel()[valid_indexes],
                y_coords.ravel()[valid_indexes],
                flat_scores,
            )
        )
        ranked_indexes = valid_indexes[order]
        pool_size = max(count, math.ceil(len(ranked_indexes) * percentile / 100.0))
        pool_indexes = ranked_indexes[:pool_size]

        pool_points = [
            CandidatePoint(
                x=int(x_coords.ravel()[index]),
                y=int(y_coords.ravel()[index]),
                point_3d_mm=(
                    float(x_mm.ravel()[index]),
                    float(y_mm.ravel()[index]),
                    float(z_mm.ravel()[index]),
                ),
                score=float(score.ravel()[index]),
            )
            for index in pool_indexes
        ]
        return spatially_distributed_points(pool_points, count)

    def _depth_array_to_cm(self, raw_depth: np.ndarray) -> np.ndarray:
        """Use a vectorized converter when available, with a scalar fallback."""
        try:
            converted = np.asarray(self.depth_to_cm(raw_depth), dtype=np.float64)
            if converted.shape == raw_depth.shape:
                return converted
        except (TypeError, ValueError):
            pass

        def convert(value):
            result = self.depth_to_cm(float(value))
            return np.nan if result is None else result

        return np.vectorize(convert, otypes=[np.float64])(raw_depth)

    def _sample_region(self, region: Region) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left, top, right, bottom = region.clipped_bounds(self.display_size)
        if right - left < 2 or bottom - top < 2:
            raise MeasurementError("Region is too small")

        x_coords, y_coords = np.meshgrid(
            np.arange(left, right, dtype=np.int32),
            np.arange(top, bottom, dtype=np.int32),
        )
        depth_height, depth_width = self.depth.shape
        display_width, display_height = self.display_size
        depth_x = x_coords * (depth_width - 1) / max(1, display_width - 1)
        depth_y = y_coords * (depth_height - 1) / max(1, display_height - 1)
        raw_depth = _bilinear_sample_array(self.depth, depth_x, depth_y)
        return x_coords, y_coords, raw_depth

    def _sample_teeth_mask(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        if self.teethmap is None:
            raise MeasurementError("The selected image has no teeth mask")
        teeth = np.asarray(self.teethmap.convert("L"))
        teeth_height, teeth_width = teeth.shape
        display_width, display_height = self.display_size
        teeth_x = np.rint(x_coords * (teeth_width - 1) / max(1, display_width - 1)).astype(np.int32)
        teeth_y = np.rint(y_coords * (teeth_height - 1) / max(1, display_height - 1)).astype(
            np.int32
        )
        return teeth[teeth_y, teeth_x] > 30

    def _to_3d(
        self, x: np.ndarray, y: np.ndarray, z_cm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        display_width, display_height = self.display_size
        image_width, image_height = self.image_size
        image_x = x * image_width / display_width
        image_y = y * image_height / display_height
        pixels_per_mm = np.asarray(self.pixels_per_mm(z_cm), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_mm = (image_x - image_width / 2.0) / pixels_per_mm
            y_mm = (image_y - image_height / 2.0) / pixels_per_mm
        return x_mm, y_mm, z_cm * 10.0

    def _flatness_score(
        self,
        x_mm: np.ndarray,
        y_mm: np.ndarray,
        z_mm: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid_depth = z_mm[valid]
        lower, upper = np.percentile(valid_depth, (10, 90))
        central = valid & (z_mm >= lower) & (z_mm <= upper)
        count = _box_sum(central.astype(np.float64), 5)

        coordinates = (x_mm, y_mm, z_mm)
        means = []
        for coordinate in coordinates:
            means.append(_box_sum(np.where(central, coordinate, 0.0), 5) / np.maximum(count, 1))

        covariance = np.empty((*z_mm.shape, 3, 3), dtype=np.float64)
        for row in range(3):
            for column in range(row, 3):
                product_sum = _box_sum(
                    np.where(
                        central,
                        coordinates[row] * coordinates[column],
                        0.0,
                    ),
                    5,
                )
                value = product_sum / np.maximum(count, 1) - means[row] * means[column]
                covariance[..., row, column] = value
                covariance[..., column, row] = value

        eigenvalues = np.linalg.eigvalsh(covariance)
        score = np.sqrt(np.maximum(eigenvalues[..., 0], 0.0))
        flat_valid = central & (count >= 9) & np.isfinite(score)
        score[~flat_valid] = np.inf
        return score, flat_valid


def _box_sum(values: np.ndarray, size: int) -> np.ndarray:
    return cv2.boxFilter(
        values,
        ddepth=-1,
        ksize=(size, size),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )


def _bilinear_sample_array(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = np.clip(x.astype(np.float64), 0, width - 1)
    y = np.clip(y.astype(np.float64), 0, height - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fraction_x = x - x0
    fraction_y = y - y0
    weights = (
        (1 - fraction_x) * (1 - fraction_y),
        fraction_x * (1 - fraction_y),
        (1 - fraction_x) * fraction_y,
        fraction_x * fraction_y,
    )
    values = (
        image[y0, x0],
        image[y0, x1],
        image[y1, x0],
        image[y1, x1],
    )
    result = np.zeros_like(x, dtype=np.float64)
    invalid = np.zeros_like(x, dtype=bool)
    for value, weight in zip(values, weights, strict=True):
        result += value * weight
        invalid |= (weight > 0) & (value == 0)
    result[invalid] = np.nan
    return result


def spatially_distributed_points(points: list[CandidatePoint], count: int) -> list[CandidatePoint]:
    """Choose deterministic, spatially dispersed points from a ranked pool."""
    if len(points) <= count:
        return list(points)

    coordinates = np.asarray([(point.x, point.y) for point in points], dtype=float)
    selected_indexes = [0]
    minimum_distance = np.sum((coordinates - coordinates[0]) ** 2, axis=1)
    minimum_distance[0] = -1
    while len(selected_indexes) < count:
        next_index = int(np.argmax(minimum_distance))
        selected_indexes.append(next_index)
        distance = np.sum((coordinates - coordinates[next_index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected_indexes] = -1
    return [points[index] for index in selected_indexes]


def pair_candidates(
    candidates_a: list[CandidatePoint],
    candidates_b: list[CandidatePoint],
    region_a: Region,
    region_b: Region,
) -> list[tuple[CandidatePoint, CandidatePoint]]:
    """Pair endpoints in their cross-profile order to avoid crossing lines."""
    center_a = region_a.center
    center_b = region_b.center
    direction_x = center_b[0] - center_a[0]
    direction_y = center_b[1] - center_a[1]
    if direction_x == 0 and direction_y == 0:
        perpendicular = (1.0, 0.0)
    else:
        length = math.hypot(direction_x, direction_y)
        perpendicular = (-direction_y / length, direction_x / length)

    def projection(point: CandidatePoint) -> tuple[float, int, int]:
        value = point.x * perpendicular[0] + point.y * perpendicular[1]
        return value, point.x, point.y

    ordered_a = sorted(candidates_a, key=projection)
    ordered_b = sorted(candidates_b, key=projection)
    return list(zip(ordered_a, ordered_b, strict=False))
