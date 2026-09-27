"""Click-based measurement tools (vector, polyline, neck) and measurement history.

Qt-free so the editing and summing rules can be unit-tested directly. Points
are in the 480x640 display space; distance callbacks return millimetres.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

Point = tuple[float, float]


class MeasurementTool(str, Enum):
    VECTOR = "Vector"
    POLYLINE = "Polyline"
    REGION = "Region"
    # Two clicks on the neck edges; width via portrait_analyser's
    # neck_width_from_edges (capture-app photos only).
    NECK = "Neck"


@dataclass
class PolylineEditor:
    """Points of the measurement being drawn with the vector or polyline tool.

    ``max_points`` of 2 gives the vector tool (finishes itself on the second
    click); ``None`` gives the polyline tool (finished explicitly).
    """

    max_points: int | None
    points: list[Point] = field(default_factory=list)
    finished: bool = False

    def add_point(self, point: Point) -> bool:
        """Append a point; return True when this click started a new measurement."""
        started_new = not self.points or self.finished
        if self.finished:
            self.points = []
            self.finished = False
        self.points.append(point)
        if self.max_points is not None and len(self.points) >= self.max_points:
            self.finished = True
        return started_new

    def finish(self) -> bool:
        """Mark the measurement complete; needs at least one segment."""
        if len(self.points) < 2:
            return False
        self.finished = True
        return True

    def undo(self) -> None:
        if self.finished or not self.points:
            return
        self.points.pop()

    def clear(self) -> None:
        self.points = []
        self.finished = False

    def point_near(self, point: Point, tolerance: float) -> int | None:
        best = None
        for index, (x, y) in enumerate(self.points):
            distance = math.hypot(x - point[0], y - point[1])
            if distance <= tolerance and (best is None or distance < best[0]):
                best = (distance, index)
        return None if best is None else best[1]

    def move_point(self, index: int, point: Point) -> None:
        self.points[index] = point


@dataclass(frozen=True)
class SegmentResult:
    start: Point
    end: Point
    linear_mm: float | None
    surface_mm: float | None


@dataclass(frozen=True)
class PolylineResult:
    segments: tuple[SegmentResult, ...]

    @property
    def linear_mm(self) -> float | None:
        if any(segment.linear_mm is None for segment in self.segments):
            return None
        return sum(segment.linear_mm for segment in self.segments)

    @property
    def surface_mm(self) -> float | None:
        if any(segment.surface_mm is None for segment in self.segments):
            return None
        return sum(segment.surface_mm for segment in self.segments)


def measure_polyline(
    points: Sequence[Point],
    linear_mm: Callable[[Point, Point], float | None],
    surface_mm: Callable[[Point, Point], float | None],
) -> PolylineResult:
    """Measure every consecutive pair; zero-length segments are skipped."""
    segments = []
    for start, end in zip(points, points[1:], strict=False):
        if start == end:
            continue
        segments.append(SegmentResult(start, end, linear_mm(start, end), surface_mm(start, end)))
    return PolylineResult(tuple(segments))


@dataclass(frozen=True)
class HistoryEntry:
    filename: str
    tool: MeasurementTool
    points: int
    linear_mm: float | None
    surface_mm: float | None


class MeasurementHistory:
    """One row per measurement; editing a measurement replaces its row."""

    def __init__(self):
        self._entries: list[HistoryEntry] = []

    @property
    def entries(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._entries)

    def add(self, entry: HistoryEntry) -> int:
        self._entries.append(entry)
        return len(self._entries) - 1

    def replace(self, index: int, entry: HistoryEntry) -> None:
        self._entries[index] = entry

    def clear(self) -> None:
        self._entries.clear()


def format_history_tsv(
    entries: Iterable[HistoryEntry],
    number_format: Callable[[float], str],
) -> str:
    """Tab-separated rows (values in cm) ready to paste into a spreadsheet."""

    def cm(value_mm):
        return "" if value_mm is None else number_format(value_mm / 10.0)

    lines = ["File\tTool\tPoints\tLinear [cm]\tSurface [cm]"]
    for entry in entries:
        lines.append(
            f"{entry.filename}\t{entry.tool.value}\t{entry.points}"
            f"\t{cm(entry.linear_mm)}\t{cm(entry.surface_mm)}"
        )
    return "\n".join(lines)
