# Measurement Tools UI Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the FIDMAA GUI around "photo + one active measurement tool + clearly visible result": a toolbar with Vector / Polyline / Region tools, a side panel with automatic results, the current tool's result and a measurement history, and with the unused legacy depth-profile UI hidden.

**Architecture:** Pure measurement logic (polyline editing, per-segment measurement, history, TSV export) lives in a new Qt-free module `measurement_tools.py`. A new `measurement_panel.py` widget renders automatic results, the tool result, tool options and history. `app.py` gains a toolbar/menus, routes image clicks to the active tool, and delegates distance maths to its existing `vector_length_between_two_clicks()` and `surface_vector_filtered()`.

**Tech Stack:** Python 3, PySide6 (QtWidgets, QtGui actions/toolbars), Pillow, pytest (+ `qapp` fixture from `tests/conftest.py`), ruff.

**Spec:** Inline — decisions agreed with the user on 2026-09-27 (Polish conversation), summarised here:

1. The legacy blue profile line, red depth chart (right of the photo), x / y / angle (skew) controls and the reconstruction chart are **hidden** (not deleted) under *View → Advanced*.
2. Measurement tools are modes; exactly one is active. Tools: **Vector** (2 clicks), **Polyline** (enter the mode, click any number of points, finish with **Enter** or **right-click / two-finger click**), **Region** (the existing A/B patch measurement with its selector/mask controls). The old single-pixel ("Point"/"Pixel") tool is **removed**.
3. Vector and Polyline show a large, clearly visible **linear 3D** and **surface** result.
4. A **measurement history** (one row per measurement) plus a **Copy** button (tab-separated, paste into a spreadsheet).
5. **Depth display** mode moves to the toolbar and to the *View* menu.
6. On photo open, automatic results are shown when available. **Thyromental distance is removed entirely.** **FaceMesh mouth-opening detection is hidden** behind an opt-in *Analysis* menu toggle (off by default).

## Global Constraints

- Work only in the worktree `~/Programowanie/fidmaa-gui-ui-tools` on branch `ui-tools`. Never touch `~/Programowanie/fidmaa-gui` (main checkout).
- Run tests with `uv run pytest`; lint/format with `uv run ruff check --fix <files>` and `uv run ruff format <files>` (line length 100).
- Never swallow exceptions silently (no bare `except: pass`); log or surface every error.
- Fixed display geometry: main image 480x640 display space; all tool points are stored in display space (floats), clamped to `[0, 479] x [0, 639]`.
- Distances are stored in **mm** internally and displayed in **cm with 2 decimals**; spreadsheet export uses the system locale decimal separator (`QLocale().toString(value, "f", 2)`).
- Surface length uses `surface_vector_filtered(..., step)` with a user-selectable step (default **3 px**, range 2–7). The legacy unfiltered `vector_length_surface()` is not shown anywhere in the new UI.
- Keyboard shortcuts (no modifier): **V** Vector, **M** Polyline, **R** Region, **H/L/F** region selectors (existing), **Enter/Return** finish polyline, **Backspace** remove last point, **Esc** clear current measurement. **P** is no longer bound.
- Commit after each task; commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Clicking on the photo, pressing Enter/Backspace/Esc, or right-clicking **before any photo is loaded** must not raise; the result area shows a hint instead. (Test in Task 5.)
2. **Double-clicking the same spot** in Polyline mode creates a zero-length segment; it must be skipped, not crash or add 0 to the result. (Test in Task 1.)
3. A segment crossing **invalid depth** (surface `None`) must still show the linear total and show the surface as "n/a (invalid depth)"; the history cell stays empty. (Tests in Tasks 1 and 3.)
4. **Loading another photo mid-measurement** clears the in-progress points but keeps earlier history rows with their original file name. (Test in Task 5.)
5. **Enter or right-click with fewer than 2 points** does nothing and adds no history row. (Test in Task 5.)

---

## File Structure

- Create `src/fidmaa_gui/measurement_tools.py` — Qt-free: `MeasurementTool`, `PolylineEditor`, `SegmentResult`, `PolylineResult`, `measure_polyline()`, `HistoryEntry`, `MeasurementHistory`, `format_history_tsv()`.
- Create `src/fidmaa_gui/measurement_panel.py` — `MeasurementPanel(QWidget)`: automatic-results box, tool-result box with big numbers, options container, history table with Copy/Clear.
- Modify `src/fidmaa_gui/QClickableLabel.py` — add `rightClicked` signal.
- Modify `src/fidmaa_gui/app.py` — toolbar, menus, tool switching, click routing, polyline painting, automatic results, TMD removal, mouth toggle, legacy hiding.
- Create `tests/test_measurement_tools.py`, `tests/test_measurement_panel.py`, `tests/test_tool_interaction_ui.py`.
- Modify `tests/test_region_selection_ui.py` — replace references to the removed `regionSelectionButton`, `measurementControlsAction`, `pixelToolAction`.

---

### Task 1: Qt-free measurement tool logic

**Files:**
- Create: `src/fidmaa_gui/measurement_tools.py`
- Test: `tests/test_measurement_tools.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class MeasurementTool(str, Enum)`: `VECTOR = "Vector"`, `POLYLINE = "Polyline"`, `REGION = "Region"`.
  - `Point = tuple[float, float]`
  - `@dataclass class PolylineEditor(max_points: int | None, points: list[Point], finished: bool)` with `add_point(point) -> bool` (True when this click started a new measurement), `finish() -> bool`, `undo() -> None`, `clear() -> None`, `point_near(point, tolerance: float) -> int | None`, `move_point(index: int, point) -> None`.
  - `@dataclass(frozen=True) class SegmentResult(start, end, linear_mm: float, surface_mm: float | None)`
  - `@dataclass(frozen=True) class PolylineResult(segments: tuple[SegmentResult, ...])` with properties `linear_mm -> float`, `surface_mm -> float | None`.
  - `measure_polyline(points, linear_mm: Callable[[Point, Point], float], surface_mm: Callable[[Point, Point], float | None]) -> PolylineResult`
  - `@dataclass(frozen=True) class HistoryEntry(filename: str, tool: MeasurementTool, points: int, linear_mm: float | None, surface_mm: float | None)`
  - `class MeasurementHistory` with `entries -> tuple[HistoryEntry, ...]`, `add(entry) -> int`, `replace(index, entry)`, `clear()`.
  - `format_history_tsv(entries, number_format: Callable[[float], str]) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_measurement_tools.py
import pytest

from fidmaa_gui.measurement_tools import (
    HistoryEntry,
    MeasurementHistory,
    MeasurementTool,
    PolylineEditor,
    format_history_tsv,
    measure_polyline,
)


def test_vector_editor_finishes_on_second_point_and_restarts_on_third():
    editor = PolylineEditor(max_points=2)

    assert editor.add_point((10.0, 10.0)) is True
    assert editor.add_point((20.0, 10.0)) is False
    assert editor.finished
    assert editor.add_point((30.0, 30.0)) is True
    assert editor.points == [(30.0, 30.0)]
    assert not editor.finished


def test_polyline_editor_needs_two_points_to_finish():
    editor = PolylineEditor(max_points=None)
    assert editor.finish() is False
    editor.add_point((1.0, 1.0))
    assert editor.finish() is False
    assert not editor.finished
    editor.add_point((5.0, 1.0))
    editor.add_point((9.0, 1.0))
    assert not editor.finished
    assert editor.finish() is True
    assert editor.finished


def test_undo_and_clear():
    editor = PolylineEditor(max_points=None)
    editor.add_point((1.0, 1.0))
    editor.add_point((2.0, 2.0))
    editor.undo()
    assert editor.points == [(1.0, 1.0)]
    editor.clear()
    assert editor.points == []
    assert not editor.finished


def test_undo_does_nothing_on_finished_measurement():
    editor = PolylineEditor(max_points=2)
    editor.add_point((1.0, 1.0))
    editor.add_point((2.0, 2.0))
    editor.undo()
    assert editor.points == [(1.0, 1.0), (2.0, 2.0)]


def test_point_near_picks_closest_within_tolerance():
    editor = PolylineEditor(max_points=None)
    editor.add_point((10.0, 10.0))
    editor.add_point((14.0, 10.0))
    assert editor.point_near((13.0, 10.0), tolerance=6) == 1
    assert editor.point_near((100.0, 100.0), tolerance=6) is None
    editor.move_point(1, (50.0, 50.0))
    assert editor.points[1] == (50.0, 50.0)


def test_measure_polyline_sums_segments_and_skips_zero_length():
    points = [(0.0, 0.0), (3.0, 4.0), (3.0, 4.0), (3.0, 8.0)]
    result = measure_polyline(
        points,
        linear_mm=lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5,
        surface_mm=lambda a, b: 2 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5,
    )
    assert len(result.segments) == 2
    assert result.linear_mm == pytest.approx(9.0)
    assert result.surface_mm == pytest.approx(18.0)


def test_measure_polyline_surface_is_none_when_any_segment_is_invalid():
    result = measure_polyline(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        linear_mm=lambda a, b: 1.0,
        surface_mm=lambda a, b: None if a[0] == 1.0 else 1.0,
    )
    assert result.linear_mm == pytest.approx(2.0)
    assert result.surface_mm is None
    assert result.segments[1].surface_mm is None


def test_history_add_replace_and_tsv():
    history = MeasurementHistory()
    first = history.add(HistoryEntry("a.heic", MeasurementTool.VECTOR, 2, 123.4, 130.0))
    second = history.add(HistoryEntry("a.heic", MeasurementTool.POLYLINE, 5, 200.0, None))
    history.replace(first, HistoryEntry("a.heic", MeasurementTool.VECTOR, 2, 100.0, 110.0))

    assert (first, second) == (0, 1)
    tsv = format_history_tsv(history.entries, number_format=lambda v: f"{v:.2f}")
    assert tsv.splitlines() == [
        "File\tTool\tPoints\tLinear [cm]\tSurface [cm]",
        "a.heic\tVector\t2\t10.00\t11.00",
        "a.heic\tPolyline\t5\t20.00\t",
    ]
    history.clear()
    assert history.entries == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_measurement_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fidmaa_gui.measurement_tools'`

- [ ] **Step 3: Write the implementation**

```python
# src/fidmaa_gui/measurement_tools.py
"""Click-based measurement tools (vector, polyline) and measurement history.

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
    linear_mm: float
    surface_mm: float | None


@dataclass(frozen=True)
class PolylineResult:
    segments: tuple[SegmentResult, ...]

    @property
    def linear_mm(self) -> float:
        return sum(segment.linear_mm for segment in self.segments)

    @property
    def surface_mm(self) -> float | None:
        if any(segment.surface_mm is None for segment in self.segments):
            return None
        return sum(segment.surface_mm for segment in self.segments)


def measure_polyline(
    points: Sequence[Point],
    linear_mm: Callable[[Point, Point], float],
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_measurement_tools.py -v`
Expected: 8 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/fidmaa_gui/measurement_tools.py tests/test_measurement_tools.py
uv run ruff format src/fidmaa_gui/measurement_tools.py tests/test_measurement_tools.py
git add src/fidmaa_gui/measurement_tools.py tests/test_measurement_tools.py
git commit -m "Add Qt-free polyline measurement and history logic"
```

---

### Task 2: Right-click signal on the clickable label

**Files:**
- Modify: `src/fidmaa_gui/QClickableLabel.py`
- Test: `tests/test_region_selection_ui.py` (append)

**Interfaces:**
- Produces: `QClickableLabel.rightClicked = Signal(QPointF)`, emitted on right-button release (macOS two-finger click arrives as a right button).

- [ ] **Step 1: Write the failing test** (append to `tests/test_region_selection_ui.py`)

```python
def test_clickable_label_emits_right_click(qapp):
    label = QClickableLabel()
    label.resize(100, 100)
    label.show()
    right_clicks = []
    left_clicks = []
    label.rightClicked.connect(right_clicks.append)
    label.clicked.connect(left_clicks.append)

    QTest.mouseClick(label, Qt.RightButton, pos=QPointF(12, 34).toPoint())

    assert len(right_clicks) == 1
    assert (right_clicks[0].x(), right_clicks[0].y()) == (12, 34)
    assert left_clicks == []
    label.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_region_selection_ui.py::test_clickable_label_emits_right_click -v`
Expected: FAIL with `AttributeError: 'QClickableLabel' object has no attribute 'rightClicked'`

- [ ] **Step 3: Implement** — in `QClickableLabel`, add the signal next to the others and handle it in `mouseReleaseEvent`:

```python
    rightClicked = QtCore.Signal(QPointF)
```

```python
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.released.emit(event.position())
            if self._press_position is not None:
                movement = event.position() - self._press_position
                if movement.manhattanLength() <= 3:
                    self.clicked.emit(event.position())
            self._press_position = None
        elif event.button() == Qt.RightButton:
            self.rightClicked.emit(event.position())
        super().mouseReleaseEvent(event)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_region_selection_ui.py -v`
Expected: all pass

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/fidmaa_gui/QClickableLabel.py tests/test_region_selection_ui.py
uv run ruff format src/fidmaa_gui/QClickableLabel.py tests/test_region_selection_ui.py
git add src/fidmaa_gui/QClickableLabel.py tests/test_region_selection_ui.py
git commit -m "Emit rightClicked from QClickableLabel"
```

---

### Task 3: Measurement side panel widget

**Files:**
- Create: `src/fidmaa_gui/measurement_panel.py`
- Test: `tests/test_measurement_panel.py`

**Interfaces:**
- Consumes (Task 1): `HistoryEntry`, `MeasurementHistory`, `MeasurementTool`, `format_history_tsv`.
- Produces: `class MeasurementPanel(QWidget)` with attributes `autoLabel: QLabel`, `autoWarningLabel: QLabel`, `resultGroup: QGroupBox`, `linearValueLabel: QLabel`, `surfaceValueLabel: QLabel`, `detailsLabel: QLabel`, `hintLabel: QLabel`, `optionsLayout: QVBoxLayout`, `historyTable: QTableWidget`, `copyButton`, `clearHistoryButton`, `history: MeasurementHistory`; methods:
  - `set_automatic_results(lines: list[str], warning: str | None) -> None`
  - `set_tool_result(title: str, linear_mm: float | None, surface_mm: float | None, details: list[str], hint: str) -> None`
  - `add_history(entry: HistoryEntry) -> int`, `replace_history(index: int, entry: HistoryEntry) -> None`
  - `copy_history() -> str` (also puts the text on the clipboard), `clear_history() -> None`
  - module function `format_cm(value_mm: float | None) -> str` → `"12,34 cm"` per locale, `"n/a (invalid depth)"` for None.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_measurement_panel.py
from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QApplication

from fidmaa_gui.measurement_panel import MeasurementPanel, format_cm
from fidmaa_gui.measurement_tools import HistoryEntry, MeasurementTool


def test_format_cm_uses_locale_and_handles_invalid_depth(qapp):
    QLocale.setDefault(QLocale(QLocale.Polish, QLocale.Poland))
    try:
        assert format_cm(123.4) == "12,34 cm"
        assert format_cm(None) == "n/a (invalid depth)"
    finally:
        QLocale.setDefault(QLocale.c())


def test_tool_result_shows_big_linear_and_surface_values(qapp):
    panel = MeasurementPanel()
    panel.set_tool_result("Vector", 123.4, None, ["Δ depth: 3.0 mm"], "Click two points.")

    assert panel.resultGroup.title() == "Vector"
    assert panel.linearValueLabel.text() == format_cm(123.4)
    assert panel.surfaceValueLabel.text() == "n/a (invalid depth)"
    assert "Δ depth" in panel.detailsLabel.text()
    assert panel.hintLabel.text() == "Click two points."
    assert panel.linearValueLabel.font().pointSize() > panel.hintLabel.font().pointSize()

    panel.set_tool_result("Vector", None, None, [], "Click two points.")
    assert panel.linearValueLabel.text() == "—"
    panel.close()


def test_history_rows_replace_and_copy_to_clipboard(qapp):
    QLocale.setDefault(QLocale.c())
    panel = MeasurementPanel()
    first = panel.add_history(HistoryEntry("a.heic", MeasurementTool.VECTOR, 2, 100.0, 110.0))
    panel.add_history(HistoryEntry("a.heic", MeasurementTool.POLYLINE, 4, 200.0, None))
    panel.replace_history(first, HistoryEntry("a.heic", MeasurementTool.VECTOR, 2, 90.0, 95.0))

    assert panel.historyTable.rowCount() == 2
    assert panel.historyTable.item(0, 3).text() == "9.00"
    assert panel.historyTable.item(1, 4).text() == ""

    text = panel.copy_history()
    assert QApplication.clipboard().text() == text
    assert text.splitlines()[1] == "a.heic\tVector\t2\t9.00\t9.50"

    panel.clear_history()
    assert panel.historyTable.rowCount() == 0
    assert panel.history.entries == ()
    panel.close()


def test_automatic_results_and_warning(qapp):
    panel = MeasurementPanel()
    panel.set_automatic_results(["Neck circumference: 36.40 cm"], "Low confidence")
    assert "36.40" in panel.autoLabel.text()
    assert panel.autoWarningLabel.text() == "⚠ Low confidence"
    assert not panel.autoWarningLabel.isHidden()

    panel.set_automatic_results(["Pose: NEUTRAL_NECK"], None)
    assert panel.autoWarningLabel.isHidden()
    panel.close()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_measurement_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fidmaa_gui.measurement_panel'`

- [ ] **Step 3: Implement**

```python
# src/fidmaa_gui/measurement_panel.py
"""Side panel: automatic results, the active tool's result, options and history."""

from __future__ import annotations

from PySide6.QtCore import QLocale
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .measurement_tools import HistoryEntry, MeasurementHistory, format_history_tsv

HISTORY_HEADERS = ("File", "Tool", "Points", "Linear [cm]", "Surface [cm]")


def _number(value_cm: float) -> str:
    return QLocale().toString(float(value_cm), "f", 2)


def format_cm(value_mm: float | None) -> str:
    if value_mm is None:
        return "n/a (invalid depth)"
    return f"{_number(value_mm / 10.0)} cm"


class MeasurementPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = MeasurementHistory()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        auto_group = QGroupBox("Automatic")
        auto_layout = QVBoxLayout(auto_group)
        self.autoLabel = QLabel("Open a photo to run the automatic analysis.")
        self.autoLabel.setWordWrap(True)
        auto_layout.addWidget(self.autoLabel)
        self.autoWarningLabel = QLabel()
        self.autoWarningLabel.setWordWrap(True)
        self.autoWarningLabel.setStyleSheet("color: #b36b00; font-weight: bold;")
        self.autoWarningLabel.hide()
        auto_layout.addWidget(self.autoWarningLabel)
        layout.addWidget(auto_group)

        self.resultGroup = QGroupBox("Vector")
        result_layout = QVBoxLayout(self.resultGroup)
        values = QFormLayout()
        self.linearValueLabel = QLabel("—")
        self.surfaceValueLabel = QLabel("—")
        for label in (self.linearValueLabel, self.surfaceValueLabel):
            font = label.font()
            font.setPointSize(font.pointSize() + 8)
            font.setBold(True)
            label.setFont(font)
        values.addRow("Linear (3D):", self.linearValueLabel)
        values.addRow("Surface:", self.surfaceValueLabel)
        result_layout.addLayout(values)
        self.detailsLabel = QLabel()
        self.detailsLabel.setWordWrap(True)
        result_layout.addWidget(self.detailsLabel)
        self.hintLabel = QLabel()
        self.hintLabel.setWordWrap(True)
        self.hintLabel.setStyleSheet("color: gray;")
        result_layout.addWidget(self.hintLabel)
        layout.addWidget(self.resultGroup)

        self.optionsLayout = QVBoxLayout()
        layout.addLayout(self.optionsLayout)

        history_group = QGroupBox("History")
        history_layout = QVBoxLayout(history_group)
        self.historyTable = QTableWidget(0, len(HISTORY_HEADERS))
        self.historyTable.setHorizontalHeaderLabels(HISTORY_HEADERS)
        self.historyTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.historyTable.verticalHeader().setVisible(False)
        history_layout.addWidget(self.historyTable)
        buttons = QHBoxLayout()
        self.copyButton = QPushButton("Copy")
        self.copyButton.setToolTip("Copy all rows as tab-separated text for a spreadsheet")
        self.copyButton.clicked.connect(self.copy_history)
        self.clearHistoryButton = QPushButton("Clear")
        self.clearHistoryButton.clicked.connect(self.clear_history)
        buttons.addWidget(self.copyButton)
        buttons.addWidget(self.clearHistoryButton)
        history_layout.addLayout(buttons)
        layout.addWidget(history_group, stretch=1)

    def set_automatic_results(self, lines, warning):
        self.autoLabel.setText("\n".join(lines))
        if warning:
            self.autoWarningLabel.setText(f"⚠ {warning}")
            self.autoWarningLabel.show()
        else:
            self.autoWarningLabel.clear()
            self.autoWarningLabel.hide()

    def set_tool_result(self, title, linear_mm, surface_mm, details, hint):
        self.resultGroup.setTitle(title)
        if linear_mm is None:
            self.linearValueLabel.setText("—")
            self.surfaceValueLabel.setText("—")
        else:
            self.linearValueLabel.setText(format_cm(linear_mm))
            self.surfaceValueLabel.setText(format_cm(surface_mm))
        self.detailsLabel.setText("\n".join(details))
        self.hintLabel.setText(hint)

    def add_history(self, entry: HistoryEntry) -> int:
        index = self.history.add(entry)
        self._refresh_history()
        return index

    def replace_history(self, index: int, entry: HistoryEntry) -> None:
        self.history.replace(index, entry)
        self._refresh_history()

    def clear_history(self, *args) -> None:
        self.history.clear()
        self._refresh_history()

    def copy_history(self, *args) -> str:
        text = format_history_tsv(self.history.entries, number_format=_number)
        QApplication.clipboard().setText(text)
        return text

    def _refresh_history(self) -> None:
        entries = self.history.entries
        self.historyTable.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            cells = (
                entry.filename,
                entry.tool.value,
                str(entry.points),
                "" if entry.linear_mm is None else _number(entry.linear_mm / 10.0),
                "" if entry.surface_mm is None else _number(entry.surface_mm / 10.0),
            )
            for column, text in enumerate(cells):
                self.historyTable.setItem(row, column, QTableWidgetItem(text))
        if entries:
            self.historyTable.scrollToBottom()
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_measurement_panel.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/fidmaa_gui/measurement_panel.py tests/test_measurement_panel.py
uv run ruff format src/fidmaa_gui/measurement_panel.py tests/test_measurement_panel.py
git add src/fidmaa_gui/measurement_panel.py tests/test_measurement_panel.py
git commit -m "Add measurement side panel with results and history"
```

---

### Task 4: Toolbar, tool switching, side panel and depth display in toolbar/View menu

Replaces the region dock, the "Show measurement controls" button/action and the Pixel tool with exclusive tool actions; moves region options into the panel; moves Depth display to the toolbar and View menu.

**Files:**
- Modify: `src/fidmaa_gui/app.py` (`__init__`, `_create_region_measurement_panel`, `_create_tools_menu`, `keyPressEvent`, `_activate_pixel_tool` (delete), `_activate_region_selector`, `clear_region_measurement`, `_calculate_region_measurement`, `_loadImage`)
- Modify: `tests/test_region_selection_ui.py`
- Test: `tests/test_tool_interaction_ui.py` (create)

**Interfaces:**
- Consumes: Task 1 `MeasurementTool`, `PolylineEditor`, `HistoryEntry`; Task 3 `MeasurementPanel`.
- Produces on `MainWindow`:
  - `current_tool: MeasurementTool` (default `VECTOR`), `set_tool(tool, *args) -> None`
  - `toolActions: dict[MeasurementTool, QAction]`, `toolBar: QToolBar`, `openAction`, `open3DAction`
  - `measurementPanel: MeasurementPanel`, `regionOptionsWidget: QWidget`, `lineOptionsWidget: QWidget`, `surfaceStepSpin: QSpinBox` (2–7, default 3)
  - `polyline_editor: PolylineEditor`, `_tool_history_index: int | None`
  - `_record_history(tool, points, linear_mm, surface_mm) -> None`
  - `_update_tool_result() -> None` (stub here that only shows the hint; Task 5 fills in vector/polyline measurement)
  - `depthDisplayActions: dict[DepthDisplayMode, QAction]` (View → Depth display); `depthDisplayCombo` and `depthContourStepSpin` keep their names and now live on the toolbar.
  - Removed: `regionDock`, `regionSelectionButton`, `measurementControlsAction`, `pixelToolAction`, `_activate_pixel_tool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_interaction_ui.py
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from fidmaa_gui.app import MainWindow
from fidmaa_gui.depth_visualization import DepthDisplayMode
from fidmaa_gui.measurement_tools import MeasurementTool
from fidmaa_gui.region_measurement import SelectionMode


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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tool_interaction_ui.py -v`
Expected: FAIL with `AttributeError: 'MainWindow' object has no attribute 'current_tool'`

- [ ] **Step 3: Implement in `app.py`**

3a. Imports — add to the PySide6 imports and local imports:

```python
from PySide6.QtGui import QAction, QActionGroup, QColor
```

add `QToolBar` to the `PySide6.QtWidgets` import list, and:

```python
from .measurement_panel import MeasurementPanel
from .measurement_tools import (
    HistoryEntry,
    MeasurementTool,
    PolylineEditor,
    measure_polyline,
)
```

(`measure_polyline` is used in Task 5; importing it now is fine only if ruff does not flag it — if ruff flags it as unused, add it in Task 5 instead.)

3b. `__init__` — replace

```python
        self.load_ui()
        self._create_zoom_panel()
        self._create_region_measurement_panel()
        self._create_tools_menu()
```

with

```python
        self.current_tool = MeasurementTool.VECTOR
        self.polyline_editor = PolylineEditor(max_points=2)
        self._tool_history_index = None
        self._point_drag_index = None
        self._suppress_next_click = False
        self.load_ui()
        self._create_zoom_panel()
        self._create_measurement_panel()
        self._create_toolbar_and_menus()
```

and at the end of `__init__`, immediately **before** the existing `self.redrawImage()` / `self.redrawZoom()` calls, add:

```python
        self.set_tool(MeasurementTool.VECTOR)
```

(`set_tool` triggers `redrawZoom()` via `_set_region_selection_enabled`, which needs `self.smallImage`, `self.depthmap`, `self.portrait` etc. to exist already.)

3c. Replace the whole `_create_region_measurement_panel` method with `_create_measurement_panel` (region widgets are unchanged except: no dock, no selection button, no depth group; they go into `regionOptionsWidget` inside the panel; the panel is inserted into the form's right-hand column above `dataOutputEdit`):

```python
    def _create_measurement_panel(self):
        self.measurementPanel = MeasurementPanel()

        self.lineOptionsWidget = QGroupBox("Surface sampling")
        line_form = QFormLayout(self.lineOptionsWidget)
        self.surfaceStepSpin = QSpinBox()
        self.surfaceStepSpin.setRange(2, 7)
        self.surfaceStepSpin.setValue(3)
        self.surfaceStepSpin.setSuffix(" px")
        self.surfaceStepSpin.setToolTip(
            "Median-filtered surface profile is sampled every N display pixels."
        )
        line_form.addRow("Step:", self.surfaceStepSpin)
        self.surfaceStepSpin.valueChanged.connect(self._update_tool_result)
        self.measurementPanel.optionsLayout.addWidget(self.lineOptionsWidget)

        self.regionOptionsWidget = QWidget()
        layout = QVBoxLayout(self.regionOptionsWidget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.regionModeACombo, self.regionMaskACombo = self._create_region_options(
            layout, "Region A", SelectionMode.LOCAL_PEAK
        )
        self.regionModeBCombo, self.regionMaskBCombo = self._create_region_options(
            layout, "Region B", SelectionMode.LOCAL_VALLEY
        )

        settings_group = QGroupBox("Sampling")
        settings_form = QFormLayout(settings_group)
        self.regionRadiusSpin = QSpinBox()
        self.regionRadiusSpin.setRange(10, 120)
        self.regionRadiusSpin.setValue(20)
        self.regionRadiusSpin.setSuffix(" px")
        self.regionRadiusSpin.setToolTip(
            "Shared radius of both click-to-place measurement patches."
        )
        settings_form.addRow("Patch radius:", self.regionRadiusSpin)
        self.regionPercentileSpin = QSpinBox()
        self.regionPercentileSpin.setRange(5, 20)
        self.regionPercentileSpin.setValue(5)
        self.regionPercentileSpin.setSuffix(" %")
        settings_form.addRow("Candidate pool:", self.regionPercentileSpin)
        self.regionVectorCountSpin = QSpinBox()
        self.regionVectorCountSpin.setRange(5, 30)
        self.regionVectorCountSpin.setValue(10)
        settings_form.addRow("Profiles:", self.regionVectorCountSpin)
        layout.addWidget(settings_group)

        clear_button = QPushButton("Clear regions")
        clear_button.clicked.connect(self.clear_region_measurement)
        layout.addWidget(clear_button)

        self.regionResultEdit = QPlainTextEdit()
        self.regionResultEdit.setReadOnly(True)
        self.regionResultEdit.setMinimumWidth(270)
        self.regionResultEdit.setPlainText(
            "Click once to place region A, then once to place region B."
        )
        layout.addWidget(self.regionResultEdit, stretch=1)
        self.measurementPanel.optionsLayout.addWidget(self.regionOptionsWidget)

        column = self.ui.findChild(QVBoxLayout, "verticalLayout")
        column.insertWidget(column.indexOf(self.ui.dataOutputEdit), self.measurementPanel)

        for widget in (
            self.regionModeACombo,
            self.regionMaskACombo,
            self.regionModeBCombo,
            self.regionMaskBCombo,
        ):
            widget.currentIndexChanged.connect(self._region_settings_changed)
        self.regionPercentileSpin.valueChanged.connect(self._region_settings_changed)
        self.regionVectorCountSpin.valueChanged.connect(self._region_settings_changed)
        self.regionRadiusSpin.valueChanged.connect(self._region_radius_changed)
```

3d. Replace `_create_tools_menu` with `_create_toolbar_and_menus`:

```python
    def _create_toolbar_and_menus(self):
        self.toolBar = QToolBar("Tools", self)
        self.toolBar.setObjectName("mainToolBar")
        self.toolBar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.openAction = QAction("Open…", self)
        self.openAction.setShortcut(QtGui.QKeySequence.Open)
        self.openAction.triggered.connect(self.loadJPEG)
        self.open3DAction = QAction("3D view", self)
        self.open3DAction.triggered.connect(self.open3DView)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.openAction)
        file_menu.addAction(self.open3DAction)

        self.toolBar.addAction(self.openAction)
        self.toolBar.addAction(self.open3DAction)
        self.toolBar.addSeparator()

        tools_menu = self.menuBar().addMenu("&Tools")
        self.toolActionGroup = QActionGroup(self)
        self.toolActionGroup.setExclusive(True)
        self.toolActions = {}
        tool_details = (
            (MeasurementTool.VECTOR, Qt.Key_V, "Vector: two clicks, linear and surface length"),
            (
                MeasurementTool.POLYLINE,
                Qt.Key_M,
                "Polyline: click points, finish with Enter or right-click",
            ),
            (MeasurementTool.REGION, Qt.Key_R, "Region: two patches A and B"),
        )
        for tool, key, tip in tool_details:
            action = QAction(tool.value, self)
            action.setCheckable(True)
            action.setShortcut(QtGui.QKeySequence(key))
            action.setShortcutContext(Qt.WindowShortcut)
            action.setStatusTip(tip)
            action.setToolTip(tip)
            action.triggered.connect(partial(self.set_tool, tool))
            self.toolActionGroup.addAction(action)
            self.toolActions[tool] = action
            tools_menu.addAction(action)
            self.toolBar.addAction(action)

        tools_menu.addSeparator()
        action_details = (
            (SelectionMode.HIGHEST, Qt.Key_H, "Highest region tool"),
            (SelectionMode.LOWEST, Qt.Key_L, "Lowest region tool"),
            (SelectionMode.FLATTEST, Qt.Key_F, "Flattest region tool"),
            (SelectionMode.LOCAL_PEAK, None, "Local peak region tool"),
            (SelectionMode.LOCAL_VALLEY, None, "Local valley region tool"),
            (SelectionMode.AREA, None, "Uniform area region tool"),
        )
        self.regionToolActions = {}
        for mode, key, text in action_details:
            action = tools_menu.addAction(text)
            if key is not None:
                action.setShortcut(QtGui.QKeySequence(key))
                action.setShortcutContext(Qt.WindowShortcut)
            action.setStatusTip(f"Use {mode.value.lower()} candidates for the active region")
            action.triggered.connect(partial(self._activate_region_selector, mode))
            self.regionToolActions[mode] = action

        self.toolBar.addSeparator()
        self.toolBar.addWidget(QLabel(" Depth: "))
        self.depthDisplayCombo = QComboBox()
        for mode in DepthDisplayMode:
            self.depthDisplayCombo.addItem(mode.value, mode)
        self.depthDisplayCombo.setCurrentIndex(
            self.depthDisplayCombo.findData(DepthDisplayMode.COLOR)
        )
        self.depthDisplayCombo.setToolTip(
            "Display-only enhancement; measurement data is never remapped."
        )
        self.toolBar.addWidget(self.depthDisplayCombo)
        self.depthContourStepSpin = QSpinBox()
        self.depthContourStepSpin.setRange(1, 8)
        self.depthContourStepSpin.setValue(4)
        self.depthContourStepSpin.setSuffix(" levels")
        self.depthContourStepSpin.setToolTip(
            "Draw one contour for every N raw depth levels. Measurements are unchanged."
        )
        self.depthContourStepSpin.setEnabled(False)
        self.toolBar.addWidget(self.depthContourStepSpin)

        self.viewMenu = self.menuBar().addMenu("&View")
        depth_menu = self.viewMenu.addMenu("Depth display")
        depth_group = QActionGroup(self)
        depth_group.setExclusive(True)
        self.depthDisplayActions = {}
        for mode in DepthDisplayMode:
            action = QAction(mode.value, self)
            action.setCheckable(True)
            action.triggered.connect(
                partial(
                    lambda m, *args: self.depthDisplayCombo.setCurrentIndex(
                        self.depthDisplayCombo.findData(m)
                    ),
                    mode,
                )
            )
            depth_group.addAction(action)
            depth_menu.addAction(action)
            self.depthDisplayActions[mode] = action
        self.depthDisplayActions[DepthDisplayMode.COLOR].setChecked(True)

        self.depthDisplayCombo.currentIndexChanged.connect(self._depth_display_changed)
        self.depthContourStepSpin.valueChanged.connect(self._depth_display_changed)
```

3e. `_depth_display_changed` — also sync the View menu:

```python
    def _depth_display_changed(self, *args):
        mode = self.depthDisplayCombo.currentData()
        self.depthContourStepSpin.setEnabled(mode == DepthDisplayMode.COLOR_CONTOURS)
        self.depthDisplayActions[mode].setChecked(True)
        self.redrawZoom()
```

3f. Tool switching, history and result stub (add after `_create_toolbar_and_menus`):

```python
    TOOL_HINTS = {
        MeasurementTool.VECTOR: "Click two points. Drag a point to adjust it. Esc clears.",
        MeasurementTool.POLYLINE: (
            "Click points; finish with Enter or right-click. "
            "Backspace removes the last point, Esc clears."
        ),
        MeasurementTool.REGION: "Click to place region A, then region B. Drag to move.",
    }

    def set_tool(self, tool, *args):
        self.current_tool = tool
        self.toolActions[tool].setChecked(True)
        self.polyline_editor = PolylineEditor(
            max_points=2 if tool == MeasurementTool.VECTOR else None
        )
        self._tool_history_index = None
        self._point_drag_index = None
        self.regionOptionsWidget.setVisible(tool == MeasurementTool.REGION)
        self.lineOptionsWidget.setVisible(tool != MeasurementTool.REGION)
        self._set_region_selection_enabled(tool == MeasurementTool.REGION)
        self._update_tool_result()

    def _record_history(self, tool, points, linear_mm, surface_mm):
        entry = HistoryEntry(
            filename=os.path.basename(self.filename) if self.filename else "",
            tool=tool,
            points=points,
            linear_mm=linear_mm,
            surface_mm=surface_mm,
        )
        if self._tool_history_index is None:
            self._tool_history_index = self.measurementPanel.add_history(entry)
        else:
            self.measurementPanel.replace_history(self._tool_history_index, entry)

    def _update_tool_result(self, *args):
        if self.current_tool == MeasurementTool.REGION:
            self._show_region_tool_result()
            return
        self.measurementPanel.set_tool_result(
            self.current_tool.value, None, None, [], self.TOOL_HINTS[self.current_tool]
        )

    def _show_region_tool_result(self):
        result = self.region_measurement_result
        linear = result.linear_stats() if result is not None else None
        surface = result.surface_stats() if result is not None else None
        self.measurementPanel.set_tool_result(
            MeasurementTool.REGION.value,
            linear.mean_mm if linear is not None else None,
            surface.mean_mm if surface is not None else None,
            ["Mean of region profiles; details below."] if result is not None else [],
            self.TOOL_HINTS[MeasurementTool.REGION],
        )
```

3g. `keyPressEvent` — replace the body's `Qt.Key_P` branch with tool keys (keep the H/L/F block as is):

```python
    def keyPressEvent(self, event):
        """Keep the single-key measurement tools reliable across Qt platforms."""
        if event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
            tool_by_key = {
                Qt.Key_V: MeasurementTool.VECTOR,
                Qt.Key_M: MeasurementTool.POLYLINE,
                Qt.Key_R: MeasurementTool.REGION,
            }
            tool = tool_by_key.get(event.key())
            if tool is not None:
                self.toolActions[tool].trigger()
                event.accept()
                return

            mode_by_key = {
                Qt.Key_H: SelectionMode.HIGHEST,
                Qt.Key_L: SelectionMode.LOWEST,
                Qt.Key_F: SelectionMode.FLATTEST,
            }
            mode = mode_by_key.get(event.key())
            if mode is not None:
                self.regionToolActions[mode].trigger()
                event.accept()
                return

        super().keyPressEvent(event)
```

3h. Delete `_activate_pixel_tool`. In `_activate_region_selector` replace `self.regionSelectionButton.setChecked(True)` with:

```python
        if self.current_tool != MeasurementTool.REGION:
            self.set_tool(MeasurementTool.REGION)
```

3i. `clear_region_measurement` — replace the `_region_target` expression and reset the history row:

```python
        self._region_target = (
            "a" if getattr(self, "current_tool", None) == MeasurementTool.REGION else None
        )
        self._tool_history_index = None
```

3j. `_calculate_region_measurement` — at its end (after `self.regionResultEdit.setPlainText("\n".join(lines))`) add:

```python
        self._record_history(
            MeasurementTool.REGION,
            2,
            linear.mean_mm if linear is not None else None,
            surface.mean_mm if surface is not None else None,
        )
        self._show_region_tool_result()
```

3k. `connect_ui` — hide the form's Load / 3D buttons (the toolbar replaces them):

```python
        self.ui.loadJPEGButton.hide()
        self.ui.open3DViewButton.hide()
```

3l. `_loadImage` — after `self.clear_region_measurement(repaint=False)` add `self.set_tool(self.current_tool)` so an in-progress measurement is dropped when another photo is opened (history rows are kept).

- [ ] **Step 4: Update `tests/test_region_selection_ui.py`**

Add `from fidmaa_gui.measurement_tools import MeasurementTool` to its imports, then:

```bash
sed -i '' 's/window.regionSelectionButton.setChecked(True)/window.set_tool(MeasurementTool.REGION)/; s/window.regionSelectionButton.setChecked(False)/window.set_tool(MeasurementTool.VECTOR)/' tests/test_region_selection_ui.py
```

Then edit by hand:
- In `test_single_click_places_a_fixed_radius_patch` replace the two lines asserting `regionSelectionButton.isChecked()` / `measurementControlsAction.isChecked()` with `assert window.current_tool == MeasurementTool.REGION`.
- In `test_single_key_tool_shortcuts_and_tools_menu`: replace `assert window.regionSelectionButton.isChecked()` with `assert window.current_tool == MeasurementTool.REGION`; replace the `QTest.keyClick(window, Qt.Key_P)` block (click + 3 asserts) with

```python
    QTest.keyClick(window, Qt.Key_V)
    assert window._region_target is None
    assert window.current_tool == MeasurementTool.VECTOR
```

  and delete the line asserting `pixelToolAction.shortcut()`.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass (previous 147 + Tasks 1–4 tests)

- [ ] **Step 6: Manual smoke check**

Run: `uv run fidmaa_gui` (file dialog opens; cancel it). Expected: toolbar with Open…, 3D view, Vector/Polyline/Region, Depth combo; panel with Automatic / Vector / Surface sampling / History in the right column; no "Region measurement" dock. Close the window.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check --fix src/fidmaa_gui/app.py tests/
uv run ruff format src/fidmaa_gui/app.py tests/
git add src/fidmaa_gui/app.py tests/
git commit -m "Add measurement toolbar, tool modes and side panel"
```

---

### Task 5: Vector and Polyline interaction, result and painting

**Files:**
- Modify: `src/fidmaa_gui/app.py` (`_update_tool_result`, `connect_ui`, `keyPressEvent`, `redrawImage`, `_paint_zoom_region_overlay`, delete `setMidlinePoint`)
- Test: `tests/test_tool_interaction_ui.py` (append)

**Interfaces:**
- Consumes: Task 1 `measure_polyline`, `PolylineEditor`; Task 2 `rightClicked`; Task 4 `set_tool`, `_record_history`, `polyline_editor`, `surfaceStepSpin`, `measurementPanel`.
- Produces on `MainWindow`: `_on_image_clicked(point)`, `_on_image_right_clicked(point)`, `_finish_polyline()`, `_begin_point_drag(point)`, `_drag_point(point)`, `_finish_point_drag(point)`, `_measure_current_polyline() -> PolylineResult | None`, `_paint_polyline_overlay(painter, point_mapper)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_tool_interaction_ui.py`)

```python
from PIL import Image
from portrait_analyser.depth_sampling import median_filter_depthmap
from PySide6.QtCore import QPointF

from fidmaa_gui.measurement_tools import PolylineEditor


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


def test_loading_new_photo_clears_points_but_keeps_history(qapp):
    window = _window_with_flat_depth()
    window._on_image_clicked(QPointF(100, 100))
    window._on_image_clicked(QPointF(100, 300))
    window._on_image_clicked(QPointF(150, 150))

    window.set_tool(window.current_tool)  # what _loadImage does before loading

    assert window.polyline_editor.points == []
    assert len(window.measurementPanel.history.entries) == 1
    assert isinstance(window.polyline_editor, PolylineEditor)
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
```

Also add to the imports at the top of the file: `from PySide6 import QtGui` and `from PySide6.QtCore import QPoint`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tool_interaction_ui.py -v`
Expected: new tests FAIL with `AttributeError: 'MainWindow' object has no attribute '_on_image_clicked'`

- [ ] **Step 3: Implement in `app.py`**

3a. Replace the Task 4 `_update_tool_result` with the full version and add measurement helpers:

```python
    def _measure_current_polyline(self):
        points = self.polyline_editor.points
        if self.depthmap is None or not hasattr(self, "image") or len(points) < 2:
            return None
        step = self.surfaceStepSpin.value()
        return measure_polyline(
            points,
            linear_mm=lambda a, b: self.vector_length_between_two_clicks(a[0], a[1], b[0], b[1]),
            surface_mm=lambda a, b: self.surface_vector_filtered(a[0], a[1], b[0], b[1], step),
        )

    def _update_tool_result(self, *args):
        if self.current_tool == MeasurementTool.REGION:
            self._show_region_tool_result()
            return
        hint = self.TOOL_HINTS[self.current_tool]
        if self.depthmap is None:
            hint = "Open a photo with depth data first. " + hint
        result = self._measure_current_polyline()
        if result is None or not result.segments:
            self.measurementPanel.set_tool_result(self.current_tool.value, None, None, [], hint)
            return

        details = []
        if self.current_tool == MeasurementTool.VECTOR:
            (ax, ay), (bx, by) = self.polyline_editor.points[:2]
            depth_a_mm = self.get_depthmap_distance(self.get_depthmap_value(ax, ay)) * 10
            depth_b_mm = self.get_depthmap_distance(self.get_depthmap_value(bx, by)) * 10
            details.append(f"Δ depth: {depth_b_mm - depth_a_mm:+.1f} mm")
        else:
            state = "finished" if self.polyline_editor.finished else "in progress"
            details.append(f"{len(result.segments)} segments ({state})")
            for number, segment in enumerate(result.segments, start=1):
                details.append(
                    f"  {number}. {format_cm(segment.linear_mm)} / {format_cm(segment.surface_mm)}"
                )
        details.append(f"Surface step: {self.surfaceStepSpin.value()} px (median filtered)")
        self.measurementPanel.set_tool_result(
            self.current_tool.value, result.linear_mm, result.surface_mm, details, hint
        )
        if self.polyline_editor.finished:
            self._record_history(
                self.current_tool,
                len(self.polyline_editor.points),
                result.linear_mm,
                result.surface_mm,
            )
```

and add `format_cm` to the panel import: `from .measurement_panel import MeasurementPanel, format_cm`.

3b. Click, right-click, drag handlers and polyline finishing:

```python
    POINT_GRAB_TOLERANCE = 6

    @staticmethod
    def _clamped_display_point(point):
        return (
            float(clamp(point.x(), 0, const.MAIN_IMAGE_WIDTH)),
            float(clamp(point.y(), 0, const.MAIN_IMAGE_HEIGHT)),
        )

    def _on_image_clicked(self, point, *args):
        if self.current_tool == MeasurementTool.REGION:
            return
        if self._suppress_next_click:
            self._suppress_next_click = False
            return
        x, y = self._clamped_display_point(point)
        if self.polyline_editor.add_point((x, y)):
            self._tool_history_index = None
        self.ui.xValue.setValue(int(x))
        self.ui.yValue.setValue(int(y))
        self._update_tool_result()
        self._redraw_tool_overlay()

    def _on_image_right_clicked(self, point, *args):
        self._finish_polyline()

    def _finish_polyline(self):
        if self.current_tool != MeasurementTool.POLYLINE:
            return
        if self.polyline_editor.finished or not self.polyline_editor.finish():
            return
        self._update_tool_result()
        self._redraw_tool_overlay()

    def _begin_point_drag(self, point, *args):
        if self.current_tool == MeasurementTool.REGION:
            return
        index = self.polyline_editor.point_near(
            self._clamped_display_point(point), self.POINT_GRAB_TOLERANCE
        )
        self._point_drag_index = index
        # A press on an existing point is a drag; the click emitted on release
        # (when the pointer moved <= 3 px) must not add a new point.
        self._suppress_next_click = index is not None

    def _drag_point(self, point, *args):
        if self._point_drag_index is None:
            return
        self.polyline_editor.move_point(self._point_drag_index, self._clamped_display_point(point))
        self._update_tool_result()
        self._redraw_tool_overlay()

    def _finish_point_drag(self, point, *args):
        if self._point_drag_index is None:
            return
        self._drag_point(point)
        self._point_drag_index = None

    def _redraw_tool_overlay(self):
        if self.portrait:
            self.redrawImage(force=True)
        self.redrawZoom()
```

3c. `keyPressEvent` — before the final `super().keyPressEvent(event)` add:

```python
        if self.current_tool != MeasurementTool.REGION:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._finish_polyline()
                event.accept()
                return
            if event.key() in (Qt.Key_Backspace, Qt.Key_Escape):
                if event.key() == Qt.Key_Backspace:
                    self.polyline_editor.undo()
                else:
                    self.polyline_editor.clear()
                    self._tool_history_index = None
                self._update_tool_result()
                self._redraw_tool_overlay()
                event.accept()
                return
```

3d. Painting — add the method:

```python
    POLYLINE_COLOR = QColor(255, 220, 0, 230)

    def _paint_polyline_overlay(self, painter, point_mapper):
        points = self.polyline_editor.points
        if self.current_tool == MeasurementTool.REGION or not points:
            return
        painter.save()
        try:
            mapped = [point_mapper(point) for point in points]
            painter.setPen(QtGui.QPen(self.POLYLINE_COLOR, 2))
            painter.setBrush(Qt.NoBrush)
            for start, end in zip(mapped, mapped[1:], strict=False):
                painter.drawLine(start, end)
            for number, point in enumerate(mapped, start=1):
                painter.setPen(Qt.NoPen)
                painter.setBrush(self.POLYLINE_COLOR)
                painter.drawEllipse(point, 4, 4)
                painter.setPen(self.POLYLINE_COLOR)
                painter.drawText(point + QPoint(6, -6), str(number))
        finally:
            painter.restore()
```

In `redrawImage`, right after `self._paint_region_overlay(painter)` add:

```python
            self._paint_polyline_overlay(painter, lambda p: QPoint(round(p[0]), round(p[1])))
```

In `_paint_zoom_region_overlay`, compute `show_polyline = self.current_tool != MeasurementTool.REGION and bool(self.polyline_editor.points)`, change the early return to `if not show_regions and not show_neck and not show_polyline: return`, and inside the `try:` after the `if show_neck:` block add:

```python
            if show_polyline:
                self._paint_polyline_overlay(painter, display_point)
```

3e. `connect_ui` — replace `self.ui.imageLabel.clicked.connect(self.setMidlinePoint)` with:

```python
        self.ui.imageLabel.clicked.connect(self._on_image_clicked)
        self.ui.imageLabel.rightClicked.connect(self._on_image_right_clicked)
        self.ui.imageLabel.pressed.connect(self._begin_point_drag)
        self.ui.imageLabel.dragged.connect(self._drag_point)
        self.ui.imageLabel.released.connect(self._finish_point_drag)
```

Delete the now-unused `setMidlinePoint` method.

Note: `redrawImage` still updates the legacy x/y line and output text from `xValue`/`yValue`; `_on_image_clicked` keeps those spin boxes following the last click so the hidden legacy chart (Task 7) stays meaningful.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Manual smoke check**

Run: `uv run fidmaa_gui <path to any .heic portrait>`. Vector: click nose tip and chin → big linear/surface values in cm, one history row; drag a point → the row updates. M: click 5 points along the jaw, press Enter (then try a two-finger click on a second polyline) → rows appear; Copy → paste into a spreadsheet shows 5 columns. Close.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
uv run ruff format src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
git add src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
git commit -m "Add vector and polyline measurement tools"
```

---

### Task 6: Automatic results; remove TMD; FaceMesh mouth opening opt-in

**Files:**
- Modify: `src/fidmaa_gui/app.py` (imports, `_run_detection_and_update`, `redrawImage`, new `_create_analysis_menu`, `_compute_mouth_measurement`, `_update_automatic_results`, `_diagnostics_text`)
- Test: `tests/test_tool_interaction_ui.py` (append)

**Interfaces:**
- Consumes: Task 3 `MeasurementPanel.set_automatic_results`; Task 4 `self.viewMenu` (only for ordering) and menu bar.
- Produces on `MainWindow`: `mouthOpeningAction: QAction` (checkable, default unchecked), `forceExtendedNeckAction: QAction`, `_compute_mouth_measurement() -> None`, `_update_automatic_results() -> None`, `_diagnostics_text() -> str`. Removed: `tmd_mm`, `compute_tmd_3d` import, `last_5_distances_vect`, `last_5_distances_srfc`, the chin→neck dashed TMD drawing.

- [ ] **Step 1: Write the failing tests** (append)

```python
from types import SimpleNamespace

from portrait_analyser.pose import PortraitPose


def _neutral_window():
    window = _window_with_flat_depth()
    window.portrait = SimpleNamespace(
        teeth_bbox=None, teethmap=None, incisor_measurement=None, skinmap=None
    )
    window.neck_midpoint = SimpleNamespace(pose=PortraitPose.NEUTRAL_NECK)
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
    window.neck_midpoint = SimpleNamespace(pose=PortraitPose.OPEN_MOUTH)
    window.mouth_measurement = SimpleNamespace(distance_3d_mm=42.0)

    window._update_automatic_results()
    assert "Mouth opening" not in window.measurementPanel.autoLabel.text()

    window.mouthOpeningAction.setChecked(True)
    window._update_automatic_results()
    assert "Mouth opening" in window.measurementPanel.autoLabel.text()
    window.close()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tool_interaction_ui.py -v -k "automatic or low_confidence or no_face or mouth"`
Expected: FAIL with `AttributeError: ... '_update_automatic_results'`

- [ ] **Step 3: Implement in `app.py`**

3a. Delete `from portrait_analyser.tmd import compute_tmd_3d`. In `__init__` delete `self.last_5_distances_vect = []` and `self.last_5_distances_srfc = []`, and initialise the detection results so the panel can be refreshed before any photo:

```python
        self.neck_midpoint = None
        self.mouth_measurement = None
        self.segmentation_debug = None
```

(place them next to `self.neck_arc = None`). After `self._create_toolbar_and_menus()` add `self._create_analysis_menu()`.

3b. Analysis menu:

```python
    def _create_analysis_menu(self):
        analysis_menu = self.menuBar().addMenu("&Analysis")
        self.forceExtendedNeckAction = analysis_menu.addAction("Force extended neck analysis")
        self.forceExtendedNeckAction.setStatusTip(
            "Skip MediaPipe and re-run analysis using segmentation/depth profile method"
        )
        self.forceExtendedNeckAction.triggered.connect(self.forceExtendedNeckAnalysis)
        analysis_menu.addSeparator()
        self.mouthOpeningAction = analysis_menu.addAction(
            "Mouth opening from FaceMesh (experimental)"
        )
        self.mouthOpeningAction.setCheckable(True)
        self.mouthOpeningAction.setChecked(False)
        self.mouthOpeningAction.setStatusTip(
            "FaceMesh lip landmarks can be wrong; only used when teeth detection fails"
        )
        self.mouthOpeningAction.toggled.connect(self._mouth_opening_toggled)

    def _mouth_opening_toggled(self, *args):
        if self.portrait:
            self._compute_mouth_measurement()
            self.redrawImage(force=True)
        self._update_automatic_results()
```

3c. In `_run_detection_and_update`: delete `self.tmd_mm = None` and the whole `# Compute thyromental distance` block (the `if` starting with `self.neck_midpoint and self.portrait.depthmap and self.neck_midpoint.x is not None and ... in (PortraitPose.EXTENDED_NECK, PortraitPose.NEUTRAL_NECK)` through the `print("TMD computation failed ...")` line). Replace the `# FaceMesh mouth opening fallback` block with a call `self._compute_mouth_measurement()` and move that block into:

```python
    def _compute_mouth_measurement(self):
        """FaceMesh mouth opening fallback, only when the user opted in."""
        self.mouth_measurement = None
        if not self.mouthOpeningAction.isChecked():
            return
        if (
            self.neck_midpoint
            and self.neck_midpoint.pose == PortraitPose.OPEN_MOUTH
            and self.portrait.incisor_measurement is None
            and self.face_mesh_debug
            and self.portrait.depthmap
        ):
            self.mouth_measurement = compute_mouth_measurement_from_facemesh(
                landmarks=self.face_mesh_debug.landmarks,
                depthmap=self.portrait.depthmap,
                photo_w=self.image.size[0],
                photo_h=self.image.size[1],
                float_min=self.portrait.floatValueMin,
                float_max=self.portrait.floatValueMax,
            )
            if self.mouth_measurement is None or self.mouth_measurement.distance_3d_mm is None:
                print("FaceMesh mouth measurement: depth conversion failed")
```

At the end of `_run_detection_and_update`, after `self.redrawImage()`, add:

```python
        self._update_automatic_results()
        self.ui.dataOutputEdit.setPlainText(self._diagnostics_text())
```

3d. In `redrawImage` delete the `# Draw chin point and TMD line (always, no checkbox)` block (the `if` through `painter.drawLine(QPoint(int(cx), int(cy)), QPoint(int(mx), int(my)))`).

3e. Replace the whole `if self.depthmap:` block at the end of `redrawImage` (from `if self.depthmap:` / `mouse_x = clamp(mouse_x, 0, img_w)` down to `self.ui.dataOutputEdit.appendPlainText(txt.strip())`) with only the legacy click bookkeeping still used by the red line / chart:

```python
        if self.depthmap:
            self.last_click_x = mouse_x
            self.last_click_y = mouse_y
```

3f. Add the automatic-results and diagnostics builders. `_diagnostics_text` is the pose-gated part of the removed block, moved verbatim except: the TMD lines are dropped, the FaceMesh mouth line is only included when `self.mouthOpeningAction.isChecked()`, and the legacy incisor distance recomputes its endpoints:

```python
    def _update_automatic_results(self):
        pose = self.neck_midpoint.pose if self.neck_midpoint else None
        if self.portrait is None and pose is None and self.filename is None:
            self.measurementPanel.set_automatic_results(
                ["Open a photo to run the automatic analysis."], None
            )
            return
        if pose is None:
            self.measurementPanel.set_automatic_results(
                ["Pose: not detected"],
                "Face not detected. Try Analysis → Force extended neck analysis.",
            )
            return

        lines = [f"Pose: {pose.value.upper()}"]
        warning = None
        if pose == PortraitPose.NEUTRAL_NECK:
            if self.neck_arc is not None:
                lines.append(f"Neck circumference: {format_cm(self.neck_arc.circumference_mm)}")
            else:
                lines.append("Neck circumference: unavailable (no stable neck level)")
            if self.neck_measurement_auto is not None:
                lines.append(
                    "Neck (skin-matte arc): "
                    f"{format_cm(self.neck_measurement_auto.circumference_mm)}"
                )
                if (
                    self.neck_confidence_ratio is not None
                    and self.neck_confidence_ratio >= const.NECK_LOW_CONFIDENCE_RATIO_THRESHOLD
                ):
                    warning = (
                        "Skin-matte neck arc: low confidence (may include shoulder/chest skin)."
                    )
        elif pose == PortraitPose.OPEN_MOUTH:
            incisor = self.portrait.incisor_measurement if self.portrait else None
            if incisor is not None and incisor.distance_3d_mm is not None:
                lines.append(f"Incisor distance: {format_cm(incisor.distance_3d_mm)}")
            else:
                lines.append("Incisor distance: teeth not detected")
            if (
                self.mouthOpeningAction.isChecked()
                and self.mouth_measurement is not None
                and self.mouth_measurement.distance_3d_mm is not None
            ):
                lines.append(
                    "Mouth opening (FaceMesh, experimental): "
                    f"{format_cm(self.mouth_measurement.distance_3d_mm)}"
                )
        elif pose == PortraitPose.EXTENDED_NECK:
            sd = self.segmentation_debug
            if sd is not None and sd.neck_width_front_arc_mm:
                lines.append(f"Neck width (3D front arc): {format_cm(sd.neck_width_front_arc_mm)}")
                lines.append(f"Neck width (3D straight): {format_cm(sd.neck_width_straight_mm)}")
        self.measurementPanel.set_automatic_results(lines, warning)

    def _diagnostics_text(self):
        pose = self.neck_midpoint.pose if self.neck_midpoint else None
        txt = f"*** {pose.value.upper() if pose else 'UNKNOWN'} ***"
        # ... move here, unchanged, the three pose branches of the removed block:
        #     `if pose == PortraitPose.OPEN_MOUTH:` / `elif pose == PortraitPose.NEUTRAL_NECK:`
        #     / `elif pose == PortraitPose.EXTENDED_NECK:` — all their `txt += ...` lines.
        return txt.strip()
```

When moving the OPEN_MOUTH branch, replace `*auto_id_click_1, *auto_id_click_2` with endpoints computed locally:

```python
                    if self.portrait.incisor_distance:
                        img_size = (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT)
                        teeth_size = self.portrait.teethmap.size
                        start = translate_coordinates_to_other_image(
                            self.portrait.incisor_distance[:2], teeth_size, img_size
                        )
                        end = translate_coordinates_to_other_image(
                            self.portrait.incisor_distance[2:4], teeth_size, img_size
                        )
                        txt += "\nAutomatic incisor distance (legacy): "
                        txt += "%.2f cm" % (
                            self.vector_length_between_two_clicks(*start, *end) / 10.0
                        )
```

and wrap the `[FaceMesh] Mouth opening` lines in `if self.mouthOpeningAction.isChecked():`. The `_diagnostics_text` comment block above is the only place in this plan where code is moved rather than written out: the moved lines are the existing `txt += ...` statements between `if pose == PortraitPose.OPEN_MOUTH:` and the removed `if self.tmd_mm is not None:` in the current `redrawImage`, kept byte-for-byte apart from the two edits just listed.

3g. `_loadImage`: at the start (after `self.filename = fileName`) nothing else changes; `_run_detection_and_update` now refreshes the panel.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass. Also `grep -n "tmd\|TMD\|Thyromental" src/fidmaa_gui/app.py` → no output.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
uv run ruff format src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
git add src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
git commit -m "Show automatic results in panel, drop TMD, make FaceMesh mouth opt-in"
```

---

### Task 7: Hide legacy depth-profile UI under View → Advanced

**Files:**
- Modify: `src/fidmaa_gui/app.py` (`_create_toolbar_and_menus` or a new `_create_advanced_menu`, `redrawImage`, `connect_ui`)
- Test: `tests/test_tool_interaction_ui.py` (append)

**Interfaces:**
- Consumes: Task 4 `self.viewMenu`.
- Produces on `MainWindow`: `depthProfileAction: QAction` (checkable, default off), `diagnosticsAction: QAction` (checkable, default off), `_legacy_profile_widgets() -> list[QWidget]`, `_set_depth_profile_visible(visible: bool)`.

- [ ] **Step 1: Write the failing test** (append)

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_tool_interaction_ui.py::test_legacy_depth_profile_ui_is_hidden_by_default_and_toggleable -v`
Expected: FAIL with `AttributeError: ... '_legacy_profile_widgets'`

- [ ] **Step 3: Implement in `app.py`**

```python
    def _create_advanced_menu(self):
        advanced = self.viewMenu.addMenu("Advanced")
        self.depthProfileAction = advanced.addAction("Depth profile chart (blue line, x/y/angle)")
        self.depthProfileAction.setCheckable(True)
        self.depthProfileAction.toggled.connect(self._set_depth_profile_visible)
        self.diagnosticsAction = advanced.addAction("Diagnostics text")
        self.diagnosticsAction.setCheckable(True)
        self.diagnosticsAction.toggled.connect(self.ui.dataOutputEdit.setVisible)
        self._set_depth_profile_visible(False)
        self.ui.dataOutputEdit.setVisible(False)

    def _legacy_profile_widgets(self):
        return [
            self.ui.chartLabel,
            self.ui.label_3,
            self.ui.xValue,
            self.ui.label_4,
            self.ui.yValue,
            self.ui.label_5,
            self.ui.angleValue,
            self.ui.angleSlider,
            self.reconstructionLabel,
        ]

    def _set_depth_profile_visible(self, visible):
        for widget in self._legacy_profile_widgets():
            widget.setVisible(visible)
        if self.portrait:
            self.redrawImage(force=True)
```

Call `self._create_advanced_menu()` in `__init__` right after `self._create_analysis_menu()`.

In `redrawImage`, draw the blue profile line and the red last-click line only when the chart is shown — wrap these existing lines:

```python
            p1 = findPoint(x, y, direction=-1, angle=angle)
            p2 = findPoint(x, y, direction=1, angle=angle)
            if self.depthProfileAction.isChecked():
                painter.drawLine(p1, p2)

                if self.last_click_x is not None:
                    painter.setPen(QColor(255, 0, 0, 127))
                    painter.drawLine(
                        QPoint(mouse_x, mouse_y),
                        QPoint(self.last_click_x, self.last_click_y),
                    )
```

(`p1`/`p2` stay computed because the chart section below uses them.) `redrawImage` is called from `__init__` before the menus exist only if `self.portrait` is set — it returns early otherwise, so `self.depthProfileAction` always exists by the time this line runs.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Manual smoke check**

Run: `uv run fidmaa_gui <path to a .heic>`. Expected: no red chart, no blue line, no x/y/angle, no reconstruction box, no diagnostics text by default; View → Advanced toggles bring them back. Automatic box shows pose and neck circumference for a neutral-neck photo. Close.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
uv run ruff format src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
git add src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
git commit -m "Hide legacy depth-profile UI under View > Advanced"
```

---

### Task 8: FaceMesh landmarks in zoom views (drawn after scaling)

User request (2026-09-27): FaceMesh points can be shown **only on the main photo** or **everywhere** (main photo + zoomed photo, skin matte, depth map, hairless depth). In zoom views the crop is scaled **first** and the landmarks are painted **afterwards** at display resolution, so points and index labels stay sharp — never paint landmarks into the source image and then enlarge it.

The zoom painters (`_paintZoomedMap` for depth and hairless depth, `paintZoomedImage`, `paintZoomedSkinmap`, `paintHairlessDepthFull`) already call `_paint_zoom_region_overlay(painter, source_size, crop_box, display_rect)` on the scaled canvas; this task paints landmarks from that hook. The teeth zoom does not use the hook and is out of scope.

**Files:**
- Modify: `src/fidmaa_gui/app.py` (`_create_advanced_menu` neighbour: new `_create_face_mesh_menu`, `_paint_zoom_region_overlay`, `connect_ui`)
- Test: `tests/test_tool_interaction_ui.py` (append)

**Interfaces:**
- Consumes: Task 4 `self.viewMenu`; existing `self.ui.showFaceMeshLandmarksCheckBox`, `self.face_mesh_debug.landmarks` (photo-space `(x, y)` list), `self.image.size`, `_zoom_display_point`.
- Produces on `MainWindow`: `faceMeshModeActions: dict[str, QAction]` with keys `"off"`, `"main"`, `"everywhere"` (exclusive, View → FaceMesh landmarks); `_face_mesh_in_zoom() -> bool`; `_paint_face_mesh_zoom(painter, display_point) -> None`.
- Behaviour: the existing checkbox stays the on/off switch for the main photo and is kept in sync with the menu (`off` ⇔ unchecked; choosing `main`/`everywhere` checks it; unchecking it selects `off`; checking it selects `main` unless `everywhere` was already selected). Default `off` (as today).

- [ ] **Step 1: Write the failing tests** (append)

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tool_interaction_ui.py -v -k face_mesh`
Expected: FAIL with `AttributeError: ... 'faceMeshModeActions'`

- [ ] **Step 3: Implement in `app.py`**

3a. Menu and sync (call `self._create_face_mesh_menu()` in `__init__` right after `self._create_advanced_menu()`):

```python
    def _create_face_mesh_menu(self):
        menu = self.viewMenu.addMenu("FaceMesh landmarks")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.faceMeshModeActions = {}
        for key, text in (
            ("off", "Off"),
            ("main", "Main photo only"),
            ("everywhere", "Main photo and zoom views"),
        ):
            action = QAction(text, self)
            action.setCheckable(True)
            action.triggered.connect(partial(self._face_mesh_mode_selected, key))
            group.addAction(action)
            menu.addAction(action)
            self.faceMeshModeActions[key] = action
        self.faceMeshModeActions["off"].setChecked(True)
        self.ui.showFaceMeshLandmarksCheckBox.toggled.connect(self._face_mesh_checkbox_toggled)

    def _face_mesh_mode_selected(self, key, *args):
        self.faceMeshModeActions[key].setChecked(True)
        self.ui.showFaceMeshLandmarksCheckBox.setChecked(key != "off")
        self.redrawZoom()

    def _face_mesh_checkbox_toggled(self, checked):
        if not checked:
            self.faceMeshModeActions["off"].setChecked(True)
        elif self.faceMeshModeActions["off"].isChecked():
            self.faceMeshModeActions["main"].setChecked(True)
        self.redrawZoom()

    def _face_mesh_in_zoom(self):
        return (
            self.faceMeshModeActions["everywhere"].isChecked()
            and self.ui.showFaceMeshLandmarksCheckBox.isChecked()
            and self.face_mesh_debug is not None
        )
```

`_run_detection_and_update` already unchecks the checkbox when FaceMesh is missing; through `_face_mesh_checkbox_toggled` that selects `off` — no extra code. Initialise `self.face_mesh_debug = None` in `__init__` next to `self.neck_arc = None` if it is not initialised there yet.

3b. Painting at display resolution — add:

```python
    FACE_MESH_ZOOM_COLOR = QColor(255, 165, 0, 220)

    def _paint_face_mesh_zoom(self, painter, display_point):
        """Paint FaceMesh on an already scaled zoom canvas (sharp dots and labels)."""
        main_size = (const.MAIN_IMAGE_WIDTH, const.MAIN_IMAGE_HEIGHT)
        bounds = painter.device().rect() if painter.device() is not None else None
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        for index, landmark in enumerate(self.face_mesh_debug.landmarks):
            main_point = translate_coordinates_to_other_image(landmark, self.image.size, main_size)
            point = display_point(main_point)
            if bounds is not None and not bounds.contains(point):
                continue
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.FACE_MESH_ZOOM_COLOR)
            painter.drawEllipse(point, 3, 3)
            painter.setPen(self.FACE_MESH_ZOOM_COLOR)
            painter.setBrush(Qt.NoBrush)
            painter.drawText(point + QPoint(5, -3), str(index))
```

Label text is drawn to the upper-right of the dot, so the pixel 6 px to the right of the dot centre on the dot's row stays background (the test relies on that).

3c. In `_paint_zoom_region_overlay`: compute `show_face_mesh = self._face_mesh_in_zoom()`, extend the early return to `if not show_regions and not show_neck and not show_polyline and not show_face_mesh: return`, and inside `try:` add as the **last** block (on top of everything else):

```python
            if show_face_mesh:
                self._paint_face_mesh_zoom(painter, display_point)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Manual smoke check**

Run: `uv run fidmaa_gui <path to a .heic>`; View → FaceMesh landmarks → *Main photo and zoom views*; hover the lips: zoomed photo, skin matte, depth and hairless depth show sharp orange dots with readable indices. *Main photo only* removes them from the zooms. Close.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
uv run ruff format src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
git add src/fidmaa_gui/app.py tests/test_tool_interaction_ui.py
git commit -m "Show FaceMesh landmarks in zoom views at display resolution"
```
