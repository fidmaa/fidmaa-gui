"""Side panel: automatic results, the active tool's result, options and history."""

from __future__ import annotations

from PySide6.QtCore import QLocale, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .measurement_tools import (
    NECK_HISTORY_SURFACE_IS_CIRCUMFERENCE,
    HistoryEntry,
    MeasurementHistory,
    format_history_tsv,
)

HISTORY_HEADERS = ("File", "Tool", "Points", "Linear [cm]", "Surface [cm]")


DEFAULT_VALUE_LABELS = ("Linear (3D):", "Surface:")

SURFACE_COLUMN_HELP = "Vector/Polyline: length along the depth surface." + (
    " Neck: circumference — the arc-circle model (chord + surface arc, 3 or more"
    " points) when it fits, otherwise the midpoint of the width model's ellipse"
    " range (b/a 0.85–0.90 assumed)."
    if NECK_HISTORY_SURFACE_IS_CIRCUMFERENCE
    else " Neck: empty."
)


def _number(value_cm: float, decimals: int = 2) -> str:
    return QLocale().toString(float(value_cm), "f", decimals)


def format_decimal(value: float, decimals: int = 2) -> str:
    """A plain number in the user's locale (e.g. "1,280")."""
    return _number(value, decimals)


def format_cm(value_mm: float | None, decimals: int = 2) -> str:
    if value_mm is None:
        return "n/a (invalid depth)"
    return f"{_number(value_mm / 10.0, decimals)} cm"


def format_neck_circumference(ellipse_mm, circle_mm) -> str:
    """Circumference text, e.g. "≈ 39,1–40,2 cm (ellipse) · 42,2 cm (circle)".

    The ellipse range assumes a population b/a ratio; the circle model is
    π·W (a model, not a bound). Either part is left out when the library did
    not give it.
    """
    parts = []
    if ellipse_mm is not None:
        low, high = ellipse_mm
        parts.append(f"≈ {_number(low / 10.0, 1)}–{_number(high / 10.0, 1)} cm (ellipse)")
    if circle_mm is not None:
        parts.append(f"{format_cm(circle_mm, 1)} (circle)")
    return " · ".join(parts) if parts else "n/a"


class MeasurementPanel(QWidget):
    MINIMUM_WIDTH = 340
    HISTORY_MINIMUM_HEIGHT = 220

    historyCleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = MeasurementHistory()
        self.setMinimumWidth(self.MINIMUM_WIDTH)
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
        values.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.linearValueLabel = QLabel("—")
        self.surfaceValueLabel = QLabel("—")
        for label in (self.linearValueLabel, self.surfaceValueLabel):
            font = label.font()
            font.setPointSize(font.pointSize() + 5)
            font.setBold(True)
            label.setFont(font)
        values.addRow(DEFAULT_VALUE_LABELS[0], self.linearValueLabel)
        values.addRow(DEFAULT_VALUE_LABELS[1], self.surfaceValueLabel)
        self._valuesLayout = values
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
        self.historyTable.horizontalHeaderItem(3).setToolTip(
            "Vector/Polyline: straight 3D length. Neck: neck width."
        )
        self.historyTable.horizontalHeaderItem(4).setToolTip(SURFACE_COLUMN_HELP)
        self.historyTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.historyTable.verticalHeader().setVisible(False)
        self.historyTable.setMinimumHeight(self.HISTORY_MINIMUM_HEIGHT)
        header = self.historyTable.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
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

    def set_automatic_results(self, lines, warning, tooltip=None):
        self.autoLabel.setText("\n".join(lines))
        self.autoLabel.setToolTip(tooltip or "")
        if warning:
            self.autoWarningLabel.setText(f"⚠ {warning}")
            self.autoWarningLabel.show()
        else:
            self.autoWarningLabel.clear()
            self.autoWarningLabel.hide()

    def value_label_texts(self) -> tuple[str, str]:
        """Current captions of the two big values (e.g. "Linear (3D):")."""
        return tuple(
            self._valuesLayout.labelForField(label).text()
            for label in (self.linearValueLabel, self.surfaceValueLabel)
        )

    def set_tool_result(
        self,
        title,
        linear_mm,
        surface_mm,
        details,
        hint,
        unavailable=None,
        *,
        labels=DEFAULT_VALUE_LABELS,
        texts=None,
    ):
        """Show the active tool's result.

        ``unavailable`` replaces both values with that text (e.g.
        "n/a (implausible depth)") when the photo cannot be measured at all.
        ``labels`` captions the two big values (the Neck tool shows "Width
        (3D):" / "Circumference:"); ``texts`` gives both values as ready-made
        strings instead of formatting ``linear_mm`` / ``surface_mm``.
        """
        self.resultGroup.setTitle(title)
        for label, caption in zip(
            (self.linearValueLabel, self.surfaceValueLabel), labels, strict=True
        ):
            self._valuesLayout.labelForField(label).setText(caption)
        if unavailable:
            self.linearValueLabel.setText(unavailable)
            self.surfaceValueLabel.setText(unavailable)
        elif not details:
            # No measurement has been attempted yet (blank slate): dashes,
            # not "n/a (invalid depth)" which implies a real measurement
            # failed.
            self.linearValueLabel.setText("—")
            self.surfaceValueLabel.setText("—")
        elif texts is not None:
            self.linearValueLabel.setText(texts[0])
            self.surfaceValueLabel.setText(texts[1])
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
        self.historyCleared.emit()

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
