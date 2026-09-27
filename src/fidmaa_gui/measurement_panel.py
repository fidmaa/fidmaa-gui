"""Side panel: automatic results, the active tool's result, options and history."""

from __future__ import annotations

from PySide6.QtCore import QLocale, Signal
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
    historyCleared = Signal()

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
        if not details:
            # No measurement has been attempted yet (blank slate): dashes,
            # not "n/a (invalid depth)" which implies a real measurement
            # failed.
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
