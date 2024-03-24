from PySide6 import QtCore
from PySide6.QtCore import QPointF
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QLabel


class QClickableLabel(QLabel):
    clicked = QtCore.Signal(QPointF)

    def __init__(self, parent=None):
        QLabel.__init__(self, parent=parent)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        self.clicked.emit(event.position())

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if ev.buttons():
            self.clicked.emit(ev.position())
