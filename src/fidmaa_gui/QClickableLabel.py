from PySide6 import QtCore
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QLabel


class QClickableLabel(QLabel):
    clicked = QtCore.Signal(QPointF)
    pressed = QtCore.Signal(QPointF)
    dragged = QtCore.Signal(QPointF)
    released = QtCore.Signal(QPointF)
    pointerMoved = QtCore.Signal(QPointF)

    def __init__(self, parent=None):
        QLabel.__init__(self, parent=parent)
        self.setMouseTracking(True)
        self._press_position = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_position = event.position()
            self.pressed.emit(event.position())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        self.pointerMoved.emit(ev.position())
        if ev.buttons() & Qt.LeftButton:
            self.dragged.emit(ev.position())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.released.emit(event.position())
            if self._press_position is not None:
                movement = event.position() - self._press_position
                if movement.manhattanLength() <= 3:
                    self.clicked.emit(event.position())
            self._press_position = None
        super().mouseReleaseEvent(event)
