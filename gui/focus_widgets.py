from __future__ import annotations

from PySide6.QtWidgets import QComboBox


class FocusWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()
