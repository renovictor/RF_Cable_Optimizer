from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QRect, Qt, QThread, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QMovie, QPainter
from PySide6.QtWidgets import QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget

from version import APP_VERSION


def _resource_path(filename: str) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base_path / filename


@dataclass(frozen=True)
class _ModuleToLoad:
    label: str
    module_name: str
    optional: bool = False


class ModulePreloader(QThread):
    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    loading_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._modules = [
            _ModuleToLoad("PySerial", "serial"),
            _ModuleToLoad("Pandas", "pandas"),
            _ModuleToLoad("OpenPyXL", "openpyxl"),
            _ModuleToLoad("NumPy", "numpy", optional=True),
            _ModuleToLoad("Matplotlib", "matplotlib", optional=True),
            _ModuleToLoad("Pillow/PIL", "PIL", optional=True),
            _ModuleToLoad("python-docx", "docx", optional=True),
            _ModuleToLoad("guardian_tlog", "guardian_tlog"),
            _ModuleToLoad("jsonl_to_excel", "jsonl_to_excel"),
        ]

    def run(self) -> None:
        total = len(self._modules)
        for index, mod in enumerate(self._modules, start=1):
            self.status_changed.emit(f"Loading {mod.label}...")
            try:
                importlib.import_module(mod.module_name)
            except ModuleNotFoundError:
                if not mod.optional:
                    self.status_changed.emit(f"{mod.label} not installed (continuing).")
            except Exception:
                if not mod.optional:
                    self.status_changed.emit(f"{mod.label} failed to load (continuing).")
            self.progress_changed.emit(index, total)
            time.sleep(0.08)
        self.status_changed.emit("Finalizing startup...")
        self.loading_finished.emit()


class SplashScreen(QWidget):
    def __init__(self) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setObjectName("splashRoot")
        self.setFixedSize(760, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 18)
        layout.setSpacing(14)

        self.logo = QLabel()
        self.logo.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self._logo_movie: QMovie | None = None
        logo_path = _resource_path("ASM-logo-small.gif")
        if logo_path.exists():
            movie = QMovie(str(logo_path))
            if movie.isValid():
                self.logo.setMovie(movie)
                self._logo_movie = movie
                movie.start()
            else:
                self.logo.setText("ASM")
        else:
            self.logo.setText("ASM")
        self.logo.setStyleSheet("background: transparent;")
        layout.addWidget(self.logo)

        self.app_name = QLabel("RF Cable Optimizer")
        self.app_name.setAlignment(Qt.AlignCenter)
        self.app_name.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self.app_name.setStyleSheet("color: #f8fafc; background: transparent;")
        layout.addWidget(self.app_name)

        self.version = QLabel(f"{APP_VERSION}  |  Global Search International  |  ASM")
        self.version.setAlignment(Qt.AlignCenter)
        self.version.setStyleSheet("color: #e2e8f0; font-size: 12px; background: transparent;")
        layout.addWidget(self.version)

        self.status = QLabel("Initializing...")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: #f1f5f9; font-size: 13px; background: transparent;")
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #c4b5fd;
                border-radius: 6px;
                text-align: center;
                color: #f8fafc;
                background-color: rgba(15, 23, 42, 0.55);
                padding: 2px;
                min-height: 22px;
            }
            QProgressBar::chunk {
                background-color: #c084fc;
                border-radius: 5px;
            }
            """
        )
        layout.addWidget(self.progress)

        layout.addStretch(1)

        self.footer = QLabel("Author: RF Cable Optimizer Team    |    Contact: rf_cable_optimizer@asmpt.com")
        self.footer.setAlignment(Qt.AlignCenter)
        self.footer.setStyleSheet(
            "color: #ede9fe; font-size: 11px; background-color: rgba(30, 41, 59, 0.55); "
            "border-top: 1px solid rgba(221, 214, 254, 0.35); padding: 8px;"
        )
        layout.addWidget(self.footer)

    def center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo: QRect = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + (geo.height() - self.height()) // 2
        self.move(QPoint(x, y))

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def set_progress(self, current: int, total: int) -> None:
        if total <= 0:
            self.progress.setValue(0)
            return
        value = int(current * 100 / total)
        self.progress.setValue(max(0, min(100, value)))

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor("#0b1b3a"))   # dark navy
        gradient.setColorAt(0.5, QColor("#1e1b4b"))   # indigo
        gradient.setColorAt(1.0, QColor("#c084fc"))   # light purple
        painter.fillRect(self.rect(), gradient)
        super().paintEvent(event)


class StartupCoordinator(QObject):
    def __init__(self, app: QApplication) -> None:
        super().__init__()
        self._app = app
        self._splash = SplashScreen()
        self._preloader = ModulePreloader()
        self._minimum_elapsed = False
        self._loading_finished = False
        self._main_window = None

        self._preloader.status_changed.connect(self._splash.set_status)
        self._preloader.progress_changed.connect(self._splash.set_progress)
        self._preloader.loading_finished.connect(self._on_loading_finished)

    def start(self) -> None:
        self._splash.center_on_screen()
        self._splash.show()
        QTimer.singleShot(5000, self._on_minimum_elapsed)
        self._preloader.start()

    def _on_minimum_elapsed(self) -> None:
        self._minimum_elapsed = True
        self._try_launch_main_window()

    def _on_loading_finished(self) -> None:
        self._loading_finished = True
        self._try_launch_main_window()

    def _try_launch_main_window(self) -> None:
        if not (self._minimum_elapsed and self._loading_finished):
            return
        self._splash.set_status("Launching main window...")
        self._splash.set_progress(100, 100)
        self._preloader.wait(2000)

        from gui.main_window import MainWindow

        self._main_window = MainWindow()
        self._main_window.show()
        self._splash.close()
