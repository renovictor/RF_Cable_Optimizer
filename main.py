from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gui.splash_screen import StartupCoordinator


def main() -> int:
    app = QApplication(sys.argv)
    startup = StartupCoordinator(app)
    app.startup_coordinator = startup  # Keep a strong reference for app lifetime.
    startup.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
