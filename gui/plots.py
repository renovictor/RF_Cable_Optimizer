from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 7.0, height: float = 5.0, dpi: int = 100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)

