from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from PySide6.QtCore import Signal


def _nan_complex_array(size: int) -> np.ndarray:
    return np.full(size, np.nan + 1j * np.nan, dtype=np.complex128)


def _z_to_gamma(z_norm: np.ndarray) -> np.ndarray:
    out = _nan_complex_array(len(z_norm))
    denom = z_norm + 1.0
    valid = np.isfinite(z_norm.real) & np.isfinite(z_norm.imag) & (np.abs(denom) > 1e-12)
    np.divide(z_norm - 1.0, denom, out=out, where=valid)
    return out


class SmithChartCanvas(FigureCanvasQTAgg):
    probePositionSelected = Signal(float)

    def __init__(self, width: float = 6.5, height: float = 6.5, dpi: int = 100) -> None:
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#0b1220")

        self._gamma = _nan_complex_array(0)
        self._z_positions = np.array([], dtype=float)
        self._line_collection: LineCollection | None = None
        self._probe_artist = None
        self._endpoint_match = None
        self._endpoint_load = None
        self._historical_artist = None
        self._measurement_artists: list = []
        self._measurement_labels: list = []
        self._probe_annotation = None
        self._hover_annotation = None
        self._hover_formatter: Callable[[int], str] | None = None
        self._build_static_grid()
        self.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.mpl_connect("button_press_event", self._on_mouse_click)

    def _build_static_grid(self) -> None:
        self.ax.clear()
        self._line_collection = None
        self._probe_artist = None
        self._endpoint_match = None
        self._endpoint_load = None
        self._historical_artist = None
        self._measurement_artists = []
        self._measurement_labels = []
        self.ax.set_facecolor("#0b1220")
        theta = np.linspace(0, 2 * np.pi, 720)
        self.ax.plot(np.cos(theta), np.sin(theta), color="#9ca3af", linewidth=1.3)
        self.ax.axhline(0.0, color="#64748b", linewidth=0.8, alpha=0.8)
        self.ax.axvline(0.0, color="#334155", linewidth=0.7, alpha=0.6)

        grid_color = "#475569"
        r_values = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
        x_values = [0.2, 0.5, 1.0, 2.0, 5.0]

        for r in r_values:
            center = r / (1.0 + r) if r >= 0 else 0.0
            radius = 1.0 / (1.0 + r) if r >= 0 else 1.0
            circle = np.exp(1j * theta) * radius + center
            mask = np.abs(circle) <= 1.001
            self.ax.plot(circle.real[mask], circle.imag[mask], color=grid_color, linewidth=0.6, alpha=0.55)
            label_x = center + radius
            if abs(label_x) <= 1.02:
                self.ax.text(label_x - 0.03, 0.02, f"{r:g}", color="#94a3b8", fontsize=7)

        r_span = np.linspace(0.0, 12.0, 600)
        for x in x_values:
            for sign in (1.0, -1.0):
                z_norm = r_span + 1j * sign * x
                gamma = _z_to_gamma(z_norm)
                valid = np.isfinite(gamma.real) & np.isfinite(gamma.imag) & (np.abs(gamma) <= 1.001)
                self.ax.plot(gamma.real[valid], gamma.imag[valid], color=grid_color, linewidth=0.6, alpha=0.55)

        self.ax.text(0.02, 0.02, "Matched", color="#93c5fd", fontsize=8)
        self.ax.text(0.88, 0.03, "Open", color="#94a3b8", fontsize=8)
        self.ax.text(-0.98, 0.03, "Short", color="#94a3b8", fontsize=8)
        self.ax.text(0.55, 0.87, "Inductive +jX", color="#94a3b8", fontsize=8)
        self.ax.text(0.55, -0.92, "Capacitive -jX", color="#94a3b8", fontsize=8)
        self.ax.text(-0.96, 0.95, "Match Output -> Cable End", color="#67e8f9", fontsize=8)
        self.ax.set_xlim(-1.05, 1.05)
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("Re(Γ)")
        self.ax.set_ylabel("Im(Γ)")
        self.ax.grid(False)
        self._probe_annotation = self.ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            color="#fde047",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#111827", "edgecolor": "#fde047", "alpha": 0.9},
        )
        self._probe_annotation.set_visible(False)
        self._hover_annotation = self.ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(10, -15),
            textcoords="offset points",
            color="#e2e8f0",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#111827", "edgecolor": "#64748b", "alpha": 0.9},
        )
        self._hover_annotation.set_visible(False)

    def set_hover_formatter(self, formatter: Callable[[int], str]) -> None:
        self._hover_formatter = formatter

    def clear_chart(self) -> None:
        self._gamma = _nan_complex_array(0)
        self._z_positions = np.array([], dtype=float)
        self._build_static_grid()
        self.refresh_chart()

    def set_trajectory(self, gamma_values: np.ndarray, cable_positions: np.ndarray, title: str = "") -> None:
        self._gamma = np.asarray(gamma_values, dtype=np.complex128)
        self._z_positions = np.asarray(cable_positions, dtype=float)
        self._build_static_grid()

        valid = np.isfinite(self._gamma.real) & np.isfinite(self._gamma.imag)
        points = np.column_stack((self._gamma.real[valid], self._gamma.imag[valid]))
        if len(points) >= 2:
            segments = np.stack([points[:-1], points[1:]], axis=1)
            cmap = colormaps["winter"]
            colors = cmap(np.linspace(0.2, 0.95, len(segments)))
            self._line_collection = LineCollection(segments, colors=colors, linewidths=2.2, alpha=0.95, zorder=3)
            self.ax.add_collection(self._line_collection)
            arrow_step = max(1, len(points) // 10)
            for idx in range(arrow_step, len(points), arrow_step):
                p0 = points[idx - 1]
                p1 = points[idx]
                self.ax.annotate(
                    "",
                    xy=(p1[0], p1[1]),
                    xytext=(p0[0], p0[1]),
                    arrowprops={"arrowstyle": "->", "color": "#67e8f9", "linewidth": 0.8, "alpha": 0.55},
                )

        self.ax.set_title(title or "Cable Position Trajectory")

    def set_endpoints(self, match_gamma: complex, load_gamma: complex) -> None:
        if np.isfinite(match_gamma.real) and np.isfinite(match_gamma.imag):
            (self._endpoint_match,) = self.ax.plot(
                [match_gamma.real], [match_gamma.imag], marker="o", markersize=7, color="#22d3ee", zorder=7
            )
            self.ax.text(match_gamma.real + 0.03, match_gamma.imag + 0.03, "Match Output", color="#67e8f9", fontsize=8)
        if np.isfinite(load_gamma.real) and np.isfinite(load_gamma.imag):
            (self._endpoint_load,) = self.ax.plot(
                [load_gamma.real], [load_gamma.imag], marker="s", markersize=7, color="#22c55e", zorder=7
            )
            self.ax.text(load_gamma.real + 0.03, load_gamma.imag - 0.05, "Cable End", color="#86efac", fontsize=8)

    def set_probe(self, position_m: float, gamma: complex, impedance: complex, invalid: bool = False) -> None:
        if not (np.isfinite(gamma.real) and np.isfinite(gamma.imag)):
            self._probe_annotation.set_visible(False)
            return
        marker = "*" if invalid else "D"
        color = "#f43f5e" if invalid else "#fde047"
        if self._probe_artist is None:
            (self._probe_artist,) = self.ax.plot(
                [gamma.real],
                [gamma.imag],
                marker=marker,
                markersize=11,
                color=color,
                markeredgecolor="#111827",
                zorder=9,
            )
        else:
            self._probe_artist.set_data([gamma.real], [gamma.imag])
            self._probe_artist.set_marker(marker)
            self._probe_artist.set_color(color)
        self._probe_annotation.xy = (gamma.real, gamma.imag)
        self._probe_annotation.set_text(
            f"Probe @ {position_m:.3f} m\nZ={impedance.real:+.3f}{impedance.imag:+.3f}j Ω"
            if np.isfinite(impedance.real) and np.isfinite(impedance.imag)
            else f"Probe @ {position_m:.3f} m\nZ invalid"
        )
        self._probe_annotation.set_visible(True)

    def set_measurement_markers(self, positions_m: list[float], gamma_values: list[complex]) -> None:
        for artist in self._measurement_artists:
            artist.remove()
        self._measurement_artists = []
        for label in self._measurement_labels:
            label.remove()
        self._measurement_labels = []

        for idx, (pos, gamma) in enumerate(zip(positions_m, gamma_values), start=1):
            if not (np.isfinite(gamma.real) and np.isfinite(gamma.imag)):
                continue
            (marker_artist,) = self.ax.plot(
                [gamma.real], [gamma.imag], marker="o", markersize=5, color="#dbeafe", alpha=0.9, zorder=6
            )
            text_artist = self.ax.text(gamma.real + 0.015, gamma.imag + 0.015, f"{idx}", color="#dbeafe", fontsize=7)
            marker_artist.set_gid(f"measure-{idx}@{pos:.4f}m")
            self._measurement_artists.append(marker_artist)
            self._measurement_labels.append(text_artist)

    def set_historical_region(self, gamma_values: np.ndarray, enabled: bool) -> None:
        if self._historical_artist is not None:
            self._historical_artist.remove()
            self._historical_artist = None
        if not enabled:
            return
        valid = np.isfinite(gamma_values.real) & np.isfinite(gamma_values.imag)
        if np.count_nonzero(valid) < 2:
            return
        (self._historical_artist,) = self.ax.plot(
            gamma_values.real[valid],
            gamma_values.imag[valid],
            color="#ef4444",
            linewidth=4.0,
            alpha=0.45,
            zorder=4,
        )

    def refresh_chart(self) -> None:
        self.draw_idle()

    def export_png(self, path: str) -> None:
        self.figure.savefig(path, dpi=180, bbox_inches="tight")

    def _nearest_index(self, event) -> int | None:
        if len(self._gamma) == 0 or event.xdata is None or event.ydata is None:
            return None
        valid = np.isfinite(self._gamma.real) & np.isfinite(self._gamma.imag)
        if not np.any(valid):
            return None
        valid_indices = np.flatnonzero(valid)
        points = np.column_stack((self._gamma.real[valid], self._gamma.imag[valid]))
        screen_points = self.ax.transData.transform(points)
        target = np.array([event.x, event.y], dtype=float)
        distances = np.linalg.norm(screen_points - target, axis=1)
        nearest_local = int(np.argmin(distances))
        if distances[nearest_local] > 10.0:
            return None
        return int(valid_indices[nearest_local])

    def _on_mouse_move(self, event) -> None:
        idx = self._nearest_index(event)
        if idx is None or self._hover_formatter is None:
            self._hover_annotation.set_visible(False)
            self.draw_idle()
            return
        gamma = self._gamma[idx]
        self._hover_annotation.xy = (gamma.real, gamma.imag)
        self._hover_annotation.set_text(self._hover_formatter(idx))
        self._hover_annotation.set_visible(True)
        self.draw_idle()

    def _on_mouse_click(self, event) -> None:
        if event.button != 1:
            return
        idx = self._nearest_index(event)
        if idx is None or idx >= len(self._z_positions):
            return
        self.probePositionSelected.emit(float(self._z_positions[idx]))
