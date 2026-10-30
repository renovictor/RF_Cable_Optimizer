from __future__ import annotations

from gui.plots import MplCanvas
from thermal.models import ThermalSimulationResult


def draw_lumped_temperature_plot(canvas: MplCanvas, result: ThermalSimulationResult, target_temp_c: float) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.subplots(1, 1)
    ax.plot(result.time_s, result.temperature_c, color="#fb923c", linewidth=2.0, label="T(t)")
    ax.axhline(result.temperature_c[0], color="#22d3ee", linestyle=":", linewidth=1.3, label="Initial Temp")
    ax.axhline(result.steady_state_temp_c, color="#22c55e", linestyle="--", linewidth=1.3, label="Steady-State Temp")
    ax.axhline(target_temp_c, color="#ef4444", linestyle="-.", linewidth=1.3, label="Exceed Temp limit")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [deg C]")
    ax.set_title("Lumped Thermal RC Temperature Response")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    canvas.draw_idle()
