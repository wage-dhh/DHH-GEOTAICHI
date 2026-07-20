"""Create presentation figures from the fresh Example 3.3 dynamic outputs.

This is postprocessing only. It reads current fresh-run CSV files and FLAC3D
reference curves, and writes images/tables under dynamic/presentation_figures.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC_DIR = ROOT / "output" / "example3_3_flac3d_free_field_new" / "dynamic"
VALIDATION_DIR = DYNAMIC_DIR / "validation"
REFERENCE_DIR = ROOT / "data" / "reference" / "flac3d_example3_3_free_field"
GEOMETRY_DIR = ROOT / "output" / "example3_3_flac3d_free_field_new" / "geometry"
BOUNDARY_DIR = ROOT / "output" / "example3_3_flac3d_free_field_new" / "boundary"
OUT_DIR = DYNAMIC_DIR / "presentation_figures"

VELOCITY_HISTORY = DYNAMIC_DIR / "velocity_history.csv"
INPUT_WAVE_HISTORY = DYNAMIC_DIR / "input_wave_history.csv"
FREE_FIELD_FORCE_HISTORY = DYNAMIC_DIR / "free_field_force_history.csv"
ERROR_STATISTICS = VALIDATION_DIR / "error_statistics.csv"
GRID_INTERFACE_PAIR = BOUNDARY_DIR / "grid_interface_pair.csv"


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    data = np.atleast_1d(data)
    return {name: np.asarray(data[name], dtype=float) for name in (data.dtype.names or [])}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def velocity_point(point_id: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [r for r in read_rows(VELOCITY_HISTORY) if int(r["point_id"]) == point_id]
    return np.asarray([float(r["time"]) for r in rows]), np.asarray([float(r["vx"]) for r in rows])


def reference_curve(filename: str) -> tuple[np.ndarray, np.ndarray]:
    data = read_numeric_csv(REFERENCE_DIR / filename)
    return data["time_s"], data["vx"]


def error_row(case_prefix: str) -> dict[str, str]:
    for row in read_rows(ERROR_STATISTICS):
        if row["case"].startswith(case_prefix):
            return row
    raise KeyError(case_prefix)


def annotate_metrics(ax, row: dict[str, str], loc: tuple[float, float] = (0.02, 0.95)) -> None:
    text = "\n".join(
        [
            f"peak error = {100.0 * float(row['error_peak']):.2f}%",
            f"NRMSE = {float(row['nrmse']):.3f}",
            f"corr = {float(row['correlation']):.3f}",
        ]
    )
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#777", "alpha": 0.86})


def finish_plot(path: Path, title: str) -> Path:
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("vx (m/s)")
    plt.grid(True, alpha=0.28)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def figure_main_top() -> Path:
    t_mpm, v_mpm = velocity_point(0)
    t_ref, v_ref = reference_curve("reference_fig_3_9_flac_main.csv")
    row = error_row("main_top")
    path = OUT_DIR / "MPM_vs_FLAC3D_main_top_velocity.png"
    plt.figure(figsize=(9, 5.3))
    plt.plot(t_ref, v_ref, color="#111111", lw=2.2, label="FLAC3D reference")
    plt.plot(t_mpm, v_mpm, color="#0b7285", lw=2.0, label="GeoTaichi MPM")
    annotate_metrics(plt.gca(), row)
    return finish_plot(path, "Main model top velocity comparison")


def figure_free_field() -> Path:
    t_left, v_left = velocity_point(1)
    t_right, v_right = velocity_point(2)
    t_ref, v_ref = reference_curve("reference_fig_3_9_flac_free.csv")
    path = OUT_DIR / "MPM_vs_FLAC3D_free_field_velocity.png"
    plt.figure(figsize=(9, 5.3))
    plt.plot(t_ref, v_ref, color="#111111", lw=2.2, label="FLAC3D free-field")
    plt.plot(t_left, v_left, color="#1c7ed6", lw=1.9, label="MPM left FF")
    plt.plot(t_right, v_right, color="#e67700", lw=1.7, ls="--", label="MPM right FF")
    left = error_row("left_free")
    right = error_row("right_free")
    text = "\n".join(
        [
            f"left: peak err {100.0 * float(left['error_peak']):.2f}%, NRMSE {float(left['nrmse']):.3f}, corr {float(left['correlation']):.3f}",
            f"right: peak err {100.0 * float(right['error_peak']):.2f}%, NRMSE {float(right['nrmse']):.3f}, corr {float(right['correlation']):.3f}",
        ]
    )
    plt.gca().text(0.02, 0.95, text, transform=plt.gca().transAxes, va="top", ha="left", fontsize=9, bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#777", "alpha": 0.86})
    return finish_plot(path, "Free-field response comparison")


def figure_error_curve() -> Path:
    t_mpm, v_mpm = velocity_point(0)
    t_ref, v_ref_raw = reference_curve("reference_fig_3_9_flac_main.csv")
    v_ref = np.interp(t_mpm, t_ref, v_ref_raw)
    diff = v_mpm - v_ref
    rmse = math.sqrt(float(np.mean(diff**2)))
    nrmse = rmse / float(np.max(v_ref) - np.min(v_ref))
    path = OUT_DIR / "velocity_error_curve.png"
    plt.figure(figsize=(9, 5.3))
    plt.plot(t_mpm, diff, color="#c92a2a", lw=1.8, label="MPM - FLAC3D")
    plt.axhline(0.0, color="#333", lw=0.8)
    plt.gca().text(0.02, 0.95, f"RMSE = {rmse:.5g}\nNRMSE = {nrmse:.3f}", transform=plt.gca().transAxes, va="top", ha="left", fontsize=10, bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#777", "alpha": 0.86})
    return finish_plot(path, "Velocity error history")


def figure_wave_propagation() -> Path:
    inp = read_numeric_csv(INPUT_WAVE_HISTORY)
    t_mid_l, v_mid_l = velocity_point(3)
    t_mid_r, v_mid_r = velocity_point(4)
    t_top, v_top = velocity_point(0)
    path = OUT_DIR / "wave_propagation_comparison.png"
    plt.figure(figsize=(9, 5.3))
    plt.plot(inp["time"], inp["applied_vx"], color="#495057", lw=1.8, label="bottom applied vx")
    plt.plot(t_mid_l, v_mid_l, color="#5c940d", lw=1.7, label="middle left boundary")
    plt.plot(t_mid_r, v_mid_r, color="#2f9e44", lw=1.5, ls="--", label="middle right boundary")
    plt.plot(t_top, v_top, color="#0b7285", lw=2.0, label="top center")
    return finish_plot(path, "Shear wave propagation histories")


def force_balance_summary_by_time() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    sum_main: list[float] = []
    sum_ff: list[float] = []
    balance: list[float] = []
    current = None
    m = 0.0
    f = 0.0
    with FREE_FIELD_FORCE_HISTORY.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rt = row["time"]
            if current is None:
                current = rt
            if rt != current:
                times.append(float(current))
                sum_main.append(m)
                sum_ff.append(f)
                balance.append(m + f)
                current = rt
                m = 0.0
                f = 0.0
            m += float(row["Fx_main"])
            f += float(row["Fx_ff"])
        if current is not None:
            times.append(float(current))
            sum_main.append(m)
            sum_ff.append(f)
            balance.append(m + f)
    return np.asarray(times), np.asarray(sum_main), np.asarray(sum_ff), np.asarray(balance)


def figure_force_balance() -> Path:
    t, main, ff, balance = force_balance_summary_by_time()
    path = OUT_DIR / "free_field_force_balance.png"
    plt.figure(figsize=(9, 5.3))
    plt.plot(t, main, color="#1864ab", lw=1.6, label="sum(Fmain)")
    plt.plot(t, ff, color="#e67700", lw=1.6, label="sum(Fff)")
    plt.plot(t, balance, color="#c92a2a", lw=1.2, label="total balance error")
    plt.gca().text(0.02, 0.95, f"max |balance| = {float(np.max(np.abs(balance))):.3g}", transform=plt.gca().transAxes, va="top", ha="left", fontsize=10, bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#777", "alpha": 0.86})
    return finish_plot(path, "Free-field dashpot force balance")


def figure_model_boundary_condition() -> Path:
    pairs = read_rows(GRID_INTERFACE_PAIR)
    path = OUT_DIR / "model_boundary_condition.png"
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    ax.fill_between([0, 2], [0, 0], [5, 5], color="#d8e2dc", alpha=0.9, label="main model")
    ax.fill_between([2, 3], [0, 0], [5, 2], color="#d8e2dc", alpha=0.9)
    ax.fill_between([3, 4], [0, 0], [2, 5], color="#d8e2dc", alpha=0.9)
    ax.fill_between([4, 6], [0, 0], [5, 5], color="#d8e2dc", alpha=0.9)
    ax.axvspan(-0.375, -0.25, color="#a5d8ff", alpha=0.9, label="left free-field")
    ax.axvspan(6.25, 6.375, color="#ffd8a8", alpha=0.9, label="right free-field")
    ax.axvspan(-0.25, 0.0, color="#f1f3f5", alpha=0.9, label="gap")
    ax.axvspan(6.0, 6.25, color="#f1f3f5", alpha=0.9)
    for row in pairs[:: max(1, len(pairs) // 28)]:
        ax.plot([float(row["main_x"]), float(row["ff_x"])], [float(row["main_z"]), float(row["ff_z"])], color="#364fc7", lw=0.7, alpha=0.7)
    monitors = [
        (2.0, 5.0, "main top"),
        (-0.25, 5.0, "left FF top"),
        (6.25, 5.0, "right FF top"),
        (0.0, 2.5, "left boundary"),
        (6.0, 2.5, "right boundary"),
    ]
    for x, z, label in monitors:
        ax.scatter([x], [z], s=60, color="#c92a2a", zorder=5)
        ax.text(x + 0.06, z + 0.08, label, fontsize=8, color="#7f1d1d")
    ax.annotate("lateral dashpot grid pairs\nFx=rho Cs A(vff-vmain)", xy=(-0.12, 3.0), xytext=(-1.25, 4.3), arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=9)
    ax.annotate("bottom shear wave input", xy=(3.0, 0.0), xytext=(2.05, -0.55), arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=9)
    ax.set_xlim(-1.45, 7.2)
    ax.set_ylim(-0.75, 5.55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("2D model and boundary conditions")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_summary() -> Path:
    rows = []
    name_map = {
        "main_top_center_vs_FLAC_main": "main top",
        "left_free_field_top_vs_FLAC_free": "left free field",
        "right_free_field_top_vs_FLAC_free": "right free field",
    }
    for row in read_rows(ERROR_STATISTICS):
        rows.append(
            {
                "case": name_map.get(row["case"], row["case"]),
                "peak_error": row["error_peak"],
                "arrival_error": row["arrival_time_error"],
                "NRMSE": row["nrmse"],
                "correlation": row["correlation"],
                "status": row["reproduction_status"],
            }
        )
    return write_csv(OUT_DIR / "presentation_summary.csv", ["case", "peak_error", "arrival_error", "NRMSE", "correlation", "status"], rows)


def write_notes(figures: list[Path], summary: Path) -> Path:
    path = OUT_DIR / "presentation_notes.md"
    lines = [
        "# Presentation Notes",
        "",
        "## 1. Model Overview",
        "",
        "The current result is the fresh GeoTaichi MPM dynamic run for FLAC3D Example 3.3 shear wave loading with Kohler-style MPM free-field side boundaries.",
        "",
        "## 2. Free-Field Boundary Method",
        "",
        "The side boundaries use grid-level free-field coupling between main boundary grid nodes and free-field grid nodes. The dynamic solve was not rerun for these figures.",
        "",
        "## 3. Lateral Force Formula",
        "",
        "`Fx = rho * Cs * A * (vff - vmain)`",
        "",
        "## 4. Current Validation Results",
        "",
        "Peak velocities are close to the FLAC3D digitized reference, while NRMSE and correlation remain outside the current reproduction thresholds. The free-field action-reaction balance is numerically exact in the current history.",
        "",
        f"- summary table: {summary}",
        "",
        "## 5. Likely Error Sources",
        "",
        "- MPM and FLAC3D discretization differences",
        "- history point position differences, especially current free-field sample x=-0.25 vs documented FLAC3D x=-1",
        "- numerical dissipation and phase differences from the simplified MPM dynamic update",
        "",
        "## Generated Figures",
        "",
        *(f"- {figure}" for figure in figures),
        "",
        "## Data Sources",
        "",
        f"- {VELOCITY_HISTORY}",
        f"- {INPUT_WAVE_HISTORY}",
        f"- {FREE_FIELD_FORCE_HISTORY}",
        f"- {ERROR_STATISTICS}",
        f"- {REFERENCE_DIR / 'reference_fig_3_9_flac_main.csv'}",
        f"- {REFERENCE_DIR / 'reference_fig_3_9_flac_free.csv'}",
        f"- {GRID_INTERFACE_PAIR}",
        f"- existing geometry preview: {GEOMETRY_DIR / 'geometry_preview.png'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = [
        figure_main_top(),
        figure_free_field(),
        figure_error_curve(),
        figure_wave_propagation(),
        figure_force_balance(),
        figure_model_boundary_condition(),
    ]
    summary = write_summary()
    notes = write_notes(figures, summary)
    print(f"presentation_dir={OUT_DIR}")
    for figure in figures:
        print(f"figure={figure}")
    print(f"summary={summary}")
    print(f"notes={notes}")


if __name__ == "__main__":
    main()
