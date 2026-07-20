"""Postprocess Kohler-style MPM shear-wave histories for FLAC3D Example 3.3 trend validation.

Reads the current two-point top-monitor history and writes Figure 3.9
trend-comparison figures plus equivalent-validation metrics.
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
OUTPUT_DIR = ROOT / "output" / "example3_3_flac3d_free_field_new"
RESULTS_DIR = OUTPUT_DIR / "results"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"
REFERENCE_DIR = ROOT / "data" / "reference" / "flac3d_example3_3_free_field"
HISTORY_ALL = RESULTS_DIR / "history_all_points.csv"
HISTORY_CANDIDATES = [
    HISTORY_ALL,
    ROOT / "output" / "example3_3_2d_kohler_native_mpm_solver" / "new_geometry_corrected_dynamic_run" / "history.csv",
]

MPM_CURVE_COLUMN_CANDIDATES = {
    "MPM-soil-column-top": (
        "soil_column_top_vx",
        "free_field_top_vx",
        "mpm_left_ff_column_top_xvel",
    ),
    "MPM-main-model-top": (
        "main_model_top_vx",
        "main_top_vx",
        "mpm_main_grid_top_xvel",
    ),
}

REFERENCE_FILES = {
    "FLAC-column": "reference_fig_3_9_flac_column.csv",
    "FLAC-free": "reference_fig_3_9_flac_free.csv",
    "FLAC-main": "reference_fig_3_9_flac_main.csv",
}

METRIC_CASES = [
    ("MPM-soil-column-top vs FLAC-free", "MPM-soil-column-top", "FLAC-free"),
    ("MPM-main-model-top vs FLAC-main", "MPM-main-model-top", "FLAC-main"),
]

TREND_CORRELATION_MIN = 0.85
TREND_NRMSE_MAX = 0.20
TREND_PEAK_ERROR_MAX_PERCENT = 25.0
TREND_PHASE_LAG_MAX_S = 0.0005


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"No rows in {path}")
    data = np.atleast_1d(data)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names or []}


def resolve_history_path() -> Path:
    for path in HISTORY_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(f"- {path}" for path in HISTORY_CANDIDATES)
    raise FileNotFoundError(f"No supported history file found. Checked:\n{candidates}")


def resolve_mpm_curves(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    missing: dict[str, tuple[str, ...]] = {}
    for label, candidates in MPM_CURVE_COLUMN_CANDIDATES.items():
        for column in candidates:
            if column in data:
                curves[label] = data[column][mask]
                break
        else:
            missing[label] = candidates
    if missing:
        details = "; ".join(f"{label}: {columns}" for label, columns in missing.items())
        raise KeyError(f"Missing top-monitor history columns. Expected one candidate per monitor: {details}")
    return curves


def read_reference(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    data = read_csv_columns(path)
    if "time_s" not in data or "vx" not in data:
        return None
    return data["time_s"], data["vx"]


def reference_curves() -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
    return {label: read_reference(REFERENCE_DIR / filename) for label, filename in REFERENCE_FILES.items()}


def trend_status(correlation: float, nrmse: float, peak_error_percent: float, phase_lag_s: float) -> str:
    values_are_finite = all(math.isfinite(value) for value in (correlation, nrmse, peak_error_percent, phase_lag_s))
    passed = (
        values_are_finite
        and correlation >= TREND_CORRELATION_MIN
        and nrmse <= TREND_NRMSE_MAX
        and abs(peak_error_percent) <= TREND_PEAK_ERROR_MAX_PERCENT
        and abs(phase_lag_s) <= TREND_PHASE_LAG_MAX_S
    )
    return "PASS" if passed else "PARTIAL"


def metric_values(model_t: np.ndarray, model_v: np.ndarray, ref_t: np.ndarray, ref_v: np.ndarray) -> dict[str, float | str]:
    ref_interp = np.interp(model_t, ref_t, ref_v, left=np.nan, right=np.nan)
    mask = np.isfinite(ref_interp)
    if not np.any(mask):
        return {
            "validation_type": "equivalent_mpm_trend",
            "peak_mpm": math.nan,
            "peak_ref": math.nan,
            "peak_error_percent": math.nan,
            "nrmse": math.nan,
            "correlation": math.nan,
            "phase_lag_s": math.nan,
            "trend_pass_fail": "PARTIAL",
            "strict_pass_fail": "PARTIAL",
        }
    t = model_t[mask]
    m = model_v[mask]
    r = ref_interp[mask]
    peak_mpm = float(np.max(np.abs(m)))
    peak_ref = float(np.max(np.abs(r)))
    peak_error_percent = 100.0 * (peak_mpm - peak_ref) / peak_ref if peak_ref > 0.0 else math.nan
    rmse = math.sqrt(float(np.mean((m - r) ** 2)))
    denom = float(np.max(r) - np.min(r))
    nrmse = rmse / denom if denom > 0.0 else math.nan
    correlation = float(np.corrcoef(m, r)[0, 1]) if m.size > 1 and np.std(m) > 0.0 and np.std(r) > 0.0 else math.nan
    m0 = m - float(np.mean(m))
    r0 = r - float(np.mean(r))
    if m0.size > 1 and np.std(m0) > 0.0 and np.std(r0) > 0.0:
        lag_index = int(np.argmax(np.correlate(m0, r0, mode="full")) - (r0.size - 1))
        dt = float(np.median(np.diff(t))) if t.size > 1 else math.nan
        phase_lag_s = lag_index * dt
    else:
        phase_lag_s = math.nan
    return {
        "validation_type": "equivalent_mpm_trend",
        "peak_mpm": peak_mpm,
        "peak_ref": peak_ref,
        "peak_error_percent": peak_error_percent,
        "nrmse": nrmse,
        "correlation": correlation,
        "phase_lag_s": phase_lag_s,
        "trend_pass_fail": trend_status(correlation, nrmse, peak_error_percent, phase_lag_s),
        "strict_pass_fail": "PARTIAL",
    }


def write_metrics(rows: list[dict[str, float | str]]) -> Path:
    path = REPORTS_DIR / "reference_metrics.csv"
    fieldnames = [
        "case",
        "validation_type",
        "peak_mpm",
        "peak_ref",
        "peak_error_percent",
        "nrmse",
        "correlation",
        "phase_lag_s",
        "trend_pass_fail",
        "strict_pass_fail",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def style_axes(title: str) -> None:
    plt.xlim(0.0, 0.015)
    plt.xlabel("Time (s)")
    plt.ylabel("x velocity")
    plt.title(title)
    plt.grid(True, alpha=0.28)
    plt.legend(fontsize=8)
    plt.tight_layout()


def plot_compare(t: np.ndarray, mpm: dict[str, np.ndarray], refs: dict[str, tuple[np.ndarray, np.ndarray] | None]) -> Path:
    path = FIGURE_DIR / "fig_3_9_x_velocity_profiles_compare.png"
    plt.figure(figsize=(8.8, 5.2))
    colors = ["#0b3954", "#8d0801", "#f2a541", "#087e8b"]
    linestyles = ["-", "--", "-.", ":"]
    for (label, values), color, ls in zip(mpm.items(), colors, linestyles):
        plt.plot(t, values, label=label, color=color, linestyle=ls, linewidth=1.9)
    for label, ref in refs.items():
        if ref is not None:
            rt, rv = ref
            mask = (rt >= 0.0) & (rt <= 0.015)
            plt.plot(rt[mask], rv[mask], label=label, linewidth=1.35, alpha=0.78)
    style_axes("Figure 3.9 equivalent MPM trend comparison")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_mpm_only(t: np.ndarray, mpm: dict[str, np.ndarray]) -> Path:
    path = FIGURE_DIR / "fig_3_9_mpm_only.png"
    plt.figure(figsize=(8.8, 5.2))
    for label, values in mpm.items():
        plt.plot(t, values, label=label, linewidth=1.9)
    style_axes("Figure 3.9 MPM top-monitor x-velocity histories")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_reference_only(refs: dict[str, tuple[np.ndarray, np.ndarray] | None]) -> Path:
    path = FIGURE_DIR / "fig_3_9_reference_only.png"
    plt.figure(figsize=(8.8, 5.2))
    for label, ref in refs.items():
        if ref is not None:
            rt, rv = ref
            mask = (rt >= 0.0) & (rt <= 0.015)
            plt.plot(rt[mask], rv[mask], label=label, linewidth=1.8)
    style_axes("Figure 3.9 FLAC3D reference-only histories")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_error_curves(t: np.ndarray, mpm: dict[str, np.ndarray], refs: dict[str, tuple[np.ndarray, np.ndarray] | None]) -> Path:
    path = FIGURE_DIR / "fig_3_9_error_curves.png"
    plt.figure(figsize=(8.8, 5.2))
    for case, mpm_label, ref_label in METRIC_CASES:
        ref = refs.get(ref_label)
        if ref is None:
            continue
        rt, rv = ref
        err = mpm[mpm_label] - np.interp(t, rt, rv, left=np.nan, right=np.nan)
        plt.plot(t, err, label=case, linewidth=1.5)
    plt.axhline(0.0, color="#222222", linewidth=0.8)
    style_axes("Figure 3.9 equivalent MPM minus FLAC3D error curves")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def missing_metric_row() -> dict[str, float | str]:
    return {
        "validation_type": "equivalent_mpm_trend",
        "peak_mpm": math.nan,
        "peak_ref": math.nan,
        "peak_error_percent": math.nan,
        "nrmse": math.nan,
        "correlation": math.nan,
        "phase_lag_s": math.nan,
        "trend_pass_fail": "PARTIAL",
        "strict_pass_fail": "PARTIAL",
    }


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    history_path = resolve_history_path()
    data = read_csv_columns(history_path)
    t = data["time"] if "time" in data else data["time_s"]
    mask = (t >= 0.0) & (t <= 0.015)
    t_plot = t[mask]
    mpm = resolve_mpm_curves(data, mask)
    refs = reference_curves()

    compare_path = plot_compare(t_plot, mpm, refs)
    mpm_only_path = plot_mpm_only(t_plot, mpm)
    ref_only_path = plot_reference_only(refs)
    error_path = plot_error_curves(t_plot, mpm, refs)

    metric_rows: list[dict[str, float | str]] = []
    for case, mpm_label, ref_label in METRIC_CASES:
        ref = refs.get(ref_label)
        row: dict[str, float | str] = {"case": case}
        row.update(missing_metric_row() if ref is None else metric_values(t_plot, mpm[mpm_label], *ref))
        metric_rows.append(row)
    metrics_path = write_metrics(metric_rows)

    print(f"History source: {history_path}")
    print(f"Figure 3.9 comparison: {compare_path}")
    print(f"Figure 3.9 MPM-only: {mpm_only_path}")
    print(f"Figure 3.9 reference-only: {ref_only_path}")
    print(f"Figure 3.9 error curves: {error_path}")
    print(f"Reference metrics: {metrics_path}")


if __name__ == "__main__":
    main()

