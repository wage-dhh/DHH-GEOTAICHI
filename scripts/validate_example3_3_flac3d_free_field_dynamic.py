"""Postprocess the fresh dynamic Example 3.3 free-field run.

Reads only:
  output/example3_3_flac3d_free_field_new/dynamic/
and optional FLAC3D reference curves under:
  data/reference/flac3d_example3_3_free_field/

It does not modify or rerun geometry, boundary, input, material, or dynamics.
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

VELOCITY_HISTORY = DYNAMIC_DIR / "velocity_history.csv"
INPUT_WAVE_HISTORY = DYNAMIC_DIR / "input_wave_history.csv"
FORCE_HISTORY = DYNAMIC_DIR / "free_field_force_history.csv"
MATERIAL_PARAMETERS = DYNAMIC_DIR / "material_parameters.csv"
VTK_DIR = DYNAMIC_DIR / "vtk"

POINT_NAMES = {
    0: "main_top_center",
    1: "left_free_field_top",
    2: "right_free_field_top",
    3: "main_left_boundary",
    4: "main_right_boundary",
}

REFERENCE_CASES = {
    "main_top_center_vs_FLAC_main": (0, "reference_fig_3_9_flac_main.csv", "FLAC-main"),
    "left_free_field_top_vs_FLAC_free": (1, "reference_fig_3_9_flac_free.csv", "FLAC-free"),
    "right_free_field_top_vs_FLAC_free": (2, "reference_fig_3_9_flac_free.csv", "FLAC-free"),
}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    data = np.atleast_1d(data)
    return {name: np.asarray(data[name], dtype=float) for name in (data.dtype.names or [])}


def read_velocity_rows() -> list[dict[str, str]]:
    with VELOCITY_HISTORY.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def velocity_by_point(rows: list[dict[str, str]]) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for pid in sorted({int(row["point_id"]) for row in rows}):
        subset = [row for row in rows if int(row["point_id"]) == pid]
        out[pid] = {
            "time": np.asarray([float(row["time"]) for row in subset], dtype=float),
            "vx": np.asarray([float(row["vx"]) for row in subset], dtype=float),
            "region": subset[0]["region"],
            "x": float(subset[0]["x"]),
            "z": float(subset[0]["z"]),
            "name": POINT_NAMES.get(pid, f"point_{pid}"),
        }
    return out


def read_reference(filename: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = REFERENCE_DIR / filename
    if not path.exists():
        return None
    data = read_numeric_csv(path)
    if "time_s" in data and "vx" in data:
        return data["time_s"], data["vx"]
    names = list(data)
    if len(names) >= 2:
        return data[names[0]], data[names[1]]
    return None


def first_arrival_time(time: np.ndarray, values: np.ndarray) -> float:
    if time.size == 0 or values.size == 0:
        return math.nan
    peak = float(np.nanmax(np.abs(values)))
    if not math.isfinite(peak) or peak <= 0.0:
        return math.nan
    threshold = 0.05 * peak
    indices = np.where(np.abs(values) >= threshold)[0]
    return float(time[int(indices[0])]) if indices.size else math.nan


def metric_row(case: str, point_id: int, point: dict[str, object], ref_file: str, ref_label: str) -> dict[str, object]:
    ref = read_reference(ref_file)
    if ref is None:
        return {
            "case": case,
            "mpm_point_id": point_id,
            "mpm_point": point["name"],
            "reference": ref_label,
            "reference_source": str(REFERENCE_DIR / ref_file),
            "reference_status": "NOT_AVAILABLE",
            "peak_mpm": math.nan,
            "peak_reference": math.nan,
            "error_peak": math.nan,
            "arrival_mpm": math.nan,
            "arrival_reference": math.nan,
            "arrival_time_error": math.nan,
            "nrmse": math.nan,
            "correlation": math.nan,
            "math_check_status": "NOT_AVAILABLE",
            "reproduction_status": "NOT_AVAILABLE",
        }
    mpm_t = np.asarray(point["time"], dtype=float)
    mpm_v = np.asarray(point["vx"], dtype=float)
    ref_t, ref_v = ref
    mask = (mpm_t >= float(np.nanmin(ref_t))) & (mpm_t <= float(np.nanmax(ref_t)))
    mpm_t = mpm_t[mask]
    mpm_v = mpm_v[mask]
    ref_interp = np.interp(mpm_t, ref_t, ref_v)
    finite = np.isfinite(mpm_v) & np.isfinite(ref_interp)
    mpm_t = mpm_t[finite]
    mpm_v = mpm_v[finite]
    ref_interp = ref_interp[finite]
    peak_mpm = float(np.max(np.abs(mpm_v))) if mpm_v.size else math.nan
    peak_ref = float(np.max(np.abs(ref_interp))) if ref_interp.size else math.nan
    error_peak = abs(peak_mpm - peak_ref) / peak_ref if peak_ref > 0.0 else math.nan
    arrival_mpm = first_arrival_time(mpm_t, mpm_v)
    arrival_ref = first_arrival_time(mpm_t, ref_interp)
    arrival_error = abs(arrival_mpm - arrival_ref) if math.isfinite(arrival_mpm) and math.isfinite(arrival_ref) else math.nan
    denom = float(np.max(ref_interp) - np.min(ref_interp)) if ref_interp.size else math.nan
    nrmse = math.sqrt(float(np.mean((mpm_v - ref_interp) ** 2))) / denom if denom and denom > 0.0 else math.nan
    corr = float(np.corrcoef(mpm_v, ref_interp)[0, 1]) if mpm_v.size > 1 and np.std(mpm_v) > 0.0 and np.std(ref_interp) > 0.0 else math.nan
    math_ok = all(math.isfinite(v) for v in (peak_mpm, peak_ref, error_peak, arrival_error, nrmse, corr))
    repro_ok = math_ok and error_peak <= 0.25 and nrmse <= 0.20 and corr >= 0.85
    return {
        "case": case,
        "mpm_point_id": point_id,
        "mpm_point": point["name"],
        "reference": ref_label,
        "reference_source": str(REFERENCE_DIR / ref_file),
        "reference_status": "AVAILABLE",
        "peak_mpm": peak_mpm,
        "peak_reference": peak_ref,
        "error_peak": error_peak,
        "arrival_mpm": arrival_mpm,
        "arrival_reference": arrival_ref,
        "arrival_time_error": arrival_error,
        "nrmse": nrmse,
        "correlation": corr,
        "math_check_status": "PASS" if math_ok else "FAIL",
        "reproduction_status": "PASS" if repro_ok else "NOT_PASS",
    }


def plot_series(path: Path, title: str, series: list[tuple[np.ndarray, np.ndarray, str]]) -> Path:
    plt.figure(figsize=(8.8, 5.0))
    for t, v, label in series:
        plt.plot(t, v, linewidth=1.7, label=label)
    plt.xlabel("time")
    plt.ylabel("vx")
    plt.title(title)
    plt.grid(True, alpha=0.28)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_reference_check() -> Path:
    rows = []
    for filename, point in [
        ("reference_fig_3_9_flac_main.csv", "FLAC-main"),
        ("reference_fig_3_9_flac_free.csv", "FLAC-free"),
        ("reference_fig_3_9_flac_column.csv", "FLAC-column"),
    ]:
        path = REFERENCE_DIR / filename
        ref = read_reference(filename)
        if ref is None:
            rows.append({"source": str(path), "point": point, "number_of_points": 0, "time_range": "NOT_AVAILABLE", "status": "NOT_AVAILABLE"})
        else:
            t, _ = ref
            rows.append({"source": str(path), "point": point, "number_of_points": int(t.size), "time_range": f"{float(np.min(t))}..{float(np.max(t))}", "status": "AVAILABLE"})
    return write_csv(VALIDATION_DIR / "reference_data_check.csv", ["source", "point", "number_of_points", "time_range", "status"], rows)


def write_mapping_report(points: dict[int, dict[str, object]], fields_ok: bool) -> Path:
    rows = []
    for pid, point in points.items():
        expected = POINT_NAMES.get(pid, "UNKNOWN")
        rows.append(
            {
                "point_id": pid,
                "expected_name": expected,
                "region": point["region"],
                "x": point["x"],
                "z": point["z"],
                "field_schema_status": "PASS" if fields_ok else "FAIL",
                "mapping_status": "PASS" if pid in POINT_NAMES else "CHECK",
            }
        )
    return write_csv(VALIDATION_DIR / "history_point_mapping_report.csv", ["point_id", "expected_name", "region", "x", "z", "field_schema_status", "mapping_status"], rows)


def write_force_balance_validation() -> Path:
    rows = []
    max_error = 0.0
    missing_layer_rows = 0
    with FORCE_HISTORY.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        current_time = None
        time_rows: list[dict[str, str]] = []

        def flush(group: list[dict[str, str]]) -> None:
            nonlocal max_error, missing_layer_rows
            if not group:
                return
            time_value = float(group[0]["time"])
            left_layers = {int(r["layer_id"]) for r in group if r["side"] == "left"}
            right_layers = {int(r["layer_id"]) for r in group if r["side"] == "right"}
            layer_ok = left_layers == right_layers and len(left_layers) > 0
            if not layer_ok:
                missing_layer_rows += 1
            sum_main = sum(float(r["Fx_main"]) for r in group)
            sum_ff = sum(float(r["Fx_ff"]) for r in group)
            balance = sum_main + sum_ff
            max_error = max(max_error, abs(balance), max(abs(float(r["balance_error"])) for r in group))
            rows.append(
                {
                    "time": time_value,
                    "left_layers": len(left_layers),
                    "right_layers": len(right_layers),
                    "sum_Fx_main": sum_main,
                    "sum_Fx_ff": sum_ff,
                    "balance_error": balance,
                    "layer_coverage_status": "PASS" if layer_ok else "FAIL",
                    "action_reaction_status": "PASS" if abs(balance) <= 1.0e-12 else "FAIL",
                }
            )

        for row in reader:
            row_time = row["time"]
            if current_time is None:
                current_time = row_time
            if row_time != current_time:
                flush(time_rows)
                time_rows = []
                current_time = row_time
            time_rows.append(row)
        flush(time_rows)
    path = write_csv(VALIDATION_DIR / "force_balance_validation.csv", ["time", "left_layers", "right_layers", "sum_Fx_main", "sum_Fx_ff", "balance_error", "layer_coverage_status", "action_reaction_status"], rows)
    return path


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_velocity_rows()
    required_fields = {"time", "point_id", "region", "x", "z", "vx"}
    fields_ok = bool(rows) and required_fields == set(rows[0].keys())
    points = velocity_by_point(rows)
    mapping_path = write_mapping_report(points, fields_ok)
    reference_check_path = write_reference_check()

    main_plot = plot_series(VALIDATION_DIR / "main_top_velocity.png", "main_top_center vx-time", [(points[0]["time"], points[0]["vx"], "main_top_center")])
    ff_plot = plot_series(
        VALIDATION_DIR / "free_field_velocity.png",
        "free-field top vx-time",
        [(points[1]["time"], points[1]["vx"], "left_free_field_top"), (points[2]["time"], points[2]["vx"], "right_free_field_top")],
    )
    boundary_plot = plot_series(
        VALIDATION_DIR / "boundary_velocity_comparison.png",
        "boundary and free-field velocity comparison",
        [
            (points[3]["time"], points[3]["vx"], "main_left_boundary z=2.5"),
            (points[1]["time"], points[1]["vx"], "left_free_field_top z=5.0"),
            (points[4]["time"], points[4]["vx"], "main_right_boundary z=2.5"),
            (points[2]["time"], points[2]["vx"], "right_free_field_top z=5.0"),
        ],
    )
    input_data = read_numeric_csv(INPUT_WAVE_HISTORY)
    propagation_plot = plot_series(
        VALIDATION_DIR / "wave_propagation_history.png",
        "vx histories at bottom, side-boundary mid-height, and top",
        [
            (input_data["time"], input_data["applied_vx"], "bottom applied vx"),
            (points[3]["time"], points[3]["vx"], "main_left_boundary z=2.5"),
            (points[4]["time"], points[4]["vx"], "main_right_boundary z=2.5"),
            (points[0]["time"], points[0]["vx"], "main_top_center z=5.0"),
        ],
    )

    metric_rows = [metric_row(case, point_id, points[point_id], ref_file, ref_label) for case, (point_id, ref_file, ref_label) in REFERENCE_CASES.items()]
    error_path = write_csv(
        VALIDATION_DIR / "error_statistics.csv",
        ["case", "mpm_point_id", "mpm_point", "reference", "reference_source", "reference_status", "peak_mpm", "peak_reference", "error_peak", "arrival_mpm", "arrival_reference", "arrival_time_error", "nrmse", "correlation", "math_check_status", "reproduction_status"],
        metric_rows,
    )
    force_balance_path = write_force_balance_validation()
    force_rows = read_numeric_csv(force_balance_path)
    max_force_balance = float(np.nanmax(np.abs(force_rows["balance_error"]))) if "balance_error" in force_rows else math.nan
    all_layer_ok = bool(np.all(force_rows["left_layers"] == force_rows["right_layers"])) if "left_layers" in force_rows else False

    peak_rows = []
    for pid, point in points.items():
        peak_rows.append({"point_id": pid, "point_name": point["name"], "region": point["region"], "x": point["x"], "z": point["z"], "peak_abs_vx": float(np.max(np.abs(point["vx"]))), "arrival_5pct_peak": first_arrival_time(point["time"], point["vx"])})
    peak_path = write_csv(VALIDATION_DIR / "velocity_response_summary.csv", ["point_id", "point_name", "region", "x", "z", "peak_abs_vx", "arrival_5pct_peak"], peak_rows)

    available_refs = [row for row in metric_rows if row["reference_status"] == "AVAILABLE"]
    math_pass = all(row["math_check_status"] == "PASS" for row in available_refs)
    reproduction_pass = all(row["reproduction_status"] == "PASS" for row in available_refs) if available_refs else False
    vtk_files = sorted(VTK_DIR.glob("step_*.vtu"))
    report_lines = [
        "# FLAC3D Example 3.3 Dynamic Validation Report",
        "",
        "## 1. Current Model Information",
        "",
        "This validation uses the fresh GeoTaichi MPM dynamic outputs with Kohler-style MPM free-field side boundaries.",
        "",
        f"- dynamic directory: {DYNAMIC_DIR}",
        f"- material parameters: {MATERIAL_PARAMETERS}",
        f"- input wave history: {INPUT_WAVE_HISTORY}",
        f"- velocity history: {VELOCITY_HISTORY}",
        f"- free-field force history: {FORCE_HISTORY}",
        f"- VTK files: {len(vtk_files)}",
        "",
        "No geometry, grid, particle generator, material, input wave, boundary formula, dashpot formula, or dynamic solver code was modified or rerun.",
        "",
        "## 2. Monitoring Points",
        "",
        *(f"- point_id={pid}, {point['name']}, region={point['region']}, x={point['x']}, z={point['z']}" for pid, point in points.items()),
        "",
        f"- mapping report: {mapping_path}",
        "",
        "## 3. Velocity Response Analysis",
        "",
        f"- main top plot: {main_plot}",
        f"- free-field plot: {ff_plot}",
        f"- boundary comparison plot: {boundary_plot}",
        f"- wave propagation plot: {propagation_plot}",
        f"- velocity response summary: {peak_path}",
        "",
        "The available history points include top free-field points and mid-height main side-boundary points. There are no same-height left/right free-field boundary histories at z=2.5 in the current dynamic output, so the boundary comparison plot uses the available free-field top histories explicitly labeled by z.",
        "",
        "## 4. Free-Field Boundary Check",
        "",
        f"- force balance validation: {force_balance_path}",
        f"- max action-reaction balance error: {max_force_balance}",
        f"- layer coverage left/right match: {'PASS' if all_layer_ok else 'FAIL'}",
        f"- numerical/action-reaction check: {'PASS' if max_force_balance <= 1.0e-12 and all_layer_ok else 'FAIL'}",
        "",
        "## 5. Reference Data",
        "",
        f"- reference data check: {reference_check_path}",
        "Reference curves are read from `data/reference/flac3d_example3_3_free_field/`. No old dynamic result histories or old validation metrics are read.",
        "",
        "## 6. Error Metrics",
        "",
        f"- error statistics: {error_path}",
        "",
        *(f"- {row['case']}: peak_error={row['error_peak']}, arrival_error={row['arrival_time_error']}, NRMSE={row['nrmse']}, correlation={row['correlation']}, reproduction_status={row['reproduction_status']}" for row in metric_rows),
        "",
        "## 7. Status Separation",
        "",
        f"- PASS: mathematical/numerical checks = {'PASS' if math_pass and max_force_balance <= 1.0e-12 and all_layer_ok else 'FAIL'}",
        f"- PASS: FLAC3D Example 3.3 reproduction = {'PASS' if reproduction_pass else 'NOT_PASS'}",
        "",
        "A NOT_PASS reproduction status means the current fresh dynamic result does not match the FLAC3D reference within the stated metric thresholds. No parameter tuning or model modification was performed.",
    ]
    report_path = VALIDATION_DIR / "validation_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"validation_dir={VALIDATION_DIR}")
    print(f"fields_ok={fields_ok}")
    print(f"math_checks={'PASS' if math_pass and max_force_balance <= 1.0e-12 and all_layer_ok else 'FAIL'}")
    print(f"flac3d_reproduction={'PASS' if reproduction_pass else 'NOT_PASS'}")
    print(f"max_force_balance={max_force_balance}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
