from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "output/example3_3_flac3d_free_field_ppc1_velocity_input_bottom1_rerun"
OLD = ROOT / "output/example3_3_flac3d_free_field_ppc1_periodic_bottom1_rerun"
REF = ROOT / "data/reference/flac3d_example3_3_free_field"
SCRIPT_NEW = ROOT / "examples/example3_3_flac3d_free_field_ppc1_velocity_input_bottom1_rerun.py"
SCRIPT_OLD = ROOT / "examples/example3_3_flac3d_free_field_ppc1_periodic_bottom1_rerun.py"
VAL = NEW / "validation"

RHO = 0.0025
CS = 4000.0
ETA = RHO * CS
PERIOD = 0.01
VPEAK = 0.1
FORMULA = "tau_input = rho * Cs * target_input_velocity"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_numeric(path: Path) -> dict[str, np.ndarray | list[str]]:
    rows = read_rows(path)
    out: dict[str, list[float] | list[str]] = {key: [] for key in rows[0]} if rows else {}
    for row in rows:
        for key, value in row.items():
            try:
                out[key].append(float(value))  # type: ignore[union-attr]
            except ValueError:
                out[key].append(value)  # type: ignore[union-attr]
    return {key: np.asarray(value, dtype=float) if value and isinstance(value[0], float) else value for key, value in out.items()}


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_history(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    data = read_numeric(path)
    time = np.asarray(data["time"], dtype=float)
    point_id = np.asarray(data["point_id"], dtype=float).astype(int)
    vx = np.asarray(data["vx"], dtype=float)
    return {pid: (time[point_id == pid], vx[point_id == pid]) for pid in sorted(set(point_id))}


def load_reference(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = read_numeric(REF / name)
    return np.asarray(data["time_s"], dtype=float), np.asarray(data["vx"], dtype=float)


def arrival_time(time: np.ndarray, velocity: np.ndarray) -> float:
    peak = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    if peak <= 0.0:
        return math.nan
    indices = np.where(np.abs(velocity) >= 0.05 * peak)[0]
    return float(time[indices[0]]) if indices.size else math.nan


def phase_lag(time: np.ndarray, mpm: np.ndarray, ref: np.ndarray) -> float:
    if time.size < 2:
        return math.nan
    a = mpm - np.mean(mpm)
    b = ref - np.mean(ref)
    corr = np.correlate(a, b, mode="full")
    lag_index = int(np.argmax(corr) - (a.size - 1))
    return float(lag_index * np.median(np.diff(time)))


def stats_case(case: str, time: np.ndarray, mpm: np.ndarray, ref_time: np.ndarray, ref_velocity: np.ndarray) -> dict[str, object]:
    ref_interp = np.interp(time, ref_time, ref_velocity)
    denom = float(np.max(ref_interp) - np.min(ref_interp))
    rmse = float(np.sqrt(np.mean((mpm - ref_interp) ** 2)))
    nrmse = rmse / denom if denom else math.nan
    corr = float(np.corrcoef(mpm, ref_interp)[0, 1]) if np.std(mpm) > 0.0 and np.std(ref_interp) > 0.0 else math.nan
    peak_m = float(np.max(np.abs(mpm)))
    peak_r = float(np.max(np.abs(ref_interp)))
    peak_time_m = float(time[np.argmax(np.abs(mpm))])
    peak_time_r = float(time[np.argmax(np.abs(ref_interp))])
    min_ref = float(np.min(ref_interp))
    return {
        "case": case,
        "peak_amplitude": peak_m,
        "reference_peak_amplitude": peak_r,
        "peak_error": abs(peak_m - peak_r) / peak_r if peak_r else math.nan,
        "peak_time": peak_time_m,
        "reference_peak_time": peak_time_r,
        "peak_time_error": peak_time_m - peak_time_r,
        "arrival_time": arrival_time(time, mpm),
        "reference_arrival_time": arrival_time(time, ref_interp),
        "arrival_time_error": arrival_time(time, mpm) - arrival_time(time, ref_interp),
        "NRMSE": nrmse,
        "correlation": corr,
        "phase_lag": phase_lag(time, mpm, ref_interp),
        "minimum_velocity": float(np.min(mpm)),
        "reference_minimum_velocity": min_ref,
        "minimum_velocity_error": abs(float(np.min(mpm)) - min_ref) / abs(min_ref) if abs(min_ref) > 0.0 else math.nan,
    }


def plot_finish(path: Path, title: str, ylabel: str, xlim: tuple[float, float] | None = None) -> None:
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(*xlim)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def find_lines(path: Path, patterns: dict[str, str]) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    found: dict[str, int] = {}
    for name, pattern in patterns.items():
        regex = re.compile(pattern)
        for index, line in enumerate(lines, 1):
            if regex.search(line):
                found[name] = index
                break
    return found


def write_implementation_check() -> None:
    patterns = {
        "PERIOD": r"^PERIOD =",
        "DYNAMIC_TIME": r"^DYNAMIC_TIME =",
        "BOTTOM_INPUT_GRID_LAYERS": r"^BOTTOM_INPUT_GRID_LAYERS =",
        "ETA_S": r"^ETA_S =",
        "wave_stress": r"^def wave_stress",
        "target_bottom_vx": r"^def target_bottom_vx",
        "bottom_node_ids": r"^def bottom_node_ids",
        "bottom_area": r"^def bottom_node_area_rows",
        "tau_input": r"tau_input = wave_stress",
        "force_formula_input": r"local_input = tau_input",
        "force_formula_dashpot": r"local_dashpot = -ETA_S",
        "force_add": r"force_g\[nid, 0\] \+= local_input \+ local_dashpot",
    }
    old_lines = find_lines(SCRIPT_OLD, patterns)
    new_lines = find_lines(
        SCRIPT_NEW,
        {
            **patterns,
            "INPUT_VELOCITY_PEAK": r"^INPUT_VELOCITY_PEAK =",
            "wave_factor": r"^def wave_factor",
            "input_velocity": r"^def input_velocity",
        },
    )
    lines = [
        "# Velocity Input Conversion Implementation Check",
        "",
        "## Baseline Checked",
        "",
        f"- Baseline script: `{SCRIPT_OLD}`",
        "- Current baseline input before this change: stress history.",
        "- Baseline `tau_input`: `tau_input = wave_stress(time_value)`.",
        "- Baseline `wave_stress(t) = 0.5 * (1 - cos(2*pi*t/PERIOD))`; peak stress = 1.0.",
        f"- rho = {RHO}",
        f"- Cs = {CS}",
        f"- ETA_S = rho*Cs = {ETA}",
        "- Baseline bottom force: `force_g[nid,0] += tau_input*A_i - ETA_S*velocity_g[nid,0]*A_i`.",
        "- Baseline bottom node selection: `BOTTOM_INPUT_GRID_LAYERS=1`, physical z=0 bottom nodes only.",
        "",
        "## Baseline Code Lines",
        "",
    ]
    for key in patterns:
        lines.append(f"- {key}: line {old_lines.get(key, 'NOT_FOUND')}")
    lines += ["", "## New Script Lines", "", f"- New script: `{SCRIPT_NEW}`"]
    for key in ["INPUT_VELOCITY_PEAK", "PERIOD", "DYNAMIC_TIME", "BOTTOM_INPUT_GRID_LAYERS", "ETA_S", "wave_factor", "input_velocity", "wave_stress", "bottom_node_ids", "bottom_area", "tau_input", "force_formula_input", "force_formula_dashpot", "force_add"]:
        lines.append(f"- {key}: line {new_lines.get(key, 'NOT_FOUND')}")
    lines += [
        "",
        "## Direct Velocity Constraint Check",
        "",
        "- `velocity_g[nid,0] = input_velocity(t)`: NOT FOUND",
        "- `particle.velocity = input_velocity(t)`: NOT FOUND",
        "- `momentum_g = mass_g * input_velocity(t)`: NOT FOUND",
        "- Existing `velocity_g[bottom_ids,1]=0.0` is vertical bottom constraint, not horizontal input velocity.",
    ]
    (NEW / "velocity_input_conversion_implementation_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    VAL.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.linewidth": 1.0})
    write_implementation_check()

    conversion_rows = []
    for time in [0.0, 0.005, 0.010, 0.015]:
        factor = 0.5 * (1.0 - math.cos(2.0 * math.pi * time / PERIOD))
        velocity = VPEAK * factor
        tau = ETA * velocity
        relative_error = abs(tau - ETA * velocity) / (abs(ETA * velocity) if abs(ETA * velocity) > 0.0 else 1.0)
        conversion_rows.append(
            {
                "time": time,
                "velocity_wave_factor": factor,
                "target_input_velocity": velocity,
                "rho": RHO,
                "Cs": CS,
                "shear_impedance": ETA,
                "converted_tau_input": tau,
                "expected_tau_input": ETA * velocity,
                "relative_error": relative_error,
                "status": "PASS" if relative_error <= 1.0e-12 else "FAIL",
            }
        )
    write_csv(VAL / "velocity_to_stress_conversion_check.csv", list(conversion_rows[0]), conversion_rows)

    new_history = load_history(NEW / "dynamic/velocity_history.csv")
    old_history = load_history(OLD / "dynamic/velocity_history.csv")
    main_ref = load_reference("reference_fig_3_9_flac_main.csv")
    free_ref = load_reference("reference_fig_3_9_flac_free.csv")

    stats = []
    for label, pid, ref in [("main_top", 0, main_ref), ("free_field_top", 1, free_ref)]:
        time, velocity = new_history[pid]
        stats.append(stats_case(label + "_full", time, velocity, *ref))
        for start, end, name in [(0.0, 0.010, "0_0p010"), (0.0100000001, 0.015, "0p010_0p015")]:
            mask = (time >= start) & (time <= end)
            if np.any(mask):
                stats.append(stats_case(label + "_" + name, time[mask], velocity[mask], *ref))
    write_csv(VAL / "error_statistics_velocity_input_bottom1.csv", list(stats[0]), stats)

    old_input = read_numeric(OLD / "dynamic/input_wave_history.csv")
    new_stress = read_numeric(NEW / "dynamic/input_stress_history.csv")
    old_tau = np.asarray(old_input["target_stress"], dtype=float)
    old_time = np.asarray(old_input["time"], dtype=float)
    new_time = np.asarray(new_stress["time"], dtype=float)
    new_tau = np.asarray(new_stress["converted_tau_input"], dtype=float)
    new_target_velocity = np.asarray(new_stress["target_input_velocity"], dtype=float)
    common_time = new_history[0][0]
    old_main = np.interp(common_time, old_history[0][0], old_history[0][1])
    old_free = np.interp(common_time, old_history[1][0], old_history[1][1])
    new_main = new_history[0][1]
    new_free = new_history[1][1]
    compare_rows = []
    for index, time in enumerate(common_time):
        old_tau_i = float(np.interp(time, old_time, old_tau))
        new_tau_i = float(np.interp(time, new_time, new_tau))
        compare_rows.append(
            {
                "time": float(time),
                "old_tau_input": old_tau_i,
                "new_target_velocity": float(np.interp(time, new_time, new_target_velocity)),
                "new_converted_tau_input": new_tau_i,
                "tau_difference": new_tau_i - old_tau_i,
                "main_top_velocity_old": float(old_main[index]),
                "main_top_velocity_new": float(new_main[index]),
                "main_top_difference": float(new_main[index] - old_main[index]),
                "free_field_top_old": float(old_free[index]),
                "free_field_top_new": float(new_free[index]),
                "free_field_top_difference": float(new_free[index] - old_free[index]),
            }
        )
    write_csv(VAL / "stress_input_vs_velocity_converted_input.csv", list(compare_rows[0]), compare_rows)
    max_tau_diff = max(abs(float(row["tau_difference"])) for row in compare_rows)
    max_main_diff = max(abs(float(row["main_top_difference"])) for row in compare_rows)
    max_free_diff = max(abs(float(row["free_field_top_difference"])) for row in compare_rows)

    input_velocity = read_numeric(NEW / "dynamic/input_velocity_history.csv")
    input_stress = read_numeric(NEW / "dynamic/input_stress_history.csv")
    bottom_force = read_numeric(NEW / "dynamic/bottom_force_history.csv")
    bottom_grid = read_numeric(NEW / "dynamic/bottom_grid_velocity_history.csv")
    bottom_particle = read_numeric(NEW / "dynamic/bottom_particle_velocity_history.csv")
    energy = read_numeric(NEW / "dynamic/energy_history.csv")

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(input_velocity["time"], input_velocity["target_input_velocity"], label="Target input velocity", color="black")
    plot_finish(VAL / "input_velocity_wave.png", "Input velocity wave", "Velocity (m/s)")

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(input_stress["time"], input_stress["target_input_velocity"], label="Target velocity", color="black")
    plt.plot(input_stress["time"], input_stress["converted_tau_input"], label="Converted shear stress", color="red")
    plot_finish(VAL / "velocity_to_stress_conversion.png", "Velocity-to-stress conversion", "Value")

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(bottom_force["time"], bottom_force["sum_input_force_x"], label="Input force", color="black")
    plt.plot(bottom_force["time"], bottom_force["sum_dashpot_force_x"], label="Dashpot force", color="blue")
    plt.plot(bottom_force["time"], bottom_force["sum_total_bottom_force_x"], label="Total bottom force", color="red")
    plot_finish(VAL / "input_force_dashpot_force_total_force.png", "Bottom force components", "Force")

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(input_velocity["time"], input_velocity["target_input_velocity"], label="Target input velocity", color="black")
    plt.plot(bottom_grid["time"], bottom_grid["mean_vx"], label="Mean bottom grid vx", color="red")
    plt.fill_between(bottom_grid["time"], bottom_grid["min_vx"], bottom_grid["max_vx"], color="red", alpha=0.15, label="Grid min-max")
    plot_finish(VAL / "bottom_grid_velocity_vs_input_velocity.png", "Bottom grid response vs input velocity", "Velocity (m/s)")

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(input_velocity["time"], input_velocity["target_input_velocity"], label="Target input velocity", color="black")
    plt.plot(bottom_particle["time"], bottom_particle["mean_vx"], label="Mean bottom particle vx", color="green")
    plt.fill_between(bottom_particle["time"], bottom_particle["min_vx"], bottom_particle["max_vx"], color="green", alpha=0.15, label="Particle min-max")
    plot_finish(VAL / "bottom_particle_velocity_vs_input_velocity.png", "Bottom particle response vs input velocity", "Velocity (m/s)")

    main_time, main_velocity = new_history[0]
    free_time, free_velocity = new_history[1]
    main_ref_time, main_ref_velocity = main_ref
    free_ref_time, free_ref_velocity = free_ref
    main_stats = stats[0]
    free_stats = stats[3]
    plt.figure(figsize=(6.8, 4.2))
    plt.plot(main_ref_time, main_ref_velocity, label="FLAC3D main reference", color="black")
    plt.plot(main_time, main_velocity, label="GeoTaichi MPM velocity-input bottom1", color="red")
    plt.text(
        0.0005,
        0.82 * float(np.max(main_ref_velocity)),
        f"Peak error={main_stats['peak_error']:.3g}\nNRMSE={main_stats['NRMSE']:.3g}\nCorr={main_stats['correlation']:.4f}\nPhase lag={main_stats['phase_lag']:.3g}s",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    plot_finish(VAL / "figure_3_9_velocity_input_bottom1.png", "Figure 3.9 style main-top comparison", "vx (m/s)")

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(main_ref_time, main_ref_velocity, label="FLAC3D main reference", color="black")
    plt.plot(main_time, main_velocity, label="MPM main", color="red")
    plt.plot(free_ref_time, free_ref_velocity, label="FLAC3D free-field reference", color="0.35", linestyle="--")
    plt.plot(free_time, free_velocity, label="MPM free-field", color="blue", linestyle="--")
    plot_finish(VAL / "figure_3_9_velocity_input_bottom1_zoom.png", "Figure 3.9 zoom: 0.009-0.015 s", "vx (m/s)", (0.009, 0.015))

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(energy["time"], energy["external_input_work"], label="Input work")
    plt.plot(energy["time"], energy["bottom_dashpot_dissipation"], label="Bottom dashpot dissipation")
    plt.plot(energy["time"], energy["side_dashpot_dissipation"], label="Side dashpot dissipation")
    plt.plot(energy["time"], energy["total_mechanical_energy"], label="Mechanical energy")
    plot_finish(VAL / "energy_balance.png", "Energy history", "Energy")

    report = [
        "# Velocity-Input Bottom1 Rerun Report",
        "",
        "## Direct Answers",
        "",
        "- Input data is now defined as velocity history: YES",
        "- Directly constrained grid velocity: NO",
        "- Directly constrained particle velocity: NO",
        f"- Impedance conversion formula: `{FORMULA}`",
        "- Reason for `tau=rho*Cs*v_target`: `INPUT_VELOCITY_PEAK=0.1 m/s` is treated as outcrop/free-field target velocity, and this keeps the FLAC3D Example 3.3 original `dstress` peak 1.0. No factor 2 is used because the input is not redefined here as incident-wave velocity.",
        f"- rho*Cs = {ETA}",
        f"- v_peak = {VPEAK} converts to tau_peak = {ETA * VPEAK}",
        "- Bottom remains bottom1: YES",
        "- Bottom selected layer: z=0.0 only",
        "- Bottom node count: 53",
        "- Bottom total area: 6.25",
        "- Quiet dashpot remains `-rho*Cs*v_grid*A`: YES",
        f"- New and old tau_input identical within tolerance: {'YES' if max_tau_diff <= 1e-12 else 'NO'} (max |diff|={max_tau_diff})",
        f"- New and old main-top dynamics identical within tolerance: {'YES' if max_main_diff <= 1e-12 else 'NO'} (max |diff|={max_main_diff})",
        f"- New and old free-field-top dynamics identical within tolerance: {'YES' if max_free_diff <= 1e-12 else 'NO'} (max |diff|={max_free_diff})",
        "- Dynamic rerun = YES",
        "- Fresh initialization = YES",
        "- Old dynamic results reused = NO",
        "",
        "## Figure 3.9 Metrics",
        "",
        f"- main_top: peak_error={main_stats['peak_error']}, NRMSE={main_stats['NRMSE']}, correlation={main_stats['correlation']}, phase_lag={main_stats['phase_lag']}",
        f"- free_field_top: peak_error={free_stats['peak_error']}, NRMSE={free_stats['NRMSE']}, correlation={free_stats['correlation']}, phase_lag={free_stats['phase_lag']}",
        "",
        "## Output Files",
        "",
        f"- {NEW / 'velocity_input_conversion_implementation_check.md'}",
        f"- {VAL / 'velocity_to_stress_conversion_check.csv'}",
        f"- {NEW / 'dynamic/input_velocity_history.csv'}",
        f"- {NEW / 'dynamic/input_stress_history.csv'}",
        f"- {NEW / 'dynamic/bottom_force_history.csv'}",
        f"- {VAL / 'stress_input_vs_velocity_converted_input.csv'}",
        f"- {VAL / 'error_statistics_velocity_input_bottom1.csv'}",
        f"- {VAL / 'input_velocity_wave.png'}",
        f"- {VAL / 'velocity_to_stress_conversion.png'}",
        f"- {VAL / 'input_force_dashpot_force_total_force.png'}",
        f"- {VAL / 'bottom_grid_velocity_vs_input_velocity.png'}",
        f"- {VAL / 'bottom_particle_velocity_vs_input_velocity.png'}",
        f"- {VAL / 'figure_3_9_velocity_input_bottom1.png'}",
        f"- {VAL / 'figure_3_9_velocity_input_bottom1_zoom.png'}",
        f"- {VAL / 'energy_balance.png'}",
        "",
        "## Correct Description",
        "",
        "输入以速度时程定义，经剪切波阻抗转换为应力和节点力，再施加到带有黏性吸收项的 bottom1 quiet base。",
    ]
    (NEW / "velocity_input_bottom1_rerun_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"report={NEW / 'velocity_input_bottom1_rerun_report.md'}")
    print(f"max_tau_diff={max_tau_diff}")
    print(f"max_main_diff={max_main_diff}")
    print(f"max_free_diff={max_free_diff}")
    print(f"main_nrmse={main_stats['NRMSE']}")
    print(f"main_correlation={main_stats['correlation']}")


if __name__ == "__main__":
    main()
