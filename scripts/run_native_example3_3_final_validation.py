import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import examples.example3_3_2d_kohler_native_mpm_solver as native  # noqa: E402


OUT = ROOT / "output" / "example3_3_2d_kohler_native_mpm_solver"
HISTORY_PATH = OUT / "example3_3_final_velocity_history.csv"
FIGURE_PATH = OUT / "example3_3_fig3_9_comparison.png"
REPORT_PATH = OUT / "example3_3_final_validation_report.md"
FLAC_MAIN_PATH = ROOT / "data" / "reference" / "flac3d_example3_3_free_field" / "reference_fig_3_9_flac_main_correct.csv"
FLAC_FREE_PATH = ROOT / "data" / "reference" / "flac3d_example3_3_free_field" / "reference_fig_3_9_flac_free_correct.csv"

FLAC_FIGURE_3_9_TIME = 0.015
NUMERICAL_RESPONSE_TOLERANCE = 1.0e-12


def nearest_particle(mpm: native.MPM, target_x: float, target_z: float, body_id: int) -> dict[str, float | int]:
    particle_count = int(mpm.scene.particleNum[0])
    positions = mpm.scene.particle.x.to_numpy()[:particle_count]
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count]
    computational_x = target_x - native.PHYSICAL_X_OFFSET
    best_id = -1
    best_distance = math.inf
    for particle_id in range(particle_count):
        if int(body_ids[particle_id]) != body_id:
            continue
        dx = float(positions[particle_id, 0]) - computational_x
        dz = float(positions[particle_id, 1]) - target_z
        distance = math.hypot(dx, dz)
        if distance < best_distance:
            best_distance = distance
            best_id = particle_id
    if best_id < 0:
        raise RuntimeError(f"No particle found for target=({target_x}, {target_z}), body_id={body_id}")
    return {
        "particle_id": best_id,
        "body_id": body_id,
        "target_x": target_x,
        "target_z": target_z,
        "actual_x": float(positions[best_id, 0]) + native.PHYSICAL_X_OFFSET,
        "actual_z": float(positions[best_id, 1]),
        "distance": best_distance,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_reference(path: Path) -> tuple[list[float], list[float]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    time_key = "time_s" if "time_s" in reader.fieldnames else "time"
    value_key = "vx" if "vx" in reader.fieldnames else "vx_flac3d"
    return [float(row[time_key]) for row in rows], [float(row[value_key]) for row in rows]


def interp(times: list[float], values: list[float], query: list[float]) -> list[float]:
    if not times:
        return []
    out: list[float] = []
    j = 0
    for q in query:
        while j + 1 < len(times) and times[j + 1] < q:
            j += 1
        if q <= times[0]:
            out.append(values[0])
        elif q >= times[-1]:
            out.append(values[-1])
        else:
            t0, t1 = times[j], times[j + 1]
            v0, v1 = values[j], values[j + 1]
            a = (q - t0) / (t1 - t0)
            out.append(v0 * (1.0 - a) + v1 * a)
    return out


def peak_abs(values: list[float]) -> float:
    return max((abs(v) for v in values), default=0.0)


def signed_peak(values: list[float], times: list[float], absolute_floor: float = 0.0) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    idx = max(range(len(values)), key=lambda i: abs(values[i]))
    if abs(values[idx]) <= absolute_floor:
        return math.nan, math.nan
    return values[idx], times[idx]


def arrival_time(values: list[float], times: list[float], rel: float = 0.05, absolute_floor: float = 0.0) -> tuple[float, float]:
    peak = peak_abs(values)
    threshold = max(peak * rel, absolute_floor)
    if peak <= absolute_floor:
        return math.nan, threshold
    for t, v in zip(times, values):
        if abs(v) >= threshold:
            return t, threshold
    return math.nan, threshold


def correlation(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return math.nan
    aa = a[:n]
    bb = b[:n]
    ma = sum(aa) / n
    mb = sum(bb) / n
    da = [x - ma for x in aa]
    db = [x - mb for x in bb]
    va = sum(x * x for x in da)
    vb = sum(x * x for x in db)
    if va <= 0.0 or vb <= 0.0:
        return math.nan
    return sum(x * y for x, y in zip(da, db)) / math.sqrt(va * vb)


def nrmse(model: list[float], ref: list[float]) -> float:
    n = min(len(model), len(ref))
    if n == 0:
        return math.nan
    rmse = math.sqrt(sum((model[i] - ref[i]) ** 2 for i in range(n)) / n)
    scale = max(ref[:n]) - min(ref[:n])
    if scale <= 0.0:
        return math.nan
    return rmse / scale


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    native.SIMULATION_TIME = FLAC_FIGURE_3_9_TIME
    trace = {
        "solver_core_calls": 0,
        "ul_explicit_usl_updating_calls": 0,
        "p2g_calls": 0,
        "compute_forces_calls": 0,
        "grid_update_calls": 0,
        "g2p_calls": 0,
        "velocity_gradient_calls": 0,
        "stress_update_calls": 0,
        "compliant_particle_traction_update_calls": 0,
    }
    mpm = native.build_native_mpm(trace)
    main_top = nearest_particle(mpm, 2.0, 5.0, native.MAIN_BODY_ID)
    free_top = nearest_particle(mpm, -0.25, 5.0, native.LEFT_FF_BODY_ID)
    history_rows: list[dict[str, float]] = []

    def recorder() -> None:
        mpm.native_compliant_base.record_monitor(mpm.sims, mpm.scene)
        current_time = float(mpm.sims.current_time)
        if history_rows and abs(current_time - float(history_rows[-1]["time"])) <= 1.0e-15:
            return
        history_rows.append(
            {
                "time": current_time,
                "input_velocity": native.input_velocity(current_time),
                "main_top_vx": float(mpm.scene.particle[int(main_top["particle_id"])].v[0]),
                "free_field_top_vx": float(mpm.scene.particle[int(free_top["particle_id"])].v[0]),
            }
        )

    mpm.run(function=recorder)
    native_pass = all(
        trace.get(key, 0) > 0
        for key in (
            "solver_core_calls",
            "ul_explicit_usl_updating_calls",
            "p2g_calls",
            "compute_forces_calls",
            "grid_update_calls",
            "g2p_calls",
            "velocity_gradient_calls",
            "stress_update_calls",
        )
    )
    native.write_report(mpm, trace)
    mpm.native_compliant_base.write_outputs(native_pass, trace)

    write_csv(HISTORY_PATH, ["time", "input_velocity", "main_top_vx", "free_field_top_vx"], history_rows)

    times = [float(row["time"]) for row in history_rows]
    input_velocity = [float(row["input_velocity"]) for row in history_rows]
    main_v = [float(row["main_top_vx"]) for row in history_rows]
    free_v = [float(row["free_field_top_vx"]) for row in history_rows]
    flac_main_t, flac_main_v = read_reference(FLAC_MAIN_PATH)
    flac_free_t, flac_free_v = read_reference(FLAC_FREE_PATH)
    main_at_flac = interp(times, main_v, flac_main_t) if flac_main_t else []
    free_at_flac = interp(times, free_v, flac_free_t) if flac_free_t else []

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(times, main_v, color="#d62728", linewidth=1.8, label="MPM main")
    ax.plot(times, free_v, color="#1f77b4", linewidth=1.8, label="MPM free-field")
    if flac_main_t:
        ax.plot(flac_main_t, flac_main_v, color="#8c1d18", linestyle="--", linewidth=1.3, label="FLAC3D main digitized")
    if flac_free_t:
        ax.plot(flac_free_t, flac_free_v, color="#0b4f8a", linestyle="--", linewidth=1.3, label="FLAC3D free-field digitized")
    ax.set_xlabel("time(s)")
    ax.set_ylabel("vx(m/s)")
    ax.set_title("Example 3.3 Figure 3.9 comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)

    main_peak = peak_abs(main_v)
    free_peak = peak_abs(free_v)
    input_peak = peak_abs(input_velocity)
    input_peak_signed, input_peak_time = signed_peak(input_velocity, times, NUMERICAL_RESPONSE_TOLERANCE)
    input_arrival, input_threshold = arrival_time(input_velocity, times, absolute_floor=NUMERICAL_RESPONSE_TOLERANCE)
    main_peak_signed, main_peak_time = signed_peak(main_v, times, NUMERICAL_RESPONSE_TOLERANCE)
    free_peak_signed, free_peak_time = signed_peak(free_v, times, NUMERICAL_RESPONSE_TOLERANCE)
    main_arrival, main_threshold = arrival_time(main_v, times, absolute_floor=NUMERICAL_RESPONSE_TOLERANCE)
    free_arrival, free_threshold = arrival_time(free_v, times, absolute_floor=NUMERICAL_RESPONSE_TOLERANCE)
    phase_difference = main_peak_time - free_peak_time
    amp_ratio = main_peak / free_peak if free_peak > NUMERICAL_RESPONSE_TOLERANCE else math.nan
    main_flac_nrmse = nrmse(main_at_flac, flac_main_v) if flac_main_t else math.nan
    free_flac_nrmse = nrmse(free_at_flac, flac_free_v) if flac_free_t else math.nan
    main_flac_corr = correlation(main_at_flac, flac_main_v) if flac_main_t else math.nan
    free_flac_corr = correlation(free_at_flac, flac_free_v) if flac_free_t else math.nan

    lines = [
        "# Example 3.3 Final Validation Report",
        "",
        "## Scope",
        "",
        "- Objective: full FLAC3D Example 3.3 Figure 3.9 time-window validation for the current GeoTaichi MPM boundary implementation.",
        "- Boundary implementation changed in this run: NO.",
        "- No amplitude tuning, manual scaling, or input-wave modification was applied.",
        "- Solver path: GeoTaichi native MPM solver, ULExplicitEngine, USL mapping.",
        "- Bottom compliant base: retained through native particle traction.",
        "- Lateral free-field coupling: retained through native particle traction.",
        f"- simulation_time: {FLAC_FIGURE_3_9_TIME}",
        f"- dt: {native.DT}",
        f"- recorded rows: {len(history_rows)}",
        f"- first time: {times[0] if times else math.nan}",
        f"- last time: {times[-1] if times else math.nan}",
        f"- history_csv: `{HISTORY_PATH}`",
        f"- figure: `{FIGURE_PATH}`",
        "",
        "## Monitors",
        "",
        "| monitor | particle_id | body_id | target_x | target_z | actual_x | actual_z | distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Main top | {main_top['particle_id']} | {main_top['body_id']} | {main_top['target_x']} | {main_top['target_z']} | {main_top['actual_x']} | {main_top['actual_z']} | {main_top['distance']} |",
        f"| Free-field top | {free_top['particle_id']} | {free_top['body_id']} | {free_top['target_x']} | {free_top['target_z']} | {free_top['actual_x']} | {free_top['actual_z']} | {free_top['distance']} |",
        "",
        "## Native Solver Calls",
        "",
        f"- solver core calls: {trace.get('solver_core_calls', 0)}",
        f"- ULExplicitEngine USL calls: {trace.get('ul_explicit_usl_updating_calls', 0)}",
        f"- P2G calls: {trace.get('p2g_calls', 0)}",
        f"- native apply_particle_traction_constraints calls: {trace.get('particle_traction_calls', 0)}",
        f"- side/bottom particle traction update calls: {trace.get('compliant_particle_traction_update_calls', 0)}",
        f"- compute_forces calls: {trace.get('compute_forces_calls', 0)}",
        f"- grid update calls: {trace.get('grid_update_calls', 0)}",
        f"- G2P calls: {trace.get('g2p_calls', 0)}",
        f"- stress update calls: {trace.get('stress_update_calls', 0)}",
        "",
        "## MPM Statistics",
        "",
        "| curve | peak velocity | signed peak velocity | peak time | arrival time | arrival threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Bottom input | {input_peak} | {input_peak_signed} | {input_peak_time} | {input_arrival} | {input_threshold} |",
        f"| MPM main | {main_peak} | {main_peak_signed} | {main_peak_time} | {main_arrival} | {main_threshold} |",
        f"| MPM free-field | {free_peak} | {free_peak_signed} | {free_peak_time} | {free_arrival} | {free_threshold} |",
        "",
        "## Main / Free-field",
        "",
        f"- phase difference, main peak time minus free-field peak time: {phase_difference} s",
        f"- main/free-field amplitude ratio: {amp_ratio}",
        "",
        "## FLAC3D Digitized Comparison",
        "",
        f"- FLAC main reference: `{FLAC_MAIN_PATH}`",
        f"- FLAC free-field reference: `{FLAC_FREE_PATH}`",
        f"- FLAC reference time range: {min(flac_main_t) if flac_main_t else 'NA'} to {max(flac_main_t) if flac_main_t else 'NA'} s",
        "",
        "| curve | FLAC peak velocity | NRMSE | correlation |",
        "| --- | ---: | ---: | ---: |",
        f"| main | {peak_abs(flac_main_v) if flac_main_v else math.nan} | {main_flac_nrmse} | {main_flac_corr} |",
        f"| free-field | {peak_abs(flac_free_v) if flac_free_v else math.nan} | {free_flac_nrmse} | {free_flac_corr} |",
        "",
        "## Judgment",
        "",
        f"- bottom input recorded: {'YES' if peak_abs(input_velocity) > 0.0 else 'NO'}",
        f"- top MPM response above numerical tolerance ({NUMERICAL_RESPONSE_TOLERANCE}) inside FLAC Figure 3.9 time window: {'YES' if main_peak > NUMERICAL_RESPONSE_TOLERANCE or free_peak > NUMERICAL_RESPONSE_TOLERANCE else 'NO'}",
        f"- reproduces FLAC3D Figure 3.9 amplitude/time response in current settings: {'YES' if main_flac_nrmse < 0.2 and free_flac_nrmse < 0.2 else 'NO'}",
        "- Important constraint: material parameters, geometry, input wave, bottom traction formula, and lateral free-field formula were not changed.",
        f"- Shear-wave travel estimate H/Cs: {native.HEIGHT / native.CS} s, which exceeds the FLAC Figure 3.9 0.015 s time window for the current retained parameters.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"history={HISTORY_PATH}")
    print(f"figure={FIGURE_PATH}")
    print(f"report={REPORT_PATH}")
    print(f"rows={len(history_rows)}")
    print(f"main_peak={main_peak}")
    print(f"free_field_peak={free_peak}")
    print(f"main_flac_nrmse={main_flac_nrmse}")
    print(f"free_flac_nrmse={free_flac_nrmse}")


if __name__ == "__main__":
    run()
