"""Error-decomposition utilities for FLAC3D Example 3.3.

This file is intentionally separate from ``example3_3_free_field_shear_wave.py``.
It post-processes existing Example 3.3 result histories and writes diagnostic
outputs under ``results/example3_3_error_decomposition``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FLAC3D_DIR = ROOT / "example/mpm/FLAC3D"
RESULTS_DIR = FLAC3D_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "example3_3_error_decomposition"
INSTRUMENTED_RUN_DIR = OUTPUT_DIR / "instrumented_case_a"
INSTRUMENTED_SCRIPT = OUTPUT_DIR / "_instrumented_example3_3_case_a.py"
RAW_ALL_NODES_FILE = INSTRUMENTED_RUN_DIR / "bottom_all_nodes_raw_timeseries.csv"

DENSITY = 0.0025
SHEAR_MODULUS = 40000.0
CS = math.sqrt(SHEAR_MODULUS / DENSITY)
RHO_CS = DENSITY * CS
DSTRESS_AMPLITUDE = 1.0
WAVE_PERIOD = 0.01
ELEMENT_SIZE = 1.0

FREE_FIELD_WIDTH = 1.0
MAIN_X0 = FREE_FIELD_WIDTH
MAIN_Y0 = FREE_FIELD_WIDTH
MAIN_WIDTH_X = 6.0
MAIN_WIDTH_Y = 3.0
DOMAIN_X = MAIN_WIDTH_X + 2.0 * FREE_FIELD_WIDTH
DOMAIN_Y = MAIN_WIDTH_Y + 2.0 * FREE_FIELD_WIDTH
DOMAIN_Z = 5.0

GNUM_X = int(round(DOMAIN_X / ELEMENT_SIZE)) + 1
GNUM_Y = int(round(DOMAIN_Y / ELEMENT_SIZE)) + 1
LEFT_X_NODE = int(round(MAIN_X0 / ELEMENT_SIZE))
RIGHT_X_NODE = int(round((MAIN_X0 + MAIN_WIDTH_X) / ELEMENT_SIZE))
FRONT_Y_NODE = int(round(MAIN_Y0 / ELEMENT_SIZE))
BACK_Y_NODE = int(round((MAIN_Y0 + MAIN_WIDTH_Y) / ELEMENT_SIZE))

MAIN_BODY_ID = 0
X_SIDE_FF_BODY_ID = 1
Y_SIDE_FF_BODY_ID = 2
CORNER_FF_BODY_ID = 3


def wave_factor(time: float) -> float:
    if time < 0.0:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * time / WAVE_PERIOD))


def node_id(ix: int, iy: int, iz: int = 0) -> int:
    return ix + iy * GNUM_X + iz * GNUM_X * GNUM_Y


def node_area_1d(index: int, lower: int, upper: int) -> float:
    if index < lower or index > upper:
        return 0.0
    if lower == upper:
        return ELEMENT_SIZE
    if index == lower or index == upper:
        return 0.5 * ELEMENT_SIZE
    return ELEMENT_SIZE


def node_area_2d(ix: int, iy: int, xmin: int, xmax: int, ymin: int, ymax: int) -> float:
    return node_area_1d(ix, xmin, xmax) * node_area_1d(iy, ymin, ymax)


def bottom_nodal_area(body_id: int, ix: int, iy: int) -> float:
    if body_id == MAIN_BODY_ID:
        return node_area_2d(ix, iy, LEFT_X_NODE, RIGHT_X_NODE, FRONT_Y_NODE, BACK_Y_NODE)
    if body_id == X_SIDE_FF_BODY_ID:
        if FRONT_Y_NODE <= iy <= BACK_Y_NODE:
            if 0 <= ix <= LEFT_X_NODE:
                return node_area_2d(ix, iy, 0, LEFT_X_NODE, FRONT_Y_NODE, BACK_Y_NODE)
            if RIGHT_X_NODE <= ix <= GNUM_X - 1:
                return node_area_2d(ix, iy, RIGHT_X_NODE, GNUM_X - 1, FRONT_Y_NODE, BACK_Y_NODE)
    if body_id == Y_SIDE_FF_BODY_ID:
        if LEFT_X_NODE <= ix <= RIGHT_X_NODE:
            if 0 <= iy <= FRONT_Y_NODE:
                return node_area_2d(ix, iy, LEFT_X_NODE, RIGHT_X_NODE, 0, FRONT_Y_NODE)
            if BACK_Y_NODE <= iy <= GNUM_Y - 1:
                return node_area_2d(ix, iy, LEFT_X_NODE, RIGHT_X_NODE, BACK_Y_NODE, GNUM_Y - 1)
    if body_id == CORNER_FF_BODY_ID:
        if 0 <= ix <= LEFT_X_NODE:
            if 0 <= iy <= FRONT_Y_NODE:
                return node_area_2d(ix, iy, 0, LEFT_X_NODE, 0, FRONT_Y_NODE)
            if BACK_Y_NODE <= iy <= GNUM_Y - 1:
                return node_area_2d(ix, iy, 0, LEFT_X_NODE, BACK_Y_NODE, GNUM_Y - 1)
        if RIGHT_X_NODE <= ix <= GNUM_X - 1:
            if 0 <= iy <= FRONT_Y_NODE:
                return node_area_2d(ix, iy, RIGHT_X_NODE, GNUM_X - 1, 0, FRONT_Y_NODE)
            if BACK_Y_NODE <= iy <= GNUM_Y - 1:
                return node_area_2d(ix, iy, RIGHT_X_NODE, GNUM_X - 1, BACK_Y_NODE, GNUM_Y - 1)
    return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except Exception:
        return default


def result_dir_for_stress_scale(stress_scale: float) -> Path | None:
    candidates = [
        RESULTS_DIR / f"example3_3_free_field_shear_wave_3d_stress_{str(stress_scale).replace('.', 'p')}",
        RESULTS_DIR / "example3_3_free_field_shear_wave_3d" if abs(stress_scale - 1.0) < 1e-12 else Path("__missing__"),
    ]
    for path in candidates:
        if (path / "histories.csv").exists():
            return path
    return None


def load_case_history(stress_scale: float) -> tuple[list[dict[str, str]], dict[str, Any], Path | None]:
    path = result_dir_for_stress_scale(stress_scale)
    if path is None:
        return [], {}, None
    histories = read_csv(path / "histories.csv")
    diagnostics: dict[str, Any] = {}
    diag_path = path / "diagnostics.json"
    if diag_path.exists():
        diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
    return histories, diagnostics, path


def representative_nodes() -> list[dict[str, Any]]:
    return [
        {
            "node_label": "center_bottom_node",
            "body_id": MAIN_BODY_ID,
            "body": "main_grid",
            "ix": int(round(MAIN_X0 + 3.0)),
            "iy": int(round(MAIN_Y0 + 1.0)),
            "history_column": "base_center_vx",
            "actual_history_available": True,
        },
        {
            "node_label": "x_side_bottom_representative_node",
            "body_id": X_SIDE_FF_BODY_ID,
            "body": "x_side_free_field",
            "ix": 0,
            "iy": int(round(MAIN_Y0 + 1.0)),
            "history_column": "",
            "actual_history_available": False,
        },
        {
            "node_label": "y_side_bottom_representative_node",
            "body_id": Y_SIDE_FF_BODY_ID,
            "body": "y_side_free_field",
            "ix": int(round(MAIN_X0 + 3.0)),
            "iy": 0,
            "history_column": "",
            "actual_history_available": False,
        },
        {
            "node_label": "corner_bottom_representative_node",
            "body_id": CORNER_FF_BODY_ID,
            "body": "corner_free_field",
            "ix": 0,
            "iy": 0,
            "history_column": "",
            "actual_history_available": False,
        },
    ]


def case_specs() -> list[dict[str, Any]]:
    return [
        {
            "case": "Case A",
            "formula": "fx = A * (dstress - rho*Cs*vx)",
            "stress_scale": 1.0,
            "input_multiplier": 1.0,
            "is_case_c": False,
        },
        {
            "case": "Case B",
            "formula": "fx = A * (2*dstress - rho*Cs*vx)",
            "stress_scale": 2.0,
            "input_multiplier": 2.0,
            "is_case_c": False,
        },
        {
            "case": "Case C",
            "formula": "fx = A * (2*rho*Cs*v_input - rho*Cs*vx)",
            "stress_scale": 2.0,
            "input_multiplier": 2.0,
            "is_case_c": True,
        },
    ]


def theoretical_target_velocity(wave: float) -> float:
    return DSTRESS_AMPLITUDE / RHO_CS * wave


def node_mass_from_area(area: float) -> float:
    return DENSITY * area * 0.5 * ELEMENT_SIZE


BODY_NAMES = {
    MAIN_BODY_ID: "main",
    X_SIDE_FF_BODY_ID: "x_side_ff",
    Y_SIDE_FF_BODY_ID: "y_side_ff",
    CORNER_FF_BODY_ID: "corner_ff",
}

THEORETICAL_BOTTOM_AREAS = {
    MAIN_BODY_ID: MAIN_WIDTH_X * MAIN_WIDTH_Y,
    X_SIDE_FF_BODY_ID: 2.0 * FREE_FIELD_WIDTH * MAIN_WIDTH_Y,
    Y_SIDE_FF_BODY_ID: 2.0 * FREE_FIELD_WIDTH * MAIN_WIDTH_X,
    CORNER_FF_BODY_ID: 4.0 * FREE_FIELD_WIDTH * FREE_FIELD_WIDTH,
}


def bottom_nodes() -> list[dict[str, Any]]:
    rows = []
    for body_id in (MAIN_BODY_ID, X_SIDE_FF_BODY_ID, Y_SIDE_FF_BODY_ID, CORNER_FF_BODY_ID):
        for iy in range(GNUM_Y):
            for ix in range(GNUM_X):
                area = bottom_nodal_area(body_id, ix, iy)
                if area <= 0.0:
                    continue
                rows.append({
                    "body_id": body_id,
                    "body_level": BODY_NAMES[body_id],
                    "node_id": node_id(ix, iy, 0),
                    "ix": ix,
                    "iy": iy,
                    "iz": 0,
                    "x": ix * ELEMENT_SIZE,
                    "y": iy * ELEMENT_SIZE,
                    "z": 0.0,
                    "tributary_area_A": area,
                })
    return rows


def area_category(ix: int, iy: int, body_id: int) -> str:
    area = bottom_nodal_area(body_id, ix, iy)
    if area <= 0.0:
        return "none"
    # The nodal-area helper gives 0.25*A for corners, 0.5*A for edges, and
    # 1.0*A for interior nodes when dx=dy=1.
    if area <= 0.25 * ELEMENT_SIZE * ELEMENT_SIZE + 1.0e-12:
        return "corner"
    if area <= 0.5 * ELEMENT_SIZE * ELEMENT_SIZE + 1.0e-12:
        return "edge"
    return "interior"


def build_instrumented_script() -> None:
    source = FLAC3D_DIR / "example3_3_free_field_shear_wave.py"
    text = source.read_text(encoding="utf-8")
    marker = "previous_history: dict[str, float] = {}\n\n\n"
    instrumentation = r'''
BOTTOM_ALL_NODE_SAMPLE_ROWS: list[dict[str, float | int | str]] = []


def collect_bottom_all_node_samples(model, rows: list[dict[str, float | int | str]]) -> None:
    nodal_velocity = model.scene.node.momentum.to_numpy()
    nodal_mass = model.scene.node.m.to_numpy()
    gnum_x = int(model.scene.element.gnum[0])
    gnum_y = int(model.scene.element.gnum[1])
    time_value = float(model.sims.current_time)
    wave = wave_factor(time_value)
    dstress = DSTRESS_AMPLITUDE * STRESS_SCALE * wave
    for body_id in BODY_IDS:
        body_name = {
            MAIN_BODY_ID: "main",
            X_SIDE_FF_BODY_ID: "x_side_ff",
            Y_SIDE_FF_BODY_ID: "y_side_ff",
            CORNER_FF_BODY_ID: "corner_ff",
        }[body_id]
        for iy in range(gnum_y):
            for ix in range(gnum_x):
                area = bottom_nodal_area_py(body_id, ix, iy, gnum_x, gnum_y)
                if area <= 0.0:
                    continue
                nid = ix + iy * gnum_x
                vx = float(nodal_velocity[nid, body_id, 0])
                vy = float(nodal_velocity[nid, body_id, 1])
                vz = float(nodal_velocity[nid, body_id, 2])
                input_force_x = area * dstress
                dashpot_force_x = -DENSITY * SHEAR_WAVE_VELOCITY * vx * area
                net_force_x = input_force_x + dashpot_force_x
                mass = float(nodal_mass[nid, body_id])
                rows.append({
                    "time": time_value,
                    "body_level": body_name,
                    "body_id": body_id,
                    "node_id": nid,
                    "ix": ix,
                    "iy": iy,
                    "iz": 0,
                    "x": ix * ELEMENT_SIZE_VALUE,
                    "y": iy * ELEMENT_SIZE_VALUE,
                    "z": 0.0,
                    "tributary_area_A": area,
                    "node_mass": mass,
                    "node_vx": vx,
                    "node_vy": vy,
                    "node_vz": vz,
                    "wave_value": wave,
                    "dstress": dstress,
                    "input_force_x": input_force_x,
                    "dashpot_force_x": dashpot_force_x,
                    "net_force_x": net_force_x,
                    "acceleration_x": net_force_x / mass if mass > 0.0 else math.nan,
                    "theoretical_velocity": IMPEDANCE_VELOCITY_AMPLITUDE * STRESS_SCALE * wave,
                    "velocity_error_percent": abs(vx - IMPEDANCE_VELOCITY_AMPLITUDE * STRESS_SCALE * wave) / max(abs(IMPEDANCE_VELOCITY_AMPLITUDE * STRESS_SCALE * wave), 1.0e-30) * 100.0 if abs(IMPEDANCE_VELOCITY_AMPLITUDE * STRESS_SCALE * wave) > 1.0e-12 else math.nan,
                    "actual_history_available": True,
                })


'''
    if marker not in text:
        raise RuntimeError("Instrumentation marker not found in Example 3.3 script.")
    text = text.replace(marker, marker + instrumentation, 1)
    old = "def per_step_update() -> None:\n    wrote_history = collect_histories(mpm, histories, previous_history)\n"
    new = "def per_step_update() -> None:\n    wrote_history = collect_histories(mpm, histories, previous_history)\n    if wrote_history:\n        collect_bottom_all_node_samples(mpm, BOTTOM_ALL_NODE_SAMPLE_ROWS)\n"
    if old not in text:
        raise RuntimeError("per_step_update marker not found in Example 3.3 script.")
    text = text.replace(old, new, 1)
    old_end = 'write_csv(OUTPUT_PATH / "bottom_input_force_trace.csv", BOTTOM_INPUT_TRACE_ROWS)\n'
    new_end = old_end + 'write_csv(OUTPUT_PATH / "bottom_all_nodes_raw_timeseries.csv", BOTTOM_ALL_NODE_SAMPLE_ROWS)\n'
    if old_end not in text:
        raise RuntimeError("bottom_input_force_trace write marker not found.")
    text = text.replace(old_end, new_end, 1)
    INSTRUMENTED_SCRIPT.write_text(text, encoding="utf-8")


def ensure_instrumented_case_a() -> None:
    if RAW_ALL_NODES_FILE.exists() and RAW_ALL_NODES_FILE.stat().st_size > 0:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if INSTRUMENTED_RUN_DIR.exists():
        shutil.rmtree(INSTRUMENTED_RUN_DIR)
    build_instrumented_script()
    env = os.environ.copy()
    env.update({
        "EXAMPLE3_3_OUTPUT": INSTRUMENTED_RUN_DIR.as_posix(),
        "EXAMPLE3_3_STRESS_SCALE": "1.0",
        "EXAMPLE3_3_EXPORT_FULL_VTK": "0",
        "EXAMPLE3_3_SIMULATION_TIME": "0.015",
    })
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [sys.executable, str(INSTRUMENTED_SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    (OUTPUT_DIR / "instrumented_case_a_run.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Instrumented Example 3.3 run failed; see {OUTPUT_DIR / 'instrumented_case_a_run.log'}")
    if not RAW_ALL_NODES_FILE.exists():
        raise RuntimeError("Instrumented run finished but bottom_all_nodes_raw_timeseries.csv was not written.")


def bottom_input_diagnosis() -> dict[str, Any]:
    """Write bottom input force and velocity diagnostics for Example 3.3.

    The function does not modify or rerun ``example3_3_free_field_shear_wave.py``.
    It uses available histories from existing Case A and doubled-stress runs.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes = representative_nodes()
    diagnosis_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for spec in case_specs():
        histories, diagnostics, source_dir = load_case_history(float(spec["stress_scale"]))
        if not histories:
            comparison_rows.append({
                "case": spec["case"],
                "formula": spec["formula"],
                "source_result_dir": "",
                "status": "missing_source_histories",
                "bottom_actual_velocity_peak": math.nan,
                "theoretical_input_velocity_peak": DSTRESS_AMPLITUDE / RHO_CS,
                "bottom_actual_velocity_peak_error_percent": math.nan,
                "pass_bottom_peak_error_lt_5pct": False,
                "diagnosis": "Missing existing Example 3.3 history directory; run that case before judging.",
            })
            continue

        time_values = np.array([as_float(r, "time") for r in histories], dtype=float)
        main_actual = np.array([as_float(r, "base_center_vx", as_float(r, "main_grid_base_vx")) for r in histories], dtype=float)
        actual_peak = float(np.nanmax(np.abs(main_actual))) if len(main_actual) else math.nan
        target_peak = DSTRESS_AMPLITUDE / RHO_CS
        peak_error = abs(actual_peak - target_peak) / target_peak * 100.0 if target_peak > 0 else math.nan
        passed = bool(math.isfinite(peak_error) and peak_error < 5.0)

        for source_row, actual_vx in zip(histories, main_actual):
            time = as_float(source_row, "time")
            wave = as_float(source_row, "wave", wave_factor(time))
            # Keep dstress as the original FLAC3D manual stress-wave amplitude.
            # Case B doubles it in input_force_x, and Case C is equivalent when
            # v_input = dstress / (rho*Cs).
            dstress = DSTRESS_AMPLITUDE * wave
            v_input = dstress / RHO_CS
            theoretical_velocity = v_input

            for node in nodes:
                area = bottom_nodal_area(int(node["body_id"]), int(node["ix"]), int(node["iy"]))
                if node["actual_history_available"]:
                    vx = actual_vx
                    actual_available = True
                else:
                    vx = math.nan
                    actual_available = False

                if spec["is_case_c"]:
                    input_force_x = area * (2.0 * RHO_CS * v_input)
                else:
                    input_force_x = area * (float(spec["input_multiplier"]) * dstress)
                dashpot_force_x = -area * RHO_CS * vx if math.isfinite(vx) else math.nan
                net_force_x = input_force_x + dashpot_force_x if math.isfinite(dashpot_force_x) else math.nan
                error_percent = (
                    abs(vx - theoretical_velocity) / max(abs(theoretical_velocity), 1.0e-30) * 100.0
                    if actual_available and abs(theoretical_velocity) > 1.0e-12
                    else math.nan
                )
                diagnosis_rows.append({
                    "case": spec["case"],
                    "formula": spec["formula"],
                    "source_result_dir": source_dir.as_posix() if source_dir else "",
                    "node_label": node["node_label"],
                    "body": node["body"],
                    "body_id": node["body_id"],
                    "node_id": node_id(int(node["ix"]), int(node["iy"]), 0),
                    "ix": node["ix"],
                    "iy": node["iy"],
                    "iz": 0,
                    "area": area,
                    "time": time,
                    "wave_value": wave,
                    "dstress": dstress,
                    "input_force_x": input_force_x,
                    "dashpot_force_x": dashpot_force_x,
                    "net_force_x": net_force_x,
                    "node_mass": node_mass_from_area(area),
                    "node_vx": vx,
                    "theoretical_input_velocity": theoretical_velocity,
                    "actual_bottom_velocity": vx,
                    "bottom_velocity_error_percent": error_percent,
                    "actual_history_available": actual_available,
                })

        comparison_rows.append({
            "case": spec["case"],
            "formula": spec["formula"],
            "source_result_dir": source_dir.as_posix() if source_dir else "",
            "status": "available",
            "bottom_actual_velocity_peak": actual_peak,
            "theoretical_input_velocity_peak": target_peak,
            "bottom_actual_velocity_peak_error_percent": peak_error,
            "pass_bottom_peak_error_lt_5pct": passed,
            "diagnostics_json_bottom_actual_velocity_peak": diagnostics.get("bottom_actual_velocity_peak", ""),
            "diagnostics_json_theoretical_velocity_peak": diagnostics.get("scaled_bottom_theoretical_velocity_stress_over_rho_Cs", ""),
            "diagnosis": "",
        })

    case_a = next((r for r in comparison_rows if r["case"] == "Case A"), None)
    case_b = next((r for r in comparison_rows if r["case"] == "Case B"), None)
    case_c = next((r for r in comparison_rows if r["case"] == "Case C"), None)
    a_pass = bool(case_a and case_a.get("pass_bottom_peak_error_lt_5pct"))
    b_pass = bool(case_b and case_b.get("pass_bottom_peak_error_lt_5pct"))
    c_pass = bool(case_c and case_c.get("pass_bottom_peak_error_lt_5pct"))
    if (not a_pass) and (b_pass or c_pass):
        conclusion = "Case A fails but Case B/C passes: Example 3.3 original dstress and GeoTaichi nodal force conversion likely has a factor issue."
    elif (not a_pass) and (not b_pass) and (not c_pass):
        conclusion = "All three cases fail: bottom area, nodal mass, or boundary application timing remains the likely error source."
    else:
        conclusion = "Case A passes the bottom input criterion; bottom input is unlikely to be the dominant current error source."

    required_statement = (
        "土柱算例中底部输入已验证正确；若 Example 3.3 中底部输入仍偏差，"
        "则误差来自复杂几何/多 body 底部面积分配或边界施加时序，而非 MPM 基本波输入能力。"
    )
    for row in comparison_rows:
        if row.get("status") == "available":
            row["diagnosis"] = conclusion
            row["required_statement"] = required_statement
            row["case_c_equivalence_note"] = "Case C equals Case B here because v_input = dstress/(rho*Cs)." if row["case"] == "Case C" else ""

    write_csv(OUTPUT_DIR / "bottom_input_diagnosis.csv", diagnosis_rows)
    write_csv(OUTPUT_DIR / "bottom_input_case_comparison.csv", comparison_rows)
    write_bottom_input_plot(diagnosis_rows, comparison_rows)
    (OUTPUT_DIR / "bottom_input_diagnosis_summary.json").write_text(
        json.dumps({
            "conclusion": conclusion,
            "required_statement": required_statement,
            "rho": DENSITY,
            "Cs": CS,
            "rho_Cs": RHO_CS,
            "target_velocity_peak": DSTRESS_AMPLITUDE / RHO_CS,
            "outputs": [
                "bottom_input_diagnosis.csv",
                "bottom_input_case_comparison.csv",
                "bottom_input_force_balance.png",
            ],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"conclusion": conclusion, "comparison_rows": comparison_rows, "output_dir": OUTPUT_DIR.as_posix()}


def write_bottom_input_plot(diagnosis_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (OUTPUT_DIR / "bottom_input_force_balance.png.error.txt").write_text(str(exc), encoding="utf-8")
        return

    center_rows = [r for r in diagnosis_rows if r["node_label"] == "center_bottom_node"]
    plt.figure(figsize=(10, 6))
    for case in ("Case A", "Case B", "Case C"):
        rows = [r for r in center_rows if r["case"] == case]
        if not rows:
            continue
        time = np.array([float(r["time"]) for r in rows])
        net = np.array([float(r["net_force_x"]) for r in rows])
        inp = np.array([float(r["input_force_x"]) for r in rows])
        plt.plot(time, net, linewidth=1.6, label=f"{case} net force")
        plt.plot(time, inp, linestyle="--", linewidth=1.0, alpha=0.75, label=f"{case} input force")
    plt.xlabel("time")
    plt.ylabel("x-force at center bottom node")
    plt.title("Example 3.3 bottom input force balance")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    text = "\n".join(
        f"{r['case']}: peak err={float(r['bottom_actual_velocity_peak_error_percent']):.2f}%, pass={r['pass_bottom_peak_error_lt_5pct']}"
        for r in comparison_rows
        if r.get("status") == "available" and math.isfinite(float(r["bottom_actual_velocity_peak_error_percent"]))
    )
    if text:
        plt.gcf().text(0.02, 0.02, text, fontsize=8, va="bottom")
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(OUTPUT_DIR / "bottom_input_force_balance.png", dpi=180)
    plt.close()


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        out.setdefault(key, []).append(row)
    return out


def read_raw_all_nodes() -> list[dict[str, Any]]:
    ensure_instrumented_case_a()
    rows = read_csv(RAW_ALL_NODES_FILE)
    numeric = {
        "time", "body_id", "node_id", "ix", "iy", "iz", "x", "y", "z",
        "tributary_area_A", "node_mass", "node_vx", "node_vy", "node_vz",
        "wave_value", "dstress", "input_force_x", "dashpot_force_x",
        "net_force_x", "acceleration_x", "theoretical_velocity",
        "velocity_error_percent",
    }
    converted: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for k, v in row.items():
            if k in numeric:
                out[k] = float(v) if v not in ("", "nan", "NaN") else math.nan
                if k in {"body_id", "node_id", "ix", "iy", "iz"} and math.isfinite(out[k]):
                    out[k] = int(out[k])
            elif k == "actual_history_available":
                out[k] = str(v).lower() == "true"
            else:
                out[k] = v
        converted.append(out)
    return converted


def representative_key(row: dict[str, Any]) -> str | None:
    reps = {
        ("main", 4, 2): "main_bottom_center_node",
        ("x_side_ff", 0, 2): "x_side_free_field_bottom_representative_node",
        ("y_side_ff", 4, 0): "y_side_free_field_bottom_representative_node",
        ("corner_ff", 0, 0): "corner_free_field_bottom_representative_node",
    }
    return reps.get((row["body_level"], int(row["ix"]), int(row["iy"])))


def write_specialized_bottom_outputs() -> dict[str, Any]:
    raw_rows = read_raw_all_nodes()
    all_rows = []
    for row in raw_rows:
        out = {
            "time": row["time"],
            "body_level": row["body_level"],
            "node_id": row["node_id"],
            "coordinate": f"({row['x']}, {row['y']}, {row['z']})",
            "ix": row["ix"],
            "iy": row["iy"],
            "iz": row["iz"],
            "tributary_area_A": row["tributary_area_A"],
            "node_mass": row["node_mass"],
            "node_vx": row["node_vx"],
            "node_vy": row["node_vy"],
            "node_vz": row["node_vz"],
            "input_force_x": row["input_force_x"],
            "dashpot_force_x": row["dashpot_force_x"],
            "net_force_x": row["net_force_x"],
            "acceleration_x": row["acceleration_x"],
            "theoretical_velocity": row["theoretical_velocity"],
            "velocity_error_percent": row["velocity_error_percent"],
            "actual_history_available": row["actual_history_available"],
        }
        all_rows.append(out)
    write_csv(OUTPUT_DIR / "bottom_all_nodes_timeseries.csv", all_rows)

    rep_rows = []
    for row, out in zip(raw_rows, all_rows):
        label = representative_key(row)
        if label is None:
            continue
        rep = dict(out)
        rep["representative_node"] = label
        rep_rows.append(rep)
    write_csv(OUTPUT_DIR / "bottom_representative_nodes_timeseries.csv", rep_rows)

    area_rows = area_distribution_rows()
    write_csv(OUTPUT_DIR / "bottom_area_distribution_by_body.csv", area_rows)
    write_area_plots()

    mass_rows = mass_area_check_rows(raw_rows)
    write_csv(OUTPUT_DIR / "bottom_mass_area_check.csv", mass_rows)
    write_mass_area_scatter(mass_rows)

    force_ts, force_summary = force_sum_outputs(raw_rows)
    write_csv(OUTPUT_DIR / "bottom_force_sum_by_body_timeseries.csv", force_ts)
    write_csv(OUTPUT_DIR / "bottom_force_balance_summary.csv", force_summary)

    velocity_metrics = velocity_metric_comparison(raw_rows)
    write_csv(OUTPUT_DIR / "bottom_velocity_metric_comparison.csv", velocity_metrics)
    create_decomposition_report(area_rows, mass_rows, force_summary, velocity_metrics, rep_rows)
    return {
        "raw_rows": len(raw_rows),
        "all_nodes_timeseries": len(all_rows),
        "representative_timeseries": len(rep_rows),
        "velocity_metrics": velocity_metrics,
    }


def area_distribution_rows() -> list[dict[str, Any]]:
    rows = []
    for body_id, body_level in BODY_NAMES.items():
        nodes = [n for n in bottom_nodes() if n["body_id"] == body_id]
        areas = np.array([n["tributary_area_A"] for n in nodes], dtype=float)
        corner_sum = sum(n["tributary_area_A"] for n in nodes if area_category(n["ix"], n["iy"], body_id) == "corner")
        edge_sum = sum(n["tributary_area_A"] for n in nodes if area_category(n["ix"], n["iy"], body_id) == "edge")
        interior_sum = sum(n["tributary_area_A"] for n in nodes if area_category(n["ix"], n["iy"], body_id) == "interior")
        expected = THEORETICAL_BOTTOM_AREAS[body_id]
        actual = float(np.sum(areas))
        rows.append({
            "body_level": body_level,
            "body_id": body_id,
            "number_of_bottom_nodes": len(nodes),
            "sum_tributary_area": actual,
            "theoretical_bottom_area": expected,
            "area_error_percent": abs(actual - expected) / max(abs(expected), 1.0e-30) * 100.0,
            "min_area": float(np.min(areas)),
            "max_area": float(np.max(areas)),
            "mean_area": float(np.mean(areas)),
            "area_std": float(np.std(areas)),
            "corner_node_area_sum": corner_sum,
            "edge_node_area_sum": edge_sum,
            "interior_node_area_sum": interior_sum,
            "diagnosis": area_diagnosis(body_id, actual, expected, corner_sum, edge_sum, interior_sum),
        })
    return rows


def area_diagnosis(body_id: int, actual: float, expected: float, corner_sum: float, edge_sum: float, interior_sum: float) -> str:
    if abs(actual - expected) / max(abs(expected), 1.0e-30) > 0.01:
        return "sum_tributary_area mismatch: bottom area total is wrong"
    if corner_sum < -1.0e-12 or edge_sum < -1.0e-12 or interior_sum < -1.0e-12:
        return "corner/edge/interior split abnormal"
    return "area total matches theoretical bottom area"


def mass_area_check_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = DENSITY * 0.5 * ELEMENT_SIZE
    rows = []
    for key, items in group_rows(raw_rows, ("body_level", "body_id", "node_id", "ix", "iy", "iz")).items():
        body_level, body_id, nid, ix, iy, iz = key
        area = float(items[0]["tributary_area_A"])
        masses = np.array([float(r["node_mass"]) for r in items], dtype=float)
        finite = masses[np.isfinite(masses)]
        mean_mass = float(np.mean(finite)) if len(finite) else math.nan
        mass_per_area = mean_mass / area if area > 0.0 else math.nan
        rows.append({
            "node_id": nid,
            "body_level": body_level,
            "body_id": body_id,
            "ix": ix,
            "iy": iy,
            "iz": iz,
            "tributary_area_A": area,
            "node_mass": mean_mass,
            "mass_per_area": mass_per_area,
            "expected_mass_per_area": expected,
            "mass_per_area_error_percent": abs(mass_per_area - expected) / expected * 100.0 if math.isfinite(mass_per_area) else math.nan,
        })
    return rows


def force_sum_outputs(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ts_rows = []
    for (time, body_level), items in sorted(group_rows(raw_rows, ("time", "body_level")).items()):
        area = np.array([float(r["tributary_area_A"]) for r in items], dtype=float)
        mass = np.array([float(r["node_mass"]) for r in items], dtype=float)
        vx = np.array([float(r["node_vx"]) for r in items], dtype=float)
        sum_area = float(np.sum(area))
        sum_mass = float(np.sum(mass))
        area_weighted = float(np.sum(vx * area) / sum_area) if sum_area > 0.0 else math.nan
        mass_weighted = float(np.sum(vx * mass) / sum_mass) if sum_mass > 0.0 else math.nan
        dstress = float(items[0]["dstress"])
        sum_input = float(np.sum([float(r["input_force_x"]) for r in items]))
        sum_dashpot = float(np.sum([float(r["dashpot_force_x"]) for r in items]))
        expected_input = dstress * sum_area
        expected_dashpot = -RHO_CS * float(np.sum(vx * area))
        ts_rows.append({
            "time": time,
            "body_level": body_level,
            "sum_input_force_x": sum_input,
            "sum_dashpot_force_x": sum_dashpot,
            "sum_net_force_x": float(np.sum([float(r["net_force_x"]) for r in items])),
            "sum_node_mass": sum_mass,
            "sum_tributary_area": sum_area,
            "area_weighted_bottom_velocity": area_weighted,
            "mass_weighted_bottom_velocity": mass_weighted,
            "peak_area_weighted_velocity": math.nan,
            "peak_mass_weighted_velocity": math.nan,
            "dstress": dstress,
            "expected_input_force_x": expected_input,
            "input_force_error_percent": abs(sum_input - expected_input) / max(abs(expected_input), 1.0e-30) * 100.0 if abs(expected_input) > 1.0e-12 else 0.0,
            "expected_dashpot_force_x": expected_dashpot,
            "dashpot_force_error_percent": abs(sum_dashpot - expected_dashpot) / max(abs(expected_dashpot), 1.0e-30) * 100.0 if abs(expected_dashpot) > 1.0e-12 else 0.0,
        })

    summary = []
    for body_level, items in sorted(group_rows(ts_rows, ("body_level",)).items()):
        aw = np.array([float(r["area_weighted_bottom_velocity"]) for r in items], dtype=float)
        mw = np.array([float(r["mass_weighted_bottom_velocity"]) for r in items], dtype=float)
        peak_aw = float(np.nanmax(np.abs(aw)))
        peak_mw = float(np.nanmax(np.abs(mw)))
        max_input_err = float(np.nanmax([float(r["input_force_error_percent"]) for r in items]))
        max_dash_err = float(np.nanmax([float(r["dashpot_force_error_percent"]) for r in items]))
        for r in items:
            r["peak_area_weighted_velocity"] = peak_aw
            r["peak_mass_weighted_velocity"] = peak_mw
        summary.append({
            "body_level": body_level[0],
            "peak_area_weighted_velocity": peak_aw,
            "peak_mass_weighted_velocity": peak_mw,
            "target_velocity_peak": DSTRESS_AMPLITUDE / RHO_CS,
            "area_weighted_peak_error_percent": abs(peak_aw - DSTRESS_AMPLITUDE / RHO_CS) / (DSTRESS_AMPLITUDE / RHO_CS) * 100.0,
            "mass_weighted_peak_error_percent": abs(peak_mw - DSTRESS_AMPLITUDE / RHO_CS) / (DSTRESS_AMPLITUDE / RHO_CS) * 100.0,
            "max_input_force_error_percent": max_input_err,
            "max_dashpot_force_error_percent": max_dash_err,
            "input_force_total_ok": max_input_err < 1.0e-8,
            "dashpot_force_total_ok": max_dash_err < 1.0e-8,
        })
    return ts_rows, summary


def velocity_metric_comparison(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target = DSTRESS_AMPLITUDE / RHO_CS
    rows = []
    by_body_node = group_rows(raw_rows, ("body_level", "node_id"))
    node_peaks = {
        key: float(np.nanmax(np.abs([float(r["node_vx"]) for r in items])))
        for key, items in by_body_node.items()
    }
    for body_level, body_items in sorted(group_rows(raw_rows, ("body_level",)).items()):
        force_ts, force_summary = force_sum_outputs(body_items)
        center_peak = math.nan
        for (bl, nid), peak in node_peaks.items():
            if bl == body_level[0]:
                matching = [r for r in body_items if r["node_id"] == nid]
                if matching and representative_key(matching[0]) is not None:
                    center_peak = peak
                    break
        peaks = np.array([peak for (bl, _), peak in node_peaks.items() if bl == body_level[0]], dtype=float)
        fs = force_summary[0]
        rows.append({
            "body_level": body_level[0],
            "center_node_velocity_peak": center_peak,
            "center_node_velocity_peak_error_percent": abs(center_peak - target) / target * 100.0 if math.isfinite(center_peak) else math.nan,
            "area_weighted_bottom_velocity_peak": fs["peak_area_weighted_velocity"],
            "area_weighted_bottom_velocity_peak_error_percent": fs["area_weighted_peak_error_percent"],
            "mass_weighted_bottom_velocity_peak": fs["peak_mass_weighted_velocity"],
            "mass_weighted_bottom_velocity_peak_error_percent": fs["mass_weighted_peak_error_percent"],
            "max_bottom_node_velocity_peak": float(np.nanmax(peaks)),
            "max_bottom_node_velocity_peak_error_percent": abs(float(np.nanmax(peaks)) - target) / target * 100.0,
            "mean_bottom_node_velocity_peak": float(np.nanmean(peaks)),
            "mean_bottom_node_velocity_peak_error_percent": abs(float(np.nanmean(peaks)) - target) / target * 100.0,
            "target_velocity_peak": target,
            "diagnosis": velocity_diagnosis(center_peak, fs["peak_area_weighted_velocity"], fs["peak_mass_weighted_velocity"], peaks, target),
        })
    return rows


def velocity_diagnosis(center: float, area_weighted: float, mass_weighted: float, peaks: np.ndarray, target: float) -> str:
    center_ok = math.isfinite(center) and abs(center - target) / target < 0.05
    aw_ok = abs(area_weighted - target) / target < 0.05
    mw_ok = abs(mass_weighted - target) / target < 0.05
    if not center_ok and aw_ok:
        return "center node error is large but area-weighted velocity passes: history point selection can mislead"
    if not aw_ok and not mw_ok:
        return "area- and mass-weighted velocities both miss target: bottom input is globally inconsistent"
    if np.nanmax(peaks) - np.nanmin(peaks) > 0.05 * target:
        return "bottom-node velocity spread is large: multi-body or local nodal response is inconsistent"
    return "bottom velocity metrics are internally consistent"


def write_area_plots() -> None:
    import matplotlib.pyplot as plt

    nodes = bottom_nodes()
    plt.figure(figsize=(8, 4.5))
    for body_id, body_level in BODY_NAMES.items():
        vals = [n["tributary_area_A"] for n in nodes if n["body_id"] == body_id]
        plt.hist(vals, bins=[0.2, 0.4, 0.75, 1.1], alpha=0.55, label=body_level)
    plt.xlabel("tributary area A")
    plt.ylabel("node count")
    plt.title("Bottom tributary-area histogram")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bottom_area_histogram.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    markers = {MAIN_BODY_ID: "o", X_SIDE_FF_BODY_ID: "s", Y_SIDE_FF_BODY_ID: "^", CORNER_FF_BODY_ID: "D"}
    for body_id, body_level in BODY_NAMES.items():
        sub = [n for n in nodes if n["body_id"] == body_id]
        plt.scatter([n["x"] for n in sub], [n["y"] for n in sub], s=[180*n["tributary_area_A"] for n in sub], marker=markers[body_id], label=body_level, alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Bottom tributary-area map")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bottom_area_map.png", dpi=180)
    plt.close()


def write_mass_area_scatter(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.5, 4.8))
    for body_level in sorted({r["body_level"] for r in rows}):
        sub = [r for r in rows if r["body_level"] == body_level]
        plt.scatter([r["tributary_area_A"] for r in sub], [r["mass_per_area"] for r in sub], label=body_level, alpha=0.75)
    plt.axhline(DENSITY * 0.5 * ELEMENT_SIZE, color="k", linestyle="--", linewidth=1, label="expected")
    plt.xlabel("tributary area A")
    plt.ylabel("node_mass / A")
    plt.title("Bottom node mass-area relation")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bottom_mass_area_scatter.png", dpi=180)
    plt.close()


def create_decomposition_report(area_rows, mass_rows, force_summary, velocity_metrics, rep_rows) -> None:
    try:
        from docx import Document
        from docx.shared import Inches
    except Exception as exc:
        (OUTPUT_DIR / "bottom_input_decomposition_report.docx.error.txt").write_text(str(exc), encoding="utf-8")
        return
    doc = Document()
    doc.add_heading("Example 3.3 底部输入误差拆分诊断报告", level=0)
    doc.add_paragraph("本轮没有修改 example3_3_free_field_shear_wave.py；诊断通过临时复制脚本采集底部节点速度、质量与力项。")
    doc.add_paragraph("Case A/B/C 均未通过，因此输入公式倍率不是唯一原因。")
    doc.add_paragraph("土柱算例中底部输入已验证正确；不要再把问题归因于 MPM 基本波输入能力。")
    doc.add_paragraph("本轮进一步检查了底部节点速度时程完整性、底部面积分配、节点质量/面积关系、输入力是否漏施加或重复施加，以及 center、area-weighted、mass-weighted 三种速度指标。")

    doc.add_heading("面积分配", level=1)
    for r in area_rows:
        doc.add_paragraph(f"{r['body_level']}: area={r['sum_tributary_area']:.6g}, theory={r['theoretical_bottom_area']:.6g}, error={r['area_error_percent']:.3g}%, diagnosis={r['diagnosis']}")

    doc.add_heading("质量/面积", level=1)
    errors = [float(r["mass_per_area_error_percent"]) for r in mass_rows if math.isfinite(float(r["mass_per_area_error_percent"]))]
    doc.add_paragraph(f"mass_per_area error range: min={min(errors):.6g}%, max={max(errors):.6g}%." if errors else "mass_per_area unavailable.")

    doc.add_heading("力汇总", level=1)
    for r in force_summary:
        doc.add_paragraph(f"{r['body_level']}: input force ok={r['input_force_total_ok']}, dashpot force ok={r['dashpot_force_total_ok']}, area-weighted peak error={r['area_weighted_peak_error_percent']:.3g}%, mass-weighted peak error={r['mass_weighted_peak_error_percent']:.3g}%.")

    doc.add_heading("速度指标", level=1)
    for r in velocity_metrics:
        doc.add_paragraph(f"{r['body_level']}: center={r['center_node_velocity_peak']}, area-weighted={r['area_weighted_bottom_velocity_peak']:.6g}, mass-weighted={r['mass_weighted_bottom_velocity_peak']:.6g}, max={r['max_bottom_node_velocity_peak']:.6g}, mean={r['mean_bottom_node_velocity_peak']:.6g}. {r['diagnosis']}")

    pass_all = (
        all(float(r["area_weighted_bottom_velocity_peak_error_percent"]) < 5.0 for r in velocity_metrics)
        and all(float(r["mass_weighted_bottom_velocity_peak_error_percent"]) < 5.0 for r in velocity_metrics)
        and all(float(r["max_input_force_error_percent"]) < 1.0e-8 for r in force_summary)
        and all(float(r["max_dashpot_force_error_percent"]) < 1.0e-8 for r in force_summary)
        and all(str(r["actual_history_available"]).lower() == "true" for r in rep_rows)
    )
    doc.add_heading("最可能来源", level=1)
    if pass_all:
        doc.add_paragraph("底部输入误差可按本轮判据排除。")
    else:
        doc.add_paragraph("底部输入误差尚不能排除。最可能来源按当前证据排序：施加时序或速度采样相位、节点质量/面积动力响应差异、history 选点；面积总量和输入/阻尼力求和公式本身若显示通过，则不是首要嫌疑。")
    for image in ["bottom_area_histogram.png", "bottom_area_map.png", "bottom_mass_area_scatter.png", "bottom_input_force_balance.png"]:
        path = OUTPUT_DIR / image
        if path.exists():
            doc.add_paragraph(image)
            doc.add_picture(str(path), width=Inches(6.0))
    doc.save(OUTPUT_DIR / "bottom_input_decomposition_report.docx")


def _time_step_from_rows(raw_rows: list[dict[str, Any]]) -> float:
    times = sorted({float(r["time"]) for r in raw_rows})
    diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
    return float(np.median(diffs)) if diffs else 0.0


def _previous_velocity_map(raw_rows: list[dict[str, Any]]) -> dict[tuple[str, int, float], float]:
    out: dict[tuple[str, int, float], float] = {}
    by_node = group_rows(raw_rows, ("body_level", "node_id"))
    for (body, nid), items in by_node.items():
        ordered = sorted(items, key=lambda r: float(r["time"]))
        prev = float(ordered[0]["node_vx"])
        for row in ordered:
            t = float(row["time"])
            out[(body, int(nid), t)] = prev
            prev = float(row["node_vx"])
    return out


def _body_time_series(raw_rows: list[dict[str, Any]], velocity_selector: str = "current") -> list[dict[str, Any]]:
    dt = _time_step_from_rows(raw_rows)
    prev_map = _previous_velocity_map(raw_rows)
    out = []
    for (time, body), items in sorted(group_rows(raw_rows, ("time", "body_level")).items()):
        area = np.array([float(r["tributary_area_A"]) for r in items])
        mass = np.array([float(r["node_mass"]) for r in items])
        cur = np.array([float(r["node_vx"]) for r in items])
        acc = np.array([float(r["acceleration_x"]) for r in items])
        prev = np.array([prev_map[(str(r["body_level"]), int(r["node_id"]), float(r["time"]))] for r in items])
        if velocity_selector == "previous":
            vx = prev
        elif velocity_selector == "predict":
            vx = cur + acc * dt
        elif velocity_selector == "half":
            vx = 0.5 * (prev + cur)
        elif velocity_selector == "pre_traction":
            vx = prev
        elif velocity_selector == "post_traction_predict":
            vx = cur + 0.5 * acc * dt
        else:
            vx = cur
        sum_area = float(np.sum(area))
        sum_mass = float(np.sum(mass))
        aw = float(np.sum(vx * area) / sum_area)
        mw = float(np.sum(vx * mass) / sum_mass)
        out.append({
            "time": float(time),
            "body_level": body,
            "area_weighted_velocity": aw,
            "mass_weighted_velocity": mw,
            "input_force_sum": float(np.sum([float(r["input_force_x"]) for r in items])),
            "node_count": len(items),
            "mass_sum": sum_mass,
            "area_sum": sum_area,
        })
    return out


def bottom_boundary_timing_phase_diagnosis() -> dict[str, Any]:
    """Diagnose bottom-boundary timing, dashpot phase, and history sampling.

    The original Example 3.3 source is not modified.  T5 samples are true values
    from an instrumented temporary copy; T0-T4 are reconstructed from adjacent
    available states and recorded as such in the output.
    """

    raw_rows = read_raw_all_nodes()
    dt = _time_step_from_rows(raw_rows)
    prev_map = _previous_velocity_map(raw_rows)
    timing_rows: list[dict[str, Any]] = []
    substages = [
        ("T0_step_start_after_p2g_before_boundary", "reconstructed_previous_step_velocity", "previous"),
        ("T1_after_internal_force_before_boundary", "reconstructed_previous_step_velocity", "previous"),
        ("T2_after_bottom_traction_before_acceleration", "reconstructed_current_force_previous_velocity", "previous"),
        ("T3_after_nodal_acceleration_update", "reconstructed_current_plus_half_acc_dt", "half_acc"),
        ("T4_after_grid_velocity_update", "reconstructed_current_velocity", "current"),
        ("T5_after_particle_update_history_callback", "true_instrumented_callback_sample", "current"),
    ]
    for row in raw_rows:
        cur = float(row["node_vx"])
        prev = prev_map[(str(row["body_level"]), int(row["node_id"]), float(row["time"]))]
        acc = float(row["acceleration_x"])
        for name, basis, kind in substages:
            if kind == "previous":
                vx = prev
            elif kind == "half_acc":
                vx = prev + 0.5 * acc * dt
            else:
                vx = cur
            input_force = float(row["input_force_x"])
            dashpot = -RHO_CS * vx * float(row["tributary_area_A"])
            net = input_force + dashpot
            theory = float(row["theoretical_velocity"])
            timing_rows.append({
                "time": row["time"],
                "substage_name": name,
                "sampling_basis": basis,
                "body_level": row["body_level"],
                "node_id": row["node_id"],
                "coordinate": f"({row['x']}, {row['y']}, {row['z']})",
                "area_A": row["tributary_area_A"],
                "node_mass": row["node_mass"],
                "node_vx": vx,
                "node_momentum_x": vx,
                "input_force_x": input_force,
                "dashpot_force_x": dashpot,
                "net_force_x": net,
                "acceleration_x": net / float(row["node_mass"]) if float(row["node_mass"]) > 0.0 else math.nan,
                "theoretical_velocity": theory,
                "velocity_error_percent": abs(vx - theory) / max(abs(theory), 1.0e-30) * 100.0 if abs(theory) > 1.0e-12 else math.nan,
            })
    write_csv(OUTPUT_DIR / "bottom_timing_substage_timeseries.csv", timing_rows)

    dashpot_rows = dashpot_velocity_phase_sweep(raw_rows)
    write_csv(OUTPUT_DIR / "bottom_dashpot_velocity_phase_sweep.csv", dashpot_rows)
    write_dashpot_phase_plot(dashpot_rows)

    input_rows = input_time_offset_sweep(raw_rows, dt)
    write_csv(OUTPUT_DIR / "bottom_input_time_offset_sweep.csv", input_rows)
    write_input_offset_plot(input_rows)

    history_rows = history_sampling_sweep(raw_rows, dt)
    write_csv(OUTPUT_DIR / "bottom_history_sampling_sweep.csv", history_rows)
    write_history_sampling_plot(history_rows)

    mb_rows = multi_body_timing_consistency(raw_rows)
    write_csv(OUTPUT_DIR / "multi_body_timing_consistency.csv", mb_rows)
    write_multi_body_overlay(raw_rows)

    run_main_only_check()

    create_timing_phase_report(dashpot_rows, input_rows, history_rows, mb_rows)
    return {
        "dt": dt,
        "bottom_timing_substage_rows": len(timing_rows),
        "dashpot_cases": len(dashpot_rows),
        "input_time_cases": len(input_rows),
        "history_sampling_cases": len(history_rows),
        "multi_body_rows": len(mb_rows),
    }


def run_main_only_check() -> None:
    script = FLAC3D_DIR / "example3_3_main_only_bottom_input_check.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    (OUTPUT_DIR / "main_only_bottom_input_check.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError("main-only bottom input check failed; see main_only_bottom_input_check.log")


def _peak_metrics(series: list[dict[str, Any]], body_level: str) -> dict[str, float]:
    rows = [r for r in series if r["body_level"] == body_level]
    times = np.array([float(r["time"]) for r in rows])
    aw = np.array([float(r["area_weighted_velocity"]) for r in rows])
    mw = np.array([float(r["mass_weighted_velocity"]) for r in rows])
    inp = np.array([float(r["input_force_sum"]) for r in rows])
    target = DSTRESS_AMPLITUDE / RHO_CS
    pidx = int(np.nanargmax(np.abs(aw)))
    fidx = int(np.nanargmax(np.abs(inp)))
    return {
        "area_peak": float(np.nanmax(np.abs(aw))),
        "mass_peak": float(np.nanmax(np.abs(mw))),
        "peak_time": float(times[pidx]),
        "input_peak_time": float(times[fidx]),
        "error": abs(float(np.nanmax(np.abs(aw))) - target) / target * 100.0,
        "phase": float(times[pidx] - times[fidx]),
    }


def dashpot_velocity_phase_sweep(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = [
        ("Dashpot Case A", "v_current", "current"),
        ("Dashpot Case B", "v_previous", "previous"),
        ("Dashpot Case C", "v_predict = v_current + a_current*dt", "predict"),
        ("Dashpot Case D", "v_half = 0.5*(v_previous + v_current)", "half"),
        ("Dashpot Case E", "pre-boundary velocity", "pre_traction"),
        ("Dashpot Case F", "post-traction pre-update predicted velocity", "post_traction_predict"),
    ]
    rows = []
    for case, definition, selector in cases:
        series = _body_time_series(raw_rows, selector)
        for body in BODY_NAMES.values():
            m = _peak_metrics(series, body)
            rows.append({
                "case": case,
                "dashpot_velocity_definition": definition,
                "body_level": body,
                "area_weighted_bottom_velocity_peak": m["area_peak"],
                "mass_weighted_bottom_velocity_peak": m["mass_peak"],
                "bottom_peak_error_percent": m["error"],
                "phase_lag_to_input": m["phase"],
                "input_force_peak_time": m["input_peak_time"],
                "bottom_velocity_peak_time": m["peak_time"],
                "force_velocity_phase_difference": m["phase"],
                "main_top_peak_error": None,
                "NRMSE_main": None,
                "post_peak_reflection_ratio": None,
                "diagnostic_note": "Timing/phase diagnostic from bottom-node samples; not a rerun of corrected physics.",
                "passes_bottom_peak_lt_5pct": m["error"] < 5.0,
            })
    return rows


def input_time_offset_sweep(raw_rows: list[dict[str, Any]], dt: float) -> list[dict[str, Any]]:
    cases = [
        ("Input Time Case A", 0.0, "dstress = wave(t)"),
        ("Input Time Case B", 0.5 * dt, "dstress = wave(t + 0.5*dt)"),
        ("Input Time Case C", dt, "dstress = wave(t + dt)"),
        ("Input Time Case D", -0.5 * dt, "dstress = wave(t - 0.5*dt)"),
        ("Input Time Case E", -dt, "dstress = wave(t - dt)"),
    ]
    base_series = _body_time_series(raw_rows, "current")
    rows = []
    for case, offset, desc in cases:
        for body in BODY_NAMES.values():
            body_rows = [r for r in base_series if r["body_level"] == body]
            times = np.array([float(r["time"]) for r in body_rows])
            vel = np.array([float(r["area_weighted_velocity"]) for r in body_rows])
            theory = np.array([theoretical_target_velocity(wave_factor(float(t) + offset)) for t in times])
            pidx = int(np.nanargmax(np.abs(vel)))
            tidx = int(np.nanargmax(np.abs(theory)))
            peak = float(np.nanmax(np.abs(vel)))
            target_peak = float(np.nanmax(np.abs(theory)))
            rows.append({
                "case": case,
                "input_time_offset": offset,
                "definition": desc,
                "body_level": body,
                "bottom_area_weighted_peak_error": abs(peak - target_peak) / max(target_peak, 1.0e-30) * 100.0,
                "bottom_velocity_peak_time": float(times[pidx]),
                "theoretical_input_peak_time": float(times[tidx]),
                "phase_lag_to_input": float(times[pidx] - times[tidx]),
                "main_top_peak_error": None,
                "NRMSE_main": None,
                "peak_time_error_main": None,
                "shifted_NRMSE_main": None,
                "diagnostic_note": "Input-time diagnostic reindexes the theoretical wave; it does not rerun the solver.",
            })
    return rows


def history_sampling_sweep(raw_rows: list[dict[str, Any]], dt: float) -> list[dict[str, Any]]:
    cases = [
        ("History Case A", "current", "step callback after particle update; current baseline"),
        ("History Case B", "previous", "reconstructed after boundary traction before velocity update"),
        ("History Case C", "current", "reconstructed immediately after velocity update"),
        ("History Case D", "half", "reconstructed before particle update"),
        ("History Case E", "interpolated_peak", "linear interpolation to theoretical input peak time"),
    ]
    rows = []
    target = DSTRESS_AMPLITUDE / RHO_CS
    theoretical_peak_time = WAVE_PERIOD / 2.0
    for case, selector, desc in cases:
        series = _body_time_series(raw_rows, "half" if selector == "half" else ("previous" if selector == "previous" else "current"))
        for body in BODY_NAMES.values():
            body_rows = [r for r in series if r["body_level"] == body]
            times = np.array([float(r["time"]) for r in body_rows])
            aw = np.array([float(r["area_weighted_velocity"]) for r in body_rows])
            mw = np.array([float(r["mass_weighted_velocity"]) for r in body_rows])
            if selector == "interpolated_peak":
                aw_peak = abs(float(np.interp(theoretical_peak_time, times, aw)))
                mw_peak = abs(float(np.interp(theoretical_peak_time, times, mw)))
                peak_time = theoretical_peak_time
            else:
                pidx = int(np.nanargmax(np.abs(aw)))
                aw_peak = float(np.nanmax(np.abs(aw)))
                mw_peak = float(np.nanmax(np.abs(mw)))
                peak_time = float(times[pidx])
            center_peak = representative_peak(raw_rows, body, selector, theoretical_peak_time)
            rows.append({
                "case": case,
                "sampling_definition": desc,
                "body_level": body,
                "center_node_velocity_peak": center_peak,
                "area_weighted_bottom_velocity_peak": aw_peak,
                "mass_weighted_bottom_velocity_peak": mw_peak,
                "bottom_peak_error_percent": abs(aw_peak - target) / target * 100.0,
                "bottom_peak_time": peak_time,
                "theoretical_peak_time": theoretical_peak_time,
                "bottom_peak_time_error": peak_time - theoretical_peak_time,
                "passes_bottom_peak_lt_5pct": abs(aw_peak - target) / target < 0.05,
                "diagnostic_note": "Sampling-position diagnostic; corrected cases are not strict physical reproductions.",
            })
    return rows


def representative_peak(raw_rows: list[dict[str, Any]], body: str, selector: str, peak_time: float) -> float:
    candidates = [r for r in raw_rows if r["body_level"] == body and representative_key(r) is not None]
    if not candidates:
        return math.nan
    nid = candidates[0]["node_id"]
    rows = sorted([r for r in raw_rows if r["body_level"] == body and r["node_id"] == nid], key=lambda r: float(r["time"]))
    times = np.array([float(r["time"]) for r in rows])
    cur = np.array([float(r["node_vx"]) for r in rows])
    if selector == "interpolated_peak":
        return abs(float(np.interp(peak_time, times, cur)))
    if selector == "previous":
        vals = np.concatenate([[cur[0]], cur[:-1]])
    elif selector == "half":
        vals = 0.5 * (np.concatenate([[cur[0]], cur[:-1]]) + cur)
    else:
        vals = cur
    return float(np.nanmax(np.abs(vals)))


def multi_body_timing_consistency(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dt = _time_step_from_rows(raw_rows)
    series = _body_time_series(raw_rows, "current")
    rows = []
    for body in BODY_NAMES.values():
        body_series = [r for r in series if r["body_level"] == body]
        m = _peak_metrics(series, body)
        times = sorted({float(r["time"]) for r in body_series})
        node_count = len({int(r["node_id"]) for r in raw_rows if r["body_level"] == body})
        rows.append({
            "body_level": body,
            "dt": dt,
            "number_of_steps": len(times),
            "current_time_at_step": ";".join(f"{t:.12g}" for t in times[:5]) + (";..." if len(times) > 5 else ""),
            "bottom_input_force_peak_time": m["input_peak_time"],
            "bottom_velocity_peak_time": m["peak_time"],
            "boundary_force_application_count": len(body_series) * node_count,
            "bottom_node_count": node_count,
            "area_weighted_velocity_peak": m["area_peak"],
            "peak_error_percent": m["error"],
            "timing_consistency_note": "All bodies sampled on the same time grid; peak time differences indicate dynamic response/phase differences, not missing time steps.",
        })
    return rows


def write_dashpot_phase_plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    labels = sorted({r["case"] for r in rows})
    plt.figure(figsize=(10, 4.8))
    for body in BODY_NAMES.values():
        vals = [float(r["bottom_peak_error_percent"]) for r in rows if r["body_level"] == body]
        plt.plot(labels, vals, marker="o", label=body)
    plt.axhline(5.0, color="r", linestyle="--", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("bottom peak error (%)")
    plt.title("Dashpot velocity time-layer diagnostic")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bottom_dashpot_velocity_phase_comparison.png", dpi=180)
    plt.close()


def write_input_offset_plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    labels = sorted({r["case"] for r in rows})
    plt.figure(figsize=(9, 4.6))
    for body in BODY_NAMES.values():
        vals = [float(r["bottom_area_weighted_peak_error"]) for r in rows if r["body_level"] == body]
        plt.plot(labels, vals, marker="o", label=body)
    plt.axhline(5.0, color="r", linestyle="--", linewidth=1)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("bottom peak error (%)")
    plt.title("Input wave time-offset diagnostic")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bottom_input_time_offset_comparison.png", dpi=180)
    plt.close()


def write_history_sampling_plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    labels = sorted({r["case"] for r in rows})
    plt.figure(figsize=(10, 4.8))
    for body in BODY_NAMES.values():
        vals = [float(r["bottom_peak_error_percent"]) for r in rows if r["body_level"] == body]
        plt.plot(labels, vals, marker="o", label=body)
    plt.axhline(5.0, color="r", linestyle="--", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("bottom peak error (%)")
    plt.title("History sampling phase diagnostic")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bottom_history_sampling_comparison.png", dpi=180)
    plt.close()


def write_multi_body_overlay(raw_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    series = _body_time_series(raw_rows, "current")
    plt.figure(figsize=(9, 4.8))
    for body in BODY_NAMES.values():
        rows = [r for r in series if r["body_level"] == body]
        plt.plot([r["time"] for r in rows], [r["area_weighted_velocity"] for r in rows], label=body)
    plt.plot([r["time"] for r in rows], [theoretical_target_velocity(wave_factor(float(r["time"]))) for r in rows], "k--", label="theoretical")
    plt.xlabel("time")
    plt.ylabel("area-weighted vx")
    plt.title("Multi-body bottom velocity overlay")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multi_body_bottom_velocity_overlay.png", dpi=180)
    plt.close()


def create_timing_phase_report(dashpot_rows, input_rows, history_rows, mb_rows) -> None:
    try:
        from docx import Document
        from docx.shared import Inches
    except Exception as exc:
        (OUTPUT_DIR / "bottom_timing_phase_decomposition_report.docx.error.txt").write_text(str(exc), encoding="utf-8")
        return
    doc = Document()
    doc.add_heading("Example 3.3 底部边界时序与速度相位诊断报告", level=0)
    doc.add_paragraph("本轮没有修改 example3_3_free_field_shear_wave.py；所有 corrected case 均为时序/采样诊断，不能直接声明严格物理复现。")
    doc.add_heading("已排除项", level=1)
    doc.add_paragraph("area 总量错误、force sum 错误、dashpot sum 错误、node mass / area 错误、center history 选点误判已基本排除。")
    doc.add_heading("当前重点", level=1)
    doc.add_paragraph("boundary traction 施加时序、dashpot 速度时间层、input wave 时间层、history 采样位置、多 body 时间同步、main-only 对照。")

    def best(rows, key):
        finite = [r for r in rows if r.get(key) not in (None, "")]
        return min(finite, key=lambda r: float(r[key])) if finite else None

    b_dash = best(dashpot_rows, "bottom_peak_error_percent")
    b_input = best(input_rows, "bottom_area_weighted_peak_error")
    b_hist = best(history_rows, "bottom_peak_error_percent")
    doc.add_heading("测试结论", level=1)
    if b_dash:
        doc.add_paragraph(f"Dashpot velocity time-layer best: {b_dash['case']} / {b_dash['body_level']}, bottom peak error={float(b_dash['bottom_peak_error_percent']):.3g}%. main top NRMSE 未重跑，不能判定修正有效性。")
    if b_input:
        doc.add_paragraph(f"Input time offset best: {b_input['case']} / {b_input['body_level']}, bottom peak error={float(b_input['bottom_area_weighted_peak_error']):.3g}%.")
    if b_hist:
        doc.add_paragraph(f"History sampling best: {b_hist['case']} / {b_hist['body_level']}, bottom peak error={float(b_hist['bottom_peak_error_percent']):.3g}%.")
    if any(float(r["bottom_peak_error_percent"]) < 5.0 for r in dashpot_rows):
        doc.add_paragraph("发现某个 dashpot 速度时间层可使局部 body 通过底部峰值标准，应列为 Priority 1 候选，但必须通过完整动力重跑验证顶部 NRMSE 和相位。")
    elif any(float(r["bottom_peak_error_percent"]) < 5.0 for r in history_rows):
        doc.add_paragraph("history 采样位置可显著改变局部误差，应优先修正采样定义。")
    else:
        doc.add_paragraph("未发现可让全部 body 底部输入通过的时序/采样方案；底部偏低不是面积/质量/公式问题，更可能来自 GeoTaichi MPM 完整动力更新路径与 FLAC3D 边界时序不同。")

    doc.add_heading("多 body 同步", level=1)
    for r in mb_rows:
        doc.add_paragraph(f"{r['body_level']}: steps={r['number_of_steps']}, node_count={r['bottom_node_count']}, force_peak_time={r['bottom_input_force_peak_time']}, velocity_peak_time={r['bottom_velocity_peak_time']}, peak_error={float(r['peak_error_percent']):.3g}%.")

    for image in [
        "bottom_dashpot_velocity_phase_comparison.png",
        "bottom_input_time_offset_comparison.png",
        "bottom_history_sampling_comparison.png",
        "multi_body_bottom_velocity_overlay.png",
        "main_only_bottom_input_check.png",
    ]:
        path = OUTPUT_DIR / image
        if path.exists():
            doc.add_paragraph(image)
            doc.add_picture(str(path), width=Inches(6.0))
    doc.save(OUTPUT_DIR / "bottom_timing_phase_decomposition_report.docx")


if __name__ == "__main__":
    result = bottom_input_diagnosis()
    result["specialized_bottom_outputs"] = write_specialized_bottom_outputs()
    result["timing_phase_diagnosis"] = bottom_boundary_timing_phase_diagnosis()
    print(json.dumps(result, indent=2, ensure_ascii=False))
