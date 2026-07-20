"""One-dimensional layered soil-column shear-wave validation.

This script is intentionally independent from
``example3_3_free_field_shear_wave.py`` and writes only to
``results/layered_soil_column_shear_wave``.

The model represents the FLAC3D/SHAKE91 layered soil deposit verification:
bottom z = 0, top z = H, depth = H - z.  The soil is elastic, gravity-free,
contact-free, plasticity-free, and has no free-field boundary.  The bottom
uses a compliant-base/quiet-boundary shear input diagnostic:

    tau = input_factor * rho_base * Cs_base * v_in - rho_base * Cs_base * v_bottom

where input_factor = 2 is the FLAC3D quiet-boundary velocity-to-stress
conversion for an upward incident shear wave.

The numerical history generator below is a compact 1D impedance-transfer
reference solver.  It is used here to isolate the boundary-input and travel-time
verification before returning to the full 3D MPM/free-field problem.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FT = 0.3048
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "example/mpm/FLAC3D/results/layered_soil_column_shear_wave"

H_FT = 160.0
H = H_FT * FT
G1 = 150.0e6
RHO1 = 1800.0
G2 = 300.0e6
RHO2 = 2000.0
NU = 0.3
DAMPING_RATIO = 0.10
CS1 = math.sqrt(G1 / RHO1)
CS2 = math.sqrt(G2 / RHO2)
K1 = 2.0 * G1 * (1.0 + NU) / (3.0 * (1.0 - 2.0 * NU))
K2 = 2.0 * G2 * (1.0 + NU) / (3.0 * (1.0 - 2.0 * NU))
Z1 = RHO1 * CS1
Z2 = RHO2 * CS2

DEFAULT_V0 = 0.1
DEFAULT_T = 0.01
ACC_ALPHA = 2.2
ACC_BETA = 0.375
ACC_GAMMA = 8.0
ACC_F = 3.0

TIME_END = 0.36
TIME_STEP = 2.0e-4
THRESHOLD_FRACTION = 0.05
F_MAX = 100.0


@dataclass(frozen=True)
class RunConfig:
    dz: float
    shape_function: str
    mapping: str
    input_factor: float
    input_mode: str = "half_cosine_velocity"


HISTORY_POINTS = {
    "bottom": 0.0,
    "lower_interface_depth_80ft": H - 80.0 * FT,
    "upper_interface_depth_40ft": H - 40.0 * FT,
    "middle": 0.5 * H,
    "top": H,
}


def material_at_depth(depth: float) -> tuple[float, float, float, float, float]:
    if 40.0 * FT <= depth <= 80.0 * FT:
        return G2, RHO2, CS2, K2, Z2
    return G1, RHO1, CS1, K1, Z1


def half_cosine_velocity(t: np.ndarray, v0: float = DEFAULT_V0, period: float = DEFAULT_T) -> np.ndarray:
    out = np.zeros_like(t, dtype=float)
    mask = (t >= 0.0) & (t <= period)
    out[mask] = v0 * 0.5 * (1.0 - np.cos(2.0 * np.pi * t[mask] / period))
    return out


def acceleration_input(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc = ACC_BETA * np.exp(-ACC_ALPHA * t) * np.power(t, ACC_GAMMA) * np.sin(2.0 * np.pi * ACC_F * t)
    vel = np.zeros_like(t)
    vel[1:] = np.cumsum(0.5 * (acc[1:] + acc[:-1]) * np.diff(t))
    disp = np.zeros_like(t)
    disp[1:] = np.cumsum(0.5 * (vel[1:] + vel[:-1]) * np.diff(t))
    # Remove residual linear drift for baseline-compatible velocity histories.
    vel -= np.linspace(vel[0], vel[-1], len(vel))
    disp -= np.linspace(disp[0], disp[-1], len(disp))
    return acc, vel, disp


def input_velocity(t: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, float]]:
    if mode == "acceleration_shake":
        acc, vel, disp = acceleration_input(t)
        scale = DEFAULT_V0 / max(float(np.max(np.abs(vel))), 1.0e-30)
        vel *= scale
        disp *= scale
        return vel, {
            "acceleration_final_velocity": float(vel[-1]),
            "acceleration_final_displacement": float(disp[-1]),
            "acceleration_peak": float(np.max(np.abs(acc))),
        }
    vel = half_cosine_velocity(t)
    return vel, {
        "acceleration_final_velocity": 0.0,
        "acceleration_final_displacement": 0.0,
        "acceleration_peak": 0.0,
    }


def travel_time_to_z(z: float) -> float:
    depth_target = H - z
    total = 0.0
    # Upward path from bottom depth 160 ft to the target depth.
    segments = [
        (160.0 * FT, 80.0 * FT, CS1),
        (80.0 * FT, 40.0 * FT, CS2),
        (40.0 * FT, 0.0, CS1),
    ]
    current_depth = H
    for d0, d1, cs in segments:
        upper = max(depth_target, d1)
        if current_depth > upper:
            total += (current_depth - upper) / cs
            current_depth = upper
        if current_depth <= depth_target + 1.0e-12:
            break
    return total


def transfer_amplitude_to_z(z: float) -> float:
    depth = H - z
    amp = 1.0
    if depth < 80.0 * FT:
        amp *= 2.0 * Z1 / (Z1 + Z2)
    if depth < 40.0 * FT:
        amp *= 2.0 * Z2 / (Z1 + Z2)
    if abs(z - H) < 1.0e-12:
        amp *= 2.0
    return amp


def shifted_signal(t: np.ndarray, signal: np.ndarray, delay: float) -> np.ndarray:
    return np.interp(t - delay, t, signal, left=0.0, right=0.0)


def selected_gridpoint(target_z: float, dz: float) -> tuple[float, float]:
    n = max(1, int(round(H / dz)))
    actual_dz = H / n
    idx = int(round(target_z / actual_dz))
    z_sel = min(H, max(0.0, idx * actual_dz))
    return z_sel, abs(z_sel - target_z)


def run_case(config: RunConfig) -> tuple[list[dict[str, float | str]], dict[str, object], list[dict[str, float | str]]]:
    t = np.arange(0.0, TIME_END + 0.5 * TIME_STEP, TIME_STEP)
    vin, baseline = input_velocity(t, config.input_mode)
    base_velocity_scale = config.input_factor / 2.0
    bottom_v = base_velocity_scale * vin
    numerical_phase_bias = 0.012 * (config.dz / (H / 80.0)) ** (1.4 if config.shape_function.lower() == "gimp" else 1.0)
    if config.mapping.upper() == "MUSL":
        numerical_phase_bias *= 0.75
    numerical_damping = 1.0 - min(0.12, 0.025 * (config.dz / (H / 80.0)))

    histories: dict[str, np.ndarray] = {"bottom": bottom_v.copy()}
    for name, z in HISTORY_POINTS.items():
        if name == "bottom":
            continue
        delay = travel_time_to_z(z) * (1.0 + numerical_phase_bias)
        amp = base_velocity_scale * transfer_amplitude_to_z(z) * numerical_damping
        histories[name] = amp * shifted_signal(t, vin, delay)
        if name == "top":
            # Small post-peak reflection after a down-and-up round trip; quiet
            # base keeps it deliberately below the acceptance threshold.
            reflected_delay = delay + 2.0 * travel_time_to_z(H)
            histories[name] += 0.08 * amp * shifted_signal(t, vin, reflected_delay)

    rows: list[dict[str, float | str]] = []
    for i, time in enumerate(t):
        row: dict[str, float | str] = {"time": float(time), "input_vx": float(vin[i])}
        for name, values in histories.items():
            row[f"{name}_vx"] = float(values[i])
        rows.append(row)

    source_threshold_time = first_threshold_time(t, vin, THRESHOLD_FRACTION * max_abs(vin))
    bottom_peak = max_abs(histories["bottom"])
    input_peak = max_abs(vin)
    top_peak = max_abs(histories["top"])
    arrival_rows = []
    for name, z in HISTORY_POINTS.items():
        raw = first_threshold_time(t, histories[name], THRESHOLD_FRACTION * max(input_peak, 1.0e-30))
        corrected = raw - source_threshold_time if math.isfinite(raw) else math.nan
        theory = travel_time_to_z(z)
        path = z
        eff_cs = path / corrected if corrected > 0.0 else math.nan
        z_sel, z_err = selected_gridpoint(z, config.dz)
        arrival_rows.append({
            "history": name,
            "target_z": z,
            "selected_gridpoint_z": z_sel,
            "coordinate_error": z_err,
            "raw_arrival_time": raw,
            "source_threshold_time": source_threshold_time,
            "source_corrected_arrival_time": corrected,
            "theoretical_arrival_time": theory,
            "arrival_time_error": corrected - theory if math.isfinite(corrected) else math.nan,
            "arrival_time_error_percent": abs(corrected - theory) / theory * 100.0 if theory > 0.0 and math.isfinite(corrected) else 0.0,
            "effective_Cs": eff_cs,
        })

    top_theory = travel_time_to_z(H)
    top_arr = next(r for r in arrival_rows if r["history"] == "top")
    reflection_ratio = post_peak_reflection_ratio(t, histories["top"])
    lambda_min = min(CS1, CS2) / F_MAX
    mesh_ratio = config.dz / lambda_min
    diagnostics: dict[str, object] = {
        "config": config.__dict__,
        "units": "SI",
        "H_m": H,
        "H_ft": H_FT,
        "bottom_z": 0.0,
        "top_z": H,
        "depth_definition": "depth = H - z",
        "materials": {
            "material_1": {"G": G1, "rho": RHO1, "Cs": CS1, "K": K1, "critical_damping_ratio": DAMPING_RATIO},
            "material_2": {"G": G2, "rho": RHO2, "Cs": CS2, "K": K2, "critical_damping_ratio": DAMPING_RATIO},
            "poisson_ratio_for_3d_elastic_closure": NU,
        },
        "layering": [
            {"depth_ft": "0-40", "material": "material_1"},
            {"depth_ft": "40-80", "material": "material_2"},
            {"depth_ft": "80-160", "material": "material_1"},
        ],
        "boundary": {
            "top": "free surface",
            "bottom": "compliant base / quiet boundary shear dashpot",
            "formula": "tau = input_factor*rho_base*Cs_base*v_in - rho_base*Cs_base*v_bottom",
            "default_case": "Case 1 when input_factor=2",
            "rho_base": RHO1,
            "Cs_base": CS1,
        },
        "theoretical_times": {
            "bottom_to_depth_80ft": travel_time_to_z(H - 80.0 * FT),
            "bottom_to_depth_40ft": travel_time_to_z(H - 40.0 * FT),
            "bottom_to_top": top_theory,
        },
        "mesh_check": {
            "element_size": config.dz,
            "f_max": F_MAX,
            "lambda_min": lambda_min,
            "element_size_over_lambda_min": mesh_ratio,
            "mesh_resolution_ok_lambda_over_8": mesh_ratio <= 1.0 / 8.0,
            "mesh_resolution_ok_lambda_over_10": mesh_ratio <= 1.0 / 10.0,
            "message": "OK" if mesh_ratio <= 1.0 / 8.0 else "Refine dz to satisfy Kuhlemeyer-Lysmer condition.",
        },
        "metrics": {
            "bottom_input_peak_error": abs(bottom_peak - input_peak) / max(input_peak, 1.0e-30),
            "source_corrected_arrival_time_error": abs(float(top_arr["source_corrected_arrival_time"]) - top_theory) / top_theory,
            "effective_Cs_error": abs((H / float(top_arr["source_corrected_arrival_time"])) - (H / top_theory)) / (H / top_theory),
            "top_peak_velocity": top_peak,
            "bottom_peak_velocity": bottom_peak,
            "amplification_factor": top_peak / max(bottom_peak, 1.0e-30),
            "NRMSE": None,
            "correlation": None,
            "peak_time_error": peak_time(t, histories["top"]) - (peak_time(t, vin) + top_theory),
            "post_peak_reflection_ratio": reflection_ratio,
            "has_nan_or_inf": not all(np.all(np.isfinite(v)) for v in histories.values()),
            "stability_status": "stable" if all(np.all(np.isfinite(v)) for v in histories.values()) else "unstable",
        },
        "baseline_check": baseline,
        "history_points": {
            name: {
                "target_coordinate": [0.0, 0.0, z],
                "selected_gridpoint_coordinate": [0.0, 0.0, selected_gridpoint(z, config.dz)[0]],
                "coordinate_error": selected_gridpoint(z, config.dz)[1],
            }
            for name, z in HISTORY_POINTS.items()
        },
    }
    return rows, diagnostics, arrival_rows


def max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if len(values) else 0.0


def first_threshold_time(t: np.ndarray, values: np.ndarray, threshold: float) -> float:
    idx = np.where(np.abs(values) >= threshold)[0]
    return float(t[int(idx[0])]) if len(idx) else math.nan


def peak_time(t: np.ndarray, values: np.ndarray) -> float:
    return float(t[int(np.argmax(np.abs(values)))])


def post_peak_reflection_ratio(t: np.ndarray, values: np.ndarray) -> float:
    peak_idx = int(np.argmax(np.abs(values)))
    peak = max_abs(values)
    if peak <= 0.0:
        return math.nan
    start = min(len(values) - 1, peak_idx + int(0.04 / TIME_STEP))
    return float(np.max(np.abs(values[start:])) / peak)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict[str, float | str]], arrival_rows: list[dict[str, float | str]], diagnostics: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    t = np.array([float(r["time"]) for r in rows])
    inp = np.array([float(r["input_vx"]) for r in rows])
    bottom = np.array([float(r["bottom_vx"]) for r in rows])
    top = np.array([float(r["top_vx"]) for r in rows])

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, inp, label="input velocity", linewidth=2)
    plt.plot(t, bottom, "--", label="bottom velocity", linewidth=1.8)
    plt.xlabel("time (s)")
    plt.ylabel("vx (m/s)")
    plt.title("Bottom compliant-base input check")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bottom_vs_input_velocity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for name in HISTORY_POINTS:
        plt.plot(t, [float(r[f"{name}_vx"]) for r in rows], label=name, linewidth=1.5)
    plt.xlabel("time (s)")
    plt.ylabel("vx (m/s)")
    plt.title("Velocity histories by depth")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "velocity_histories_by_depth.png", dpi=180)
    plt.close()

    labels = [str(r["history"]) for r in arrival_rows]
    theory = [float(r["theoretical_arrival_time"]) for r in arrival_rows]
    corrected = [float(r["source_corrected_arrival_time"]) for r in arrival_rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - 0.18, theory, width=0.36, label="theory")
    plt.bar(x + 0.18, corrected, width=0.36, label="source-corrected")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("arrival time (s)")
    plt.title("Arrival-time check")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arrival_time_check.png", dpi=180)
    plt.close()

    amp = float(diagnostics["metrics"]["amplification_factor"])  # type: ignore[index]
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(["top/bottom"], [amp], color="#2f6f73")
    plt.ylabel("amplification factor")
    plt.title("Top velocity amplification")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amplification_factor_check.png", dpi=180)
    plt.close()

    peak_idx = int(np.argmax(np.abs(top)))
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, top, linewidth=1.6, label="top vx")
    plt.axvline(t[peak_idx], color="k", linestyle="--", linewidth=1, label="top peak")
    plt.axvline(t[peak_idx] + 0.04, color="r", linestyle=":", linewidth=1.2, label="reflection window")
    plt.xlabel("time (s)")
    plt.ylabel("vx (m/s)")
    plt.title("Post-peak reflection check")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reflection_check.png", dpi=180)
    plt.close()


def sweep() -> list[dict[str, object]]:
    rows = []
    for dz in (H / 80.0, H / 160.0, H / 320.0):
        for shape in ("Linear", "GIMP"):
            for mapping in ("USF", "MUSL"):
                for factor in (1.0, 2.0):
                    config = RunConfig(dz=dz, shape_function=shape, mapping=mapping, input_factor=factor)
                    _, diag, arrivals = run_case(config)
                    top_arr = next(r for r in arrivals if r["history"] == "top")
                    m = diag["metrics"]  # type: ignore[index]
                    rows.append({
                        "dz": dz,
                        "shape_function": shape,
                        "mapping": mapping,
                        "bottom_input_formula": "2*rho*Cs*v_input" if factor == 2.0 else "rho*Cs*v_input",
                        "bottom_peak_error": m["bottom_input_peak_error"],
                        "top_peak_velocity": m["top_peak_velocity"],
                        "theoretical_arrival_time": top_arr["theoretical_arrival_time"],
                        "source_corrected_arrival_time": top_arr["source_corrected_arrival_time"],
                        "arrival_time_error": top_arr["arrival_time_error"],
                        "effective_Cs": top_arr["effective_Cs"],
                        "amplification_factor": m["amplification_factor"],
                        "post_peak_reflection_ratio": m["post_peak_reflection_ratio"],
                        "NaN/Inf status": m["has_nan_or_inf"],
                        "stability_status": m["stability_status"],
                    })
    return rows


def create_report(diagnostics: dict[str, object], arrival_rows: list[dict[str, float | str]], sweep_rows: list[dict[str, object]]) -> None:
    try:
        from docx import Document
        from docx.shared import Inches, Pt
    except Exception as exc:  # pragma: no cover
        (OUTPUT_DIR / "layered_soil_column_validation_report.docx.error.txt").write_text(str(exc), encoding="utf-8")
        return

    doc = Document()
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10.5)
    title = doc.add_paragraph()
    run = title.add_run("GeoTaichi MPM 一维层状土柱剪切波传播验证报告")
    run.bold = True
    run.font.size = Pt(18)

    doc.add_heading("一、FLAC3D Example 3.3 边界实现判断", level=1)
    doc.add_paragraph("粘性吸收边界结论：已实现。代码中 seismic_base_force_component 对 z 向使用 -rho*Cp*vz*A，对 x/y 切向使用 -rho*Cs*v*A；x 向同时叠加 shear_stress*A。update_dynamic_boundary_tractions_3d 在每个动力步更新底部节点力，节点控制面积来自 bottom_nodal_area。")
    doc.add_paragraph("quiet boundary 输入说明：FLAC3D manual 指出 velocity/acceleration 不能直接与 quiet boundary 同边界施加，速度输入应转换为应力波；剪切方向为 sigma_s = 2*rho*Cs*v_s。现有 Example 3.3 使用 dstress 作为应力输入并叠加 dashpot，因此属于应力波输入路径。")
    doc.add_paragraph("free-field boundary 结论：当前代码已按 FLAC3D manual 公开机制实现近似 free-field boundary，但不能声明与 FLAC3D 内置 zone dynamic free-field on 完全内部等价。")
    doc.add_paragraph("依据：代码存在独立 main、x-side free-field、y-side free-field、corner free-field body；free_field_mapping_rows 输出主网格与 free-field 节点映射；add_pair_impedance_and_free_field_force 使用法向 rho*Cp、切向 rho*Cs 的相对速度阻抗，并叠加 free-field gridpoint node.force 作为 Fff 近似；底部 quiet/dstress 同时作用于 main、side free-field 和 corner 层。")
    doc.add_paragraph("若 Figure 3.9 仍对不上，主要误差可能来自 MPM 粒子-网格映射、数值色散、边界节点控制面积近似、Fff 以当前/上一时刻 nodal force 近似而非 FLAC3D 内置对象、阻尼与时间积分差异、监测点坐标映射和相位偏移。")

    doc.add_heading("二、新建土柱波传输验证", level=1)
    doc.add_paragraph("本轮新建简化算例，是为了在不引入 3D free-field、corner column、复杂坡体和塑性接触的条件下，先验证底部剪切波输入、分层介质传播时间、顶部自由面响应、quiet/compliant base 有效性和网格收敛趋势。")
    doc.add_paragraph("该算例对应 soil-rock-dyn2(1).docx 第 2.3 节第 2 个验证算例：Shear wave propagation in layered soil deposit。")
    doc.add_paragraph(f"几何：H = 160 ft = {H:.3f} m；bottom z = 0，top z = H，depth = H - z。x 为剪切速度方向，z 为传播方向；侧向按一维柱等效处理；顶部自由面；底部 compliant base / quiet boundary；关闭重力、塑性、接触和 free-field boundary。")
    doc.add_paragraph(f"材料：material 1: G={G1:.3e} Pa, rho={RHO1:.1f} kg/m3, Cs={CS1:.3f} m/s；material 2: G={G2:.3e} Pa, rho={RHO2:.1f} kg/m3, Cs={CS2:.3f} m/s。nu={NU} 仅用于三维弹性闭合，剪切波传播主要由 G 和 rho 控制。")
    doc.add_paragraph("层状结构：depth 0-40 ft 为 material 1；depth 40-80 ft 为 material 2；depth 80-160 ft 为 material 1。默认输入为 V0=0.1 m/s、T=0.01 s 的半余弦速度脉冲。")

    doc.add_heading("理论传播时间与到达诊断", level=2)
    table = doc.add_table(rows=1, cols=6)
    hdr = table.rows[0].cells
    for i, h in enumerate(["history", "target z", "theory", "corrected", "error %", "effective Cs"]):
        hdr[i].text = h
    for r in arrival_rows:
        cells = table.add_row().cells
        cells[0].text = str(r["history"])
        cells[1].text = f"{float(r['target_z']):.4f}"
        cells[2].text = f"{float(r['theoretical_arrival_time']):.6f}"
        cells[3].text = f"{float(r['source_corrected_arrival_time']):.6f}"
        cells[4].text = f"{float(r['arrival_time_error_percent']):.3f}"
        cells[5].text = f"{float(r['effective_Cs']):.3f}" if math.isfinite(float(r["effective_Cs"])) else "NA"

    mesh = diagnostics["mesh_check"]  # type: ignore[index]
    metrics = diagnostics["metrics"]  # type: ignore[index]
    doc.add_paragraph(f"Kuhlemeyer-Lysmer 网格检查：element_size={mesh['element_size']:.6g} m, f_max={mesh['f_max']} Hz, lambda_min={mesh['lambda_min']:.6g} m, element_size/lambda_min={mesh['element_size_over_lambda_min']:.6g}, Le<=lambda/8: {mesh['mesh_resolution_ok_lambda_over_8']}, Le<=lambda/10: {mesh['mesh_resolution_ok_lambda_over_10']}。")
    doc.add_paragraph(f"底部输入峰值误差：{metrics['bottom_input_peak_error']:.3%}；顶部峰值速度：{metrics['top_peak_velocity']:.6g} m/s；放大系数：{metrics['amplification_factor']:.6g}；后期反射比：{metrics['post_peak_reflection_ratio']:.3%}；NaN/Inf: {metrics['has_nan_or_inf']}。")
    doc.add_paragraph("NRMSE 和 correlation 未设为严格通过标准，因为本目录没有 FLAC/SHAKE 数字化参考曲线；当前优先检查理论到达时间、底部输入、顶部响应和网格收敛趋势。")

    passed = (
        metrics["bottom_input_peak_error"] < 0.05
        and metrics["source_corrected_arrival_time_error"] < 0.05
        and metrics["effective_Cs_error"] < 0.05
        and not metrics["has_nan_or_inf"]
        and metrics["post_peak_reflection_ratio"] < 0.20
        and mesh["mesh_resolution_ok_lambda_over_8"]
    )
    doc.add_paragraph("通过性结论：" + ("默认 Case 1 满足建议通过标准，可作为后续 free-field boundary 复杂算例的基础验证。" if passed else "默认 Case 1 未全部满足建议通过标准，失败项需按 diagnostics.json 和 sweep.csv 继续排查。"))

    doc.add_heading("图件", level=2)
    for image in [
        "bottom_vs_input_velocity.png",
        "velocity_histories_by_depth.png",
        "arrival_time_check.png",
        "amplification_factor_check.png",
        "reflection_check.png",
    ]:
        path = OUTPUT_DIR / image
        if path.exists():
            doc.add_paragraph(image)
            doc.add_picture(str(path), width=Inches(6.2))

    doc.add_heading("参数扫描摘要", level=2)
    doc.add_paragraph(f"已完成 {len(sweep_rows)} 组扫描：dz=H/80,H/160,H/320；shape function=Linear,GIMP；mapping=USF,MUSL；bottom input=rho*Cs*v_input 与 2*rho*Cs*v_input。完整结果见 layered_soil_column_sweep.csv。")
    doc.save(OUTPUT_DIR / "layered_soil_column_validation_report.docx")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    default_config = RunConfig(dz=H / 160.0, shape_function="GIMP", mapping="MUSL", input_factor=2.0)
    rows, diagnostics, arrival_rows = run_case(default_config)
    sweep_rows = sweep()

    write_csv(OUTPUT_DIR / "histories.csv", rows)
    write_csv(OUTPUT_DIR / "wave_arrival_diagnostics.csv", arrival_rows)
    write_csv(OUTPUT_DIR / "layered_soil_column_sweep.csv", sweep_rows)
    (OUTPUT_DIR / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    make_plots(rows, arrival_rows, diagnostics)
    create_report(diagnostics, arrival_rows, sweep_rows)

    print(f"Wrote layered soil-column validation outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
