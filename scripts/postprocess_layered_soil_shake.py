"""Postprocess FLAC3D/SHAKE layered soil MPM verification outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "layered_soil_shake"
FIG = OUT / "figures"
REF = ROOT / "data" / "reference" / "flac3d_manual_layered_soil"
G0 = 9.80665
MANUAL_TOP_FLAC3D_PEAK_G = 0.160
MANUAL_TOP_SHAKE91_PEAK_G = 0.156


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except (TypeError, ValueError):
                cols[k].append(math.nan)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def peak_abs(a: np.ndarray) -> float:
    finite = a[np.isfinite(a)]
    return float(np.max(np.abs(finite))) if finite.size else math.nan


def compare_metrics(cur_t: np.ndarray, cur_y: np.ndarray, ref_t: np.ndarray, ref_y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(ref_t) & np.isfinite(ref_y)
    ref_t = ref_t[mask]
    ref_y = ref_y[mask]
    if ref_t.size == 0:
        return {"peak_reference": math.nan, "peak_error_percent": math.nan, "NRMSE": math.nan, "correlation": math.nan}
    interp = np.interp(ref_t, cur_t, cur_y)
    peak_ref = peak_abs(ref_y)
    peak_cur = peak_abs(interp)
    span = float(np.max(ref_y) - np.min(ref_y))
    rmse = float(np.sqrt(np.mean((interp - ref_y) ** 2)))
    corr = float(np.corrcoef(interp, ref_y)[0, 1]) if np.std(interp) > 0.0 and np.std(ref_y) > 0.0 else math.nan
    return {
        "peak_reference": peak_ref,
        "peak_error_percent": 100.0 * (peak_cur - peak_ref) / peak_ref if peak_ref else math.nan,
        "NRMSE": rmse / span if span > 0.0 else math.nan,
        "correlation": corr,
    }


def maybe_ref(filename: str, time_col: str, value_col: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = REF / filename
    if not path.exists():
        return None, None
    data = read_csv(path)
    if time_col not in data or value_col not in data:
        return None, None
    return data[time_col], data[value_col]


def plot_line(path: Path, title: str, xlabel: str, ylabel: str, t: np.ndarray, y: np.ndarray, label: str, ref: tuple[np.ndarray | None, np.ndarray | None] = (None, None), ylim: tuple[float, float] | None = None) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.5, 5.2), dpi=180)
    plt.plot(t, y, color="#155a8a", linewidth=1.5, label=label)
    rt, ry = ref
    if rt is not None and ry is not None:
        plt.plot(rt, ry, "k--", linewidth=1.3, label="FLAC3D/SHAKE91 reference")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.xlim(float(np.nanmin(t)), float(np.nanmax(t)))
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fmt(v: float, digits: int = 6) -> str:
    return "NA" if not math.isfinite(float(v)) else f"{float(v):.{digits}g}"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    top = read_csv(OUT / "history_top_acceleration.csv")
    strain = read_csv(OUT / "history_shear_strain_35ft.csv")
    stress = read_csv(OUT / "history_shear_stress_35ft.csv")
    inp = read_csv(OUT / "input_motion.csv")
    meta_path = OUT / "model_metadata.json"
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    refs = {
        "top": maybe_ref("reference_top_acceleration.csv", "time_s", "ax_g"),
        "strain": maybe_ref("reference_shear_strain_35ft.csv", "time_s", "gamma_xy_percent"),
        "stress": maybe_ref("reference_shear_stress_35ft.csv", "time_s", "tau_xy_kpa"),
    }

    plot_line(
        FIG / "fig_3_49_input_acceleration.png",
        "Input acceleration at bottom of model",
        "Time (s)",
        "Acceleration (g)",
        inp["time_s"],
        inp["input_acc_g"],
        "MPM input",
    )
    plot_line(
        FIG / "fig_3_51_top_acceleration_compare.png",
        "Horizontal acceleration at top of model",
        "Time (s)",
        "Acceleration (g)",
        top["time_s"],
        top["ax_g"],
        "GeoTaichi MPM",
        refs["top"],
    )
    plot_line(
        FIG / "fig_3_52_shear_strain_35ft_compare.png",
        "Shear strain at 35 ft depth",
        "Time (s)",
        "Shear strain (%)",
        strain["time_s"],
        strain["gamma_xy_percent"],
        "GeoTaichi MPM",
        refs["strain"],
    )
    plot_line(
        FIG / "fig_3_53_shear_stress_35ft_compare.png",
        "Shear stress at 35 ft depth",
        "Time (s)",
        "Shear stress (kPa)",
        stress["time_s"],
        stress["tau_xy_kpa"],
        "GeoTaichi MPM",
        refs["stress"],
        ylim=(-30.0, 30.0),
    )

    m_top = compare_metrics(top["time_s"], top["ax_g"], refs["top"][0], refs["top"][1]) if refs["top"][0] is not None else {}
    m_strain = compare_metrics(strain["time_s"], strain["gamma_xy_percent"], refs["strain"][0], refs["strain"][1]) if refs["strain"][0] is not None else {}
    m_stress = compare_metrics(stress["time_s"], stress["tau_xy_kpa"], refs["stress"][0], refs["stress"][1]) if refs["stress"][0] is not None else {}

    top_peak = peak_abs(top["ax_g"])
    strain_peak = peak_abs(strain["gamma_xy_percent"])
    stress_peak = peak_abs(stress["tau_xy_kpa"])
    top_manual_error = 100.0 * (top_peak - MANUAL_TOP_FLAC3D_PEAK_G) / MANUAL_TOP_FLAC3D_PEAK_G
    top_status = "better than 5%" if abs(top_manual_error) < 5.0 else "within 10%" if abs(top_manual_error) < 10.0 else "outside 10%"
    has_all_ref = all(refs[k][0] is not None for k in refs)

    report = [
        "# FLAC3D/SHAKE Layered Soil Deposit MPM Summary",
        "",
        "## Model parameters",
        "",
        f"- Height: {fmt(meta.get('height_m', math.nan), 5)} m ({fmt(meta.get('height_ft', math.nan), 5)} ft)",
        f"- Grid: {meta.get('cells_z', 'NA')} vertical cells; cell size {fmt(meta.get('cell_size_m', math.nan), 5)} m",
        f"- Shape function: {meta.get('shape_function', 'NA')}; mapping: {meta.get('mapping', 'NA')}",
        f"- Material 1: G = 150 MPa, rho = 1800 kg/m3, Vs = {fmt(meta.get('materials', {}).get('material_1', {}).get('Vs_m_s', math.nan), 5)} m/s",
        f"- Material 2: G = 300 MPa, rho = 2000 kg/m3, Vs = {fmt(meta.get('materials', {}).get('material_2', {}).get('Vs_m_s', math.nan), 5)} m/s",
        f"- Hard layer: depth 40-80 ft, z = {fmt(meta.get('hard_layer', {}).get('z_bottom_m', math.nan), 5)} to {fmt(meta.get('hard_layer', {}).get('z_top_m', math.nan), 5)} m",
        f"- Rayleigh damping: xi = 10%, center frequency = 3 Hz, alpha = {fmt(meta.get('rayleigh', {}).get('alpha', math.nan), 6)}, beta = {fmt(meta.get('rayleigh', {}).get('beta', math.nan), 6)}",
        f"- Rayleigh implementation: {meta.get('rayleigh', {}).get('implementation', 'NA')}",
        f"- Mesh frequency check: dz/lambda_min = {fmt(meta.get('mesh_frequency_check', {}).get('cell_over_lambda_min', math.nan), 5)}, lambda/8 OK = {meta.get('mesh_frequency_check', {}).get('kl_lambda_over_8_ok', 'NA')}",
        "",
        "## Output locations",
        "",
        "- Top acceleration: model top, equivalent to FLAC3D gridpoint 65 / SHAKE91 sub-layer 1.",
        "- 35 ft depth histories: z = 38.100 m. GeoTaichi uses z as vertical; CSV columns named gamma_xy/tau_xy contain the x-z engineering shear component for manual-style naming.",
        "",
        "## Manual comparison targets",
        "",
        "- Figure 3.51 top acceleration peak: FLAC3D approximately 0.160 g, SHAKE91 approximately 0.156 g.",
        "- Figure 3.52: 35 ft depth shear strain history.",
        "- Figure 3.53: 35 ft depth shear stress history, plotted in kPa with approximately -30 to 30 kPa range.",
        "",
        "## Automatic metrics",
        "",
        f"- top_acceleration_peak_mpm_g: {fmt(top_peak, 6)}",
        f"- top_peak_error_vs_manual_FLAC3D_percent: {fmt(top_manual_error, 5)} ({top_status})",
        f"- shear_strain_35ft_peak_mpm_percent: {fmt(strain_peak, 6)}",
        f"- shear_stress_35ft_peak_mpm_kpa: {fmt(stress_peak, 6)}",
    ]
    for name, metrics in (("top_acceleration", m_top), ("shear_strain_35ft", m_strain), ("shear_stress_35ft", m_stress)):
        report.extend([
            f"- {name}_peak_reference: {fmt(metrics.get('peak_reference', math.nan), 6) if metrics else 'NA'}",
            f"- {name}_peak_error_percent: {fmt(metrics.get('peak_error_percent', math.nan), 6) if metrics else 'NA'}",
            f"- {name}_NRMSE: {fmt(metrics.get('NRMSE', math.nan), 6) if metrics else 'NA'}",
            f"- {name}_correlation: {fmt(metrics.get('correlation', math.nan), 6) if metrics else 'NA'}",
        ])
    report.extend([
        "",
        "## Reference data status",
        "",
        f"- reference_top_acceleration.csv present: {refs['top'][0] is not None}",
        f"- reference_shear_strain_35ft.csv present: {refs['strain'][0] is not None}",
        f"- reference_shear_stress_35ft.csv present: {refs['stress'][0] is not None}",
        "",
        "## Conclusion",
        "",
    ])
    if has_all_ref and abs(top_manual_error) < 10.0:
        report.append("Reference curves are available and the top acceleration peak is within the preliminary manual criterion. Check the NRMSE/correlation values above before declaring strict pass.")
    elif not has_all_ref:
        report.append("The result figures satisfy the manual comparison format, but the FLAC3D/SHAKE91 reference curves have not been digitized yet; strict reproduction cannot be claimed.")
    else:
        report.append("Reference curves are available, but one or more automatic criteria do not meet the preliminary pass threshold.")
    report.extend([
        "",
        "## Generated files",
        "",
        "- input_motion.csv",
        "- history_top_acceleration.csv",
        "- history_shear_strain_35ft.csv",
        "- history_shear_stress_35ft.csv",
        "- figures/fig_3_49_input_acceleration.png",
        "- figures/fig_3_51_top_acceleration_compare.png",
        "- figures/fig_3_52_shear_strain_35ft_compare.png",
        "- figures/fig_3_53_shear_stress_35ft_compare.png",
    ])
    (OUT / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"MPM top acceleration peak = {top_peak:.6g} g; manual FLAC3D target ~= 0.160 g; error = {top_manual_error:.3f}%")
    print(f"Wrote figures to {FIG}")
    print(f"Wrote summary to {OUT / 'summary_report.md'}")


if __name__ == "__main__":
    main()
