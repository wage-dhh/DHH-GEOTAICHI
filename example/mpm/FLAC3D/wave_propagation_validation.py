"""Wave propagation validation for FLAC3D Example 3.3 reproduction.

This script is intentionally post-processing only.  It reads the current
GeoTaichi histories and the digitized FLAC3D Figure 3.9 curve, then checks
whether the shear wave input, vertical travel time, top response, free-field
matching, and late-time reflections are consistent with the target example.
"""

from __future__ import annotations

import csv
import html
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
import os


RESULT_DIR = Path(
    os.environ.get(
        "WAVE_VALIDATION_RESULT_DIR",
        (ROOT / "example/mpm/FLAC3D/results/example3_3_free_field_shear_wave_3d").as_posix(),
    )
)
BASELINE_RESULT_DIR = os.environ.get("WAVE_VALIDATION_BASELINE_DIR")
SINGLE_PULSE_RESULT_DIR = os.environ.get("WAVE_VALIDATION_SINGLE_PULSE_DIR")
HISTORIES_CSV = RESULT_DIR / "histories.csv"
REFERENCE_CSV = (
    ROOT
    / "example/mpm/FLAC3D/results/example3_3_free_field_shear_wave/strict_verification/flac3d_fig3_9_digitized.csv"
)

RHO = 0.0025
CS = 4000.0
STRESS_PEAK = 1.0
HEIGHT = 5.0
WAVE_PERIOD = 0.01


def input_wave_threshold_time(threshold_fraction: float) -> float:
    if threshold_fraction <= 0.0:
        return 0.0
    if threshold_fraction >= 1.0:
        return 0.5 * WAVE_PERIOD
    return WAVE_PERIOD / (2.0 * math.pi) * math.acos(1.0 - 2.0 * threshold_fraction)


def read_csv_float(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows: list[dict[str, float]] = []
        for row in csv.DictReader(file):
            converted: dict[str, float] = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = math.nan
            rows.append(converted)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def peak_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


def first_arrival_time(time: np.ndarray, values: np.ndarray, threshold: float) -> float:
    ids = np.flatnonzero(np.abs(values) >= threshold)
    return float(time[int(ids[0])]) if ids.size else math.nan


def status_bottom(error_percent: float) -> str:
    ae = abs(error_percent)
    if ae < 5.0:
        return "pass"
    if ae <= 10.0:
        return "acceptable"
    return "fail"


def status_peak(error_percent: float) -> str:
    ae = abs(error_percent)
    if ae < 5.0:
        return "strict_pass"
    if ae < 10.0:
        return "trend_good"
    return "strict_fail"


def status_ratio(ratio: float) -> str:
    if 0.95 <= ratio <= 1.05:
        return "pass"
    if 0.90 <= ratio <= 1.10:
        return "acceptable"
    return "fail"


def status_reflection(ratio: float) -> str:
    if ratio < 0.10:
        return "small"
    if ratio <= 0.20:
        return "moderate"
    return "large"


def best_time_shift(ref_time: np.ndarray, ref_v: np.ndarray, cur_time: np.ndarray, cur_v: np.ndarray) -> tuple[float, float, float]:
    interp = np.interp(ref_time, cur_time, cur_v)
    ref_range = float(np.max(ref_v) - np.min(ref_v))
    rmse_before = float(np.sqrt(np.mean((interp - ref_v) ** 2)))
    nrmse_before = 100.0 * rmse_before / ref_range if ref_range > 0.0 else math.inf
    best_shift = 0.0
    best_rmse = math.inf
    for shift in np.linspace(-0.003, 0.003, 601):
        shifted = np.interp(ref_time, cur_time - shift, cur_v)
        rmse = float(np.sqrt(np.mean((shifted - ref_v) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = float(shift)
    nrmse_after = 100.0 * best_rmse / ref_range if ref_range > 0.0 else math.inf
    return best_shift, nrmse_before, nrmse_after


def curve_metrics(ref_time: np.ndarray, ref_v: np.ndarray, cur_time: np.ndarray, cur_v: np.ndarray) -> dict[str, float]:
    interp = np.interp(ref_time, cur_time, cur_v)
    error = interp - ref_v
    ref_range = float(np.max(ref_v) - np.min(ref_v))
    rmse = float(np.sqrt(np.mean(error**2)))
    nrmse = 100.0 * rmse / ref_range if ref_range > 0.0 else math.inf
    corr = float(np.corrcoef(ref_v, interp)[0, 1]) if np.std(ref_v) > 0.0 and np.std(interp) > 0.0 else math.nan
    ref_peak_idx = int(np.argmax(np.abs(ref_v)))
    cur_peak_idx = int(np.argmax(np.abs(cur_v)))
    ref_peak = float(abs(ref_v[ref_peak_idx]))
    cur_peak = float(abs(cur_v[cur_peak_idx]))
    _, _, shifted = best_time_shift(ref_time, ref_v, cur_time, cur_v)
    return {
        "peak_time_error": float(cur_time[cur_peak_idx] - ref_time[ref_peak_idx]),
        "NRMSE": nrmse,
        "correlation": corr,
        "shifted_NRMSE": shifted,
        "main_peak_error": 100.0 * (cur_peak - ref_peak) / ref_peak if ref_peak > 0.0 else math.inf,
    }


def build_phase_sampling_sweep(
    ref_time: np.ndarray,
    ref_v: np.ndarray,
    time: np.ndarray,
    main_top: np.ndarray,
    ratios: tuple[float, float, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for correction in (0.0, -0.00025, -0.00050, -0.00075, -0.000762, -0.00090):
        shifted_time = time + correction
        m = curve_metrics(ref_time, ref_v, shifted_time, main_top)
        rows.append(
            {
                "case": f"history_time_correction_{correction:g}",
                "sampling_mode": "A_step_callback_shifted_time_axis",
                "history_time_correction": correction,
                "wave_time_offset": "",
                "peak_time_error": m["peak_time_error"],
                "NRMSE": m["NRMSE"],
                "correlation": m["correlation"],
                "shifted_NRMSE": m["shifted_NRMSE"],
                "main_peak_error": m["main_peak_error"],
                "main_to_x_side_ff_peak_ratio": ratios[0],
                "main_to_y_side_ff_peak_ratio": ratios[1],
                "main_to_corner_ff_peak_ratio": ratios[2],
            }
        )
    for correction in (0.0, -0.00025, -0.00050, -0.00075, -0.000762, -0.00090):
        target_time = time + correction
        interp_v = np.interp(target_time, time, main_top)
        m = curve_metrics(ref_time, ref_v, target_time, interp_v)
        rows.append(
            {
                "case": f"linear_interpolation_to_{correction:g}",
                "sampling_mode": "D_linear_interpolation_current_previous_velocity",
                "history_time_correction": correction,
                "wave_time_offset": "",
                "peak_time_error": m["peak_time_error"],
                "NRMSE": m["NRMSE"],
                "correlation": m["correlation"],
                "shifted_NRMSE": m["shifted_NRMSE"],
                "main_peak_error": m["main_peak_error"],
                "main_to_x_side_ff_peak_ratio": ratios[0],
                "main_to_y_side_ff_peak_ratio": ratios[1],
                "main_to_corner_ff_peak_ratio": ratios[2],
            }
        )
    return rows


def calc_reflection_ratio(time: np.ndarray, values: np.ndarray, start: float, peak: float) -> float:
    mask = time >= start
    if not np.any(mask) or peak <= 0.0:
        return math.nan
    return peak_abs(values[mask]) / peak


def plot_outputs(summary: dict[str, Any], time: np.ndarray, histories: dict[str, np.ndarray], ref_time: np.ndarray, ref_v: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9.0, 4.8), dpi=180)
    plt.plot(time, histories["bottom"], label="main grid bottom vx")
    plt.axhline(summary["bottom_input"]["theoretical_velocity"], color="k", linestyle="--", linewidth=1.0, label="theory +0.1")
    plt.axhline(-summary["bottom_input"]["theoretical_velocity"], color="k", linestyle=":", linewidth=1.0, label="theory -0.1")
    plt.xlabel("time (s)")
    plt.ylabel("vx")
    plt.title("Bottom Input Velocity Check")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "wave_input_bottom_check.png")
    plt.close()

    plt.figure(figsize=(9.0, 4.8), dpi=180)
    plt.plot(time, histories["top"], label="main grid top vx")
    plt.axvline(summary["wave_speed"]["theoretical_arrival_time"], color="k", linestyle="--", label="theoretical arrival")
    plt.axvline(summary["wave_speed"]["current_arrival_time"], color="r", linestyle=":", label="current first arrival")
    plt.xlabel("time (s)")
    plt.ylabel("vx")
    plt.title("Wave Arrival Time Check")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "wave_arrival_time_check.png")
    plt.close()

    plt.figure(figsize=(9.0, 4.8), dpi=180)
    plt.plot(time, histories["top"], label="main grid top")
    plt.plot(time, histories["x_side"], label="x-side free-field top")
    plt.plot(time, histories["y_side"], label="y-side free-field top")
    plt.plot(time, histories["corner"], label="corner free-field top")
    plt.xlabel("time (s)")
    plt.ylabel("vx")
    plt.title("Main vs Free-Field Velocity Check")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "main_vs_freefield_velocity_check.png")
    plt.close()

    plt.figure(figsize=(9.5, 5.2), dpi=180)
    plt.plot(ref_time, ref_v, "k-", linewidth=2.0, label="FLAC3D Figure 3.9 digitized")
    plt.plot(time, histories["top"], label="GeoTaichi main top")
    plt.plot(time, histories["x_side"], label="GeoTaichi x-side FF")
    plt.plot(time, histories["y_side"], label="GeoTaichi y-side FF")
    plt.plot(time, histories["corner"], label="GeoTaichi corner FF")
    plt.xlabel("time (s)")
    plt.ylabel("vx")
    plt.title("Figure 3.9 Wave Validation")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "figure_3_9_wave_validation.png")
    plt.close()


def write_raw_vs_corrected_plot(
    path: Path,
    baseline_dir: Path,
    corrected_time: np.ndarray,
    corrected_top: np.ndarray,
    ref_time: np.ndarray,
    ref_v: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    baseline_histories = read_csv_float(baseline_dir / "histories.csv")
    raw_time = np.array([row["time"] for row in baseline_histories], dtype=float)
    raw_top = np.array([row["main_grid_top_vx"] for row in baseline_histories], dtype=float)
    plt.figure(figsize=(9.5, 5.2), dpi=180)
    plt.plot(ref_time, ref_v, "k-", linewidth=2.0, label="FLAC3D Figure 3.9 digitized")
    plt.plot(raw_time, raw_top, color="#b44", linewidth=1.4, label="raw GeoTaichi main top")
    plt.plot(corrected_time, corrected_top, color="#1769aa", linewidth=1.6, label="corrected GeoTaichi main top")
    plt.xlabel("time (s)")
    plt.ylabel("vx")
    plt.title("Figure 3.9 Raw vs Corrected")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def write_phase_best_plot(path: Path, ref_time: np.ndarray, ref_v: np.ndarray, time: np.ndarray, main_top: np.ndarray, sweep_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    best = min(sweep_rows, key=lambda row: float(row["NRMSE"]))
    best_time = time + float(best["history_time_correction"])
    plt.figure(figsize=(9.5, 5.2), dpi=180)
    plt.plot(ref_time, ref_v, "k-", linewidth=2.0, label="FLAC3D Figure 3.9")
    plt.plot(time, main_top, color="#b44", linewidth=1.3, label="current")
    plt.plot(best_time, main_top, color="#1769aa", linewidth=1.5, label=f"best phase case: {best['case']}")
    plt.xlabel("time (s)")
    plt.ylabel("vx")
    plt.title("Phase Sampling Best Case")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return best


def write_periodic_vs_single_plot(path: Path, periodic_dir: Path, single_dir: Path, ref_time: np.ndarray, ref_v: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    periodic_rows = read_csv_float(periodic_dir / "histories.csv")
    single_rows = read_csv_float(single_dir / "histories.csv")
    pt = np.array([row["time"] for row in periodic_rows], dtype=float)
    pv = np.array([row["main_grid_top_vx"] for row in periodic_rows], dtype=float)
    st = np.array([row["time"] for row in single_rows], dtype=float)
    sv = np.array([row["main_grid_top_vx"] for row in single_rows], dtype=float)
    plt.figure(figsize=(9.5, 5.2), dpi=180)
    plt.plot(ref_time, ref_v, "k-", linewidth=2.0, label="FLAC3D Figure 3.9")
    plt.plot(pt, pv, color="#1769aa", label="periodic wave")
    plt.plot(st, sv, color="#c47f00", label="single pulse wave")
    plt.axvline(WAVE_PERIOD, color="gray", linestyle="--", linewidth=1.0, label="T=0.01 s")
    plt.xlabel("time (s)")
    plt.ylabel("vx")
    plt.title("Periodic vs Single Pulse Input")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_docx(path: Path, summary: dict[str, Any]) -> None:
    def p(text: str, style: str | None = None) -> str:
        style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
        return f"<w:p><w:pPr>{style_xml}</w:pPr><w:r><w:t>{html.escape(text)}</w:t></w:r></w:p>"

    def table(rows: list[list[str]]) -> str:
        parts = ['<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/></w:tblPr>']
        for row in rows:
            parts.append("<w:tr>")
            for item in row:
                parts.append(
                    '<w:tc><w:tcPr><w:tcW w:w="2600" w:type="dxa"/></w:tcPr>'
                    + p(item)
                    + "</w:tc>"
                )
            parts.append("</w:tr>")
        parts.append("</w:tbl>")
        return "".join(parts)

    sections = [
        p("FLAC3D Example 3.3 Wave Propagation Validation Report", "Title"),
        p("本报告只验证波传输正确性，不声明当前结果已经完全复现 FLAC3D Figure 3.9。"),
        p("Graded Conclusions", "Heading1"),
        table(
            [
                ["check", "status", "key result"],
                ["底部输入波", summary["bottom_input"]["status"], f"peak error = {summary['bottom_input']['bottom_peak_error_percent']:.3f}%"],
                ["波速传播", summary["wave_speed"]["status"], f"effective Cs = {summary['wave_speed']['effective_Cs']:.3f}"],
                ["顶部幅值", summary["top_amplitude"]["status"], f"peak error = {summary['top_amplitude']['peak_error_percent_main']:.3f}%"],
                ["相位", summary["phase"]["status"], f"best shift = {summary['phase']['best_shift']:.6g} s"],
                ["free-field 一致性", summary["free_field_consistency"]["status"], f"ratios = {summary['free_field_consistency']['main_to_x_side_ff_peak_ratio']:.4f}, {summary['free_field_consistency']['main_to_y_side_ff_peak_ratio']:.4f}, {summary['free_field_consistency']['main_to_corner_ff_peak_ratio']:.4f}"],
                ["边界反射", summary["reflection"]["status"], f"post-peak ratio = {summary['reflection']['post_peak_reflection_ratio']:.4f}"],
                ["Figure 3.9 严格复现", summary["strict_figure_3_9"]["status"], summary["strict_figure_3_9"]["reason"]],
            ]
        ),
        p("Bottom Input Wave", "Heading1"),
        p(json.dumps(summary["bottom_input"], ensure_ascii=False, indent=2)),
        p("Wave Speed Propagation", "Heading1"),
        p(json.dumps(summary["wave_speed"], ensure_ascii=False, indent=2)),
        p("Top Amplitude", "Heading1"),
        p(json.dumps(summary["top_amplitude"], ensure_ascii=False, indent=2)),
        p("Phase", "Heading1"),
        p(json.dumps(summary["phase"], ensure_ascii=False, indent=2)),
        p("Free-Field Consistency", "Heading1"),
        p(json.dumps(summary["free_field_consistency"], ensure_ascii=False, indent=2)),
        p("Boundary Reflection", "Heading1"),
        p(json.dumps(summary["reflection"], ensure_ascii=False, indent=2)),
        p("Output Files", "Heading1"),
        p("wave_propagation_diagnostics.csv; wave_propagation_summary.json; wave_input_bottom_check.png; wave_arrival_time_check.png; main_vs_freefield_velocity_check.png; figure_3_9_wave_validation.png"),
    ]
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        + "".join(sections)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>'
        + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Microsoft YaHei"/><w:sz w:val="20"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>'
        "</w:styles>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles)


def write_fixed_docx(path: Path, baseline: dict[str, Any], corrected: dict[str, Any]) -> None:
    def p(text: str, style: str | None = None) -> str:
        style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
        return f"<w:p><w:pPr>{style_xml}</w:pPr><w:r><w:t>{html.escape(text)}</w:t></w:r></w:p>"

    def table(rows: list[list[str]]) -> str:
        parts = ['<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/></w:tblPr>']
        for row in rows:
            parts.append("<w:tr>")
            for item in row:
                parts.append(
                    '<w:tc><w:tcPr><w:tcW w:w="2200" w:type="dxa"/></w:tcPr>'
                    + p(item)
                    + "</w:tc>"
                )
            parts.append("</w:tr>")
        parts.append("</w:tbl>")
        return "".join(parts)

    rows = [
        ["metric", "raw", "corrected", "target/status"],
        [
            "bottom_peak_error_percent",
            f"{baseline['bottom_input']['bottom_peak_error_percent']:.3f}",
            f"{corrected['bottom_input']['bottom_peak_error_percent']:.3f}",
            "target < 5%",
        ],
        [
            "effective_Cs_raw_5pct",
            f"{baseline['wave_speed']['effective_Cs']:.3f}",
            f"{corrected['wave_speed']['effective_Cs']:.3f}",
            "raw 5% criterion retained",
        ],
        [
            "effective_Cs_corrected_1pct",
            f"{baseline['wave_speed']['input_corrected_effective_Cs_1pct']:.3f}",
            f"{corrected['wave_speed']['input_corrected_effective_Cs_1pct']:.3f}",
            "target near 4000",
        ],
        [
            "peak_time_error",
            f"{baseline['phase']['peak_time_error']:.6g}",
            f"{corrected['phase']['peak_time_error']:.6g}",
            "target near 0",
        ],
        [
            "NRMSE_main",
            f"{baseline['phase']['NRMSE_before_shift']:.3f}",
            f"{corrected['phase']['NRMSE_before_shift']:.3f}",
            "target < 10%",
        ],
        [
            "post_peak_reflection_ratio",
            f"{baseline['reflection']['post_peak_reflection_ratio']:.3f}",
            f"{corrected['reflection']['post_peak_reflection_ratio']:.3f}",
            "original late window",
        ],
        [
            "post_input_reflection_ratio",
            f"{baseline['reflection']['post_input_reflection_ratio']:.3f}",
            f"{corrected['reflection']['post_input_reflection_ratio']:.3f}",
            "after input pulse window",
        ],
        [
            "main/free-field ratio x/y/corner",
            f"{baseline['free_field_consistency']['main_to_x_side_ff_peak_ratio']:.3f}/{baseline['free_field_consistency']['main_to_y_side_ff_peak_ratio']:.3f}/{baseline['free_field_consistency']['main_to_corner_ff_peak_ratio']:.3f}",
            f"{corrected['free_field_consistency']['main_to_x_side_ff_peak_ratio']:.3f}/{corrected['free_field_consistency']['main_to_y_side_ff_peak_ratio']:.3f}/{corrected['free_field_consistency']['main_to_corner_ff_peak_ratio']:.3f}",
            "target 0.95-1.05",
        ],
    ]
    body = [
        p("FLAC3D Example 3.3 Wave Propagation Fixed Validation Report", "Title"),
        p("本报告对比修正前 raw 与 corrected 结果。修正包括 stress_scale=1.15、wave_factor(t+dt)、history_time=current_time-dt。"),
        p("结论必须分级：底部幅值已修正到目标范围；free-field 一致性保持通过；原始 5% 到达时间、相位、NRMSE 和反射仍未全部达到严格目标。"),
        p("Before/After Metrics", "Heading1"),
        table(rows),
        p("Figure", "Heading1"),
        p("Raw 与 corrected 对比图已生成：figure_3_9_raw_vs_corrected.png。"),
        p("Failure Items Still Visible", "Heading1"),
        p(corrected["strict_figure_3_9"]["reason"]),
    ]
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        + "".join(body)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>'
        + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Microsoft YaHei"/><w:sz w:val="20"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>'
        "</w:styles>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles)


def write_phase_reflection_docx(path: Path, summary: dict[str, Any], best_phase: dict[str, Any] | None, comparison_rows: list[dict[str, Any]]) -> None:
    def p(text: str, style: str | None = None) -> str:
        style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
        return f"<w:p><w:pPr>{style_xml}</w:pPr><w:r><w:t>{html.escape(text)}</w:t></w:r></w:p>"

    def table(rows: list[list[str]]) -> str:
        parts = ['<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/></w:tblPr>']
        for row in rows:
            parts.append("<w:tr>")
            for item in row:
                parts.append('<w:tc><w:tcPr><w:tcW w:w="2300" w:type="dxa"/></w:tcPr>' + p(item) + "</w:tc>")
            parts.append("</w:tr>")
        parts.append("</w:tbl>")
        return "".join(parts)

    conclusion_rows = [
        ["item", "status", "metric"],
        ["幅值", summary["bottom_input"]["status"], f"bottom error {summary['bottom_input']['bottom_peak_error_percent']:.3f}%, main error {summary['top_amplitude']['peak_error_percent_main']:.3f}%"],
        ["raw arrival", summary["wave_speed"]["status"], f"raw effective Cs {summary['wave_speed']['effective_Cs']:.1f}"],
        ["source-corrected arrival", "pass" if abs(summary["wave_speed"]["source_corrected_effective_Cs_5pct"] - CS) / CS < 0.15 else "check", f"corrected Cs {summary['wave_speed']['source_corrected_effective_Cs_5pct']:.1f}"],
        ["相位", "best_effort" if best_phase else "not_available", f"best {best_phase['case'] if best_phase else ''}, NRMSE {float(best_phase['NRMSE']) if best_phase else math.nan:.3f}%"],
        ["反射/后期振荡", summary["reflection"]["status"], f"post_peak {summary['reflection']['post_peak_reflection_ratio']:.3f}"],
        ["Figure 3.9 raw", summary["strict_figure_3_9"]["status"], summary["strict_figure_3_9"]["reason"]],
        ["Figure 3.9 corrected/best-effort", "best_effort", f"shifted NRMSE {summary['phase']['NRMSE_after_shift']:.3f}%"],
    ]
    body = [
        p("Wave Propagation Phase and Reflection Fixed Report", "Title"),
        p("raw arrival time 包含输入半余弦波自身上升时间；source-corrected arrival time 才更接近传播时间。"),
        p("Conclusions", "Heading1"),
        table(conclusion_rows),
        p("Periodic vs Single Pulse", "Heading1"),
        table(
            [["case", "post_peak_reflection_ratio", "NRMSE", "peak_time_error", "main_peak_error"]]
            + [
                [
                    str(row["case"]),
                    f"{float(row['post_peak_reflection_ratio']):.3f}",
                    f"{float(row['NRMSE']):.3f}",
                    f"{float(row['peak_time_error']):.6g}",
                    f"{float(row['main_peak_error']):.3f}",
                ]
                for row in comparison_rows
            ]
        )
        if comparison_rows
        else p("Single-pulse comparison was not provided."),
        p("Generated Files", "Heading1"),
        p("wave_arrival_source_corrected.csv; phase_sampling_sweep.csv; reflection_source_diagnostics.csv; periodic_vs_single_pulse_comparison.csv; figure_phase_sampling_best.png; figure_periodic_vs_single_pulse.png"),
    ]
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        + "".join(body)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>'
        + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Microsoft YaHei"/><w:sz w:val="20"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>'
        "</w:styles>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles)


def main() -> None:
    histories_rows = read_csv_float(HISTORIES_CSV)
    reference_rows = read_csv_float(REFERENCE_CSV)
    time = np.array([row["time"] for row in histories_rows], dtype=float)
    ref_time = np.array([row["time"] for row in reference_rows], dtype=float)
    ref_v = np.array([row["vx_flac3d"] for row in reference_rows], dtype=float)
    histories = {
        "bottom": np.array([row["main_grid_base_vx"] for row in histories_rows], dtype=float),
        "top": np.array([row["main_grid_top_vx"] for row in histories_rows], dtype=float),
        "x_side": np.array([row["x_parallel_side_free_field_top_vx"] for row in histories_rows], dtype=float),
        "y_side": np.array([row["y_parallel_side_free_field_top_vx"] for row in histories_rows], dtype=float),
        "corner": np.array([row["corner_free_field_top_vx"] for row in histories_rows], dtype=float),
        "mid": np.array([row.get("main_mid_height_vx", math.nan) for row in histories_rows], dtype=float),
        "left_x_side": np.array([row.get("left_x_side_ff_top_vx", math.nan) for row in histories_rows], dtype=float),
        "right_x_side": np.array([row.get("right_x_side_ff_top_vx", math.nan) for row in histories_rows], dtype=float),
        "front_y_side": np.array([row.get("front_y_side_ff_top_vx", math.nan) for row in histories_rows], dtype=float),
        "back_y_side": np.array([row.get("back_y_side_ff_top_vx", math.nan) for row in histories_rows], dtype=float),
        "left_front_corner": np.array([row.get("left_front_corner_column_top_vx", math.nan) for row in histories_rows], dtype=float),
        "left_back_corner": np.array([row.get("left_back_corner_column_top_vx", math.nan) for row in histories_rows], dtype=float),
        "right_front_corner": np.array([row.get("right_front_corner_column_top_vx", math.nan) for row in histories_rows], dtype=float),
        "right_back_corner": np.array([row.get("right_back_corner_column_top_vx", math.nan) for row in histories_rows], dtype=float),
    }
    dt_values = np.diff(time)
    dt = float(np.median(dt_values[dt_values > 0.0])) if np.any(dt_values > 0.0) else math.nan

    v_theory = STRESS_PEAK / (RHO * CS)
    bottom_peak = peak_abs(histories["bottom"])
    bottom_error = 100.0 * (bottom_peak - v_theory) / v_theory

    top_peak = peak_abs(histories["top"])
    arrival_threshold = 0.05 * max(top_peak, 1.0e-30)
    current_arrival = first_arrival_time(time, histories["top"], arrival_threshold)
    source_threshold_time_5pct = input_wave_threshold_time(0.05)
    source_corrected_arrival_5pct = current_arrival - source_threshold_time_5pct if math.isfinite(current_arrival) else math.nan
    source_corrected_effective_cs_5pct = HEIGHT / source_corrected_arrival_5pct if source_corrected_arrival_5pct > 0 else math.nan
    one_percent_arrival = first_arrival_time(time, histories["top"], 0.01 * max(top_peak, 1.0e-30))
    theoretical_arrival = HEIGHT / CS
    effective_cs = HEIGHT / current_arrival if current_arrival > 0.0 else math.nan
    arrival_error = current_arrival - theoretical_arrival if math.isfinite(current_arrival) else math.nan
    time_step_error_count = abs(arrival_error) / dt if math.isfinite(arrival_error) and dt > 0.0 else math.nan
    wave_speed_status = "pass" if math.isfinite(time_step_error_count) and time_step_error_count <= 2.0 else "fail"
    input_time_1pct = input_wave_threshold_time(0.01)
    input_corrected_travel_1pct = one_percent_arrival - input_time_1pct if math.isfinite(one_percent_arrival) else math.nan
    input_corrected_effective_cs_1pct = HEIGHT / input_corrected_travel_1pct if input_corrected_travel_1pct > 0.0 else math.nan

    ref_peak_idx = int(np.argmax(np.abs(ref_v)))
    cur_peak_idx = int(np.argmax(np.abs(histories["top"])))
    reference_peak = float(abs(ref_v[ref_peak_idx]))
    current_peak = top_peak
    peak_error = 100.0 * (current_peak - reference_peak) / reference_peak
    reference_peak_time = float(ref_time[ref_peak_idx])
    current_peak_time = float(time[cur_peak_idx])
    peak_time_error = current_peak_time - reference_peak_time
    shift, nrmse_before, nrmse_after = best_time_shift(ref_time, ref_v, time, histories["top"])
    phase_status = "phase_error_dominant" if nrmse_after < 0.7 * nrmse_before else "not_shift_dominant"

    x_peak = peak_abs(histories["x_side"])
    y_peak = peak_abs(histories["y_side"])
    corner_peak = peak_abs(histories["corner"])
    ratio_x = current_peak / x_peak if x_peak > 0.0 else math.inf
    ratio_y = current_peak / y_peak if y_peak > 0.0 else math.inf
    ratio_corner = current_peak / corner_peak if corner_peak > 0.0 else math.inf
    free_field_status = (
        "pass"
        if all(status_ratio(ratio) == "pass" for ratio in (ratio_x, ratio_y, ratio_corner))
        else "fail"
    )

    late_start = current_peak_time + 0.002
    late_mask = time >= late_start
    late_max = peak_abs(histories["top"][late_mask]) if np.any(late_mask) else math.nan
    reflection_ratio = late_max / current_peak if current_peak > 0.0 and math.isfinite(late_max) else math.nan
    post_input_late_start = WAVE_PERIOD + theoretical_arrival + 0.002
    post_input_mask = time >= post_input_late_start
    post_input_late_max = peak_abs(histories["top"][post_input_mask]) if np.any(post_input_mask) else math.nan
    post_input_reflection_ratio = (
        post_input_late_max / current_peak if current_peak > 0.0 and math.isfinite(post_input_late_max) else math.nan
    )

    strict_status = (
        "strict_pass"
        if abs(peak_error) < 5.0 and nrmse_before < 10.0 and abs(peak_time_error) <= 2.0 * dt
        else "strict_fail"
    )
    strict_reason = (
        f"peak_error={peak_error:.3f}%, NRMSE={nrmse_before:.3f}%, "
        f"peak_time_error={peak_time_error:.6g}s"
    )

    summary: dict[str, Any] = {
        "inputs": {
            "histories_csv": HISTORIES_CSV.as_posix(),
            "reference_csv": REFERENCE_CSV.as_posix(),
            "rho": RHO,
            "Cs": CS,
            "stress_peak": STRESS_PEAK,
            "height": HEIGHT,
            "history_dt_median": dt,
        },
        "bottom_input": {
            "theoretical_velocity": v_theory,
            "bottom_actual_velocity_peak": bottom_peak,
            "bottom_peak_error_percent": bottom_error,
            "status": status_bottom(bottom_error),
        },
        "wave_speed": {
            "theoretical_arrival_time": theoretical_arrival,
            "current_arrival_time": current_arrival,
            "raw_arrival_time_5pct": current_arrival,
            "source_threshold_time_5pct": source_threshold_time_5pct,
            "source_corrected_arrival_time_5pct": source_corrected_arrival_5pct,
            "source_corrected_effective_Cs_5pct": source_corrected_effective_cs_5pct,
            "arrival_time_error": arrival_error,
            "effective_Cs": effective_cs,
            "time_step_error_count": time_step_error_count,
            "one_percent_arrival_time": one_percent_arrival,
            "input_wave_threshold_time_1pct": input_time_1pct,
            "input_corrected_travel_time_1pct": input_corrected_travel_1pct,
            "input_corrected_effective_Cs_1pct": input_corrected_effective_cs_1pct,
            "status": wave_speed_status,
        },
        "top_amplitude": {
            "reference_main_peak": reference_peak,
            "current_main_peak": current_peak,
            "peak_error_percent_main": peak_error,
            "status": status_peak(peak_error),
        },
        "phase": {
            "reference_peak_time": reference_peak_time,
            "current_peak_time": current_peak_time,
            "peak_time_error": peak_time_error,
            "best_shift": shift,
            "NRMSE_before_shift": nrmse_before,
            "NRMSE_after_shift": nrmse_after,
            "status": phase_status,
        },
        "free_field_consistency": {
            "main_peak": current_peak,
            "x_side_ff_peak": x_peak,
            "y_side_ff_peak": y_peak,
            "corner_ff_peak": corner_peak,
            "main_to_x_side_ff_peak_ratio": ratio_x,
            "main_to_y_side_ff_peak_ratio": ratio_y,
            "main_to_corner_ff_peak_ratio": ratio_corner,
            "status": free_field_status,
        },
        "reflection": {
            "late_time_start": late_start,
            "late_time_max_abs_velocity": late_max,
            "post_peak_reflection_ratio": reflection_ratio,
            "post_input_late_time_start": post_input_late_start,
            "post_input_late_time_max_abs_velocity": post_input_late_max,
            "post_input_reflection_ratio": post_input_reflection_ratio,
            "status": status_reflection(reflection_ratio) if math.isfinite(reflection_ratio) else "unknown",
        },
        "strict_figure_3_9": {
            "status": strict_status,
            "reason": strict_reason,
        },
    }

    diagnostics_rows = [
        {"section": section, "metric": key, "value": value}
        for section, data in summary.items()
        if isinstance(data, dict)
        for key, value in data.items()
        if not isinstance(value, dict)
    ]
    write_csv(RESULT_DIR / "wave_propagation_diagnostics.csv", diagnostics_rows)
    write_csv(
        RESULT_DIR / "wave_arrival_source_corrected.csv",
        [
            {
                "threshold_fraction": frac,
                "raw_arrival_time": first_arrival_time(time, histories["top"], frac * max(top_peak, 1.0e-30)),
                "source_threshold_time": input_wave_threshold_time(frac),
                "source_corrected_arrival_time": first_arrival_time(time, histories["top"], frac * max(top_peak, 1.0e-30)) - input_wave_threshold_time(frac),
                "source_corrected_effective_Cs": HEIGHT / (first_arrival_time(time, histories["top"], frac * max(top_peak, 1.0e-30)) - input_wave_threshold_time(frac)),
            }
            for frac in (0.01, 0.02, 0.05, 0.10, 0.20)
        ],
    )
    ratios = (ratio_x, ratio_y, ratio_corner)
    phase_rows = build_phase_sampling_sweep(ref_time, ref_v, time, histories["top"], ratios)
    write_csv(RESULT_DIR / "phase_sampling_sweep.csv", phase_rows)
    best_phase = write_phase_best_plot(RESULT_DIR / "figure_phase_sampling_best.png", ref_time, ref_v, time, histories["top"], phase_rows)
    source_rows = []
    for name, values in (
        ("bottom_main", histories["bottom"]),
        ("mid_height_main", histories["mid"]),
        ("top_main", histories["top"]),
        ("left_x_side_ff_top", histories["left_x_side"]),
        ("right_x_side_ff_top", histories["right_x_side"]),
        ("front_y_side_ff_top", histories["front_y_side"]),
        ("back_y_side_ff_top", histories["back_y_side"]),
        ("left_front_corner_column_top", histories["left_front_corner"]),
        ("left_back_corner_column_top", histories["left_back_corner"]),
        ("right_front_corner_column_top", histories["right_front_corner"]),
        ("right_back_corner_column_top", histories["right_back_corner"]),
    ):
        p = peak_abs(values)
        source_rows.append(
            {
                "location": name,
                "peak_abs_velocity": p,
                "late_start_post_peak": late_start,
                "late_reflection_ratio_post_peak": calc_reflection_ratio(time, values, late_start, p),
                "late_start_post_input": post_input_late_start,
                "late_reflection_ratio_post_input": calc_reflection_ratio(time, values, post_input_late_start, p),
            }
        )
    write_csv(RESULT_DIR / "reflection_source_diagnostics.csv", source_rows)
    (RESULT_DIR / "wave_propagation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_outputs(summary, time, histories, ref_time, ref_v)
    write_docx(RESULT_DIR / "wave_propagation_report.docx", summary)
    comparison_rows: list[dict[str, Any]] = []
    if SINGLE_PULSE_RESULT_DIR:
        single_dir = Path(SINGLE_PULSE_RESULT_DIR)
        write_periodic_vs_single_plot(RESULT_DIR / "figure_periodic_vs_single_pulse.png", RESULT_DIR, single_dir, ref_time, ref_v)
        for case, directory in (("periodic", RESULT_DIR), ("single_pulse", single_dir)):
            case_summary = json.loads((directory / "wave_propagation_summary.json").read_text(encoding="utf-8"))
            comparison_rows.append(
                {
                    "case": case,
                    "post_peak_reflection_ratio": case_summary["reflection"]["post_peak_reflection_ratio"],
                    "NRMSE": case_summary["phase"]["NRMSE_before_shift"],
                    "peak_time_error": case_summary["phase"]["peak_time_error"],
                    "main_peak_error": case_summary["top_amplitude"]["peak_error_percent_main"],
                }
            )
        write_csv(RESULT_DIR / "periodic_vs_single_pulse_comparison.csv", comparison_rows)
    else:
        write_csv(RESULT_DIR / "periodic_vs_single_pulse_comparison.csv", [])
    write_phase_reflection_docx(RESULT_DIR / "wave_propagation_report_phase_reflection_fixed.docx", summary, best_phase, comparison_rows)
    if BASELINE_RESULT_DIR:
        baseline_dir = Path(BASELINE_RESULT_DIR)
        baseline_summary_path = baseline_dir / "wave_propagation_summary.json"
        if not baseline_summary_path.exists():
            raise FileNotFoundError(f"baseline summary is required before fixed comparison: {baseline_summary_path}")
        baseline_summary = json.loads((baseline_dir / "wave_propagation_summary.json").read_text(encoding="utf-8"))
        write_raw_vs_corrected_plot(
            RESULT_DIR / "figure_3_9_raw_vs_corrected.png",
            baseline_dir,
            time,
            histories["top"],
            ref_time,
            ref_v,
        )
        write_fixed_docx(RESULT_DIR / "wave_propagation_report_fixed.docx", baseline_summary, summary)


if __name__ == "__main__":
    main()
