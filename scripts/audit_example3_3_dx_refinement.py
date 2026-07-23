import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "example3_3_2d_kohler_native_mpm_solver"
REF = ROOT / "data" / "reference" / "flac3d_example3_3_free_field"

RUNS = [
    {
        "dx": 0.25,
        "label": "dx=0.25 m",
        "run_dir": OUT / "latest_2d_height4m_centered_v_platform_top_monitors",
        "geometry": "geometry_only_corrected",
    },
    {
        "dx": 0.125,
        "label": "dx=0.125 m",
        "run_dir": OUT / "latest_2d_dx0p125_height4m_centered_v_platform_top_monitors",
        "geometry": "geometry_only_corrected_dx0p125_centered_v",
    },
]

TOP_CASES = [
    "soil_grid_top vs FLAC3D free-field",
    "main_grid_top_manual vs FLAC3D main",
]

REQUIRED_FILES = [
    "geometry_corrected_dynamic_validation_report.md",
    "geometry_validation_checks.source.csv",
    "material_parameter_check.csv",
    "history.csv",
    "grid_monitor_velocity.csv",
    "phase_alignment_summary.csv",
    "bottom_boundary_force_check.csv",
    "side_boundary_force_check.csv",
    "grid_bottom_node_check.csv",
    "grid_side_pair_check.csv",
    "figure3_9_geometry_corrected.png",
    "figure3_9_doc_reference_style.png",
    "figure3_9_phase_aligned.png",
    "top_grid_monitor_vs_flac3d.png",
    "bottom_monitor_comparison.png",
    "particles_dynamic.vtu",
    "grid_dynamic.vtr",
    "interface_dynamic.vtk",
]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_xy(path: Path, x_col: str, y_col: str) -> tuple[list[float], list[float]]:
    rows = read_csv_dicts(path)
    return [float(r[x_col]) for r in rows], [float(r[y_col]) for r in rows]


def geometry_status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    rows = read_csv_dicts(path)
    failing = [r for r in rows if r.get("status") != "PASS"]
    return "PASS" if not failing else "FAIL"


def file_status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    if path.is_file() and path.stat().st_size <= 0:
        return "EMPTY"
    return "PASS"


def collect_phase_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in RUNS:
        phase_path = run["run_dir"] / "phase_alignment_summary.csv"
        for row in read_csv_dicts(phase_path):
            if row["case"] not in TOP_CASES:
                continue
            rows.append(
                {
                    "dx": run["dx"],
                    "case": row["case"],
                    "nrmse": row["nrmse"],
                    "correlation": row["correlation"],
                    "peak_time_error_s": row["peak_time_error"],
                    "best_time_shift_s": row["best_time_shift"],
                    "nrmse_after_time_shift": row["nrmse_after_time_shift"],
                    "peak_error_m_s": row["peak_error"],
                }
            )
    return rows


def collect_audit_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in RUNS:
        run_dir = run["run_dir"]
        rows.append(
            {
                "dx": run["dx"],
                "run_dir": run_dir,
                "check": "run directory",
                "target": run_dir,
                "status": "PASS" if run_dir.is_dir() else "MISSING",
                "detail": "",
            }
        )
        geom_check = run_dir / "geometry_validation_checks.source.csv"
        rows.append(
            {
                "dx": run["dx"],
                "run_dir": run_dir,
                "check": "geometry validation checks",
                "target": geom_check,
                "status": geometry_status(geom_check),
                "detail": "all geometry rows must be PASS",
            }
        )
        for filename in REQUIRED_FILES:
            path = run_dir / filename
            rows.append(
                {
                    "dx": run["dx"],
                    "run_dir": run_dir,
                    "check": "required artifact",
                    "target": path,
                    "status": file_status(path),
                    "detail": filename,
                }
            )
    return rows


def collect_delta_rows(phase_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_case: dict[str, dict[float, dict[str, object]]] = {}
    for row in phase_rows:
        by_case.setdefault(str(row["case"]), {})[float(row["dx"])] = row

    rows: list[dict[str, object]] = []
    for case, values in by_case.items():
        coarse = values[0.25]
        refined = values[0.125]
        nrmse_delta = float(refined["nrmse"]) - float(coarse["nrmse"])
        corr_delta = float(refined["correlation"]) - float(coarse["correlation"])
        peak_time_abs_delta = abs(float(refined["peak_time_error_s"])) - abs(float(coarse["peak_time_error_s"]))
        rows.append(
            {
                "case": case,
                "coarse_dx": 0.25,
                "refined_dx": 0.125,
                "nrmse_delta_refined_minus_coarse": nrmse_delta,
                "nrmse_change_percent": 100.0 * nrmse_delta / float(coarse["nrmse"]),
                "correlation_delta_refined_minus_coarse": corr_delta,
                "abs_peak_time_error_delta_s": peak_time_abs_delta,
                "abs_peak_time_error_change_percent": 100.0
                * peak_time_abs_delta
                / abs(float(coarse["peak_time_error_s"])),
            }
        )
    return rows


def plot_top_grid_comparison() -> None:
    free_t, free_v = read_xy(REF / "reference_fig_3_9_flac_free_correct.csv", "time_s", "vx")
    main_t, main_v = read_xy(REF / "reference_fig_3_9_flac_main_correct.csv", "time_s", "vx")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    axes[0].plot(free_t, free_v, "k--", linewidth=1.8, label="FLAC3D free-field")
    axes[1].plot(main_t, main_v, "k--", linewidth=1.8, label="FLAC3D main")

    for run in RUNS:
        path = run["run_dir"] / "grid_monitor_velocity.csv"
        t, soil = read_xy(path, "time", "soil_grid_top_vx")
        _, main = read_xy(path, "time", "main_grid_top_manual_vx")
        axes[0].plot(t, soil, linewidth=1.3, label=run["label"])
        axes[1].plot(t, main, linewidth=1.3, label=run["label"])

    axes[0].set_title("Top free-field grid monitor")
    axes[1].set_title("Top main grid monitor")
    for ax in axes:
        ax.set_ylabel("x velocity (m/s)")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "latest_2d_dx_refinement_top_grid_monitor_vs_flac3d.png", dpi=180)
    plt.close(fig)


def write_summary(phase_rows: list[dict[str, object]], audit_rows: list[dict[str, object]], delta_rows: list[dict[str, object]]) -> None:
    bad = [r for r in audit_rows if r["status"] != "PASS"]
    complete_status = "PASS" if not bad else "FAIL"

    lines = [
        "# dx refinement validation summary",
        "",
        "Geometry: centered V, relative height 4.0 m, same physical dimensions for both runs.",
        "",
        f"Artifact completeness status: `{complete_status}`.",
        "",
        "| dx | case | NRMSE | correlation | peak time error (s) | best time shift (s) | shifted NRMSE | peak error (m/s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in phase_rows:
        lines.append(
            f"| {float(row['dx']):.3g} | {row['case']} | {float(row['nrmse']):.12g} | "
            f"{float(row['correlation']):.12g} | {float(row['peak_time_error_s']):.12g} | "
            f"{float(row['best_time_shift_s']):.12g} | {float(row['nrmse_after_time_shift']):.12g} | "
            f"{float(row['peak_error_m_s']):.12g} |"
        )

    lines += [
        "",
        "## Refinement Deltas",
        "",
        "| case | NRMSE change | NRMSE change (%) | correlation change | abs peak-time-error change (s) | abs peak-time-error change (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in delta_rows:
        lines.append(
            f"| {row['case']} | {float(row['nrmse_delta_refined_minus_coarse']):.12g} | "
            f"{float(row['nrmse_change_percent']):.6g} | "
            f"{float(row['correlation_delta_refined_minus_coarse']):.12g} | "
            f"{float(row['abs_peak_time_error_delta_s']):.12g} | "
            f"{float(row['abs_peak_time_error_change_percent']):.6g} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Reducing dx from 0.25 m to 0.125 m reduces raw NRMSE and improves correlation for both top grid monitors. The absolute peak-time error also drops strongly, supporting numerical dispersion / grid discretization as a primary source of the remaining phase difference.",
        "",
        "The shifted NRMSE is not monotonic after refinement because it measures residual waveform mismatch after an independently optimized time shift; use the raw NRMSE, correlation, and peak-time-error columns for the mesh-refinement conclusion.",
        "",
        "## Files",
        "",
        f"- combined metrics: `{OUT / 'latest_2d_dx_refinement_phase_summary.csv'}`",
        f"- refinement deltas: `{OUT / 'latest_2d_dx_refinement_delta_summary.csv'}`",
        f"- artifact audit: `{OUT / 'latest_2d_dx_refinement_artifact_audit.csv'}`",
        f"- top grid comparison: `{OUT / 'latest_2d_dx_refinement_top_grid_monitor_vs_flac3d.png'}`",
        f"- FLAC free-field reference: `{REF / 'reference_fig_3_9_flac_free_correct.csv'}`",
        f"- FLAC main reference: `{REF / 'reference_fig_3_9_flac_main_correct.csv'}`",
    ]
    if bad:
        lines += ["", "## Failed Checks", "", "| dx | check | target | status |", "| ---: | --- | --- | --- |"]
        for row in bad:
            lines.append(f"| {row['dx']} | {row['check']} | `{row['target']}` | {row['status']} |")

    (OUT / "latest_2d_dx_refinement_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    phase_rows = collect_phase_rows()
    audit_rows = collect_audit_rows()
    delta_rows = collect_delta_rows(phase_rows)

    write_csv(
        OUT / "latest_2d_dx_refinement_phase_summary.csv",
        [
            "dx",
            "case",
            "nrmse",
            "correlation",
            "peak_time_error_s",
            "best_time_shift_s",
            "nrmse_after_time_shift",
            "peak_error_m_s",
        ],
        phase_rows,
    )
    write_csv(
        OUT / "latest_2d_dx_refinement_artifact_audit.csv",
        ["dx", "run_dir", "check", "target", "status", "detail"],
        audit_rows,
    )
    write_csv(
        OUT / "latest_2d_dx_refinement_delta_summary.csv",
        [
            "case",
            "coarse_dx",
            "refined_dx",
            "nrmse_delta_refined_minus_coarse",
            "nrmse_change_percent",
            "correlation_delta_refined_minus_coarse",
            "abs_peak_time_error_delta_s",
            "abs_peak_time_error_change_percent",
        ],
        delta_rows,
    )
    plot_top_grid_comparison()
    write_summary(phase_rows, audit_rows, delta_rows)
    print(f"summary={OUT / 'latest_2d_dx_refinement_summary.md'}")
    print(f"audit={OUT / 'latest_2d_dx_refinement_artifact_audit.csv'}")


if __name__ == "__main__":
    main()
