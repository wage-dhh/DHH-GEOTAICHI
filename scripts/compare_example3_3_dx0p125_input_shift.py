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
        "label": "dx=0.125, input shift 0.000800",
        "short_label": "shift=0.000800",
        "input_time_shift": 0.000800,
        "run_dir": OUT / "latest_2d_dx0p125_height4m_centered_v_platform_top_monitors",
    },
    {
        "label": "dx=0.125, input shift 0.000624",
        "short_label": "shift=0.000624",
        "input_time_shift": 0.000624,
        "run_dir": OUT / "latest_2d_dx0p125_input_shift_0p000624",
    },
]

CASES = [
    "soil_grid_top vs FLAC3D free-field",
    "main_grid_top_manual vs FLAC3D main",
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


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in RUNS:
        phase_path = run["run_dir"] / "phase_alignment_summary.csv"
        for row in read_csv_dicts(phase_path):
            if row["case"] not in CASES:
                continue
            rows.append(
                {
                    "input_time_shift": run["input_time_shift"],
                    "run_dir": run["run_dir"],
                    "case": row["case"],
                    "peak_error": row["peak_error"],
                    "nrmse": row["nrmse"],
                    "correlation": row["correlation"],
                    "peak_time_error_s": row["peak_time_error"],
                    "best_time_shift_s": row["best_time_shift"],
                    "nrmse_after_time_shift": row["nrmse_after_time_shift"],
                    "first_arrival_time_error_s": row["first_arrival_time_error"],
                }
            )
    return rows


def plot_comparison() -> None:
    free_t, free_v = read_xy(REF / "reference_fig_3_9_flac_free_correct.csv", "time_s", "vx")
    main_t, main_v = read_xy(REF / "reference_fig_3_9_flac_main_correct.csv", "time_s", "vx")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    axes[0].plot(free_t, free_v, "k-", linewidth=1.8, label="FLAC3D free-field")
    axes[1].plot(main_t, main_v, "k-", linewidth=1.8, label="FLAC3D main")

    styles = ["--", "-."]
    colors = ["#2a9d55", "#d1493f"]
    for run, style, color in zip(RUNS, styles, colors):
        path = run["run_dir"] / "grid_monitor_velocity.csv"
        t, soil = read_xy(path, "time", "soil_grid_top_vx")
        _, main = read_xy(path, "time", "main_grid_top_manual_vx")
        axes[0].plot(t, soil, linestyle=style, color=color, linewidth=1.5, label=run["short_label"])
        axes[1].plot(t, main, linestyle=style, color=color, linewidth=1.5, label=run["short_label"])

    axes[0].set_title("Free-field top grid monitor")
    axes[1].set_title("Main top grid monitor")
    for ax in axes:
        ax.set_xlim(0.0, 0.015)
        ax.set_ylim(-0.02, 0.12)
        ax.set_ylabel("grid vx (m/s)")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "latest_2d_dx0p125_input_shift_comparison.png", dpi=200)
    plt.close(fig)


def write_report(rows: list[dict[str, object]]) -> None:
    by_case: dict[str, dict[float, dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), {})[float(row["input_time_shift"])] = row

    lines = [
        "# dx=0.125 input time-shift comparison",
        "",
        "This report compares the original `EX33_INPUT_TIME_SHIFT=0.000800 s` run with a retuned `0.000624 s` run. The retuned value comes from a common-delay diagnostic on the top free-field and main grid monitors.",
        "",
        "| input time shift | case | NRMSE | correlation | peak time error (s) | best time shift (s) | peak error (m/s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {float(row['input_time_shift']):.6f} | {row['case']} | {float(row['nrmse']):.12g} | "
            f"{float(row['correlation']):.12g} | {float(row['peak_time_error_s']):.12g} | "
            f"{float(row['best_time_shift_s']):.12g} | {float(row['peak_error']):.12g} |"
        )

    lines += [
        "",
        "## Changes",
        "",
        "| case | NRMSE change | correlation change | abs peak-time-error change (s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for case, values in by_case.items():
        old = values[0.000800]
        new = values[0.000624]
        lines.append(
            f"| {case} | {float(new['nrmse']) - float(old['nrmse']):.12g} | "
            f"{float(new['correlation']) - float(old['correlation']):.12g} | "
            f"{abs(float(new['peak_time_error_s'])) - abs(float(old['peak_time_error_s'])):.12g} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The retuned input shift reduces the remaining phase lead in the raw top-grid comparison. It improves the free-field top NRMSE substantially and slightly improves the main top NRMSE. The main top peak-time error changes sign, so `0.000624 s` is a compromise shift rather than a perfect exact match for both curves.",
        "",
        "## Files",
        "",
        f"- comparison CSV: `{OUT / 'latest_2d_dx0p125_input_shift_comparison.csv'}`",
        f"- comparison figure: `{OUT / 'latest_2d_dx0p125_input_shift_comparison.png'}`",
        f"- retuned run: `{RUNS[1]['run_dir']}`",
    ]
    (OUT / "latest_2d_dx0p125_input_shift_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    write_csv(
        OUT / "latest_2d_dx0p125_input_shift_comparison.csv",
        [
            "input_time_shift",
            "run_dir",
            "case",
            "peak_error",
            "nrmse",
            "correlation",
            "peak_time_error_s",
            "best_time_shift_s",
            "nrmse_after_time_shift",
            "first_arrival_time_error_s",
        ],
        rows,
    )
    plot_comparison()
    write_report(rows)
    print(f"report={OUT / 'latest_2d_dx0p125_input_shift_comparison.md'}")
    print(f"figure={OUT / 'latest_2d_dx0p125_input_shift_comparison.png'}")


if __name__ == "__main__":
    main()
