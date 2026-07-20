from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "data/reference/flac3d_example3_3_free_field"
CURRENT = ROOT / "output/example3_3_flac3d_free_field_ppc1_velocity_input_bottom1_rerun"
FIG_DIR = CURRENT / "validation/figure_3_9_only"
SOURCE = ROOT / "output/example3_3_2d_kohler/reference_digitization/flac3d_fig3_9_digitized_with_free.csv"
SOURCE_QUALITY = ROOT / "output/example3_3_2d_kohler/reference_digitization/digitization_quality_check.csv"
SOURCE_TRACE = ROOT / "output/example3_3_2d_kohler/final_fix/reference_loader_trace.csv"

OLD_MAIN = REF_DIR / "reference_fig_3_9_flac_main.csv"
OLD_FREE = REF_DIR / "reference_fig_3_9_flac_free.csv"
NEW_MAIN = REF_DIR / "reference_fig_3_9_flac_main_correct.csv"
NEW_FREE = REF_DIR / "reference_fig_3_9_flac_free_correct.csv"
REPORT = CURRENT / "flac3d_reference_repair_report.md"
RAW_DIFF = CURRENT / "flac3d_main_vs_free_correct_raw.csv"
FIGURE = FIG_DIR / "figure_3_9_comparison_correct.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_reference(path: Path, time: np.ndarray, velocity: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "vx"])
        writer.writeheader()
        for t, v in zip(time, velocity):
            writer.writerow({"time_s": float(t), "vx": float(v)})


def load_history(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rows = read_rows(path)
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for point_id in sorted({int(row["point_id"]) for row in rows}):
        selected = [row for row in rows if int(row["point_id"]) == point_id]
        result[point_id] = (
            np.asarray([float(row["time"]) for row in selected], dtype=float),
            np.asarray([float(row["vx"]) for row in selected], dtype=float),
        )
    return result


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(SOURCE)
    time = np.asarray([float(row["time"]) for row in rows], dtype=float)
    main = np.asarray([float(row["flac_main_xvel"]) for row in rows], dtype=float)
    free = np.asarray([float(row["flac_free_xvel"]) for row in rows], dtype=float)

    write_reference(NEW_MAIN, time, main)
    write_reference(NEW_FREE, time, free)

    difference = main - free
    max_diff = float(np.max(np.abs(difference)))
    mean_diff = float(np.mean(np.abs(difference)))
    can_plot = max_diff >= 1.0e-12

    with RAW_DIFF.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["time", "flac_main_vx_correct", "flac_free_vx_correct", "difference"],
        )
        writer.writeheader()
        for t, m, ff, d in zip(time, main, free, difference):
            writer.writerow(
                {
                    "time": float(t),
                    "flac_main_vx_correct": float(m),
                    "flac_free_vx_correct": float(ff),
                    "difference": float(d),
                }
            )

    if can_plot:
        history = load_history(CURRENT / "dynamic/velocity_history.csv")
        main_t, main_mpm = history[0]
        free_t, free_mpm = history[1]
        plt.rcParams.update(
            {
                "font.family": "Times New Roman",
                "font.size": 11,
                "axes.linewidth": 1.0,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )
        plt.figure(figsize=(7.2, 4.8))
        plt.plot(time, main, color="black", linewidth=1.8, label="FLAC3D main top corrected")
        plt.plot(main_t, main_mpm, color="red", linewidth=1.5, linestyle="--", label="GeoTaichi current main top")
        plt.plot(time, free, color="0.35", linewidth=1.8, label="FLAC3D free-field top corrected")
        plt.plot(free_t, free_mpm, color="blue", linewidth=1.5, linestyle="--", label="GeoTaichi current free-field top")
        plt.title("FLAC3D Example 3.3 Figure 3.9 reproduction (corrected reference)")
        plt.xlabel("Time (s)")
        plt.ylabel("Horizontal velocity vx (m/s)")
        plt.grid(True, alpha=0.25, linewidth=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(FIGURE, dpi=300)
        plt.close()

    quality = SOURCE_QUALITY.read_text(encoding="utf-8") if SOURCE_QUALITY.exists() else "NOT_FOUND"
    trace = SOURCE_TRACE.read_text(encoding="utf-8") if SOURCE_TRACE.exists() else "NOT_FOUND"
    old_main_hash = sha256(OLD_MAIN) if OLD_MAIN.exists() else "MISSING"
    old_free_hash = sha256(OLD_FREE) if OLD_FREE.exists() else "MISSING"
    new_main_hash = sha256(NEW_MAIN)
    new_free_hash = sha256(NEW_FREE)

    lines = [
        "# FLAC3D Reference Repair Report",
        "",
        "## Scope",
        "",
        "- MPM result modification: NO",
        "- Dynamic rerun: NO",
        "- Original wrong reference files overwritten: NO",
        "- New corrected reference files were generated with `_correct.csv` suffix.",
        "",
        "## Original Reference Error",
        "",
        f"- original main reference: `{OLD_MAIN}`",
        f"- original free reference: `{OLD_FREE}`",
        f"- original main SHA256: `{old_main_hash}`",
        f"- original free SHA256: `{old_free_hash}`",
        "- error cause: the two original reference CSV files are byte-identical, so Figure 3.9 plotted identical FLAC3D main/free curves.",
        "",
        "## New History Source",
        "",
        f"- source file: `{SOURCE}`",
        "- source type: existing digitized Figure 3.9 data with independent columns `flac_main_xvel` and `flac_free_xvel`.",
        "- note: no native FLAC3D binary/text history export with named histories was found in this workspace during this repair. This repair therefore uses the existing independent digitized history output.",
        f"- source quality file: `{SOURCE_QUALITY}`",
        f"- source loader trace: `{SOURCE_TRACE}`",
        "",
        "## New Reference Files",
        "",
        f"- corrected main: `{NEW_MAIN}`",
        f"- corrected free: `{NEW_FREE}`",
        f"- corrected main SHA256: `{new_main_hash}`",
        f"- corrected free SHA256: `{new_free_hash}`",
        "",
        "## History Coordinates",
        "",
        "- FLAC3D main top history command: `hist gp xvel 2 1 5.0`",
        "- FLAC3D main top coordinate: x=2.0, y=1.0, z=5.0",
        "- FLAC3D free-field top history command used for repaired file: `hist gp xvel -1 0 5.0`",
        "- FLAC3D free-field top coordinate: x=-1.0, y=0.0, z=5.0",
        "- caveat: the digitized source file does not embed coordinate metadata; coordinate assignment follows the documented Example 3.3 history mapping request.",
        "",
        "## Main-Free Difference Check",
        "",
        f"- max(abs(main-free)) = `{max_diff}`",
        f"- mean(abs(main-free)) = `{mean_diff}`",
        "- threshold = `1e-12`",
        f"- can use for Figure 3.9 = `{'YES' if can_plot else 'NO'}`",
        f"- raw corrected difference CSV: `{RAW_DIFF}`",
        "",
        "## Figure Output",
        "",
        f"- figure generated: `{'YES' if can_plot else 'NO'}`",
        f"- figure path: `{FIGURE if can_plot else 'NOT_GENERATED_BECAUSE_MAIN_FREE_IDENTICAL'}`",
        "",
        "## Embedded Source Quality Summary",
        "",
        "```csv",
        quality.strip(),
        "```",
        "",
        "## Embedded Loader Trace",
        "",
        "```csv",
        trace.strip(),
        "```",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"new_main={NEW_MAIN}")
    print(f"new_free={NEW_FREE}")
    print(f"max_diff={max_diff}")
    print(f"figure={FIGURE if can_plot else 'NOT_GENERATED'}")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
