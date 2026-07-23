import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference" / "flac3d_example3_3_free_field"
OUT = ROOT / "output" / "example3_3_2d_kohler_native_mpm_solver"

PAIRS = [
    (
        "legacy",
        REF / "reference_fig_3_9_flac_free.csv",
        REF / "reference_fig_3_9_flac_main.csv",
    ),
    (
        "corrected",
        REF / "reference_fig_3_9_flac_free_correct.csv",
        REF / "reference_fig_3_9_flac_main_correct.csv",
    ),
]


def read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return (
        np.asarray([float(r["time_s"]) for r in rows], dtype=float),
        np.asarray([float(r["vx"]) for r in rows], dtype=float),
    )


def main() -> None:
    rows: list[dict[str, object]] = []
    for label, free_path, main_path in PAIRS:
        free_t, free_v = read_xy(free_path)
        main_t, main_v = read_xy(main_path)
        main_on_free = np.interp(free_t, main_t, main_v)
        diff = free_v - main_on_free
        max_abs_diff = float(np.max(np.abs(diff)))
        rmse = math.sqrt(float(np.mean(diff * diff)))
        status = "FAIL_IDENTICAL" if max_abs_diff <= 1.0e-14 else "PASS_DISTINCT"
        rows.append(
            {
                "reference_set": label,
                "free_file": free_path,
                "main_file": main_path,
                "free_peak": float(np.max(np.abs(free_v))),
                "main_peak": float(np.max(np.abs(main_v))),
                "max_abs_free_minus_main": max_abs_diff,
                "rmse_free_minus_main": rmse,
                "status": status,
            }
        )

    csv_path = OUT / "latest_2d_flac_reference_curve_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# FLAC reference curve check",
        "",
        "| reference set | max abs free-main | RMSE free-main | status |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['reference_set']} | {float(row['max_abs_free_minus_main']):.12g} | "
            f"{float(row['rmse_free_minus_main']):.12g} | {row['status']} |"
        )
    lines += [
        "",
        "Use the `corrected` reference files for Figure 3.9 comparisons. The legacy free/main CSV pair is identical and is not suitable for distinguishing FLAC3D free-field and main responses.",
        "",
        f"- CSV: `{csv_path}`",
    ]
    md_path = OUT / "latest_2d_flac_reference_curve_check.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report={md_path}")


if __name__ == "__main__":
    main()
