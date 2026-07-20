"""Main-grid-only bottom input check for FLAC3D Example 3.3.

This is a diagnostic control, not a replacement for the full Example 3.3 run.
It keeps the Example 3.3 main-grid bottom footprint, material impedance, and
input wave, but removes x-side/y-side/corner free-field bodies and Fff coupling.
The purpose is to isolate whether a single main-grid compliant base can match
the target bottom velocity without multi-body interactions.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "example/mpm/FLAC3D/results/example3_3_error_decomposition"
OUT_CSV = OUTPUT_DIR / "main_only_bottom_input_check.csv"
OUT_PNG = OUTPUT_DIR / "main_only_bottom_input_check.png"

DENSITY = 0.0025
SHEAR_MODULUS = 40000.0
CS = math.sqrt(SHEAR_MODULUS / DENSITY)
RHO_CS = DENSITY * CS
WAVE_PERIOD = 0.01
DSTRESS = 1.0
ELEMENT_SIZE = 1.0
MAIN_WIDTH_X = 6.0
MAIN_WIDTH_Y = 3.0
DT = 0.0001443373621441424
TIME_END = 0.015


def wave(t: float) -> float:
    if t < 0.0:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * t / WAVE_PERIOD))


def main_bottom_nodes() -> list[dict[str, float]]:
    rows = []
    for iy in range(1, 5):
        for ix in range(1, 8):
            ax = 0.5 if ix in (1, 7) else 1.0
            ay = 0.5 if iy in (1, 4) else 1.0
            area = ax * ay
            rows.append({"ix": ix, "iy": iy, "area": area, "mass": DENSITY * area * 0.5 * ELEMENT_SIZE, "vx": 0.0})
    return rows


def run() -> list[dict[str, float]]:
    nodes = main_bottom_nodes()
    rows = []
    times = np.arange(0.0, TIME_END + 0.5 * DT, DT)
    for t in times:
        dstress = DSTRESS * wave(float(t))
        for n in nodes:
            input_force = n["area"] * dstress
            dashpot_force = -RHO_CS * n["vx"] * n["area"]
            net = input_force + dashpot_force
            acc = net / n["mass"]
            # Explicit single-body control update.  The full GeoTaichi MPM path
            # has additional P2G/G2P and internal-force timing; this intentionally
            # isolates the bottom dashpot/input balance.
            n["vx"] += acc * DT
        area_sum = sum(n["area"] for n in nodes)
        mass_sum = sum(n["mass"] for n in nodes)
        area_weighted = sum(n["vx"] * n["area"] for n in nodes) / area_sum
        mass_weighted = sum(n["vx"] * n["mass"] for n in nodes) / mass_sum
        rows.append({
            "time": float(t),
            "wave": wave(float(t)),
            "theoretical_velocity": wave(float(t)) * DSTRESS / RHO_CS,
            "area_weighted_velocity": area_weighted,
            "mass_weighted_velocity": mass_weighted,
            "max_node_velocity": max(abs(n["vx"]) for n in nodes),
            "sum_area": area_sum,
            "sum_mass": mass_sum,
        })
    return rows


def write_csv(rows: list[dict[str, float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    t = [r["time"] for r in rows]
    plt.figure(figsize=(8, 4.6))
    plt.plot(t, [r["theoretical_velocity"] for r in rows], "k--", label="target")
    plt.plot(t, [r["area_weighted_velocity"] for r in rows], label="area-weighted")
    plt.plot(t, [r["mass_weighted_velocity"] for r in rows], label="mass-weighted")
    plt.xlabel("time")
    plt.ylabel("bottom vx")
    plt.title("Main-only bottom input control")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    plt.close()


def main() -> None:
    rows = run()
    write_csv(rows)
    plot(rows)
    peak = max(abs(r["area_weighted_velocity"]) for r in rows)
    target = DSTRESS / RHO_CS
    print({
        "area_weighted_velocity_peak": peak,
        "target_velocity_peak": target,
        "bottom_area_weighted_velocity_peak_error_percent": abs(peak - target) / target * 100.0,
        "csv": OUT_CSV.as_posix(),
        "png": OUT_PNG.as_posix(),
    })


if __name__ == "__main__":
    main()
