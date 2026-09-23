"""Plot Kohler failure snapshots coloured by per-particle plastic strain.

The failure configurations store ``state_vars`` in each particle NPZ.  This
script selects the nearest saved state to requested dynamic times, filters the
main soil body, and writes a compact multi-panel figure plus a machine-readable
snapshot summary.  It does not modify simulation data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def snapshot_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "particles").glob("MPMParticle*.npz"))


def load_snapshot(path: Path, x_shift: float, y_shift: float) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=True) as data:
        required = {"t_current", "position", "state_vars"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing failure fields: {sorted(missing)}")
        position = np.asarray(data["position"], dtype=np.float64)
        state = data["state_vars"].item()
        if "epstrain" not in state:
            raise ValueError(f"{path} state_vars has no epstrain field")
        epstrain = np.asarray(state["epstrain"], dtype=np.float64).reshape(-1)
        body_id = np.asarray(data["bodyID"], dtype=np.int32) if "bodyID" in data else np.zeros(position.shape[0], dtype=np.int32)
        material_id = np.asarray(data["materialID"], dtype=np.int32) if "materialID" in data else np.ones(position.shape[0], dtype=np.int32)
        time_value = float(np.asarray(data["t_current"]).reshape(()))
    return {
        "time": time_value,
        "position": position - np.array([x_shift, y_shift], dtype=np.float64),
        "epstrain": epstrain,
        "body_id": body_id,
        "material_id": material_id,
    }


def nearest_snapshot(files: list[Path], target: float, x_shift: float, y_shift: float) -> tuple[Path, dict[str, np.ndarray | float]]:
    candidates: list[tuple[float, Path, dict[str, np.ndarray | float]]] = []
    for path in files:
        snapshot = load_snapshot(path, x_shift, y_shift)
        candidates.append((abs(float(snapshot["time"]) - target), path, snapshot))
    if not candidates:
        raise FileNotFoundError("No particle snapshots found")
    _, path, snapshot = min(candidates, key=lambda item: item[0])
    return path, snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--times", nargs="+", type=float, required=True, help="Requested dynamic/absolute snapshot times in seconds")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--dynamic-start", type=float, default=0.0, help="Add this absolute-time offset when selecting requested dynamic times")
    parser.add_argument("--x-shift", type=float, default=46.0)
    parser.add_argument("--y-shift", type=float, default=38.0)
    parser.add_argument("--body-id", type=int, default=0)
    parser.add_argument("--material-id", type=int, default=1)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    files = snapshot_files(run_dir)
    if not files:
        raise FileNotFoundError(f"No particle snapshots under {run_dir / 'particles'}")
    selected: list[tuple[float, Path, dict[str, np.ndarray | float]]] = []
    for target in args.times:
        path, snapshot = nearest_snapshot(files, target + args.dynamic_start, args.x_shift, args.y_shift)
        selected.append((target, path, snapshot))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )

    filtered: list[tuple[float, Path, dict[str, np.ndarray | float], np.ndarray]] = []
    maxima: list[float] = []
    for target, path, snapshot in selected:
        mask = (np.asarray(snapshot["body_id"]) == args.body_id) & (np.asarray(snapshot["material_id"]) == args.material_id)
        values = np.abs(np.asarray(snapshot["epstrain"], dtype=np.float64))
        positions = np.asarray(snapshot["position"], dtype=np.float64)
        if values.shape[0] != positions.shape[0]:
            raise ValueError(f"State/position length mismatch in {path}")
        if not np.any(mask):
            raise ValueError(f"No particles match body_id={args.body_id}, material_id={args.material_id} in {path}")
        maxima.append(float(np.max(values[mask])))
        filtered.append((target, path, snapshot, mask))

    vmax = max(maxima + [1.0e-12])
    cols = min(3, len(filtered))
    rows = int(np.ceil(len(filtered) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.0 * rows), squeeze=False, constrained_layout=True)
    scatter = None
    summary_rows: list[dict[str, float | int | str]] = []
    for index, (target, path, snapshot, mask) in enumerate(filtered):
        ax = axes[index // cols][index % cols]
        positions = np.asarray(snapshot["position"], dtype=np.float64)[mask]
        values = np.abs(np.asarray(snapshot["epstrain"], dtype=np.float64))[mask]
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=values, s=2.2, linewidths=0.0, cmap="inferno", vmin=0.0, vmax=vmax, rasterized=True)
        actual_time = float(snapshot["time"])
        ax.set_title(f"target {target:g} s, saved {actual_time:.4g} s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15, linewidth=0.4)
        summary_rows.append(
            {
                "target_time": target,
                "saved_time": actual_time,
                "file": str(path),
                "particle_count": int(np.count_nonzero(mask)),
                "plastic_particle_count": int(np.count_nonzero(values > 1.0e-12)),
                "max_epstrain": float(np.max(values)),
                "mean_epstrain": float(np.mean(values)),
            }
        )
    for index in range(len(filtered), rows * cols):
        axes[index // cols][index % cols].axis("off")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label="|equivalent plastic strain|")

    output = args.output or (run_dir / "kohler_failure_snapshots.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = args.summary or (run_dir / "kohler_failure_snapshots.csv")
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"figure={output}")
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
