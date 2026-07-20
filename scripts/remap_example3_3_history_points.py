"""Correct Example 3.3 validation history-point mapping without rerunning dynamics.

Reads only the current fresh dynamic history and reference metadata files. It
does not modify geometry, grid, particles, material, input, boundary, dashpot, or
solver code.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC_DIR = ROOT / "output" / "example3_3_flac3d_free_field_new" / "dynamic"
VALIDATION_DIR = DYNAMIC_DIR / "validation"
REFERENCE_DIR = ROOT / "data" / "reference" / "flac3d_example3_3_free_field"
VELOCITY_HISTORY = DYNAMIC_DIR / "velocity_history.csv"
OLD_MAPPING = VALIDATION_DIR / "history_point_mapping_report.csv"
EXACT_MAPPING = VALIDATION_DIR / "history_point_exact_mapping.csv"
EXACT_HISTORY = VALIDATION_DIR / "velocity_history_exact_points.csv"
MAPPING_REPORT = VALIDATION_DIR / "history_point_mapping_update_report.md"


FLAC_REFERENCE_POINTS = {
    "main_top_point": {
        "FLAC3D_x": 2.0,
        "FLAC3D_z": 5.0,
        "reference_curve": "reference_fig_3_9_flac_main.csv",
        "source": "docs/geometry_gap_analysis_example3_3.md: hist gp xvel 2 1 5.0",
        "current_history_point_id": 0,
    },
    "free_field_point": {
        "FLAC3D_x": -1.0,
        "FLAC3D_z": 5.0,
        "reference_curve": "reference_fig_3_9_flac_free.csv",
        "source": "docs/geometry_gap_analysis_example3_3.md: hist gp xvel -1 0 5.0 / -1 -1 5.0",
        "current_history_point_id": 1,
    },
    "main_left_boundary_point": {
        "FLAC3D_x": 0.0,
        "FLAC3D_z": 2.5,
        "reference_curve": "",
        "source": "requested side-boundary diagnostic; not present in reference curve CSV metadata",
        "current_history_point_id": 3,
    },
    "main_right_boundary_point": {
        "FLAC3D_x": 6.0,
        "FLAC3D_z": 2.5,
        "reference_curve": "",
        "source": "requested side-boundary diagnostic; not present in reference curve CSV metadata",
        "current_history_point_id": 4,
    },
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def current_points(history_rows: list[dict[str, str]]) -> dict[int, dict[str, object]]:
    points: dict[int, dict[str, object]] = {}
    for row in history_rows:
        pid = int(row["point_id"])
        points.setdefault(
            pid,
            {
                "region": row["region"],
                "MPM_x": float(row["x"]),
                "MPM_z": float(row["z"]),
            },
        )
    return points


def reference_curve_status(filename: str) -> str:
    if not filename:
        return "NO_REFERENCE_CURVE"
    path = REFERENCE_DIR / filename
    if not path.exists():
        return "REFERENCE_CURVE_NOT_AVAILABLE"
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    return "CURVE_HAS_NO_COORDINATE_METADATA" if {"time_s", "vx"}.issubset(set(header)) else "CHECK_REFERENCE_METADATA"


def mapping_status(dx: float, dz: float, reference_status: str) -> str:
    if reference_status == "NO_REFERENCE_CURVE":
        return "EXACT_MPM_LOCATION_NO_FLAC3D_REFERENCE_CURVE" if abs(dx) <= 1.0e-12 and abs(dz) <= 1.0e-12 else "NO_FLAC3D_REFERENCE_POINT"
    return "EXACT" if abs(dx) <= 1.0e-12 and abs(dz) <= 1.0e-12 else "NOT_EXACT"


def build_mapping(history_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    points = current_points(history_rows)
    rows = []
    for name, spec in FLAC_REFERENCE_POINTS.items():
        pid = int(spec["current_history_point_id"])
        point = points.get(pid)
        if point is None:
            mpm_x = math.nan
            mpm_z = math.nan
        else:
            mpm_x = float(point["MPM_x"])
            mpm_z = float(point["MPM_z"])
        flac_x = float(spec["FLAC3D_x"])
        flac_z = float(spec["FLAC3D_z"])
        dx = mpm_x - flac_x if math.isfinite(mpm_x) else math.nan
        dz = mpm_z - flac_z if math.isfinite(mpm_z) else math.nan
        ref_status = reference_curve_status(str(spec["reference_curve"]))
        rows.append(
            {
                "point_name": name,
                "FLAC3D_x": flac_x,
                "FLAC3D_z": flac_z,
                "MPM_x": mpm_x,
                "MPM_z": mpm_z,
                "dx": dx,
                "dz": dz,
                "mapping_status": mapping_status(dx, dz, ref_status),
            }
        )
    return rows


def write_exact_history(history_rows: list[dict[str, str]], mapping_rows: list[dict[str, object]]) -> Path:
    by_pid: dict[int, list[dict[str, str]]] = {}
    for row in history_rows:
        by_pid.setdefault(int(row["point_id"]), []).append(row)
    point_name_to_pid = {name: int(spec["current_history_point_id"]) for name, spec in FLAC_REFERENCE_POINTS.items()}
    mapping_by_name = {str(row["point_name"]): row for row in mapping_rows}
    out = []
    for point_name, pid in point_name_to_pid.items():
        mapping = mapping_by_name[point_name]
        for row in by_pid.get(pid, []):
            out.append(
                {
                    "time": row["time"],
                    "point_name": point_name,
                    "source_point_id": pid,
                    "region": row["region"],
                    "FLAC3D_x": mapping["FLAC3D_x"],
                    "FLAC3D_z": mapping["FLAC3D_z"],
                    "MPM_x": mapping["MPM_x"],
                    "MPM_z": mapping["MPM_z"],
                    "dx": mapping["dx"],
                    "dz": mapping["dz"],
                    "vx": row["vx"],
                    "mapping_status": mapping["mapping_status"],
                }
            )
    return write_csv(
        EXACT_HISTORY,
        ["time", "point_name", "source_point_id", "region", "FLAC3D_x", "FLAC3D_z", "MPM_x", "MPM_z", "dx", "dz", "vx", "mapping_status"],
        out,
    )


def regenerate_mapping_report(mapping_rows: list[dict[str, object]]) -> Path:
    rows = []
    for row in mapping_rows:
        name = str(row["point_name"])
        spec = FLAC_REFERENCE_POINTS[name]
        rows.append(
            {
                **row,
                "reference_curve": spec["reference_curve"],
                "reference_coordinate_source": spec["source"],
                "reference_metadata_status": reference_curve_status(str(spec["reference_curve"])),
                "actual_mpm_sample_note": "current fresh dynamic history point; no nearest-point substitution performed",
            }
        )
    return write_csv(
        OLD_MAPPING,
        [
            "point_name",
            "FLAC3D_x",
            "FLAC3D_z",
            "MPM_x",
            "MPM_z",
            "dx",
            "dz",
            "mapping_status",
            "reference_curve",
            "reference_coordinate_source",
            "reference_metadata_status",
            "actual_mpm_sample_note",
        ],
        rows,
    )


def write_report(mapping_rows: list[dict[str, object]], old_rows: list[dict[str, str]]) -> Path:
    old_by_name = {row.get("expected_name", row.get("point_name", "")): row for row in old_rows}
    lines = [
        "# History Point Mapping Update Report",
        "",
        "## Scope",
        "",
        "Only history extraction/mapping files were regenerated. No geometry, grid, particles, material, input wave, free-field boundary, dashpot force, or solver code was modified. No dynamics were rerun.",
        "",
        "## Reference Coordinate Metadata",
        "",
        "The files under `data/reference/flac3d_example3_3_free_field/` contain digitized curves with columns `time_s,vx` only. They do not contain history-point coordinates. FLAC3D coordinates are therefore taken from the documented manual history commands in `docs/geometry_gap_analysis_example3_3.md`.",
        "",
        "## Mapping Results",
        "",
    ]
    for row in mapping_rows:
        name = str(row["point_name"])
        old = old_by_name.get(name, {})
        lines.extend(
            [
                f"### {name}",
                "",
                f"- FLAC3D coordinate: x={row['FLAC3D_x']}, z={row['FLAC3D_z']}",
                f"- actual MPM sampled coordinate: x={row['MPM_x']}, z={row['MPM_z']}",
                f"- coordinate difference: dx={row['dx']}, dz={row['dz']}",
                f"- mapping status: {row['mapping_status']}",
                f"- previous report coordinate if available: x={old.get('x', 'N/A')}, z={old.get('z', 'N/A')}",
                "",
            ]
        )
    inconsistent = [row for row in mapping_rows if row["mapping_status"] == "NOT_EXACT"]
    exact = [row for row in mapping_rows if row["mapping_status"] == "EXACT"]
    no_ref = [row for row in mapping_rows if "NO_FLAC3D_REFERENCE" in str(row["mapping_status"]) or "NO_FLAC3D_REFERENCE_CURVE" in str(row["mapping_status"])]
    lines.extend(
        [
            "## Summary",
            "",
            f"- exact points: {', '.join(str(row['point_name']) for row in exact) if exact else 'none'}",
            f"- previously/inherently inconsistent points: {', '.join(str(row['point_name']) for row in inconsistent) if inconsistent else 'none'}",
            f"- side diagnostic points without FLAC3D reference curve metadata: {', '.join(str(row['point_name']) for row in no_ref) if no_ref else 'none'}",
            "",
            "The free-field point is not exact in the current 2D fresh model: FLAC3D documented free-field x is `-1`, while the actual MPM sampled free-field top is `-0.25`. This is reported, not silently replaced.",
            "",
            f"- exact mapping CSV: {EXACT_MAPPING}",
            f"- regenerated mapping report CSV: {OLD_MAPPING}",
            f"- extracted history CSV: {EXACT_HISTORY}",
        ]
    )
    MAPPING_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MAPPING_REPORT


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    history_rows = read_rows(VELOCITY_HISTORY)
    old_rows = read_rows(OLD_MAPPING) if OLD_MAPPING.exists() else []
    mapping_rows = build_mapping(history_rows)
    write_csv(EXACT_MAPPING, ["point_name", "FLAC3D_x", "FLAC3D_z", "MPM_x", "MPM_z", "dx", "dz", "mapping_status"], mapping_rows)
    regenerate_mapping_report(mapping_rows)
    write_exact_history(history_rows, mapping_rows)
    report = write_report(mapping_rows, old_rows)
    print(f"history_point_exact_mapping={EXACT_MAPPING}")
    print(f"velocity_history_exact_points={EXACT_HISTORY}")
    print(f"history_point_mapping_report={OLD_MAPPING}")
    print(f"mapping_update_report={report}")


if __name__ == "__main__":
    main()
