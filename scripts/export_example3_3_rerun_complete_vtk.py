"""Export ParaView-ready VTK files from existing fresh dynamic snapshots.

Postprocessing only: reads existing dynamic VTK/CSV outputs and writes a
reorganized vtk_paraview directory. It does not rerun dynamics or modify model
configuration.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import meshio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import examples.example3_3_2d_kohler_geometry_only as geom  # noqa: E402
from third_party.pyevtk.hl import gridToVTK, pointsToVTK, unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkLine


BASE = ROOT / "output" / "example3_3_flac3d_free_field_new_rerun"
DYNAMIC = BASE / "dynamic"
SRC_VTK = BASE / "vtk"
SRC_GRID = BASE / "grid" / "grid.vtr"
GRID_CHECK = BASE / "grid" / "geotaichi_grid_generation_check.csv"
SRC_PARTICLES = BASE / "geometry" / "particles_initial.vtu"
PAIR_CSV = BASE / "boundary" / "grid_interface_pair.csv"
FORCE_HISTORY = DYNAMIC / "free_field_force_history.csv"

OUT = DYNAMIC / "vtk_paraview"
GEOM_OUT = OUT / "geometry"
TIME_OUT = OUT / "time_steps"
REPORT = OUT / "vtk_export_report.md"
STEPS = sorted(int(path.stem.split("_")[1]) for path in SRC_VTK.glob("step_*.vtu"))


def read_pairs() -> list[dict[str, str]]:
    with PAIR_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def force_rows_by_step() -> dict[int, dict[int, dict[str, str]]]:
    target_times = {step: step * 2.5e-6 for step in STEPS}
    by_step = {step: {} for step in STEPS}
    with FORCE_HISTORY.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = float(row["time"])
            for step, target in target_times.items():
                if abs(t - target) <= 1.0e-12:
                    by_step[step][int(row["pair_id"])] = row
    return by_step


def read_grid_params() -> dict[str, float]:
    out: dict[str, float] = {}
    with GRID_CHECK.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["parameter"]] = float(row["value"])
    return out


def grid_axes(grid_params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(grid_params["domain_start_x"], grid_params["domain_end_x"], int(grid_params["num_grid_nodes_x"]))
    ys = np.asarray([0.0], dtype=float)
    zs = np.linspace(grid_params["domain_start_z"], grid_params["domain_end_z"], int(grid_params["num_grid_nodes_z"]))
    return xs, ys, zs


def cell_arrays(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> dict[str, np.ndarray]:
    shape = (xs.size - 1, max(ys.size - 1, 1), zs.size - 1)
    component = np.zeros(shape, dtype=np.int32)
    active = np.zeros(shape, dtype=np.int32)
    marker = np.zeros(shape, dtype=np.int32)
    pair_id = np.full(shape, -1, dtype=np.int32)
    gap_distance = np.zeros(shape, dtype=float)
    distance = np.zeros(shape, dtype=float)
    for ix in range(xs.size - 1):
        for iz in range(zs.size - 1):
            cx = 0.5 * (xs[ix] + xs[ix + 1])
            cz = 0.5 * (zs[iz] + zs[iz + 1])
            c, a, m, p, g, d = geom.classify_cell(float(cx), float(cz))
            component[ix, 0, iz] = c
            active[ix, 0, iz] = a
            marker[ix, 0, iz] = m
            pair_id[ix, 0, iz] = p
            gap_distance[ix, 0, iz] = g
            distance[ix, 0, iz] = d
    return {
        "Component Mask": component,
        "Active Cell": active,
        "Boundary Marker": marker,
        "Coupling Pair ID": pair_id,
        "Gap Distance": gap_distance,
        "Distance To Main": distance,
    }


def velocity_to_grid_arrays(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    grid_points: np.ndarray | None,
    grid_velocity: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npoints = xs.size * ys.size * zs.size
    if grid_velocity is None:
        vx = np.zeros((xs.size, ys.size, zs.size), dtype=float)
        vy = np.zeros_like(vx)
        vz = np.zeros_like(vx)
    else:
        vx = np.zeros((xs.size, ys.size, zs.size), dtype=float)
        vy = np.zeros_like(vx)
        vz = np.zeros_like(vx)
        x_index = {round(float(x), 12): i for i, x in enumerate(xs)}
        z_index = {round(float(z), 12): i for i, z in enumerate(zs)}
        assert grid_points is not None
        for point, vel in zip(grid_points, grid_velocity):
            ix = x_index[round(float(point[0]), 12)]
            iz = z_index[round(float(point[2]), 12)]
            vx[ix, 0, iz] = vel[0]
            vy[ix, 0, iz] = vel[1]
            vz[ix, 0, iz] = vel[2]
    mag = np.sqrt(vx * vx + vy * vy + vz * vz)
    return vx, vy, vz, mag


def write_grid_vtr(path_base: Path, grid_params: dict[str, float], grid_points: np.ndarray | None = None, grid_velocity: np.ndarray | None = None) -> Path:
    xs, ys, zs = grid_axes(grid_params)
    node_id = np.arange(xs.size * ys.size * zs.size, dtype=np.int32).reshape((xs.size, ys.size, zs.size), order="F")
    vx, vy, vz, mag = velocity_to_grid_arrays(xs, ys, zs, grid_points, grid_velocity)
    point_data = {
        "Node ID": np.ascontiguousarray(node_id),
        "grid_velocity_x": np.ascontiguousarray(vx),
        "grid_velocity_y": np.ascontiguousarray(vy),
        "grid_velocity_z": np.ascontiguousarray(vz),
        "velocity_magnitude": np.ascontiguousarray(mag),
    }
    cell_data = cell_arrays(xs, ys, zs)
    out = gridToVTK(str(path_base), xs, ys, zs, pointData=point_data, cellData=cell_data)
    return Path(out)


def combined_snapshot(step: int):
    return meshio.read(SRC_VTK / f"step_{step:06d}.vtu")


def flat(values: np.ndarray, dtype=float) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape(-1)


def split_snapshot(mesh) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    role = flat(mesh.point_data["role_0particle_1grid"], np.int32)
    particle_mask = role == 0
    grid_mask = role == 1
    p_points = np.asarray(mesh.points[particle_mask], dtype=float)
    g_points = np.asarray(mesh.points[grid_mask], dtype=float)
    pdata = {name: flat(values)[particle_mask] for name, values in mesh.point_data.items()}
    gdata = {name: flat(values)[grid_mask] for name, values in mesh.point_data.items()}
    return p_points, g_points, pdata, gdata


def write_particles(path_base: Path, p_points: np.ndarray, pdata: dict[str, np.ndarray], material_id: np.ndarray | None = None) -> Path:
    n = p_points.shape[0]
    vx = flat(pdata.get("particle_velocity_x", np.zeros(n)), float)
    vz = flat(pdata.get("particle_velocity_z", np.zeros(n)), float)
    velocity_mag = np.sqrt(vx * vx + vz * vz)
    region = flat(pdata.get("component_id", np.ones(n, dtype=np.int32)), np.int32) - 1
    region = np.maximum(region, 0)
    if material_id is None:
        material_id = np.ones(n, dtype=np.int32)
    material_id = flat(material_id, np.int32)
    stress_xx = flat(pdata.get("stress_xx", np.zeros(n)), float)
    stress_zz = flat(pdata.get("stress_zz", np.zeros(n)), float)
    stress_xz = flat(pdata.get("stress_xz", np.zeros(n)), float)
    zeros = np.zeros(n, dtype=float)
    out = pointsToVTK(
        str(path_base),
        np.ascontiguousarray(p_points[:, 0]),
        np.ascontiguousarray(p_points[:, 1]),
        np.ascontiguousarray(p_points[:, 2]),
        data={
            "Particle ID": np.arange(n, dtype=np.int32),
            "Region ID": np.ascontiguousarray(region.astype(np.int32)),
            "Material ID": np.ascontiguousarray(material_id),
            "Velocity": (np.ascontiguousarray(vx), zeros, np.ascontiguousarray(vz)),
            "Velocity magnitude": np.ascontiguousarray(velocity_mag),
            "Displacement": (zeros, zeros, zeros),
            "stress_xx": np.ascontiguousarray(stress_xx),
            "stress_zz": np.ascontiguousarray(stress_zz),
            "stress_xz": np.ascontiguousarray(stress_xz),
        },
    )
    return Path(out)


def write_interface(path_base: Path, pairs: list[dict[str, str]], force_by_pair: dict[int, dict[str, str]] | None = None) -> Path:
    points: list[tuple[float, float, float]] = []
    conn: list[int] = []
    offsets: list[int] = []
    types: list[int] = []
    cell_pair = []
    cell_side = []
    cell_layer = []
    cell_fx = []
    cell_vdiff = []
    point_role = []
    for pair in pairs:
        p0 = len(points)
        points.append((float(pair["main_x"]), 0.0, float(pair["main_z"])))
        points.append((float(pair["ff_x"]), 0.0, float(pair["ff_z"])))
        point_role.extend([0, 1])
        conn.extend([p0, p0 + 1])
        offsets.append(len(conn))
        types.append(VtkLine.tid)
        pid = int(pair["pair_id"])
        force = (force_by_pair or {}).get(pid, {})
        vdiff = float(force.get("v_ff", 0.0)) - float(force.get("v_main", 0.0)) if force else 0.0
        cell_pair.append(pid)
        cell_side.append(0 if pair["side"] == "left" else 1)
        cell_layer.append(int(pair["layer_id"]))
        cell_fx.append(float(force.get("Fx_main", 0.0)) if force else 0.0)
        cell_vdiff.append(vdiff)
    pts = np.asarray(points, dtype=float)
    out = unstructuredGridToVTK(
        str(path_base),
        np.ascontiguousarray(pts[:, 0]),
        np.ascontiguousarray(pts[:, 1]),
        np.ascontiguousarray(pts[:, 2]),
        connectivity=np.asarray(conn, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int32),
        cell_types=np.asarray(types, dtype=np.uint8),
        pointData={"point_role_main0_ff1": np.asarray(point_role, dtype=np.int32)},
        cellData={
            "pair_id": np.asarray(cell_pair, dtype=np.int32),
            "side_left0_right1": np.asarray(cell_side, dtype=np.int32),
            "layer_id": np.asarray(cell_layer, dtype=np.int32),
            "Fx": np.asarray(cell_fx, dtype=float),
            "velocity_difference": np.asarray(cell_vdiff, dtype=float),
        },
    )
    return Path(out)


def validate_vtk(path: Path, expected_type: str, arrays: list[str]) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    header = path.read_bytes().split(b"<AppendedData", 1)[0].decode("utf-8", errors="ignore")
    if f'type="{expected_type}"' not in header:
        return False, f"wrong type, expected {expected_type}"
    missing = [name for name in arrays if f'Name="{name}"' not in header]
    if missing:
        return False, f"missing arrays {missing}"
    return True, "readable XML header and required arrays present"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    GEOM_OUT.mkdir(parents=True, exist_ok=True)
    TIME_OUT.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs()
    forces = force_rows_by_step()
    grid_params = read_grid_params()
    init_mesh = meshio.read(SRC_PARTICLES)
    material_id = np.asarray(init_mesh.point_data.get("material_id", np.ones(init_mesh.points.shape[0], dtype=np.int32)), dtype=np.int32)

    files: list[Path] = []
    step0 = combined_snapshot(0)
    p0, _, pdata0, _ = split_snapshot(step0)
    files.append(write_particles(GEOM_OUT / "model_particles", p0, pdata0, material_id))
    files.append(write_grid_vtr(GEOM_OUT / "background_grid", grid_params))
    files.append(write_interface(GEOM_OUT / "interface_pair", pairs, forces.get(0, {})))

    for step in STEPS:
        step_dir = TIME_OUT / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        mesh = combined_snapshot(step)
        p_points, g_points, pdata, gdata = split_snapshot(mesh)
        grid_velocity = np.column_stack([
            np.asarray(gdata["grid_velocity_x"], dtype=float),
            np.zeros(len(gdata["grid_velocity_x"]), dtype=float),
            np.asarray(gdata["grid_velocity_z"], dtype=float),
        ])
        files.append(write_particles(step_dir / "particles", p_points, pdata, material_id))
        files.append(write_grid_vtr(step_dir / "grid", grid_params, g_points, grid_velocity))

    checks = []
    for path in files:
        if path.suffix == ".vtu" and "particles" in path.name or path.name == "model_particles.vtu":
            checks.append((path, *validate_vtk(path, "UnstructuredGrid", ["Particle ID", "Region ID", "Material ID", "Velocity", "Velocity magnitude", "Stress" if False else "stress_xz"])))
        elif path.suffix == ".vtu":
            checks.append((path, *validate_vtk(path, "UnstructuredGrid", ["pair_id", "layer_id", "Fx", "velocity_difference"])))
        elif path.suffix == ".vtr":
            checks.append((path, *validate_vtk(path, "RectilinearGrid", ["grid_velocity_x", "velocity_magnitude", "Component Mask", "Active Cell", "Boundary Marker"])))

    xs, ys, zs = grid_axes(grid_params)
    n_grid = xs.size * ys.size * zs.size
    n_cells = (xs.size - 1) * max(ys.size - 1, 1) * (zs.size - 1)
    n_particles = p0.shape[0]
    lines = [
        "# ParaView VTK Export Report",
        "",
        "## Summary",
        "",
        f"- particle_count: {n_particles}",
        f"- grid_node_count: {n_grid}",
        f"- grid_cell_count: {n_cells}",
        f"- time_step_count: {len(STEPS)}",
        f"- output_directory: {OUT}",
        "",
        "## Checks",
        "",
        *(f"- {path}: {'PASS' if ok else 'FAIL'} - {msg}" for path, ok, msg in checks),
        "",
        "## Generated Files",
        "",
        *(f"- {path}" for path in files),
        "",
        "## Source Files",
        "",
        f"- {SRC_PARTICLES}",
        f"- {SRC_GRID}",
        f"- {PAIR_CSV}",
        *(f"- {SRC_VTK / f'step_{step:06d}.vtu'}" for step in STEPS),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"vtk_paraview={OUT}")
    print(f"report={REPORT}")
    for path in files:
        print(f"file={path}")


if __name__ == "__main__":
    main()
