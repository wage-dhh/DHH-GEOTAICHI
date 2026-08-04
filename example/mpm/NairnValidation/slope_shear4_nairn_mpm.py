"""Configurable NairnMPM-style seismic validation driver in GeoTaichi.

The default case reproduces the NairnMPM slope-shear4 validation setup.  The
driver can also load a JSON case file through ``--config`` or
``NAIRN_MPM_CONFIG`` to vary numerical settings, materials, regions, bodies,
boundary selectors, monitors, and output names without editing the solver
stream.

This command stream follows the structure of
``example/mpm/ColumnCollapse/DPmaterial2D.py`` and adds the dynamic boundary
logic needed by the NairnMPM input
``column-shear-validation3-freecolumn/slope-shear4.fmcmd``.

Coordinate note:
    Nairn's mesh starts at x=-1, y=-0.1. GeoTaichi regions are kept inside a
    positive domain, so all physical coordinates are shifted by (+1, +0.1).
    CSV diagnostics are written back in the original Nairn physical coordinates.
"""

import csv
import argparse
import copy
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mpm.mainMPM import MPM  # noqa: E402
from src.mpm.elements.QuadrilateralElement4Nodes import QuadrilateralElement4Nodes  # noqa: E402
from src.mpm.generator.BodyGenerator import BodyGenerator  # noqa: E402
from src.utils import GlobalVariable  # noqa: E402
from src.utils.TypeDefination import vec2f, vec2i  # noqa: E402


def init_taichi_mpm(dim: int = 2, arch: str = "cpu") -> None:
    """Initialize the MPM runtime without importing optional SDF dependencies."""

    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    GlobalVariable.DIMENSION = dim
    ti_arch = ti.cpu if arch.lower() == "cpu" else ti.gpu
    ti.init(arch=ti_arch, offline_cache=True, default_fp=ti.f64, default_ip=ti.i32, log_level=ti.ERROR)


def force_three_grid_levels(scene: Any, sims: Any) -> int:
    scene.grid_level = 3
    return scene.grid_level


def allow_three_grid_inputs(scene: Any, sims: Any, grid_level: int) -> None:
    return None


ORIGINAL_BODY_GENERATE = BodyGenerator.Generate
ORIGINAL_QUAD_CREATE_NODES = QuadrilateralElement4Nodes.create_nodes


def create_nodes_with_precise_grid_count(self: QuadrilateralElement4Nodes, sims: Any, grid_size: Any) -> None:
    if self.element_type != "Q4N2D":
        ORIGINAL_QUAD_CREATE_NODES(self, sims, grid_size)
        return

    domain = np.array(sims.domain, dtype=np.float64)
    requested_grid_size = np.array(grid_size, dtype=np.float64)
    if requested_grid_size.shape == (2,) and np.allclose(requested_grid_size, ELEMENT_SIZE, rtol=1.0e-5, atol=1.0e-12):
        requested_grid_size = ELEMENT_SIZE

    cnum = np.floor((domain + 1.0e-10) / requested_grid_size).astype(np.int32)
    cnum[cnum <= 0] = 1
    self.grid_size = vec2f(domain / cnum)
    self.igrid_size = 1.0 / self.grid_size
    self.cell_volume = self.calc_volume()
    self.cnum = vec2i(cnum)
    self.gnum = self.cnum + 1
    self.gridSum = int(self.gnum[0] * self.gnum[1])
    self.cellSum = int(self.cnum[0] * self.cnum[1])
    self.set_nodal_coords()


@ti.kernel
def kernel_place_one_particle_per_cell_2d(
    grid_size: ti.types.vector(2, float),
    start_point: ti.types.vector(2, float),
    nx: ti.i32,
    ny: ti.i32,
    particle: ti.template(),
    insert_particle_num: ti.template(),
    is_in_region: ti.template(),
):
    ti.loop_config(serialize=True)
    for linear_id in range(nx * ny):
        ix = linear_id % nx
        iy = linear_id // nx
        particle_pos = (ti.Vector([ti.cast(ix, float), ti.cast(iy, float)]) + 0.5) * grid_size + start_point
        if is_in_region(particle_pos):
            old_particle = ti.atomic_add(insert_particle_num[None], 1)
            particle[old_particle] = particle_pos


def generate_with_safe_single_particle_cell(self: BodyGenerator, scene: Any, region: Any, nParticlesPerCell: int) -> None:
    if scene.is_rectangle_cell() and self.sims.dimension == 2 and int(nParticlesPerCell) == 1:
        dx = float(scene.element.grid_size[0])
        dy = float(scene.element.grid_size[1])
        nx = max(1, int(round(float(region.region_size[0]) / dx)))
        ny = max(1, int(round(float(region.region_size[1]) / dy)))
        kernel_place_one_particle_per_cell_2d(
            scene.element.grid_size,
            region.start_point,
            nx,
            ny,
            self.particle,
            self.insert_particle_num,
            region.function,
        )
        return

    ORIGINAL_BODY_GENERATE(self, scene, region, nParticlesPerCell)


def install_safe_single_particle_generator() -> None:
    BodyGenerator.Generate = generate_with_safe_single_particle_cell


def install_precise_grid_count_patch() -> None:
    QuadrilateralElement4Nodes.create_nodes = create_nodes_with_precise_grid_count


# -------------------------- Nairn input equivalents -------------------------- #
X_SHIFT = 1.0
Y_SHIFT = 0.1

DX = float(os.environ.get("NAIRN_SHEAR_DX", "0.1"))
DOMAIN = np.array([8.0, 5.2], dtype=np.float64)
ELEMENT_SIZE = np.array([DX, DX], dtype=np.float64)

DT = float(os.environ.get("NAIRN_SHEAR_DT", "5e-6"))
SIMULATION_TIME = float(os.environ.get("NAIRN_SHEAR_TIME", "0.015"))
SAVE_INTERVAL = float(os.environ.get("NAIRN_SHEAR_SAVE_INTERVAL", "1e-3"))
HISTORY_INTERVAL = float(os.environ.get("NAIRN_SHEAR_HISTORY_INTERVAL", "1e-4"))
OUTPUT_DIR = Path(os.environ.get("NAIRN_SHEAR_OUTPUT", "output/nairn_slope_shear4_geotaichi"))

MAPPING = os.environ.get("NAIRN_SHEAR_MAPPING", "USF")
SHAPE_FUNCTION = os.environ.get("NAIRN_SHEAR_SHAPE", "GIMP")
ARCH = os.environ.get("NAIRN_SHEAR_ARCH", "cpu")
STRICT_TIME_LOOP = os.environ.get("NAIRN_SHEAR_STRICT_LOOP", "1").lower() not in {"0", "false", "no"}
RUN_POSTPROCESS = os.environ.get("NAIRN_SHEAR_POSTPROCESS", "1").lower() not in {"0", "false", "no"}
SAVE_PARTICLE = os.environ.get("NAIRN_SHEAR_SAVE_PARTICLE", "1").lower() not in {"0", "false", "no"}
SAVE_GRID = os.environ.get("NAIRN_SHEAR_SAVE_GRID", "1").lower() not in {"0", "false", "no"}

MAT_FREE = 1
MAT_MAIN = 2
BODY_MAIN = 0
BODY_LEFT_FREE = 1
BODY_RIGHT_FREE = 2

E_MODULUS = 1.0e5
POISSON = 0.25
DENSITY = 2.5e-3
SHEAR_MODULUS = E_MODULUS / (2.0 * (1.0 + POISSON))
BULK_MODULUS = E_MODULUS / (3.0 * (1.0 - 2.0 * POISSON))
CS = math.sqrt(SHEAR_MODULUS / DENSITY)
CP = math.sqrt((BULK_MODULUS + 4.0 * SHEAR_MODULUS / 3.0) / DENSITY)

INPUT_STRESS_PEAK = 1.0
INPUT_FREQUENCY = 100.0
INPUT_PERIOD = 1.0 / INPUT_FREQUENCY
BOTTOM_TOL = 0.07
SIDE_TOL = 0.06

HALF_PARTICLE_SIZE = 0.5 * DX
MAIN_AREA = 28.8
BOUNDING_BOX_EPS = 1.0e-8

CASE: dict[str, Any] = {}


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


DEFAULT_CASE: dict[str, Any] = {
    "name": "nairn_slope_shear4",
    "title": "NairnMPM slope-shear4 seismic validation",
    "source": "NairnMPM slope-shear4.fmcmd translated to GeoTaichi command stream",
    "coordinate_shift": [X_SHIFT, Y_SHIFT],
    "domain": DOMAIN.tolist(),
    "dx": DX,
    "dt": DT,
    "simulation_time": SIMULATION_TIME,
    "save_interval": SAVE_INTERVAL,
    "history_interval": HISTORY_INTERVAL,
    "output_dir": OUTPUT_DIR.as_posix(),
    "output_prefix": "nairn_slope_shear4",
    "arch": ARCH,
    "mapping": MAPPING,
    "shape_function": SHAPE_FUNCTION,
    "strict_time_loop": STRICT_TIME_LOOP,
    "run_postprocess": RUN_POSTPROCESS,
    "save_particle": SAVE_PARTICLE,
    "save_grid": SAVE_GRID,
    "memory": {
        "max_material_number": 3,
        "max_particle_number": 10000,
        "max_constraint_number": {
            "max_velocity_constraint": 10,
            "max_particle_traction_constraint": 300,
        },
    },
    "materials": [
        {"MaterialID": MAT_FREE, "Density": DENSITY, "YoungModulus": E_MODULUS, "PossionRatio": POISSON},
        {"MaterialID": MAT_MAIN, "Density": DENSITY, "YoungModulus": E_MODULUS, "PossionRatio": POISSON},
    ],
    "wave_speeds": {"material_id": MAT_MAIN},
    "input_stress": {"type": "raised_cosine", "peak": INPUT_STRESS_PEAK, "frequency": INPUT_FREQUENCY},
    "regions": [
        {
            "name": "left_free_column",
            "function": "slope_left_free",
            "bbox_point": [-0.6, 0.0],
            "bbox_size": [0.1, 4.0],
            "volume": 0.4,
        },
        {
            "name": "main_slope_body",
            "function": "slope_main",
            "bbox_point": [-0.1, 0.0],
            "bbox_size": [6.2, 4.0],
            "volume": MAIN_AREA,
        },
        {
            "name": "right_free_column",
            "function": "slope_right_free",
            "bbox_point": [6.5, 0.0],
            "bbox_size": [0.1, 4.0],
            "volume": 0.4,
        },
    ],
    "bodies": [
        {"region": "left_free_column", "material_id": MAT_FREE, "body_id": BODY_LEFT_FREE},
        {"region": "main_slope_body", "material_id": MAT_MAIN, "body_id": BODY_MAIN},
        {"region": "right_free_column", "material_id": MAT_FREE, "body_id": BODY_RIGHT_FREE},
    ],
    "bottom_traction": {
        "function": "slope_bottom",
        "pressure_seed": [1.0e-30, 0.0],
    },
    "silent_boundary": {
        "bottom": True,
        "side": {
            "enabled": True,
            "body_ids": [BODY_MAIN],
            "x_values": [-0.1, 6.1],
            "y_range": [-1.0, 11.0],
            "tolerance": SIDE_TOL,
        },
    },
    "monitors": [
        {"name": "nairn_pt1", "body_id": BODY_LEFT_FREE, "point": [-0.55, 0.05]},
        {"name": "nairn_pt40", "body_id": BODY_LEFT_FREE, "point": [-0.55, 3.95]},
        {"name": "left_free_bottom", "body_id": BODY_LEFT_FREE, "point": [-0.55, 0.05]},
        {"name": "left_free_top", "body_id": BODY_LEFT_FREE, "point": [-0.55, 3.95]},
        {"name": "main_left_bottom", "body_id": BODY_MAIN, "point": [-0.05, 0.05]},
        {"name": "main_center_bottom", "body_id": BODY_MAIN, "point": [3.0, 0.05]},
        {"name": "main_valley", "body_id": BODY_MAIN, "point": [3.0, 2.05]},
        {"name": "main_left_top", "body_id": BODY_MAIN, "point": [1.0, 3.95]},
        {"name": "right_free_bottom", "body_id": BODY_RIGHT_FREE, "point": [6.55, 0.05]},
    ],
}


@ti.pyfunc
def nairn_left_free_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    return -0.6 <= xp <= -0.5 and 0.0 <= yp <= 4.0


@ti.pyfunc
def nairn_right_free_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    return 6.5 <= xp <= 6.6 and 0.0 <= yp <= 4.0


@ti.pyfunc
def nairn_main_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    in_left_edge = -0.1 <= xp <= 0.0 and 0.0 <= yp <= 4.0
    in_bottom_block = 0.0 <= xp <= 6.0 and 0.0 <= yp <= 2.0
    in_left_top = 0.0 <= xp <= 2.0 and 2.0 <= yp <= 4.0
    in_right_top = 4.0 <= xp <= 6.0 and 2.0 <= yp <= 4.0
    in_left_slope = 2.0 <= xp <= 4.0 and 0.0 <= yp <= 8.0 - 2.0 * xp
    in_right_slope = 2.0 <= xp <= 4.0 and 0.0 <= yp <= 2.0 * (xp - 2.0)
    in_right_edge = 6.0 <= xp <= 6.1 and 0.0 <= yp <= 4.0
    return in_left_edge or in_bottom_block or in_left_top or in_right_top or in_left_slope or in_right_slope or in_right_edge


@ti.pyfunc
def nairn_bottom_region(x):
    return ti.abs((x[1] - Y_SHIFT) - 0.0) <= BOTTOM_TOL


def main_region_volume() -> float:
    return MAIN_AREA


REGION_FUNCTIONS: dict[str, Any] = {
    "slope_left_free": nairn_left_free_region,
    "slope_main": nairn_main_region,
    "slope_right_free": nairn_right_free_region,
    "slope_bottom": nairn_bottom_region,
}


def resolve_region_function(name: str) -> Any:
    try:
        return REGION_FUNCTIONS[name]
    except KeyError as exc:
        known = ", ".join(sorted(REGION_FUNCTIONS))
        raise ValueError(f"Unknown region function {name!r}. Known functions: {known}") from exc


def load_case_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="JSON case configuration override.")
    parser.add_argument("--dump-default-config", type=Path, default=None, help="Write the default case JSON and exit.")
    args = parser.parse_args(argv)

    case = copy.deepcopy(DEFAULT_CASE)
    config_path = args.config or (Path(os.environ["NAIRN_MPM_CONFIG"]) if os.environ.get("NAIRN_MPM_CONFIG") else None)
    if config_path is not None:
        user_case = json.loads(config_path.read_text(encoding="utf-8-sig"))
        deep_update(case, user_case)

    if args.dump_default_config is not None:
        args.dump_default_config.parent.mkdir(parents=True, exist_ok=True)
        args.dump_default_config.write_text(json.dumps(case, indent=2), encoding="utf-8")
        raise SystemExit(0)

    # Legacy environment variables keep existing command-line workflows working.
    case["dx"] = float(os.environ.get("NAIRN_SHEAR_DX", case["dx"]))
    case["dt"] = float(os.environ.get("NAIRN_SHEAR_DT", case["dt"]))
    case["simulation_time"] = float(os.environ.get("NAIRN_SHEAR_TIME", case["simulation_time"]))
    case["save_interval"] = float(os.environ.get("NAIRN_SHEAR_SAVE_INTERVAL", case["save_interval"]))
    case["history_interval"] = float(os.environ.get("NAIRN_SHEAR_HISTORY_INTERVAL", case["history_interval"]))
    case["output_dir"] = os.environ.get("NAIRN_SHEAR_OUTPUT", case["output_dir"])
    case["mapping"] = os.environ.get("NAIRN_SHEAR_MAPPING", case["mapping"])
    case["shape_function"] = os.environ.get("NAIRN_SHEAR_SHAPE", case["shape_function"])
    case["arch"] = os.environ.get("NAIRN_SHEAR_ARCH", case["arch"])
    case["strict_time_loop"] = env_bool("NAIRN_SHEAR_STRICT_LOOP", bool(case["strict_time_loop"]))
    case["run_postprocess"] = env_bool("NAIRN_SHEAR_POSTPROCESS", bool(case["run_postprocess"]))
    case["save_particle"] = env_bool("NAIRN_SHEAR_SAVE_PARTICLE", bool(case["save_particle"]))
    case["save_grid"] = env_bool("NAIRN_SHEAR_SAVE_GRID", bool(case["save_grid"]))
    return case


def apply_case_globals(case: dict[str, Any]) -> None:
    global X_SHIFT, Y_SHIFT, DX, DOMAIN, ELEMENT_SIZE, DT, SIMULATION_TIME, SAVE_INTERVAL
    global HISTORY_INTERVAL, OUTPUT_DIR, MAPPING, SHAPE_FUNCTION, ARCH, STRICT_TIME_LOOP
    global RUN_POSTPROCESS, SAVE_PARTICLE, SAVE_GRID, E_MODULUS, POISSON, DENSITY
    global SHEAR_MODULUS, BULK_MODULUS, CS, CP, INPUT_STRESS_PEAK, INPUT_FREQUENCY
    global INPUT_PERIOD, HALF_PARTICLE_SIZE, CASE

    CASE = case
    X_SHIFT, Y_SHIFT = [float(value) for value in case["coordinate_shift"]]
    DX = float(case["dx"])
    DOMAIN = np.array(case["domain"], dtype=np.float64)
    ELEMENT_SIZE = np.array(case.get("element_size", [DX, DX]), dtype=np.float64)
    DT = float(case["dt"])
    SIMULATION_TIME = float(case["simulation_time"])
    SAVE_INTERVAL = float(case["save_interval"])
    HISTORY_INTERVAL = float(case["history_interval"])
    OUTPUT_DIR = Path(case["output_dir"])
    MAPPING = str(case["mapping"])
    SHAPE_FUNCTION = str(case["shape_function"])
    ARCH = str(case["arch"])
    STRICT_TIME_LOOP = bool(case["strict_time_loop"])
    RUN_POSTPROCESS = bool(case["run_postprocess"])
    SAVE_PARTICLE = bool(case["save_particle"])
    SAVE_GRID = bool(case["save_grid"])

    wave_material_id = int(case.get("wave_speeds", {}).get("material_id", case["materials"][0]["MaterialID"]))
    material = next((item for item in case["materials"] if int(item["MaterialID"]) == wave_material_id), case["materials"][0])
    E_MODULUS = float(material["YoungModulus"])
    POISSON = float(material["PossionRatio"])
    DENSITY = float(material["Density"])
    SHEAR_MODULUS = E_MODULUS / (2.0 * (1.0 + POISSON))
    BULK_MODULUS = E_MODULUS / (3.0 * (1.0 - 2.0 * POISSON))
    wave_speeds = case.get("wave_speeds", {})
    CS = float(wave_speeds.get("cs", math.sqrt(SHEAR_MODULUS / DENSITY)))
    CP = float(wave_speeds.get("cp", math.sqrt((BULK_MODULUS + 4.0 * SHEAR_MODULUS / 3.0) / DENSITY)))
    INPUT_STRESS_PEAK = float(case.get("input_stress", {}).get("peak", INPUT_STRESS_PEAK))
    INPUT_FREQUENCY = float(case.get("input_stress", {}).get("frequency", INPUT_FREQUENCY))
    INPUT_PERIOD = 1.0 / INPUT_FREQUENCY if INPUT_FREQUENCY != 0.0 else math.inf
    HALF_PARTICLE_SIZE = 0.5 * DX


@ti.kernel
def update_bottom_input_traction(
    bottom_count: ti.i32,
    bottom_traction_ids: ti.template(),
    input_stress: ti.f64,
    particle_traction: ti.template(),
    total_input_force_x: ti.template(),
):
    total_input_force_x[None] = 0.0
    for i in range(bottom_count):
        tid = bottom_traction_ids[i]
        particle_traction[tid].traction = ti.Vector([input_stress, 0.0])
        total_input_force_x[None] += particle_traction[tid]._compute_traction_force()[0]


@ti.kernel
def apply_nairn_silent_loads(
    total_nodes: ti.i32,
    bottom_count: ti.i32,
    bottom_particle_ids: ti.template(),
    side_count: ti.i32,
    side_particle_ids: ti.template(),
    cs: ti.f64,
    cp: ti.f64,
    half_size: ti.f64,
    particle: ti.template(),
    node: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    node_size: ti.template(),
    total_bottom_silent_x: ti.template(),
    total_bottom_silent_y: ti.template(),
    total_side_silent_x: ti.template(),
    total_side_silent_y: ti.template(),
):
    total_bottom_silent_x[None] = 0.0
    total_bottom_silent_y[None] = 0.0
    total_side_silent_x[None] = 0.0
    total_side_silent_y[None] = 0.0

    for i in range(bottom_count):
        pid = bottom_particle_ids[i]
        body_id = int(particle[pid].bodyID)
        force = ti.Vector(
            [
                -particle[pid].m * cs * particle[pid].v[0] / (2.0 * half_size),
                -particle[pid].m * cp * particle[pid].v[1] / (2.0 * half_size),
            ]
        )
        offset = pid * total_nodes
        for ln in range(offset, offset + int(node_size[pid])):
            node_id = ln_id[ln]
            node[node_id, body_id]._update_nodal_force(shape_fn[ln] * force)
        total_bottom_silent_x[None] += force[0]
        total_bottom_silent_y[None] += force[1]

    for i in range(side_count):
        pid = side_particle_ids[i]
        body_id = int(particle[pid].bodyID)
        force = ti.Vector(
            [
                -particle[pid].m * cp * particle[pid].v[0] / (2.0 * half_size),
                -particle[pid].m * cs * particle[pid].v[1] / (2.0 * half_size),
            ]
        )
        offset = pid * total_nodes
        for ln in range(offset, offset + int(node_size[pid])):
            node_id = ln_id[ln]
            node[node_id, body_id]._update_nodal_force(shape_fn[ln] * force)
        total_side_silent_x[None] += force[0]
        total_side_silent_y[None] += force[1]


def input_stress(time_value: float) -> float:
    if time_value < 0.0:
        return 0.0
    spec = CASE.get("input_stress", {}) if CASE else {}
    stress_type = spec.get("type", "raised_cosine")
    if stress_type == "raised_cosine":
        return INPUT_STRESS_PEAK * 0.5 * (1.0 - math.cos(2.0 * math.pi * INPUT_FREQUENCY * time_value))
    if stress_type == "constant":
        return INPUT_STRESS_PEAK
    if stress_type == "sine":
        return INPUT_STRESS_PEAK * math.sin(2.0 * math.pi * INPUT_FREQUENCY * time_value)
    raise ValueError(f"Unsupported input_stress.type={stress_type!r}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class NairnSeismicBoundary:
    def __init__(self, mpm: MPM, case: dict[str, Any]) -> None:
        self.mpm = mpm
        self.case = case
        self.next_history = 0.0
        self.rows: list[dict[str, float | int]] = []
        self.boundary_rows: list[dict[str, float | int]] = []

        particle_count = int(mpm.scene.particleNum[0])
        positions = mpm.scene.particle.x.to_numpy()[:particle_count]
        body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count]

        ptraction_count = int(mpm.scene.boundary.ptraction_list[0])
        traction_pids = mpm.scene.boundary.particle_traction.pid.to_numpy()[:ptraction_count].astype(np.int32)
        self.bottom_particle_ids_np = np.ascontiguousarray(traction_pids)

        traction_id_by_pid = {int(pid): i for i, pid in enumerate(traction_pids.tolist())}
        bottom_traction_ids = np.array([traction_id_by_pid[int(pid)] for pid in traction_pids], dtype=np.int32)

        physical_x = positions[:, 0] - X_SHIFT
        physical_y = positions[:, 1] - Y_SHIFT
        side_spec = case.get("silent_boundary", {}).get("side", {})
        side_enabled = bool(side_spec.get("enabled", True))
        side_body_ids = np.array(side_spec.get("body_ids", [BODY_MAIN]), dtype=np.int32)
        side_x_values = [float(value) for value in side_spec.get("x_values", [-0.1, 6.1])]
        side_y_min, side_y_max = [float(value) for value in side_spec.get("y_range", [-1.0, 11.0])]
        side_tolerance = float(side_spec.get("tolerance", SIDE_TOL))
        side_x_mask = np.zeros_like(physical_x, dtype=bool)
        for x_value in side_x_values:
            side_x_mask |= np.abs(physical_x - x_value) <= side_tolerance
        side_mask = (
            side_enabled
            & np.isin(body_ids.astype(np.int32), side_body_ids)
            & (physical_y >= side_y_min - 1.0e-12)
            & (physical_y <= side_y_max + 1.0e-12)
            & side_x_mask
        )
        self.side_particle_ids_np = np.ascontiguousarray(np.nonzero(side_mask)[0].astype(np.int32))

        self.bottom_count = int(self.bottom_particle_ids_np.size)
        self.side_count = int(self.side_particle_ids_np.size)
        self.bottom_particle_ids = ti.field(dtype=ti.i32, shape=max(1, self.bottom_count))
        self.bottom_traction_ids = ti.field(dtype=ti.i32, shape=max(1, self.bottom_count))
        self.side_particle_ids = ti.field(dtype=ti.i32, shape=max(1, self.side_count))
        if self.bottom_count:
            self.bottom_particle_ids.from_numpy(self.bottom_particle_ids_np)
            self.bottom_traction_ids.from_numpy(bottom_traction_ids)
        if self.side_count:
            self.side_particle_ids.from_numpy(self.side_particle_ids_np)

        self.total_input_force_x = ti.field(dtype=ti.f64, shape=())
        self.total_bottom_silent_x = ti.field(dtype=ti.f64, shape=())
        self.total_bottom_silent_y = ti.field(dtype=ti.f64, shape=())
        self.total_side_silent_x = ti.field(dtype=ti.f64, shape=())
        self.total_side_silent_y = ti.field(dtype=ti.f64, shape=())

        self.monitor_particles = {
            item["name"]: self._nearest_particle(int(item["body_id"]), tuple(item["point"]))
            for item in case.get("monitors", [])
        }

    def _nearest_particle(self, body_id: int, physical_xy: tuple[float, float]) -> int:
        particle_count = int(self.mpm.scene.particleNum[0])
        positions = self.mpm.scene.particle.x.to_numpy()[:particle_count]
        body_ids = self.mpm.scene.particle.bodyID.to_numpy()[:particle_count]
        target = np.array([physical_xy[0] + X_SHIFT, physical_xy[1] + Y_SHIFT], dtype=float)
        candidates = np.nonzero(body_ids == body_id)[0]
        if candidates.size == 0:
            raise RuntimeError(f"No particles found for body_id={body_id}")
        distances = np.linalg.norm(positions[candidates] - target, axis=1)
        return int(candidates[int(np.argmin(distances))])

    def update_input_traction(self, sims: Any, scene: Any) -> None:
        stress = input_stress(float(sims.current_time))
        update_bottom_input_traction(
            self.bottom_count,
            self.bottom_traction_ids,
            stress,
            scene.boundary.particle_traction,
            self.total_input_force_x,
        )

    def apply_silent_forces(self, sims: Any, scene: Any) -> None:
        apply_nairn_silent_loads(
            scene.element.grid_nodes,
            self.bottom_count,
            self.bottom_particle_ids,
            self.side_count,
            self.side_particle_ids,
            CS,
            CP,
            HALF_PARTICLE_SIZE,
            scene.particle,
            scene.node,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
            self.total_bottom_silent_x,
            self.total_bottom_silent_y,
            self.total_side_silent_x,
            self.total_side_silent_y,
        )
        self.boundary_rows.append(
            {
                "time": float(sims.current_time),
                "input_stress_x": input_stress(float(sims.current_time)),
                "total_input_force_x": float(self.total_input_force_x[None]),
                "total_bottom_silent_force_x": float(self.total_bottom_silent_x[None]),
                "total_bottom_silent_force_y": float(self.total_bottom_silent_y[None]),
                "total_side_silent_force_x": float(self.total_side_silent_x[None]),
                "total_side_silent_force_y": float(self.total_side_silent_y[None]),
                "bottom_particle_count": self.bottom_count,
                "side_particle_count": self.side_count,
            }
        )

    def record_history(self, sims: Any, scene: Any) -> None:
        time_value = float(sims.current_time)
        if time_value + 0.5 * float(sims.delta) < self.next_history:
            return

        velocities = scene.particle.v.to_numpy()[: int(scene.particleNum[0])]
        positions = scene.particle.x.to_numpy()[: int(scene.particleNum[0])]
        row: dict[str, float | int] = {
            "time": time_value,
            "input_stress_x": input_stress(time_value),
            "total_input_force_x": float(self.total_input_force_x[None]),
            "total_bottom_silent_force_x": float(self.total_bottom_silent_x[None]),
            "total_side_silent_force_x": float(self.total_side_silent_x[None]),
        }
        for name, pid in self.monitor_particles.items():
            row[f"{name}_pid"] = pid
            row[f"{name}_x"] = float(positions[pid, 0] - X_SHIFT)
            row[f"{name}_y"] = float(positions[pid, 1] - Y_SHIFT)
            row[f"{name}_vx"] = float(velocities[pid, 0])
            row[f"{name}_vy"] = float(velocities[pid, 1])
        self.rows.append(row)
        self.next_history += HISTORY_INTERVAL

    def write_outputs(self) -> None:
        prefix = str(self.case.get("output_prefix", self.case.get("name", "nairn_case")))
        history_path = OUTPUT_DIR / f"{prefix}_history.csv"
        boundary_path = OUTPUT_DIR / f"{prefix}_boundary_forces.csv"
        metadata_path = OUTPUT_DIR / f"{prefix}_metadata.json"
        report_path = OUTPUT_DIR / f"{prefix}_report.md"
        history_fields = [
            "time",
            "input_stress_x",
            "total_input_force_x",
            "total_bottom_silent_force_x",
            "total_side_silent_force_x",
        ]
        for name in self.monitor_particles:
            history_fields.extend([f"{name}_pid", f"{name}_x", f"{name}_y", f"{name}_vx", f"{name}_vy"])
        write_csv(history_path, history_fields, self.rows)
        write_csv(
            boundary_path,
            [
                "time",
                "input_stress_x",
                "total_input_force_x",
                "total_bottom_silent_force_x",
                "total_bottom_silent_force_y",
                "total_side_silent_force_x",
                "total_side_silent_force_y",
                "bottom_particle_count",
                "side_particle_count",
            ],
            self.boundary_rows,
        )
        metadata = {
            "case": self.case,
            "source": self.case.get("source", self.case.get("name")),
            "coordinate_shift": {"x": X_SHIFT, "y": Y_SHIFT},
            "dx": DX,
            "dt": DT,
            "simulation_time": SIMULATION_TIME,
            "save_interval": SAVE_INTERVAL,
            "mapping": MAPPING,
            "shape_function": SHAPE_FUNCTION,
            "material": {
                "E": E_MODULUS,
                "nu": POISSON,
                "rho": DENSITY,
                "G": SHEAR_MODULUS,
                "K": BULK_MODULUS,
                "Cs": CS,
                "Cp": CP,
            },
            "input_stress": "0.5*(1-cos(2*pi*100*t))",
            "bottom_boundary": self.case.get("bottom_traction", {}),
            "side_boundary": self.case.get("silent_boundary", {}).get("side", {}),
            "bottom_particle_count": self.bottom_count,
            "side_particle_count": self.side_count,
            "monitor_particles": self.monitor_particles,
            "strict_time_loop": STRICT_TIME_LOOP,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        lines = [
            f"# {self.case.get('title', self.case.get('name', 'Nairn case'))}",
            "",
            "This run uses the configurable Nairn-style GeoTaichi command stream.",
            "",
            "## Case",
            "",
            f"- source: `{self.case.get('source', '')}`",
            f"- mapping: `{MAPPING}`",
            f"- shape_function: `{SHAPE_FUNCTION}`",
            f"- dt: `{DT}`",
            f"- simulation_time: `{SIMULATION_TIME}`",
            "",
            "## Counts",
            "",
            f"- particles: `{int(self.mpm.scene.particleNum[0])}`",
            f"- bottom traction/silent particles: `{self.bottom_count}`",
            f"- side silent particles: `{self.side_count}`",
            f"- monitor particles: `{len(self.monitor_particles)}`",
            "",
            "## Output",
            "",
            f"- history: `{history_path}`",
            f"- boundary forces: `{boundary_path}`",
            f"- metadata: `{metadata_path}`",
            f"- native particles/grids: `{OUTPUT_DIR}`",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def register_bottom_input_traction(mpm: MPM, case: dict[str, Any]) -> None:
    traction_spec = case.get("bottom_traction", {})
    pressure_seed = traction_spec.get("pressure_seed", [1.0e-30, 0.0])
    region_function = resolve_region_function(traction_spec.get("function", "slope_bottom"))
    mpm.scene.boundary.get_essentials(mpm.scene.is_rigid, mpm.scene.psize, mpm.generator.myRegion)
    mpm.scene.boundary.set_particle_traction(
        mpm.sims,
        {"Pressure": pressure_seed, "RegionFunction": region_function},
        int(mpm.scene.particleNum[0]),
        0,
        mpm.scene.particle,
        mpm.scene.psize,
    )


def add_region_body(mpm: MPM, name: str, material_id: int, body_id: int, n_particles_per_cell: int = 1) -> None:
    mpm.add_body(
        body={
            "Template": {
                "RegionName": name,
                "nParticlesPerCell": n_particles_per_cell,
                "BodyID": body_id,
                "MaterialID": material_id,
                "ParticleStress": {
                    "GravityField": False,
                    "InternalStress": ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    "Traction": {},
                },
                "InitialVelocity": ti.Vector([0.0, 0.0]),
                "FixVelocity": ["Free", "Free"],
            }
        }
    )


def install_seismic_hooks(mpm: MPM) -> None:
    original_add_engine = mpm.add_engine

    def add_engine_with_seismic_boundary() -> None:
        original_add_engine()
        engine = mpm.enginer
        if getattr(engine, "_nairn_slope_shear_hooks_installed", False):
            return

        original_particle_traction = engine.apply_particle_traction_constraints
        original_compute_forces = engine.compute_forces

        def apply_particle_traction_constraints(sims: Any, scene: Any) -> Any:
            boundary = getattr(mpm, "nairn_seismic_boundary", None)
            if boundary is not None:
                boundary.update_input_traction(sims, scene)
            return original_particle_traction(sims, scene)

        def compute_forces_with_silent_loads(sims: Any, scene: Any) -> Any:
            result = original_compute_forces(sims, scene)
            boundary = getattr(mpm, "nairn_seismic_boundary", None)
            if boundary is not None:
                boundary.apply_silent_forces(sims, scene)
            return result

        engine.apply_particle_traction_constraints = apply_particle_traction_constraints
        engine.compute_forces = compute_forces_with_silent_loads
        engine._nairn_slope_shear_hooks_installed = True

    mpm.add_engine = add_engine_with_seismic_boundary


def run_strict_time_loop(mpm: MPM, callback) -> None:
    mpm.add_essentials({"function": callback})
    mpm.check_critical_timestep()
    solver = mpm.solver
    solver.engine.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    solver.save_file(mpm.scene)
    mpm.sims.current_print += 1
    solver.last_save_time = -0.8 * mpm.sims.delta

    start_time = time.time()
    while float(mpm.sims.current_time) < float(mpm.sims.time) - 0.5 * float(mpm.sims.delta):
        solver.core(mpm.scene, mpm.neighbor)
        new_body = mpm.generator.regenerate(mpm.scene)
        if new_body:
            raise RuntimeError("This validation command stream does not support body regeneration")
        if mpm.sims.current_time - solver.last_save_time + 0.1 * mpm.sims.delta > mpm.sims.save_interval:
            solver.save_file(mpm.scene)
            solver.last_save_time = 1.0 * mpm.sims.current_time
            mpm.sims.current_print += 1
        mpm.sims.current_time += mpm.sims.delta
        mpm.sims.current_step += 1
    end_time = time.time()

    if abs(mpm.sims.current_time - solver.last_save_time) > mpm.sims.save_interval:
        solver.save_file(mpm.scene)
        solver.last_save_time = 1.0 * mpm.sims.current_time
        mpm.sims.current_print += 1
    mpm.first_run = False
    print("Physical time = ", end_time - start_time)


def main() -> None:
    case = load_case_config()
    apply_case_globals(case)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    init_taichi_mpm(dim=2, arch=ARCH)
    install_precise_grid_count_patch()
    install_safe_single_particle_generator()

    mpm = MPM(title=case.get("title", case.get("name", "Nairn-style seismic validation")))
    install_seismic_hooks(mpm)

    mpm.set_configuration(
        domain=DOMAIN,
        dimension="2-Dimension",
        is_2DAxisy=False,
        background_damping=0.0,
        gravity=ti.Vector([0.0, 0.0]),
        alphaPIC=0.0,
        mapping=MAPPING,
        shape_function=SHAPE_FUNCTION,
        velocity_projection="PIC/FLIP",
        material_type="Solid",
        visualize=False,
    )

    mpm.set_solver(
        solver={
            "Timestep": DT,
            "SimulationTime": SIMULATION_TIME,
            "SaveInterval": SAVE_INTERVAL,
            "SavePath": OUTPUT_DIR.as_posix(),
        }
    )

    mpm.memory_allocate(
        memory=case["memory"]
    )

    mpm.add_material(model=case.get("material_model", "LinearElastic"), material=case["materials"])

    mpm.scene.find_grid_level = types.MethodType(force_three_grid_levels, mpm.scene)
    mpm.scene.check_grid_inputs = types.MethodType(allow_three_grid_inputs, mpm.scene)
    mpm.add_element(
        element={
            "ElementType": "Q4N2D",
            "ElementSize": ELEMENT_SIZE,
            "Contact": {},
        }
    )

    regions = []
    for item in case["regions"]:
        bbox_point = item["bbox_point"]
        bbox_size = item["bbox_size"]
        volume = float(item["volume"])
        regions.append(
            {
                "Name": item["name"],
                "Type": item.get("type", "UserDefined"),
                "BoundingBoxPoint": ti.Vector([float(bbox_point[0]) + X_SHIFT, float(bbox_point[1]) + Y_SHIFT]),
                "BoundingBoxSize": ti.Vector([float(bbox_size[0]) + BOUNDING_BOX_EPS, float(bbox_size[1]) + BOUNDING_BOX_EPS]),
                "RegionFunction": resolve_region_function(item["function"]),
                "RegionVolume": lambda volume=volume: volume,
            }
        )
    mpm.add_region(region=regions)

    for item in case["bodies"]:
        add_region_body(
            mpm,
            item["region"],
            int(item["material_id"]),
            int(item["body_id"]),
            int(item.get("n_particles_per_cell", 1)),
        )

    mpm.add_boundary_condition(boundary=[])
    register_bottom_input_traction(mpm, case)
    mpm.select_save_data(particle=SAVE_PARTICLE, grid=SAVE_GRID, object=False)

    mpm.nairn_seismic_boundary = NairnSeismicBoundary(mpm, case)

    if STRICT_TIME_LOOP:
        run_strict_time_loop(mpm, lambda: mpm.nairn_seismic_boundary.record_history(mpm.sims, mpm.scene))
    else:
        mpm.run(function=lambda: mpm.nairn_seismic_boundary.record_history(mpm.sims, mpm.scene))

    mpm.nairn_seismic_boundary.write_outputs()

    if RUN_POSTPROCESS:
        mpm.postprocessing(read_path=OUTPUT_DIR.as_posix(), write_background_grid=SAVE_GRID)

    prefix = str(case.get("output_prefix", case.get("name", "nairn_case")))
    print(f"history = {OUTPUT_DIR / f'{prefix}_history.csv'}")
    print(f"boundary_forces = {OUTPUT_DIR / f'{prefix}_boundary_forces.csv'}")
    print(f"report = {OUTPUT_DIR / f'{prefix}_report.md'}")


if __name__ == "__main__":
    main()
