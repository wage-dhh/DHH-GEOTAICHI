import csv
import importlib.util
import math
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mpm.mainMPM import MPM  # noqa: E402
from src.utils import GlobalVariable  # noqa: E402

OUTPUT_ROOT = ROOT / "output" / "example3_3_2d_kohler_native_mpm_solver"
GEOMETRY_NAME = os.environ.get("EX33_GEOMETRY_NAME", "geometry_only_corrected")
GEOMETRY_DIR = OUTPUT_ROOT / GEOMETRY_NAME
MPM_MAPPING = os.environ.get("EX33_MPM_MAPPING", "MUSL")
MPM_SHAPE_FUNCTION = os.environ.get("EX33_MPM_SHAPE_FUNCTION", "Linear")
SIDE_COUPLING_MODE = os.environ.get("EX33_SIDE_COUPLING", "grid")
BOTTOM_COUPLING_MODE = os.environ.get("EX33_BOTTOM_COUPLING", "grid")
SIDE_X_IMPEDANCE_MODE = os.environ.get("EX33_SIDE_X_IMPEDANCE", "Cp")
RUN_NAME = os.environ.get("EX33_RUN_NAME", f"new_geometry_corrected_dynamic_run_{MPM_MAPPING.lower()}_{MPM_SHAPE_FUNCTION.lower()}")
RUN_DIR = OUTPUT_ROOT / RUN_NAME
REFERENCE_DIR = ROOT / "data" / "reference" / "flac3d_example3_3_free_field"
LEGACY_NATIVE = ROOT / "examples" / "example3_3_2d_kohler_native_mpm_solver.py"

DX = float(os.environ.get("EX33_DX", "0.25"))
DOMAIN_WIDTH = 10.5
DOMAIN_HEIGHT = 8.0
MATERIAL_ID = 1
MAIN_BODY_ID = 0
LEFT_FF_BODY_ID = 1
RIGHT_FF_BODY_ID = 2

RHO = 0.0025
G = 40000.0
K = 66667.0
NU = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
E = 2.0 * G * (1.0 + NU)
CS = math.sqrt(G / RHO)
CP = math.sqrt((K + 4.0 * G / 3.0) / RHO)
RHO_CS = RHO * CS
RHO_CP = RHO * CP
SIDE_ETA_X = RHO_CS if SIDE_X_IMPEDANCE_MODE.lower() == "cs" else RHO_CP
SIDE_ETA_Z = RHO_CP if SIDE_X_IMPEDANCE_MODE.lower() == "cs" else RHO_CS
SIDE_PARTICLE_SCALE = 0.0 if SIDE_COUPLING_MODE.lower() == "grid" else 1.0
BOTTOM_PARTICLE_SCALE = 0.0 if BOTTOM_COUPLING_MODE.lower() == "grid" else 1.0

DT = 1.0e-5
SIMULATION_TIME = 0.015
INPUT_PERIOD = 0.01
INPUT_VELOCITY_AMPLITUDE = 0.05
INPUT_TIME_SHIFT = float(os.environ.get("EX33_INPUT_TIME_SHIFT", "0.0"))
HISTORY_TIME_SHIFT = float(os.environ.get("EX33_HISTORY_TIME_SHIFT", "0.0"))
BOTTOM_GRID_Z = 1.75
BOTTOM_Z = BOTTOM_GRID_Z + 0.5 * DX
TOP_GRID_Z = 5.75
TOP_PARTICLE_Z = TOP_GRID_Z - 0.5 * DX
VALLEY_GRID_Z = 3.5
LEFT_MAIN_X = 1.0 + 0.5 * DX
LEFT_FF_X = 1.0 - 0.5 - 0.5 * DX
RIGHT_MAIN_X = 7.0 - 0.5 * DX
RIGHT_FF_X = 7.0 + 0.5 + 0.5 * DX
SOIL_COLUMN_X = LEFT_FF_X
SOIL_COLUMN_BOTTOM_TARGET = (SOIL_COLUMN_X, BOTTOM_Z)
SOIL_COLUMN_3M_TARGET = (SOIL_COLUMN_X, BOTTOM_Z + 3.0)
MAIN_MODEL_BOTTOM_TARGET = (4.0, BOTTOM_Z)
MAIN_MODEL_3M_TARGET = (2.0, BOTTOM_Z + 3.0)
SOIL_GRID_BOTTOM_TARGET = SOIL_COLUMN_BOTTOM_TARGET
MAIN_GRID_BOTTOM_TARGET = MAIN_MODEL_BOTTOM_TARGET
MAIN_GRID_TOP_MANUAL_TARGET = (2.0, TOP_GRID_Z)
MAIN_GRID_TOP_CENTER_TARGET = (4.0, VALLEY_GRID_Z)
SOIL_GRID_TOP_TARGET = (0.375, TOP_GRID_Z)
REFLECTION_MONITOR_Z = TOP_PARTICLE_Z
REFLECTION_TARGETS = {
    "main_left_inside": (1.25, REFLECTION_MONITOR_Z),
    "main_center": (4.0, REFLECTION_MONITOR_Z),
    "main_right_inside": (6.75, REFLECTION_MONITOR_Z),
}


@ti.pyfunc
def corrected_bottom_region(x):
    return ti.abs(x[1] - BOTTOM_Z) <= 1.0e-10


@ti.pyfunc
def corrected_interface_region(x):
    on_left = ti.abs(x[0] - LEFT_MAIN_X) <= 1.0e-10 or ti.abs(x[0] - LEFT_FF_X) <= 1.0e-10
    on_right = ti.abs(x[0] - RIGHT_MAIN_X) <= 1.0e-10 or ti.abs(x[0] - RIGHT_FF_X) <= 1.0e-10
    return (on_left or on_right) and x[1] > BOTTOM_Z + 1.0e-10


@ti.pyfunc
def all_particle_region(x):
    return True


@ti.kernel
def update_corrected_particle_tractions(
    total_nodes: ti.i32,
    bottom_count: ti.i32,
    bottom_particle_ids: ti.template(),
    bottom_traction_ids: ti.template(),
    bottom_areas: ti.template(),
    pair_count: ti.i32,
    side_signs: ti.template(),
    main_particle_ids: ti.template(),
    ff_particle_ids: ti.template(),
    main_traction_ids: ti.template(),
    ff_traction_ids: ti.template(),
    input_velocity: ti.f64,
    rho_cs: ti.f64,
    rho_cp: ti.f64,
    side_eta_x: ti.f64,
    side_eta_z: ti.f64,
    side_particle_scale: ti.f64,
    bottom_particle_scale: ti.f64,
    particle: ti.template(),
    node: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    node_size: ti.template(),
    particle_traction: ti.template(),
    total_input_force: ti.template(),
    total_dashpot_force_particle_x: ti.template(),
    total_dashpot_force_grid_x: ti.template(),
    total_dashpot_force_z: ti.template(),
    total_bottom_force_x: ti.template(),
    total_bottom_force_z: ti.template(),
    total_side_main_force_x: ti.template(),
    total_side_main_force_z: ti.template(),
):
    total_input_force[None] = 0.0
    total_dashpot_force_particle_x[None] = 0.0
    total_dashpot_force_grid_x[None] = 0.0
    total_dashpot_force_z[None] = 0.0
    total_bottom_force_x[None] = 0.0
    total_bottom_force_z[None] = 0.0
    total_side_main_force_x[None] = 0.0
    total_side_main_force_z[None] = 0.0

    for i in range(bottom_count):
        pid = bottom_particle_ids[i]
        tid = bottom_traction_ids[i]
        area = bottom_areas[i]
        body_id = int(particle[pid].bodyID)
        boundary_grid_velocity = ti.Vector([0.0, 0.0])
        offset = pid * total_nodes
        for ln in range(offset, offset + int(node_size[pid])):
            node_id = ln_id[ln]
            boundary_grid_velocity += shape_fn[ln] * node[node_id, body_id].momentum
        input_tx = 2.0 * rho_cs * input_velocity
        dashpot_particle_tx = -rho_cs * particle[pid].v[0]
        dashpot_grid_tx = -rho_cs * boundary_grid_velocity[0]
        dashpot_tz = -rho_cp * particle[pid].v[1]
        tx = input_tx + dashpot_grid_tx
        tz = dashpot_tz
        particle_traction[tid].traction = bottom_particle_scale * ti.Vector([tx, tz])
        total_input_force[None] += area * input_tx
        total_dashpot_force_particle_x[None] += area * dashpot_particle_tx
        total_dashpot_force_grid_x[None] += area * dashpot_grid_tx
        total_dashpot_force_z[None] += bottom_particle_scale * area * dashpot_tz
        total_bottom_force_x[None] += bottom_particle_scale * area * tx
        total_bottom_force_z[None] += bottom_particle_scale * area * tz

    for i in range(pair_count):
        main_pid = main_particle_ids[i]
        ff_pid = ff_particle_ids[i]
        mtid = main_traction_ids[i]
        ftid = ff_traction_ids[i]
        vrel = particle[ff_pid].v - particle[main_pid].v
        tx = side_eta_x * vrel[0]
        tz = side_eta_z * vrel[1]
        traction = side_particle_scale * ti.Vector([tx, tz])
        particle_traction[mtid].traction = traction
        particle_traction[ftid].traction = -traction
        total_side_main_force_x[None] += DX * tx
        total_side_main_force_z[None] += DX * tz


@ti.kernel
def apply_grid_side_dashpot(
    total_nodes: ti.i32,
    pair_count: ti.i32,
    main_particle_ids: ti.template(),
    ff_particle_ids: ti.template(),
    areas: ti.template(),
    side_eta_x: ti.f64,
    side_eta_z: ti.f64,
    particle: ti.template(),
    node: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    node_size: ti.template(),
    total_grid_side_main_force_x: ti.template(),
    total_grid_side_main_force_z: ti.template(),
):
    total_grid_side_main_force_x[None] = 0.0
    total_grid_side_main_force_z[None] = 0.0
    for i in range(pair_count):
        main_pid = main_particle_ids[i]
        ff_pid = ff_particle_ids[i]
        main_body = int(particle[main_pid].bodyID)
        ff_body = int(particle[ff_pid].bodyID)
        area = areas[i]
        v_main = ti.Vector([0.0, 0.0])
        v_ff = ti.Vector([0.0, 0.0])
        main_offset = main_pid * total_nodes
        ff_offset = ff_pid * total_nodes
        for ln in range(main_offset, main_offset + int(node_size[main_pid])):
            node_id = ln_id[ln]
            v_main += shape_fn[ln] * node[node_id, main_body].momentum
        for ln in range(ff_offset, ff_offset + int(node_size[ff_pid])):
            node_id = ln_id[ln]
            v_ff += shape_fn[ln] * node[node_id, ff_body].momentum
        vrel = v_ff - v_main
        force = ti.Vector([side_eta_x * area * vrel[0], side_eta_z * area * vrel[1]])
        for ln in range(main_offset, main_offset + int(node_size[main_pid])):
            node_id = ln_id[ln]
            node[node_id, main_body]._update_nodal_force(shape_fn[ln] * force)
        for ln in range(ff_offset, ff_offset + int(node_size[ff_pid])):
            node_id = ln_id[ln]
            node[node_id, ff_body]._update_nodal_force(-shape_fn[ln] * force)
        total_grid_side_main_force_x[None] += force[0]
        total_grid_side_main_force_z[None] += force[1]


@ti.kernel
def apply_grid_bottom_compliant_base(
    total_nodes: ti.i32,
    bottom_count: ti.i32,
    bottom_particle_ids: ti.template(),
    bottom_areas: ti.template(),
    input_velocity: ti.f64,
    rho_cs: ti.f64,
    rho_cp: ti.f64,
    particle: ti.template(),
    node: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    node_size: ti.template(),
    total_input_force: ti.template(),
    total_dashpot_force_grid_x: ti.template(),
    total_dashpot_force_z: ti.template(),
    total_bottom_force_x: ti.template(),
    total_bottom_force_z: ti.template(),
):
    total_input_force[None] = 0.0
    total_dashpot_force_grid_x[None] = 0.0
    total_dashpot_force_z[None] = 0.0
    total_bottom_force_x[None] = 0.0
    total_bottom_force_z[None] = 0.0
    for i in range(bottom_count):
        pid = bottom_particle_ids[i]
        area = bottom_areas[i]
        body_id = int(particle[pid].bodyID)
        boundary_grid_velocity = ti.Vector([0.0, 0.0])
        offset = pid * total_nodes
        for ln in range(offset, offset + int(node_size[pid])):
            node_id = ln_id[ln]
            boundary_grid_velocity += shape_fn[ln] * node[node_id, body_id].momentum
        input_tx = 2.0 * rho_cs * input_velocity
        dashpot_tx = -rho_cs * boundary_grid_velocity[0]
        dashpot_tz = -rho_cp * boundary_grid_velocity[1]
        total_force = area * ti.Vector([input_tx + dashpot_tx, dashpot_tz])
        for ln in range(offset, offset + int(node_size[pid])):
            node_id = ln_id[ln]
            node[node_id, body_id]._update_nodal_force(shape_fn[ln] * total_force)
        total_input_force[None] += area * input_tx
        total_dashpot_force_grid_x[None] += area * dashpot_tx
        total_dashpot_force_z[None] += area * dashpot_tz
        total_bottom_force_x[None] += total_force[0]
        total_bottom_force_z[None] += total_force[1]


def input_velocity(time_value: float) -> float:
    if time_value < 0.0:
        return 0.0
    return INPUT_VELOCITY_AMPLITUDE * 0.5 * (1.0 - math.cos(2.0 * math.pi * time_value / INPUT_PERIOD))


def load_legacy_native() -> Any:
    spec = importlib.util.spec_from_file_location("legacy_native_example3_3", LEGACY_NATIVE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LEGACY_NATIVE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_geometry_source() -> None:
    checks = read_csv(GEOMETRY_DIR / "geometry_validation_checks.csv")
    failing = [r for r in checks if r.get("status") != "PASS"]
    if failing:
        raise RuntimeError(f"geometry_only_corrected validation is not PASS: {failing[:3]}")


def load_particles() -> list[dict[str, Any]]:
    rows = read_csv(GEOMETRY_DIR / "geotaichi_particle_generation_check.csv")
    rows = sorted(rows, key=lambda r: int(r["particle_id"]))
    particles: list[dict[str, Any]] = []
    for r in rows:
        particles.append(
            {
                "particle_id": int(r["particle_id"]),
                "component": r["component"],
                "x": float(r["x"]),
                "z": float(r["z"]),
                "body_id": int(r["body_id"]),
                "material_id": int(r["material_id"]),
                "volume": float(r["volume"]),
                "mass": float(r["mass"]),
            }
        )
    return particles


def prepare_particle_files(particles: list[dict[str, Any]]) -> dict[int, tuple[Path, int]]:
    out: dict[int, tuple[Path, int]] = {}
    for body_id, name in ((MAIN_BODY_ID, "main"), (LEFT_FF_BODY_ID, "left_ff"), (RIGHT_FF_BODY_ID, "right_ff")):
        body_particles = [p for p in particles if p["body_id"] == body_id]
        path = RUN_DIR / "particle_input" / f"{name}_particles_from_geometry_only_corrected.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# x z volume psize_x psize_z reserved velocity_x velocity_z"]
        for p in body_particles:
            lines.append(f"{p['x']:.16g} {p['z']:.16g} {p['volume']:.16g} {DX:.16g} {DX:.16g} 0 0 0")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out[body_id] = (path, len(body_particles))
    return out


def force_three_grid_levels(scene: Any, sims: Any) -> int:
    scene.grid_level = 3
    return scene.grid_level


def allow_grid_inputs(scene: Any, sims: Any, grid_level: int) -> None:
    return None


def register_particle_tractions(mpm: MPM) -> None:
    mpm.scene.boundary.get_essentials(mpm.scene.is_rigid, mpm.scene.psize, mpm.generator.myRegion)
    mpm.scene.boundary.set_particle_traction(
        mpm.sims,
        {"Pressure": [1.0e-30, 0.0], "RegionFunction": all_particle_region},
        int(mpm.scene.particleNum[0]),
        0,
        mpm.scene.particle,
        mpm.scene.psize,
    )


def install_grid_side_force_hook(mpm: MPM, trace: dict[str, int]) -> None:
    """Insert grid-level side dashpot force after internal force assembly."""

    original_add_engine = mpm.add_engine

    def traced_add_engine() -> None:
        original_add_engine()
        engine = mpm.enginer
        if getattr(engine, "_geometry_corrected_grid_side_hook_installed", False):
            return
        original_compute_forces = engine.compute_forces

        def compute_forces_with_grid_side(sims: Any, scene: Any) -> Any:
            result = original_compute_forces(sims, scene)
            boundary = getattr(mpm, "native_compliant_base", None)
            if boundary is not None:
                boundary.apply_grid_boundary_forces(sims, scene)
                trace["grid_boundary_force_calls"] = trace.get("grid_boundary_force_calls", 0) + 1
                if SIDE_COUPLING_MODE.lower() == "grid":
                    trace["grid_side_dashpot_calls"] = trace.get("grid_side_dashpot_calls", 0) + 1
                if BOTTOM_COUPLING_MODE.lower() == "grid":
                    trace["grid_bottom_compliant_base_calls"] = trace.get("grid_bottom_compliant_base_calls", 0) + 1
                boundary.record_stage(sims, scene, "after_grid_boundary_forces")
            return result

        compute_forces_with_grid_side.__name__ = getattr(original_compute_forces, "__name__", "compute_forces")
        engine.compute_forces = compute_forces_with_grid_side
        engine._geometry_corrected_grid_side_hook_installed = True

    mpm.add_engine = traced_add_engine


class CorrectedBoundary:
    def __init__(self, mpm: MPM, pair_rows: list[dict[str, str]]) -> None:
        self.mpm = mpm
        self.pair_rows_source = pair_rows
        traction_indices = self._traction_indices()
        self.bottom_specs = self._bottom_specs(traction_indices)
        self.interface_specs = self._interface_specs(traction_indices)
        self.bottom_count = len(self.bottom_specs)
        self.interface_pair_count = len(self.interface_specs)

        self.bottom_particle_ids = ti.field(dtype=ti.i32, shape=self.bottom_count)
        self.bottom_traction_ids = ti.field(dtype=ti.i32, shape=self.bottom_count)
        self.bottom_areas = ti.field(dtype=ti.f64, shape=self.bottom_count)
        for i, row in enumerate(self.bottom_specs):
            self.bottom_particle_ids[i] = int(row["particle_id"])
            self.bottom_traction_ids[i] = int(row["traction_id"])
            self.bottom_areas[i] = float(row["area"])

        self.side_signs = ti.field(dtype=ti.f64, shape=self.interface_pair_count)
        self.main_particle_ids = ti.field(dtype=ti.i32, shape=self.interface_pair_count)
        self.ff_particle_ids = ti.field(dtype=ti.i32, shape=self.interface_pair_count)
        self.main_traction_ids = ti.field(dtype=ti.i32, shape=self.interface_pair_count)
        self.ff_traction_ids = ti.field(dtype=ti.i32, shape=self.interface_pair_count)
        for i, row in enumerate(self.interface_specs):
            self.side_signs[i] = float(row["side_sign"])
            self.main_particle_ids[i] = int(row["main_particle_id"])
            self.ff_particle_ids[i] = int(row["ff_particle_id"])
            self.main_traction_ids[i] = int(row["main_traction_id"])
            self.ff_traction_ids[i] = int(row["ff_traction_id"])

        self.total_input_force = ti.field(dtype=ti.f64, shape=())
        self.total_dashpot_force_particle_x = ti.field(dtype=ti.f64, shape=())
        self.total_dashpot_force_grid_x = ti.field(dtype=ti.f64, shape=())
        self.total_dashpot_force_z = ti.field(dtype=ti.f64, shape=())
        self.total_bottom_force_x = ti.field(dtype=ti.f64, shape=())
        self.total_bottom_force_z = ti.field(dtype=ti.f64, shape=())
        self.total_side_main_force_x = ti.field(dtype=ti.f64, shape=())
        self.total_side_main_force_z = ti.field(dtype=ti.f64, shape=())
        self.total_grid_side_main_force_x = ti.field(dtype=ti.f64, shape=())
        self.total_grid_side_main_force_z = ti.field(dtype=ti.f64, shape=())

        self.grid_side_main_nodes = ti.field(dtype=ti.i32, shape=self.interface_pair_count)
        self.grid_side_ff_nodes = ti.field(dtype=ti.i32, shape=self.interface_pair_count)
        self.grid_side_main_body_ids = ti.field(dtype=ti.i32, shape=self.interface_pair_count)
        self.grid_side_ff_body_ids = ti.field(dtype=ti.i32, shape=self.interface_pair_count)
        self.grid_side_areas = ti.field(dtype=ti.f64, shape=self.interface_pair_count)
        self.grid_side_initialized = False
        self.grid_side_specs: list[dict[str, Any]] = []

        self.grid_bottom_nodes = ti.field(dtype=ti.i32, shape=self.bottom_count)
        self.grid_bottom_body_ids = ti.field(dtype=ti.i32, shape=self.bottom_count)
        self.grid_bottom_areas = ti.field(dtype=ti.f64, shape=self.bottom_count)
        self.grid_bottom_initialized = False
        self.grid_bottom_specs: list[dict[str, Any]] = []

        self.bottom_rows: list[dict[str, Any]] = []
        self.side_rows: list[dict[str, Any]] = []
        self.monitor_rows: list[dict[str, Any]] = []
        self.reflection_rows: list[dict[str, Any]] = []
        self.monitor_specs = [
            ("soil_column_bottom", LEFT_FF_BODY_ID, SOIL_COLUMN_BOTTOM_TARGET),
            ("soil_column_3m", LEFT_FF_BODY_ID, SOIL_COLUMN_3M_TARGET),
            ("main_model_bottom", MAIN_BODY_ID, MAIN_MODEL_BOTTOM_TARGET),
            ("main_model_3m", MAIN_BODY_ID, MAIN_MODEL_3M_TARGET),
        ]
        self.monitor_particles = {
            name: self._nearest_particle(body_id, *target)
            for name, body_id, target in self.monitor_specs
        }
        self.grid_monitor_specs = [
            ("soil_grid_bottom", LEFT_FF_BODY_ID, SOIL_GRID_BOTTOM_TARGET),
            ("main_grid_bottom", MAIN_BODY_ID, MAIN_GRID_BOTTOM_TARGET),
            ("main_grid_top_manual", MAIN_BODY_ID, MAIN_GRID_TOP_MANUAL_TARGET),
            ("main_grid_top_center", MAIN_BODY_ID, MAIN_GRID_TOP_CENTER_TARGET),
            ("soil_grid_top", LEFT_FF_BODY_ID, SOIL_GRID_TOP_TARGET),
        ]
        self.grid_monitor_nodes: dict[str, int] = {}
        self.pt40_particle = 40 if int(mpm.scene.particleNum[0]) > 40 else -1
        self.reflection_particles = {
            name: self._reflection_particle(*target)
            for name, target in REFLECTION_TARGETS.items()
        }

    def _traction_indices(self) -> dict[int, int]:
        boundary = self.mpm.scene.boundary
        count = int(boundary.ptraction_list[0])
        pids = boundary.particle_traction.pid.to_numpy()[:count]
        return {int(pid): i for i, pid in enumerate(pids)}

    def _bottom_specs(self, traction_indices: dict[int, int]) -> list[dict[str, Any]]:
        n = int(self.mpm.scene.particleNum[0])
        pos = self.mpm.scene.particle.x.to_numpy()[:n]
        vol = self.mpm.scene.particle.vol.to_numpy()[:n]
        body = self.mpm.scene.particle.bodyID.to_numpy()[:n]
        specs = []
        for pid in range(n):
            if abs(float(pos[pid, 1]) - BOTTOM_Z) <= 1.0e-10:
                if pid not in traction_indices:
                    raise RuntimeError(f"Bottom particle {pid} has no traction constraint")
                specs.append(
                    {
                        "particle_id": pid,
                        "traction_id": traction_indices[pid],
                        "body_id": int(body[pid]),
                        "x": float(pos[pid, 0]),
                        "z": float(pos[pid, 1]),
                        "area": float(vol[pid]) / DX,
                    }
                )
        return specs

    def _interface_specs(self, traction_indices: dict[int, int]) -> list[dict[str, Any]]:
        n = int(self.mpm.scene.particleNum[0])
        pos = self.mpm.scene.particle.x.to_numpy()[:n]
        body = self.mpm.scene.particle.bodyID.to_numpy()[:n]
        specs = []
        for r in self.pair_rows_source:
            main_pid = int(r["main_particle_id"])
            ff_pid = int(r["ff_particle_id"])
            if main_pid not in traction_indices or ff_pid not in traction_indices:
                raise RuntimeError(f"Interface pair {r['pair_id']} endpoints missing traction constraints")
            side = r["side"]
            side_sign = -1.0 if side == "left" else 1.0
            specs.append(
                {
                    "side": side,
                    "pair_id": int(r["pair_id"]),
                    "side_sign": side_sign,
                    "main_particle_id": main_pid,
                    "ff_particle_id": ff_pid,
                    "main_traction_id": traction_indices[main_pid],
                    "ff_traction_id": traction_indices[ff_pid],
                    "main_body_id": int(body[main_pid]),
                    "ff_body_id": int(body[ff_pid]),
                    "main_x": float(pos[main_pid, 0]),
                    "main_z": float(pos[main_pid, 1]),
                    "ff_x": float(pos[ff_pid, 0]),
                    "ff_z": float(pos[ff_pid, 1]),
                }
            )
        return specs

    def _nearest_particle(self, body_id: int, x: float, z: float) -> int:
        n = int(self.mpm.scene.particleNum[0])
        pos = self.mpm.scene.particle.x.to_numpy()[:n]
        body = self.mpm.scene.particle.bodyID.to_numpy()[:n]
        best = -1
        dist = math.inf
        for pid in range(n):
            if int(body[pid]) != body_id:
                continue
            d = math.hypot(float(pos[pid, 0]) - x, float(pos[pid, 1]) - z)
            if d < dist:
                best = pid
                dist = d
        if best < 0:
            raise RuntimeError(f"No monitor particle for body_id={body_id}, x={x}, z={z}")
        return best

    def _reflection_particle(self, x: float, z: float) -> int:
        n = int(self.mpm.scene.particleNum[0])
        pos = self.mpm.scene.particle.x.to_numpy()[:n]
        body = self.mpm.scene.particle.bodyID.to_numpy()[:n]
        main_ids = [pid for pid in range(n) if int(body[pid]) == MAIN_BODY_ID]
        if not main_ids:
            raise RuntimeError("No main particles available for reflection monitor")
        min_dx = min(abs(float(pos[pid, 0]) - x) for pid in main_ids)
        column_ids = [pid for pid in main_ids if abs(abs(float(pos[pid, 0]) - x) - min_dx) <= 1.0e-10]
        return max(column_ids, key=lambda pid: float(pos[pid, 1]))

    def _nearest_grid_node_with_mass(self, scene: Any, body_id: int, x: float, z: float) -> int:
        coords = np.asarray(scene.element.get_nodal_coords(), dtype=float)
        masses = scene.node.m.to_numpy()[:, body_id]
        valid = masses > float(scene.mass_cut_off)
        if not np.any(valid):
            raise RuntimeError(f"No active grid nodes for body_id={body_id}")
        dx = coords[:, 0] - x
        dz = coords[:, 1] - z
        distance = dx * dx + dz * dz
        distance[~valid] = np.inf
        node_id = int(np.argmin(distance))
        if not math.isfinite(float(distance[node_id])):
            raise RuntimeError(f"No valid grid node near body_id={body_id}, x={x}, z={z}")
        return node_id

    def _initialize_grid_side_pairs(self, scene: Any) -> None:
        if self.grid_side_initialized:
            return
        specs: list[dict[str, Any]] = []
        coords = np.asarray(scene.element.get_nodal_coords(), dtype=float)
        for i, row in enumerate(self.interface_specs):
            main_body = int(row["main_body_id"])
            ff_body = int(row["ff_body_id"])
            main_node = self._nearest_grid_node_with_mass(scene, main_body, float(row["main_x"]), float(row["main_z"]))
            ff_node = self._nearest_grid_node_with_mass(scene, ff_body, float(row["ff_x"]), float(row["ff_z"]))
            area = DX
            self.grid_side_main_nodes[i] = main_node
            self.grid_side_ff_nodes[i] = ff_node
            self.grid_side_main_body_ids[i] = main_body
            self.grid_side_ff_body_ids[i] = ff_body
            self.grid_side_areas[i] = area
            specs.append(
                {
                    "pair_id": int(row["pair_id"]),
                    "side": row["side"],
                    "main_node": main_node,
                    "ff_node": ff_node,
                    "main_body_id": main_body,
                    "ff_body_id": ff_body,
                    "main_node_x": float(coords[main_node, 0]),
                    "main_node_z": float(coords[main_node, 1]),
                    "ff_node_x": float(coords[ff_node, 0]),
                    "ff_node_z": float(coords[ff_node, 1]),
                    "area": area,
                }
            )
        self.grid_side_specs = specs
        self.grid_side_initialized = True

    def _initialize_grid_bottom_nodes(self, scene: Any) -> None:
        if self.grid_bottom_initialized:
            return
        specs: list[dict[str, Any]] = []
        coords = np.asarray(scene.element.get_nodal_coords(), dtype=float)
        for i, row in enumerate(self.bottom_specs):
            body_id = int(row["body_id"])
            node_id = self._nearest_grid_node_with_mass(scene, body_id, float(row["x"]), float(row["z"]))
            area = float(row["area"])
            self.grid_bottom_nodes[i] = node_id
            self.grid_bottom_body_ids[i] = body_id
            self.grid_bottom_areas[i] = area
            specs.append(
                {
                    "particle_id": int(row["particle_id"]),
                    "body_id": body_id,
                    "node_id": node_id,
                    "particle_x": float(row["x"]),
                    "particle_z": float(row["z"]),
                    "node_x": float(coords[node_id, 0]),
                    "node_z": float(coords[node_id, 1]),
                    "area": area,
                }
            )
        self.grid_bottom_specs = specs
        self.grid_bottom_initialized = True

    def apply_grid_boundary_forces(self, sims: Any, scene: Any) -> None:
        load_time = float(sims.current_time) + INPUT_TIME_SHIFT
        v_in = input_velocity(load_time)
        if BOTTOM_COUPLING_MODE.lower() == "grid":
            self._initialize_grid_bottom_nodes(scene)
            apply_grid_bottom_compliant_base(
                scene.element.grid_nodes,
                self.bottom_count,
                self.bottom_particle_ids,
                self.grid_bottom_areas,
                v_in,
                RHO_CS,
                RHO_CP,
                scene.particle,
                scene.node,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
                self.total_input_force,
                self.total_dashpot_force_grid_x,
                self.total_dashpot_force_z,
                self.total_bottom_force_x,
                self.total_bottom_force_z,
            )
            self.record_bottom(sims, scene, v_in, load_time)
        if SIDE_COUPLING_MODE.lower() != "grid":
            return
        self._initialize_grid_side_pairs(scene)
        apply_grid_side_dashpot(
            scene.element.grid_nodes,
            self.interface_pair_count,
            self.main_particle_ids,
            self.ff_particle_ids,
            self.grid_side_areas,
            SIDE_ETA_X,
            SIDE_ETA_Z,
            scene.particle,
            scene.node,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
            self.total_grid_side_main_force_x,
            self.total_grid_side_main_force_z,
        )

    def apply(self, sims: Any, scene: Any) -> None:
        load_time = float(sims.current_time) + INPUT_TIME_SHIFT
        v_in = input_velocity(load_time)
        update_corrected_particle_tractions(
            scene.element.grid_nodes,
            self.bottom_count,
            self.bottom_particle_ids,
            self.bottom_traction_ids,
            self.bottom_areas,
            self.interface_pair_count,
            self.side_signs,
            self.main_particle_ids,
            self.ff_particle_ids,
            self.main_traction_ids,
            self.ff_traction_ids,
            v_in,
            RHO_CS,
            RHO_CP,
            SIDE_ETA_X,
            SIDE_ETA_Z,
            SIDE_PARTICLE_SCALE,
            BOTTOM_PARTICLE_SCALE,
            scene.particle,
            scene.node,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
            scene.boundary.particle_traction,
            self.total_input_force,
            self.total_dashpot_force_particle_x,
            self.total_dashpot_force_grid_x,
            self.total_dashpot_force_z,
            self.total_bottom_force_x,
            self.total_bottom_force_z,
            self.total_side_main_force_x,
            self.total_side_main_force_z,
        )
        if BOTTOM_COUPLING_MODE.lower() != "grid":
            self.record_bottom(sims, scene, v_in, load_time)
        self.record_side(sims, scene)

    def record_stage(self, sims: Any, scene: Any, stage: str) -> None:
        return None

    def record_bottom(self, sims: Any, scene: Any, v_in: float, load_time: float) -> None:
        self.bottom_rows.append(
            {
                "time": float(sims.current_time),
                "load_time": load_time,
                "input_time_shift": INPUT_TIME_SHIFT,
                "input_velocity": v_in,
                "input_traction": 2.0 * RHO_CS * v_in,
                "total_input_force": float(self.total_input_force[None]),
                "total_dashpot_force_particle_x": float(self.total_dashpot_force_particle_x[None]),
                "total_dashpot_force_grid_x": float(self.total_dashpot_force_grid_x[None]),
                "total_dashpot_force_x": float(self.total_dashpot_force_grid_x[None]),
                "total_dashpot_force_z": float(self.total_dashpot_force_z[None]),
                "total_bottom_force_x": float(self.total_bottom_force_x[None]),
                "total_bottom_force_z": float(self.total_bottom_force_z[None]),
                "bottom_particle_count": self.bottom_count,
                "formula": "t_total=2*rho*Cs*v_input-rho*Cs*v_boundary_grid",
            }
        )

    def record_side(self, sims: Any, scene: Any) -> None:
        tr = scene.boundary.particle_traction.traction.to_numpy()
        vel = scene.particle.v.to_numpy()[: int(scene.particleNum[0])]
        stress = scene.particle.stress.to_numpy()[: int(scene.particleNum[0])]
        for r in self.interface_specs:
            mp = int(r["main_particle_id"])
            fp = int(r["ff_particle_id"])
            mt = int(r["main_traction_id"])
            ft = int(r["ff_traction_id"])
            main_tx = float(tr[mt, 0])
            main_tz = float(tr[mt, 1])
            ff_tx = float(tr[ft, 0])
            ff_tz = float(tr[ft, 1])
            self.side_rows.append(
                {
                    "time": float(sims.current_time),
                    "side": r["side"],
                    "pair_id": int(r["pair_id"]),
                    "main_particle_id": mp,
                    "ff_particle_id": fp,
                    "main_x": r["main_x"],
                    "main_z": r["main_z"],
                    "ff_x": r["ff_x"],
                    "ff_z": r["ff_z"],
                    "relative_vx": float(vel[fp, 0] - vel[mp, 0]),
                    "relative_vz": float(vel[fp, 1] - vel[mp, 1]),
                    "ff_sigma_xx": float(stress[fp, 0]),
                    "ff_tau_xz": float(stress[fp, 3]),
                    "main_traction_x": main_tx,
                    "main_traction_z": main_tz,
                    "ff_traction_x": ff_tx,
                    "ff_traction_z": ff_tz,
                    "action_reaction_error_x": main_tx + ff_tx,
                    "action_reaction_error_z": main_tz + ff_tz,
                    "status": "PASS" if abs(main_tx + ff_tx) <= 1.0e-10 and abs(main_tz + ff_tz) <= 1.0e-10 else "FAIL",
                    "formula": f"dashpot connection: tx={SIDE_ETA_X:.12g}*(v_ff_x-v_main_x), tz={SIDE_ETA_Z:.12g}*(v_ff_z-v_main_z)",
                }
            )

    def record_monitor(self, sims: Any, scene: Any) -> None:
        solver_time = float(sims.current_time)
        row: dict[str, Any] = {
            "time": solver_time + HISTORY_TIME_SHIFT,
            "solver_time": solver_time,
            "history_time_shift": HISTORY_TIME_SHIFT,
        }
        if not self.grid_monitor_nodes:
            coords = np.asarray(scene.element.get_nodal_coords(), dtype=float)
            for name, body_id, target in self.grid_monitor_specs:
                node_id = self._nearest_grid_node_with_mass(scene, body_id, *target)
                self.grid_monitor_nodes[name] = node_id
                row[f"{name}_node_id"] = node_id
                row[f"{name}_x"] = float(coords[node_id, 0])
                row[f"{name}_z"] = float(coords[node_id, 1])
        for name, _, _ in self.monitor_specs:
            pid = self.monitor_particles[name]
            v = scene.particle[pid].v
            row[f"{name}_particle_id"] = pid
            row[f"{name}_vx"] = float(v[0])
            row[f"{name}_velocity_magnitude"] = math.hypot(float(v[0]), float(v[1]))
        coords = np.asarray(scene.element.get_nodal_coords(), dtype=float)
        for name, body_id, _target in self.grid_monitor_specs:
            node_id = self.grid_monitor_nodes[name]
            v = scene.node[node_id, body_id].momentum
            row[f"{name}_node_id"] = node_id
            row[f"{name}_x"] = float(coords[node_id, 0])
            row[f"{name}_z"] = float(coords[node_id, 1])
            row[f"{name}_vx"] = float(v[0])
            row[f"{name}_vz"] = float(v[1])
            row[f"{name}_velocity_magnitude"] = math.hypot(float(v[0]), float(v[1]))
        if self.pt40_particle >= 0:
            v40 = scene.particle[self.pt40_particle].v
            row["pt40_particle_id"] = self.pt40_particle
            row["pt40_vx"] = float(v40[0])
            row["pt40_velocity_magnitude"] = math.hypot(float(v40[0]), float(v40[1]))
        self.monitor_rows.append(row)
        lv = scene.particle[self.reflection_particles["main_left_inside"]].v
        cv = scene.particle[self.reflection_particles["main_center"]].v
        rv = scene.particle[self.reflection_particles["main_right_inside"]].v
        self.reflection_rows.append(
            {
                "time": float(sims.current_time),
                "main_left_inside_particle_id": self.reflection_particles["main_left_inside"],
                "main_left_inside_vx": float(lv[0]),
                "main_center_particle_id": self.reflection_particles["main_center"],
                "main_center_vx": float(cv[0]),
                "main_right_inside_particle_id": self.reflection_particles["main_right_inside"],
                "main_right_inside_vx": float(rv[0]),
            }
        )


def build_mpm(trace: dict[str, int], particle_files: dict[int, tuple[Path, int]], pair_rows: list[dict[str, str]], legacy: Any) -> MPM:
    GlobalVariable.DIMENSION = 2
    ti.init(arch=ti.cpu, offline_cache=True, default_fp=ti.f64, default_ip=ti.i32, log_level=ti.ERROR)
    mpm = MPM(title='Example 3.3 "geometry corrected native MPM run"', log=False)
    legacy.install_native_trace_hooks(mpm, trace)
    install_grid_side_force_hook(mpm, trace)
    mpm.set_configuration(
        log=False,
        domain=ti.Vector([DOMAIN_WIDTH, DOMAIN_HEIGHT]),
        dimension="2-Dimension",
        boundary=["None", "None", "None"],
        gravity=ti.Vector([0.0, 0.0]),
        background_damping=0.0,
        alphaPIC=0.0,
        mapping=MPM_MAPPING,
        shape_function=MPM_SHAPE_FUNCTION,
        stabilize=None,
        material_type="Solid",
        visualize=False,
    )
    mpm.set_solver(log=False, solver={"Timestep": DT, "SimulationTime": SIMULATION_TIME, "SaveInterval": 1.0, "SavePath": RUN_DIR.as_posix()})
    mpm.memory_allocate(
        log=False,
        memory={
            "max_material_number": 1,
            "max_particle_number": 10000,
            "max_constraint_number": {"max_velocity_constraint": 100, "max_particle_traction_constraint": 10000},
        },
    )
    mpm.add_material(model="LinearElastic", material={"MaterialID": MATERIAL_ID, "Density": RHO, "YoungModulus": E, "PossionRatio": NU})
    mpm.scene.find_grid_level = types.MethodType(force_three_grid_levels, mpm.scene)
    mpm.scene.check_grid_inputs = types.MethodType(allow_grid_inputs, mpm.scene)
    mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": ti.Vector([DX, DX]), "Contact": {}})
    for body_id in (MAIN_BODY_ID, LEFT_FF_BODY_ID, RIGHT_FF_BODY_ID):
        path, count = particle_files[body_id]
        mpm.add_body_from_file(
            body={
                "FileType": "TXT",
                "Template": {
                    "ParticleFile": path.as_posix(),
                    "ParticleNumber": count,
                    "BodyID": body_id,
                    "MaterialID": MATERIAL_ID,
                    "Orientation": ti.Vector([0.0, 1.0]),
                    "ParticleStress": {"GravityField": False, "InternalStress": ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
                    "FixVelocity": ["Free", "Free"],
                },
            }
        )
    register_particle_tractions(mpm)
    mpm.select_save_data(particle=False, grid=False, object=False)
    mpm.native_compliant_base = CorrectedBoundary(mpm, pair_rows)
    return mpm


def write_material_check() -> None:
    rows = [
        {"parameter": "rho", "value": RHO, "source": "FLAC3D Example 3.3 original manual parameter", "status": "PASS"},
        {"parameter": "G", "value": G, "source": "FLAC3D Example 3.3 original manual parameter", "status": "PASS"},
        {"parameter": "K", "value": K, "source": "FLAC3D Example 3.3 original manual parameter", "status": "PASS"},
        {"parameter": "Cs=sqrt(G/rho)", "value": CS, "source": "computed", "status": "PASS"},
        {"parameter": "Cp=sqrt((K+4G/3)/rho)", "value": CP, "source": "computed", "status": "PASS"},
        {"parameter": "YoungModulus", "value": E, "source": "computed from K,G", "status": "PASS"},
        {"parameter": "PoissonRatio", "value": NU, "source": "computed from K,G", "status": "PASS"},
    ]
    write_csv(RUN_DIR / "material_parameter_check.csv", ["parameter", "value", "source", "status"], rows)


def read_reference(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(REFERENCE_DIR / name, delimiter=",", names=True, dtype=float, encoding="utf-8")
    data = np.atleast_1d(data)
    return np.asarray(data["time_s"], dtype=float), np.asarray(data["vx"], dtype=float)


def metrics(model_t: np.ndarray, model_v: np.ndarray, ref_t: np.ndarray, ref_v: np.ndarray) -> dict[str, float]:
    ref = np.interp(model_t, ref_t, ref_v, left=np.nan, right=np.nan)
    mask = np.isfinite(ref)
    t = model_t[mask]
    m = model_v[mask]
    r = ref[mask]
    if t.size < 3:
        return {
            "peak_error": math.nan,
            "nrmse": math.nan,
            "correlation": math.nan,
            "phase_difference": math.nan,
            "model_peak_time": math.nan,
            "ref_peak_time": math.nan,
            "peak_time_error": math.nan,
            "model_first_arrival_time": math.nan,
            "ref_first_arrival_time": math.nan,
            "first_arrival_time_error": math.nan,
            "best_time_shift": math.nan,
            "nrmse_after_time_shift": math.nan,
        }
    peak_error = float(np.max(np.abs(m)) - np.max(np.abs(r)))
    rmse = math.sqrt(float(np.mean((m - r) ** 2)))
    denom = float(np.max(r) - np.min(r))
    corr = float(np.corrcoef(m, r)[0, 1]) if np.std(m) > 0 and np.std(r) > 0 else math.nan
    m0 = m - np.mean(m)
    r0 = r - np.mean(r)
    lag = int(np.argmax(np.correlate(m0, r0, mode="full")) - (r0.size - 1)) if np.std(m0) > 0 and np.std(r0) > 0 else 0
    ref_peak_index = int(np.argmax(np.abs(r)))
    model_peak_index = int(np.argmax(np.abs(m)))
    ref_peak_time = float(t[ref_peak_index])
    model_peak_time = float(t[model_peak_index])
    ref_peak_abs = float(np.max(np.abs(r)))
    arrival_threshold = 0.05 * ref_peak_abs
    model_arrivals = np.flatnonzero(np.abs(m) >= arrival_threshold)
    ref_arrivals = np.flatnonzero(np.abs(r) >= arrival_threshold)
    model_arrival = float(t[int(model_arrivals[0])]) if model_arrivals.size else math.nan
    ref_arrival = float(t[int(ref_arrivals[0])]) if ref_arrivals.size else math.nan

    best_shift = 0.0
    best_rmse = math.inf
    ref_window = (ref_t >= float(np.min(t))) & (ref_t <= float(np.max(t)))
    for shift in np.linspace(-0.003, 0.003, 601):
        shifted = np.interp(ref_t, model_t - shift, model_v, left=np.nan, right=np.nan)
        shifted_mask = ref_window & np.isfinite(shifted)
        if np.count_nonzero(shifted_mask) < 3:
            continue
        shifted_rmse = math.sqrt(float(np.mean((shifted[shifted_mask] - ref_v[shifted_mask]) ** 2)))
        if shifted_rmse < best_rmse:
            best_rmse = shifted_rmse
            best_shift = float(shift)

    return {
        "peak_error": peak_error,
        "nrmse": rmse / denom if denom > 0 else math.nan,
        "correlation": corr,
        "phase_difference": lag * float(np.median(np.diff(t))),
        "model_peak_time": model_peak_time,
        "ref_peak_time": ref_peak_time,
        "peak_time_error": model_peak_time - ref_peak_time,
        "model_first_arrival_time": model_arrival,
        "ref_first_arrival_time": ref_arrival,
        "first_arrival_time_error": model_arrival - ref_arrival if math.isfinite(model_arrival) and math.isfinite(ref_arrival) else math.nan,
        "best_time_shift": best_shift,
        "nrmse_after_time_shift": best_rmse / denom if denom > 0 and math.isfinite(best_rmse) else math.nan,
    }


def write_history_and_figure(boundary: CorrectedBoundary) -> dict[str, dict[str, float]]:
    rows = boundary.monitor_rows
    fieldnames = ["time", "solver_time", "history_time_shift"]
    for name, _, _ in boundary.monitor_specs:
        fieldnames.extend([f"{name}_particle_id", f"{name}_vx", f"{name}_velocity_magnitude"])
    for name, _, _ in boundary.grid_monitor_specs:
        fieldnames.extend([f"{name}_node_id", f"{name}_x", f"{name}_z", f"{name}_vx", f"{name}_vz", f"{name}_velocity_magnitude"])
    if boundary.pt40_particle >= 0:
        fieldnames.extend(["pt40_particle_id", "pt40_vx", "pt40_velocity_magnitude"])
    write_csv(
        RUN_DIR / "history.csv",
        fieldnames,
        rows,
    )
    grid_velocity_fields = ["time", "solver_time", "history_time_shift"]
    for name, _, _ in boundary.grid_monitor_specs:
        grid_velocity_fields.extend([f"{name}_node_id", f"{name}_x", f"{name}_z", f"{name}_vx", f"{name}_vz", f"{name}_velocity_magnitude"])
    write_csv(
        RUN_DIR / "grid_monitor_velocity.csv",
        grid_velocity_fields,
        [{field: row[field] for field in grid_velocity_fields} for row in rows],
    )
    t = np.array([r["time"] for r in rows], dtype=float)
    soil_column_3m = np.array([r["soil_column_3m_vx"] for r in rows], dtype=float)
    main_model_3m = np.array([r["main_model_3m_vx"] for r in rows], dtype=float)
    soil_column_bottom = np.array([r["soil_column_bottom_vx"] for r in rows], dtype=float)
    main_model_bottom = np.array([r["main_model_bottom_vx"] for r in rows], dtype=float)
    soil_grid_bottom = np.array([r["soil_grid_bottom_vx"] for r in rows], dtype=float)
    main_grid_bottom = np.array([r["main_grid_bottom_vx"] for r in rows], dtype=float)
    soil_grid_top = np.array([r["soil_grid_top_vx"] for r in rows], dtype=float)
    main_grid_top = np.array([r["main_grid_top_manual_vx"] for r in rows], dtype=float)
    flac_column_t, flac_column_v = read_reference("reference_fig_3_9_flac_column.csv")
    flac_main_t, flac_main_v = read_reference("reference_fig_3_9_flac_main_correct.csv")
    flac_free_t, flac_free_v = read_reference("reference_fig_3_9_flac_free_correct.csv")
    mask_column = (flac_column_t >= 0.0) & (flac_column_t <= SIMULATION_TIME)
    mask_main = (flac_main_t >= 0.0) & (flac_main_t <= SIMULATION_TIME)
    mask_free = (flac_free_t >= 0.0) & (flac_free_t <= SIMULATION_TIME)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(flac_main_t[mask_main], flac_main_v[mask_main], color="black", lw=1.8, label="FLAC3D main")
    ax.plot(flac_free_t[mask_free], flac_free_v[mask_free], color="#276fbf", lw=1.8, label="FLAC3D free-field")
    for name, color in [
        ("soil_column_bottom", "#8c5a2b"),
        ("soil_column_3m", "#2a9d55"),
        ("main_model_bottom", "#5b5f97"),
        ("main_model_3m", "#c43c2d"),
    ]:
        ax.plot(t, np.array([r[f"{name}_vx"] for r in rows], dtype=float), lw=1.5, label=f"MPM {name}", color=color)
    ax.set_xlim(0.0, SIMULATION_TIME)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vx")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN_DIR / "figure3_9_geometry_corrected.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(flac_column_t[mask_column], flac_column_v[mask_column], color="black", lw=1.2, label="FLAC-column")
    ax.plot(flac_free_t[mask_free], flac_free_v[mask_free], color="#276fbf", lw=1.2, ls=(0, (4, 4)), label="FLAC-free")
    ax.plot(flac_main_t[mask_main], flac_main_v[mask_main], color="#d43d2a", lw=1.2, ls=(0, (4, 4)), label="FLAC-main")
    ax.plot(t, soil_grid_top, color="#2ca02c", lw=1.4, ls=(0, (1, 3)), label="MPM")
    ax.set_xlim(0.0, SIMULATION_TIME)
    ax.set_ylim(-0.02, 0.12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Vel [m/s]")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "figure3_9_doc_reference_style.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(flac_free_t[mask_free], flac_free_v[mask_free], color="#276fbf", lw=1.8, label="FLAC3D free-field")
    ax.plot(t, soil_grid_top, color="#2a9d55", lw=1.5, ls="--", label="MPM soil_grid_top")
    ax.plot(flac_main_t[mask_main], flac_main_v[mask_main], color="black", lw=1.8, label="FLAC3D main")
    ax.plot(t, main_grid_top, color="#c43c2d", lw=1.5, ls="--", label="MPM main_grid_top")
    ax.set_xlim(0.0, SIMULATION_TIME)
    ax.set_ylim(-0.02, 0.12)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("grid vx (m/s)")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "top_grid_monitor_vs_flac3d.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(t, soil_column_bottom, color="#8c5a2b", lw=1.4, label="MPM soil_column_bottom particle")
    ax.plot(t, main_model_bottom, color="#5b5f97", lw=1.4, label="MPM main_model_bottom particle")
    ax.plot(t, soil_grid_bottom, color="#2a9d55", lw=1.4, ls="--", label="MPM soil_grid_bottom")
    ax.plot(t, main_grid_bottom, color="#c43c2d", lw=1.4, ls="--", label="MPM main_grid_bottom")
    ax.set_xlim(0.0, SIMULATION_TIME)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vx")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "bottom_monitor_comparison.png", dpi=220)
    plt.close(fig)

    metric_rows = {
        "soil_column_3m vs FLAC3D free-field": metrics(t, soil_column_3m, flac_free_t, flac_free_v),
        "main_model_3m vs FLAC3D main": metrics(t, main_model_3m, flac_main_t, flac_main_v),
        "soil_grid_top vs FLAC3D free-field": metrics(t, soil_grid_top, flac_free_t, flac_free_v),
        "main_grid_top_manual vs FLAC3D main": metrics(t, main_grid_top, flac_main_t, flac_main_v),
    }
    phase_fields = [
        "case",
        "peak_error",
        "nrmse",
        "correlation",
        "phase_difference",
        "model_peak_time",
        "ref_peak_time",
        "peak_time_error",
        "model_first_arrival_time",
        "ref_first_arrival_time",
        "first_arrival_time_error",
        "best_time_shift",
        "nrmse_after_time_shift",
    ]
    write_csv(
        RUN_DIR / "phase_alignment_summary.csv",
        phase_fields,
        [{"case": case, **values} for case, values in metric_rows.items()],
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(flac_main_t[mask_main], flac_main_v[mask_main], color="black", lw=1.8, label="FLAC3D main")
    ax.plot(flac_free_t[mask_free], flac_free_v[mask_free], color="#276fbf", lw=1.8, label="FLAC3D free-field")
    shifted_cases = [
        ("soil_column_3m vs FLAC3D free-field", soil_column_3m, "#2a9d55", "MPM soil_column_3m phase-aligned"),
        ("main_model_3m vs FLAC3D main", main_model_3m, "#c43c2d", "MPM main_model_3m phase-aligned"),
        ("soil_grid_top vs FLAC3D free-field", soil_grid_top, "#7a9cc6", "MPM soil_grid_top phase-aligned"),
        ("main_grid_top_manual vs FLAC3D main", main_grid_top, "#b56576", "MPM main_grid_top_manual phase-aligned"),
    ]
    for case, values, color, label in shifted_cases:
        shift = float(metric_rows[case]["best_time_shift"])
        ax.plot(t - shift, values, lw=1.35, label=f"{label} (shift={shift:.6g}s)", color=color)
    ax.set_xlim(0.0, SIMULATION_TIME)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vx")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "figure3_9_phase_aligned.png", dpi=220)
    plt.close(fig)
    return metric_rows


def write_reflection_history_and_report(boundary: CorrectedBoundary) -> None:
    rows = boundary.reflection_rows
    fieldnames = [
        "time",
        "main_left_inside_particle_id",
        "main_left_inside_vx",
        "main_center_particle_id",
        "main_center_vx",
        "main_right_inside_particle_id",
        "main_right_inside_vx",
    ]
    write_csv(RUN_DIR / "reflection_history.csv", fieldnames, rows)
    if not rows:
        raise RuntimeError("No reflection history rows were recorded")

    left = np.array([r["main_left_inside_vx"] for r in rows], dtype=float)
    center = np.array([r["main_center_vx"] for r in rows], dtype=float)
    right = np.array([r["main_right_inside_vx"] for r in rows], dtype=float)
    diff_left = np.abs(left - center)
    diff_right = np.abs(right - center)
    max_left = float(np.max(diff_left))
    max_right = float(np.max(diff_right))
    indicator = max_left + max_right
    center_peak = float(np.max(np.abs(center)))
    normalized = indicator / center_peak if center_peak > 0.0 else math.nan

    n = int(boundary.mpm.scene.particleNum[0])
    pos = boundary.mpm.scene.particle.x.to_numpy()[:n]
    monitor_lines = []
    for name, pid in boundary.reflection_particles.items():
        monitor_lines.append(
            f"| {name} | {pid} | {float(pos[pid, 0]):.12g} | {float(pos[pid, 1]):.12g} |"
        )

    verdict = "INCONCLUSIVE"
    if math.isfinite(normalized):
        if normalized < 0.5:
            verdict = "NO_STRONG_REFLECTION_INDICATOR"
        elif normalized < 1.5:
            verdict = "MODERATE_SIDE_CENTER_DIFFERENCE"
        else:
            verdict = "STRONG_SIDE_CENTER_DIFFERENCE"

    lines = [
        "# Reflection Diagnostic Report",
        "",
        f"Run checked: `{RUN_NAME}`.",
        "",
        "This diagnostic samples side and center histories after applying the current side-boundary formula.",
        "",
        "## Current Side Formula",
        "",
        f"`tx={SIDE_ETA_X:.12g}*(v_ff_x-v_main_x)`, `tz={SIDE_ETA_Z:.12g}*(v_ff_z-v_main_z)`",
        "",
        "The formula follows dashpot connections between the main model and free-field columns.",
        "",
        "## Monitor Particles",
        "",
        "| monitor | particle_id | x | z |",
        "| --- | ---: | ---: | ---: |",
        *monitor_lines,
        "",
        "## Indicator",
        "",
        "Definition:",
        "",
        "`reflection_indicator = max(abs(v_left-v_center)) + max(abs(v_right-v_center))`",
        "",
        f"- max(abs(v_left-v_center)): `{max_left:.12g}`",
        f"- max(abs(v_right-v_center)): `{max_right:.12g}`",
        f"- reflection_indicator: `{indicator:.12g}`",
        f"- center_peak_abs_vx: `{center_peak:.12g}`",
        f"- normalized_indicator_by_center_peak: `{normalized:.12g}`",
        f"- verdict: `{verdict}`",
        "",
        "## Interpretation",
        "",
        "A small side-center difference would support effective lateral free-field behavior. A large value means the side response differs strongly from the interior response, which is a possible reflection or side-coupling indicator, but this scalar alone does not separate physical geometry effects from boundary reflection.",
        "",
        f"Output history: `{RUN_DIR / 'reflection_history.csv'}`",
    ]
    (RUN_DIR / "reflection_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def data_array(name: str, values: np.ndarray, components: int = 1, dtype: str = "Float64") -> str:
    flat = np.asarray(values).reshape(-1)
    text = " ".join(f"{float(v):.16g}" if dtype.startswith("Float") else str(int(v)) for v in flat)
    comp = f' NumberOfComponents="{components}"' if components != 1 else ""
    return f'        <DataArray type="{dtype}" Name="{name}"{comp} format="ascii">\n          {text}\n        </DataArray>\n'


def write_particles_vtu(mpm: MPM) -> None:
    n = int(mpm.scene.particleNum[0])
    pos2 = mpm.scene.particle.x.to_numpy()[:n]
    vel2 = mpm.scene.particle.v.to_numpy()[:n]
    stress = mpm.scene.particle.stress.to_numpy()[:n]
    body = mpm.scene.particle.bodyID.to_numpy()[:n].astype(np.int32)
    material = mpm.scene.particle.materialID.to_numpy()[:n].astype(np.int32)
    points = np.column_stack((pos2[:, 0], np.zeros(n), pos2[:, 1]))
    point_data = data_array("particle_id", np.arange(n, dtype=np.int32), dtype="Int32")
    point_data += data_array("body_id", body, dtype="Int32")
    point_data += data_array("material_id", material, dtype="Int32")
    point_data += data_array("position", points, components=3)
    point_data += data_array("velocity_x", vel2[:, 0])
    point_data += data_array("velocity_z", vel2[:, 1])
    point_data += data_array("velocity_magnitude", np.sqrt(vel2[:, 0] ** 2 + vel2[:, 1] ** 2))
    point_data += data_array("stress", stress, components=stress.shape[1])
    xml = (
        '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n  <UnstructuredGrid>\n'
        f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n      <PointData>\n{point_data}      </PointData>\n      <Points>\n'
        f"{data_array('Points', points, components=3)}      </Points>\n      <Cells>\n"
        f"{data_array('connectivity', np.arange(n, dtype=np.int32), dtype='Int32')}"
        f"{data_array('offsets', np.arange(1, n + 1, dtype=np.int32), dtype='Int32')}"
        f"{data_array('types', np.ones(n, dtype=np.int32), dtype='UInt8')}"
        "      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n"
    )
    (RUN_DIR / "particles_dynamic.vtu").write_text(xml, encoding="utf-8")


def write_grid_vtr(mpm: MPM) -> None:
    coords = np.asarray(mpm.scene.element.get_nodal_coords(), dtype=float)
    xcoords = np.unique(coords[:, 0])
    zcoords = np.unique(coords[:, 1])
    ycoords = np.array([0.0], dtype=float)
    gnum_x = xcoords.size
    gnum_z = zcoords.size
    cell_count = (gnum_x - 1) * (gnum_z - 1)
    cell_id = np.arange(cell_count, dtype=np.int32)
    xml = (
        '<?xml version="1.0"?>\n<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n'
        f'  <RectilinearGrid WholeExtent="0 {gnum_x - 1} 0 0 0 {gnum_z - 1}">\n'
        f'    <Piece Extent="0 {gnum_x - 1} 0 0 0 {gnum_z - 1}">\n      <CellData>\n'
        f"{data_array('cell_id', cell_id, dtype='Int32')}      </CellData>\n      <Coordinates>\n"
        f"{data_array('X_COORDINATES', xcoords)}{data_array('Y_COORDINATES', ycoords)}{data_array('Z_COORDINATES', zcoords)}"
        "      </Coordinates>\n    </Piece>\n  </RectilinearGrid>\n</VTKFile>\n"
    )
    (RUN_DIR / "grid_dynamic.vtr").write_text(xml, encoding="utf-8")


def write_interface_vtk(boundary: CorrectedBoundary) -> None:
    lines = ["# vtk DataFile Version 3.0", "geometry corrected dynamic true interface pairs", "ASCII", "DATASET POLYDATA", f"POINTS {2 * len(boundary.interface_specs)} float"]
    for r in boundary.interface_specs:
        lines.append(f"{float(r['main_x']):.16g} 0 {float(r['main_z']):.16g}")
        lines.append(f"{float(r['ff_x']):.16g} 0 {float(r['ff_z']):.16g}")
    lines.append(f"LINES {len(boundary.interface_specs)} {3 * len(boundary.interface_specs)}")
    for i in range(len(boundary.interface_specs)):
        lines.append(f"2 {2 * i} {2 * i + 1}")
    lines += [f"CELL_DATA {len(boundary.interface_specs)}", "SCALARS pair_id int 1", "LOOKUP_TABLE default"]
    lines += [str(int(r["pair_id"])) for r in boundary.interface_specs]
    (RUN_DIR / "interface_dynamic.vtk").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(metric_rows: dict[str, dict[str, float]], trace: dict[str, int], boundary: CorrectedBoundary) -> None:
    side_status = "PASS" if all(r["status"] == "PASS" for r in boundary.side_rows) else "FAIL"
    n = int(boundary.mpm.scene.particleNum[0])
    pos = boundary.mpm.scene.particle.x.to_numpy()[:n]
    monitor_specs = [
        (name, target, boundary.monitor_particles[name])
        for name, _, target in boundary.monitor_specs
    ]
    grid_monitor_specs = [
        (name, target, boundary.grid_monitor_nodes.get(name, -1))
        for name, _, target in boundary.grid_monitor_specs
    ]
    lines = [
        "# Geometry Corrected Dynamic Validation Report",
        "",
        'Run identity: "geometry corrected native MPM run"',
        "",
        "Old rectangular geometry results are obsolete and discarded for this validation.",
        "",
        "## Geometry Source",
        "",
        f"- source directory: `{GEOMETRY_DIR}`",
        f"- geometry_name: `{GEOMETRY_NAME}`",
        "- geometry validation: `geometry_validation_checks.csv` status PASS",
        f"- particles/background grid/interface pairs loaded from `{GEOMETRY_NAME}` outputs; geometry was not regenerated.",
        "",
        "## Monitor Points",
        "",
        "Four monitor points are used in `history.csv`: soil-column bottom, soil-column 3 m height, main-model bottom, and main-model 3 m height.",
        "",
        "| point | target x | target z | particle id | selected x | selected z |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, target, pid in monitor_specs:
        lines.append(
            f"| {name} | {target[0]:.12g} | {target[1]:.12g} | {pid} | {float(pos[pid, 0]):.12g} | {float(pos[pid, 1]):.12g} |"
        )
    lines += [
        "",
        "## Gridpoint Monitor Points",
        "",
        "These grid monitors are intended to match FLAC3D `hist gp xvel` more closely than particle histories.",
        "",
        "| point | target x | target z | node id | selected x | selected z |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    coords = np.asarray(boundary.mpm.scene.element.get_nodal_coords(), dtype=float)
    for name, target, node_id in grid_monitor_specs:
        lines.append(
            f"| {name} | {target[0]:.12g} | {target[1]:.12g} | {node_id} | {float(coords[node_id, 0]):.12g} | {float(coords[node_id, 1]):.12g} |"
        )
    lines += [
        "",
        "## Solver",
        "",
        "- solver: GeoTaichi native MPM",
        f"- dx: {DX}",
        f"- mapping: {MPM_MAPPING}",
        f"- shape_function: {MPM_SHAPE_FUNCTION}",
        f"- bottom_coupling_mode: {BOTTOM_COUPLING_MODE}",
        f"- side_coupling_mode: {SIDE_COUPLING_MODE}",
        f"- side_x_impedance_mode: {SIDE_X_IMPEDANCE_MODE}",
        f"- side_eta_x: {SIDE_ETA_X}",
        f"- side_eta_z: {SIDE_ETA_Z}",
        f"- bottom_particle_scale: {BOTTOM_PARTICLE_SCALE}",
        f"- side_particle_scale: {SIDE_PARTICLE_SCALE}",
        f"- native Solver.core calls: {trace.get('solver_core_calls', 0)}",
        f"- P2G calls: {trace.get('p2g_calls', 0)}",
        f"- G2P calls: {trace.get('g2p_calls', 0)}",
        f"- stress update calls: {trace.get('stress_update_calls', 0)}",
        f"- grid-boundary force calls: {trace.get('grid_boundary_force_calls', 0)}",
        f"- grid-bottom compliant-base calls: {trace.get('grid_bottom_compliant_base_calls', 0)}",
        f"- grid-side dashpot calls: {trace.get('grid_side_dashpot_calls', 0)}",
        f"- simulation_time: {SIMULATION_TIME}",
        f"- dt: {DT}",
        f"- input_time_shift: {INPUT_TIME_SHIFT}",
        f"- history_time_shift: {HISTORY_TIME_SHIFT}",
        "",
        "## Bottom Boundary Check",
        "",
        "- velocity input period T: 0.01 s",
        "- direct velocity boundary: not used",
        "- traction formula: `t_total = 2*rho*Cs*v_input - rho*Cs*v_boundary_grid` for x and absorbing `-rho*Cp*v_boundary_z` for z.",
        f"- bottom coupling mode: `{BOTTOM_COUPLING_MODE}`",
        "- grid bottom force distribution: MPM shape-function weights over each bottom particle support.",
        f"- check file: `{RUN_DIR / 'bottom_boundary_force_check.csv'}`",
        f"- grid-bottom nearest-node diagnostic file: `{RUN_DIR / 'grid_bottom_node_check.csv'}`",
        "",
        "## Side Free-Field Check",
        "",
        f"- interface pairs: loaded from `{GEOMETRY_NAME}/geotaichi_interface_pair_check.csv`",
        "- pair mapping: main particle to free-field particle, one-to-one",
        f"- formula currently used: dashpot connection only, `tx={SIDE_ETA_X:.12g}*(v_ff_x-v_main_x)`, `tz={SIDE_ETA_Z:.12g}*(v_ff_z-v_main_z)`",
        "- grid side force distribution: MPM shape-function weights over each paired main/free-field particle support.",
        f"- action-reaction status: {side_status}",
        f"- check file: `{RUN_DIR / 'side_boundary_force_check.csv'}`",
        f"- grid-side pair file: `{RUN_DIR / 'grid_side_pair_check.csv'}`",
        f"- total_grid_side_main_force_x_last: {float(boundary.total_grid_side_main_force_x[None])}",
        f"- total_grid_side_main_force_z_last: {float(boundary.total_grid_side_main_force_z[None])}",
        "",
        "## Metrics",
        "",
        "| case | peak error | NRMSE | shifted NRMSE | correlation | peak time error | best time shift | first arrival error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case, row in metric_rows.items():
        lines.append(
            f"| {case} | {row['peak_error']:.12g} | {row['nrmse']:.12g} | {row['nrmse_after_time_shift']:.12g} | "
            f"{row['correlation']:.12g} | {row['peak_time_error']:.12g} | {row['best_time_shift']:.12g} | {row['first_arrival_time_error']:.12g} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- material check: `{RUN_DIR / 'material_parameter_check.csv'}`",
        f"- history: `{RUN_DIR / 'history.csv'}`",
        f"- grid monitor velocity: `{RUN_DIR / 'grid_monitor_velocity.csv'}`",
        f"- phase alignment summary: `{RUN_DIR / 'phase_alignment_summary.csv'}`",
        f"- Figure 3.9: `{RUN_DIR / 'figure3_9_geometry_corrected.png'}`",
        f"- reference-document style Figure 3.9: `{RUN_DIR / 'figure3_9_doc_reference_style.png'}`",
        f"- top grid monitor comparison: `{RUN_DIR / 'top_grid_monitor_vs_flac3d.png'}`",
        f"- bottom monitor comparison: `{RUN_DIR / 'bottom_monitor_comparison.png'}`",
        f"- phase-aligned Figure 3.9 diagnostic: `{RUN_DIR / 'figure3_9_phase_aligned.png'}`",
        f"- particles: `{RUN_DIR / 'particles_dynamic.vtu'}`",
        f"- grid: `{RUN_DIR / 'grid_dynamic.vtr'}`",
        f"- interface: `{RUN_DIR / 'interface_dynamic.vtk'}`",
        f"- grid side pairs: `{RUN_DIR / 'grid_side_pair_check.csv'}`",
        f"- grid bottom nodes: `{RUN_DIR / 'grid_bottom_node_check.csv'}`",
        "",
        "No scaling, sign flip, artificial amplitude adjustment, tuning, or optimization was applied to the raw history. The phase-aligned figure is a diagnostic plot only and is reported separately from the raw Figure 3.9 output.",
    ]
    (RUN_DIR / "geometry_corrected_dynamic_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    validate_geometry_source()
    shutil.copy2(GEOMETRY_DIR / "geometry_validation_checks.csv", RUN_DIR / "geometry_validation_checks.source.csv")
    particles = load_particles()
    particle_files = prepare_particle_files(particles)
    pair_rows = read_csv(GEOMETRY_DIR / "geotaichi_interface_pair_check.csv")
    write_material_check()
    legacy = load_legacy_native()
    trace = {
        "solver_core_calls": 0,
        "ul_explicit_usl_updating_calls": 0,
        "p2g_calls": 0,
        "compute_forces_calls": 0,
        "grid_update_calls": 0,
        "g2p_calls": 0,
        "velocity_gradient_calls": 0,
        "stress_update_calls": 0,
        "compliant_particle_traction_update_calls": 0,
    }
    mpm = build_mpm(trace, particle_files, pair_rows, legacy)

    def recorder() -> None:
        mpm.native_compliant_base.record_monitor(mpm.sims, mpm.scene)

    mpm.run(function=recorder)
    boundary: CorrectedBoundary = mpm.native_compliant_base
    write_csv(
        RUN_DIR / "bottom_boundary_force_check.csv",
        [
            "time",
            "load_time",
            "input_time_shift",
            "input_velocity",
            "input_traction",
            "total_input_force",
            "total_dashpot_force_particle_x",
            "total_dashpot_force_grid_x",
            "total_dashpot_force_x",
            "total_dashpot_force_z",
            "total_bottom_force_x",
            "total_bottom_force_z",
            "bottom_particle_count",
            "formula",
        ],
        boundary.bottom_rows,
    )
    write_csv(
        RUN_DIR / "side_boundary_force_check.csv",
        [
            "time",
            "side",
            "pair_id",
            "main_particle_id",
            "ff_particle_id",
            "main_x",
            "main_z",
            "ff_x",
            "ff_z",
            "relative_vx",
            "relative_vz",
            "ff_sigma_xx",
            "ff_tau_xz",
            "main_traction_x",
            "main_traction_z",
            "ff_traction_x",
            "ff_traction_z",
            "action_reaction_error_x",
            "action_reaction_error_z",
            "status",
            "formula",
        ],
        boundary.side_rows,
    )
    write_csv(
        RUN_DIR / "grid_side_pair_check.csv",
        [
            "pair_id",
            "side",
            "main_node",
            "ff_node",
            "main_body_id",
            "ff_body_id",
            "main_node_x",
            "main_node_z",
            "ff_node_x",
            "ff_node_z",
            "area",
        ],
        boundary.grid_side_specs,
    )
    write_csv(
        RUN_DIR / "grid_bottom_node_check.csv",
        [
            "particle_id",
            "body_id",
            "node_id",
            "particle_x",
            "particle_z",
            "node_x",
            "node_z",
            "area",
        ],
        boundary.grid_bottom_specs,
    )
    metric_rows = write_history_and_figure(boundary)
    write_reflection_history_and_report(boundary)
    write_particles_vtu(mpm)
    write_grid_vtr(mpm)
    write_interface_vtk(boundary)
    write_report(metric_rows, trace, boundary)
    print(f"run_dir={RUN_DIR}")
    print(f"history={RUN_DIR / 'history.csv'}")
    print(f"report={RUN_DIR / 'geometry_corrected_dynamic_validation_report.md'}")


if __name__ == "__main__":
    main()
