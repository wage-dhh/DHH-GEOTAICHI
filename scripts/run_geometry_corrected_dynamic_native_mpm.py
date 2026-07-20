import csv
import importlib.util
import math
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

GEOMETRY_DIR = ROOT / "output" / "example3_3_2d_kohler_native_mpm_solver" / "geometry_only_corrected"
RUN_DIR = ROOT / "output" / "example3_3_2d_kohler_native_mpm_solver" / "new_geometry_corrected_dynamic_run"
REFERENCE_DIR = ROOT / "data" / "reference" / "flac3d_example3_3_free_field"
LEGACY_NATIVE = ROOT / "examples" / "example3_3_2d_kohler_native_mpm_solver.py"

DX = 0.25
GRID_CELLS_X = 42
GRID_CELLS_Z = 32
DOMAIN_WIDTH = GRID_CELLS_X * DX
DOMAIN_HEIGHT = GRID_CELLS_Z * DX
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

DT = 1.0e-5
SIMULATION_TIME = 0.015
INPUT_PERIOD = 0.01
INPUT_VELOCITY_AMPLITUDE = 0.05
BOTTOM_Z = 1.875
LEFT_MAIN_X = 1.125
LEFT_FF_X = 0.375
RIGHT_MAIN_X = 8.875
RIGHT_FF_X = 9.625
TOP_MAIN_MODEL_TARGET = (2.0, 6.75)
TOP_SOIL_COLUMN_TARGET = (0.375, 6.625)
REFLECTION_MONITOR_Z = 6.625
REFLECTION_TARGETS = {
    "main_left_inside": (1.25, REFLECTION_MONITOR_Z),
    "main_center": (5.0, REFLECTION_MONITOR_Z),
    "main_right_inside": (8.75, REFLECTION_MONITOR_Z),
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
    particle: ti.template(),
    particle_traction: ti.template(),
    total_input_force: ti.template(),
    total_dashpot_force_x: ti.template(),
    total_dashpot_force_z: ti.template(),
    total_bottom_force_x: ti.template(),
    total_bottom_force_z: ti.template(),
    total_side_main_force_x: ti.template(),
    total_side_main_force_z: ti.template(),
):
    total_input_force[None] = 0.0
    total_dashpot_force_x[None] = 0.0
    total_dashpot_force_z[None] = 0.0
    total_bottom_force_x[None] = 0.0
    total_bottom_force_z[None] = 0.0
    total_side_main_force_x[None] = 0.0
    total_side_main_force_z[None] = 0.0

    for i in range(bottom_count):
        pid = bottom_particle_ids[i]
        tid = bottom_traction_ids[i]
        area = bottom_areas[i]
        input_tx = 2.0 * rho_cs * input_velocity
        dashpot_tx = -rho_cs * particle[pid].v[0]
        dashpot_tz = -rho_cp * particle[pid].v[1]
        tx = input_tx + dashpot_tx
        tz = dashpot_tz
        particle_traction[tid].traction = ti.Vector([tx, tz])
        total_input_force[None] += area * input_tx
        total_dashpot_force_x[None] += area * dashpot_tx
        total_dashpot_force_z[None] += area * dashpot_tz
        total_bottom_force_x[None] += area * tx
        total_bottom_force_z[None] += area * tz

    for i in range(pair_count):
        main_pid = main_particle_ids[i]
        ff_pid = ff_particle_ids[i]
        mtid = main_traction_ids[i]
        ftid = ff_traction_ids[i]
        n = side_signs[i]
        vrel = particle[ff_pid].v - particle[main_pid].v
        sigma_ff = particle[ff_pid].stress
        sigma_main = particle[main_pid].stress
        stress_diff = sigma_ff - sigma_main
        tx = n * stress_diff[0] + rho_cp * vrel[0]
        tz = n * stress_diff[3] + rho_cs * vrel[1]
        traction = ti.Vector([tx, tz])
        particle_traction[mtid].traction = traction
        particle_traction[ftid].traction = -traction
        total_side_main_force_x[None] += DX * tx
        total_side_main_force_z[None] += DX * tz


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
        self.total_dashpot_force_x = ti.field(dtype=ti.f64, shape=())
        self.total_dashpot_force_z = ti.field(dtype=ti.f64, shape=())
        self.total_bottom_force_x = ti.field(dtype=ti.f64, shape=())
        self.total_bottom_force_z = ti.field(dtype=ti.f64, shape=())
        self.total_side_main_force_x = ti.field(dtype=ti.f64, shape=())
        self.total_side_main_force_z = ti.field(dtype=ti.f64, shape=())

        self.bottom_rows: list[dict[str, Any]] = []
        self.side_rows: list[dict[str, Any]] = []
        self.monitor_rows: list[dict[str, Any]] = []
        self.reflection_rows: list[dict[str, Any]] = []
        self.main_model_top_particle = self._nearest_particle(MAIN_BODY_ID, *TOP_MAIN_MODEL_TARGET)
        self.soil_column_top_particle = self._nearest_particle(LEFT_FF_BODY_ID, *TOP_SOIL_COLUMN_TARGET)
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
                specs.append({"particle_id": pid, "traction_id": traction_indices[pid], "body_id": int(body[pid]), "area": float(vol[pid]) / DX})
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

    def apply(self, sims: Any, scene: Any) -> None:
        v_in = input_velocity(float(sims.current_time))
        update_corrected_particle_tractions(
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
            scene.particle,
            scene.boundary.particle_traction,
            self.total_input_force,
            self.total_dashpot_force_x,
            self.total_dashpot_force_z,
            self.total_bottom_force_x,
            self.total_bottom_force_z,
            self.total_side_main_force_x,
            self.total_side_main_force_z,
        )
        self.record_bottom(sims, scene, v_in)
        self.record_side(sims, scene)

    def record_stage(self, sims: Any, scene: Any, stage: str) -> None:
        return None

    def record_bottom(self, sims: Any, scene: Any, v_in: float) -> None:
        self.bottom_rows.append(
            {
                "time": float(sims.current_time),
                "input_velocity": v_in,
                "input_traction": 2.0 * RHO_CS * v_in,
                "total_input_force": float(self.total_input_force[None]),
                "total_dashpot_force_x": float(self.total_dashpot_force_x[None]),
                "total_dashpot_force_z": float(self.total_dashpot_force_z[None]),
                "total_bottom_force_x": float(self.total_bottom_force_x[None]),
                "total_bottom_force_z": float(self.total_bottom_force_z[None]),
                "bottom_particle_count": self.bottom_count,
                "formula": "t_total=2*rho*Cs*v_input-rho*C*v_particle",
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
                    "formula": "sigma_ff*n+rho*C*(v_ff-v_main)",
                }
            )

    def record_monitor(self, sims: Any, scene: Any) -> None:
        mv = scene.particle[self.main_model_top_particle].v
        sv = scene.particle[self.soil_column_top_particle].v
        self.monitor_rows.append(
            {
                "time": float(sims.current_time),
                "soil_column_top_particle_id": self.soil_column_top_particle,
                "soil_column_top_vx": float(sv[0]),
                "soil_column_top_velocity_magnitude": math.hypot(float(sv[0]), float(sv[1])),
                "main_model_top_particle_id": self.main_model_top_particle,
                "main_model_top_vx": float(mv[0]),
                "main_model_top_velocity_magnitude": math.hypot(float(mv[0]), float(mv[1])),
            }
        )
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
    mpm.set_configuration(
        log=False,
        domain=ti.Vector([DOMAIN_WIDTH, DOMAIN_HEIGHT]),
        dimension="2-Dimension",
        boundary=["None", "None", "None"],
        gravity=ti.Vector([0.0, 0.0]),
        background_damping=0.0,
        alphaPIC=0.0,
        mapping="USL",
        shape_function="Linear",
        stabilize=None,
        material_type="Solid",
        visualize=False,
    )
    mpm.set_solver(log=False, solver={"Timestep": DT, "SimulationTime": SIMULATION_TIME, "SaveInterval": 1.0, "SavePath": RUN_DIR.as_posix()})
    mpm.memory_allocate(
        log=False,
        memory={
            "max_material_number": 1,
            "max_particle_number": 2000,
            "max_constraint_number": {"max_velocity_constraint": 100, "max_particle_traction_constraint": 1000},
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
        return {"peak_error": math.nan, "nrmse": math.nan, "correlation": math.nan, "phase_difference": math.nan}
    peak_error = float(np.max(np.abs(m)) - np.max(np.abs(r)))
    rmse = math.sqrt(float(np.mean((m - r) ** 2)))
    denom = float(np.max(r) - np.min(r))
    corr = float(np.corrcoef(m, r)[0, 1]) if np.std(m) > 0 and np.std(r) > 0 else math.nan
    m0 = m - np.mean(m)
    r0 = r - np.mean(r)
    lag = int(np.argmax(np.correlate(m0, r0, mode="full")) - (r0.size - 1)) if np.std(m0) > 0 and np.std(r0) > 0 else 0
    return {"peak_error": peak_error, "nrmse": rmse / denom if denom > 0 else math.nan, "correlation": corr, "phase_difference": lag * float(np.median(np.diff(t)))}


def write_history_and_figure(boundary: CorrectedBoundary) -> dict[str, dict[str, float]]:
    rows = boundary.monitor_rows
    write_csv(
        RUN_DIR / "history.csv",
        [
            "time",
            "soil_column_top_particle_id",
            "soil_column_top_vx",
            "soil_column_top_velocity_magnitude",
            "main_model_top_particle_id",
            "main_model_top_vx",
            "main_model_top_velocity_magnitude",
        ],
        rows,
    )
    t = np.array([r["time"] for r in rows], dtype=float)
    soil_column = np.array([r["soil_column_top_vx"] for r in rows], dtype=float)
    main_model = np.array([r["main_model_top_vx"] for r in rows], dtype=float)
    flac_main_t, flac_main_v = read_reference("reference_fig_3_9_flac_main.csv")
    flac_free_t, flac_free_v = read_reference("reference_fig_3_9_flac_free.csv")
    mask_main = (flac_main_t >= 0.0) & (flac_main_t <= SIMULATION_TIME)
    mask_free = (flac_free_t >= 0.0) & (flac_free_t <= SIMULATION_TIME)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(flac_main_t[mask_main], flac_main_v[mask_main], color="black", lw=1.8, label="FLAC3D main")
    ax.plot(t, main_model, color="#c43c2d", lw=1.6, label="MPM main model top")
    ax.plot(flac_free_t[mask_free], flac_free_v[mask_free], color="#276fbf", lw=1.8, label="FLAC3D free-field")
    ax.plot(t, soil_column, color="#2a9d55", lw=1.6, label="MPM soil column top")
    ax.set_xlim(0.0, SIMULATION_TIME)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vx")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN_DIR / "figure3_9_geometry_corrected.png", dpi=220)
    plt.close(fig)
    return {
        "soil_column_top vs FLAC3D free-field": metrics(t, soil_column, flac_free_t, flac_free_v),
        "main_model_top vs FLAC3D main": metrics(t, main_model, flac_main_t, flac_main_v),
    }


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
        "Run checked: `new_geometry_corrected_dynamic_run`.",
        "",
        "Only history sampling was added. Solver, geometry, material, boundary formula, and side traction formula were not changed.",
        "",
        "## Current Side Formula",
        "",
        "`sigma_ff*n + rho*C*(v_ff-v_main)`",
        "",
        "This diagnostic does not change that formula.",
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
        ("soil_column_top", TOP_SOIL_COLUMN_TARGET, boundary.soil_column_top_particle),
        ("main_model_top", TOP_MAIN_MODEL_TARGET, boundary.main_model_top_particle),
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
        "- geometry validation: `geometry_validation_checks.csv` status PASS",
        "- particles/background grid/interface pairs loaded from geometry_only_corrected outputs; geometry was not regenerated.",
        "",
        "## Monitor Points",
        "",
        "Only two monitor points are used in `history.csv`: soil-column top and main-model top.",
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
        "## Solver",
        "",
        "- solver: GeoTaichi native MPM",
        f"- native Solver.core calls: {trace.get('solver_core_calls', 0)}",
        f"- P2G calls: {trace.get('p2g_calls', 0)}",
        f"- G2P calls: {trace.get('g2p_calls', 0)}",
        f"- stress update calls: {trace.get('stress_update_calls', 0)}",
        f"- simulation_time: {SIMULATION_TIME}",
        f"- dt: {DT}",
        "",
        "## Bottom Boundary Check",
        "",
        "- velocity input period T: 0.01 s",
        "- direct velocity boundary: not used",
        "- traction formula: `t_total = 2*rho*Cs*v_input - rho*Cs*v_particle` for x and absorbing `-rho*Cp*vz` for z.",
        f"- check file: `{RUN_DIR / 'bottom_boundary_force_check.csv'}`",
        "",
        "## Side Free-Field Check",
        "",
        "- interface pairs: loaded from `geometry_only_corrected/geotaichi_interface_pair_check.csv`",
        "- pair mapping: main particle to free-field particle, one-to-one",
        "- formula currently used: `sigma_ff*n + rho*C*(v_ff-v_main)`",
        f"- action-reaction status: {side_status}",
        f"- check file: `{RUN_DIR / 'side_boundary_force_check.csv'}`",
        "",
        "## Metrics",
        "",
        "| case | peak error | NRMSE | correlation | phase difference |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case, row in metric_rows.items():
        lines.append(f"| {case} | {row['peak_error']:.12g} | {row['nrmse']:.12g} | {row['correlation']:.12g} | {row['phase_difference']:.12g} |")
    lines += [
        "",
        "## Files",
        "",
        f"- material check: `{RUN_DIR / 'material_parameter_check.csv'}`",
        f"- history: `{RUN_DIR / 'history.csv'}`",
        f"- Figure 3.9: `{RUN_DIR / 'figure3_9_geometry_corrected.png'}`",
        f"- particles: `{RUN_DIR / 'particles_dynamic.vtu'}`",
        f"- grid: `{RUN_DIR / 'grid_dynamic.vtr'}`",
        f"- interface: `{RUN_DIR / 'interface_dynamic.vtk'}`",
        "",
        "No scaling, translation, sign flip, artificial amplitude adjustment, tuning, or optimization was applied.",
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
        ["time", "input_velocity", "input_traction", "total_input_force", "total_dashpot_force_x", "total_dashpot_force_z", "total_bottom_force_x", "total_bottom_force_z", "bottom_particle_count", "formula"],
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
