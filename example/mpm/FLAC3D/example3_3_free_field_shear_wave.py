"""FLAC3D manual Example 3.3 reproduced with GeoTaichi MPM.

Source: C:/Users/Dell/Desktop/example3.3.pdf

FLAC3D command block extracted from the manual:

    def iniwave
      per = 0.01
    end
    def wave
      wave = 0.5 * (1.0 - cos(2*pi*dytime/per))
    end

    gen zone brick size 6 3 2
    gen zone brick size 2 3 2 p0 0 0 2
    gen zone brick size 2 3 2 p0 4 0 2
    gen zone wedge size 1 3 2 p0 2 0 2
    gen zone wedge size 1 3 2 p0 4 3 2 p1 3 3 2 p2 4 0 2 p3 4 3 4 &
      p4 3 0 2 p5 4 0 4

    model elastic prop bulk 66667 shear 40000 ini dens 0.0025
    set grav 0 0 -10
    fix x range x -0.01 0.01
    fix x range x 5.99 6.01
    fix y range y -0.01 0.01
    fix y range y 2.99 3.01
    fix z range z -0.1 0.1

    set dyn on
    free x y z range z -0.1 0.1
    apply nquiet squiet dquiet range z -0.1 0.1
    apply dstress 1.0 hist wave range z -0.1 0.1
    apply ff
    group ff_corner
    group ff_side range x 0 6
    group ff_side range y 0 3
    group main_grid range x 0 6 y 0 3
    set dyn time = 0
    solve age 0.015

GeoTaichi MPM does not expose FLAC3D's command-level `apply ff` free-field
boundary or `nquiet/squiet/dquiet` base boundary. This script preserves the
documented geometry, material, gravity, shear-wave function, and final dynamic
time. The 3D FLAC3D model is reduced to its x-z section because the geometry
and loading are uniform in y. One explicit side column is added at each lateral
side to represent the free-field side histories from the manual figure.
"""

from pathlib import Path
import csv
import math
import os
import sys

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geotaichi import *


OUTPUT_PATH = Path(
    os.environ.get(
        "EXAMPLE3_3_OUTPUT",
        "example/mpm/FLAC3D/results/example3_3_free_field_shear_wave",
    )
)

# Set EXAMPLE3_3_SMOKE=1 for a short assembly check. Default is the manual run.
SIMULATION_TIME = 0.001 if os.environ.get("EXAMPLE3_3_SMOKE") == "1" else 0.015
SAVE_INTERVAL = SIMULATION_TIME
HISTORY_INTERVAL = 1.0e-4

# FLAC3D material parameters from `model elastic prop bulk 66667 shear 40000
# ini dens 0.0025`.
DENSITY = 0.0025
BULK_MODULUS = 66667.0
SHEAR_MODULUS = 40000.0
GRAVITY_Z = -10.0

# FLAC3D wave function from Example 3.3.
WAVE_PERIOD = 0.01
DSTRESS_AMPLITUDE = 1.0

# Coordinate convention: GeoTaichi 2D x-y maps FLAC3D x-z. The FLAC3D y
# dimension is uniform and is represented by plane strain.
MAIN_X0 = 1.0
MAIN_WIDTH = 6.0
FREE_FIELD_WIDTH = 1.0
DOMAIN_WIDTH = MAIN_WIDTH + 2.0 * FREE_FIELD_WIDTH
DOMAIN_HEIGHT = 4.0
ELEMENT_SIZE = ti.Vector([1.0, 1.0])
PARTICLES_PER_CELL = 2

# A large initial value is intentionally provided; MPM.check_critical_timestep()
# reduces it using the existing GeoTaichi CFL and material sound speed logic.
INITIAL_TIMESTEP = 1.0

# Default follows the manual: update a bottom x-direction traction with
# dstress * wave(t). The optional impedance velocity mode is fully derived from
# tau = rho * Vs * v and is useful when testing the MPM velocity-boundary path.
LOADING_MODE = os.environ.get("EXAMPLE3_3_LOADING", "stress").lower()


def elastic_constants_from_bulk_shear(bulk, shear):
    young = 9.0 * bulk * shear / (3.0 * bulk + shear)
    poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))
    return young, poisson


YOUNG_MODULUS, POISSON_RATIO = elastic_constants_from_bulk_shear(
    BULK_MODULUS, SHEAR_MODULUS
)
SHEAR_WAVE_VELOCITY = math.sqrt(SHEAR_MODULUS / DENSITY)
P_WAVE_VELOCITY = math.sqrt((BULK_MODULUS + 4.0 * SHEAR_MODULUS / 3.0) / DENSITY)
IMPEDANCE_VELOCITY_AMPLITUDE = DSTRESS_AMPLITUDE / (
    DENSITY * SHEAR_WAVE_VELOCITY
)


def wave_factor(time):
    if time < 0.0 or time > WAVE_PERIOD:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * time / WAVE_PERIOD))


def bottom_shear_stress(time):
    return DSTRESS_AMPLITUDE * wave_factor(time)


def bottom_impedance_velocity(time):
    return IMPEDANCE_VELOCITY_AMPLITUDE * wave_factor(time)


def shifted_x(x):
    return MAIN_X0 + x


def body_template(region_name, body_id, material_id=1):
    return {
        "RegionName": region_name,
        "nParticlesPerCell": PARTICLES_PER_CELL,
        "BodyID": body_id,
        "MaterialID": material_id,
        "ParticleStress": {
            "GravityField": True,
            "InternalStress": ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "Traction": {},
        },
        "Traction": {},
        "InitialVelocity": ti.Vector([0.0, 0.0]),
        "FixVelocity": ["Free", "Free"],
    }


def in_main_geometry(x, y):
    """Return True for the FLAC3D main grid x-z section, shifted by MAIN_X0."""
    xf = x - MAIN_X0
    if 0.0 <= xf <= 6.0 and 0.0 <= y <= 2.0:
        return True
    if 0.0 <= xf <= 2.0 and 2.0 <= y <= 4.0:
        return True
    if 4.0 <= xf <= 6.0 and 2.0 <= y <= 4.0:
        return True
    if 2.0 <= xf <= 3.0 and 2.0 <= y <= 4.0 - 2.0 * (xf - 2.0):
        return True
    if 3.0 <= xf <= 4.0 and 2.0 <= y <= 2.0 + 2.0 * (xf - 3.0):
        return True
    return False


def in_free_field_geometry(x, y):
    return (
        0.0 <= y <= DOMAIN_HEIGHT
        and (0.0 <= x <= FREE_FIELD_WIDTH or MAIN_X0 + MAIN_WIDTH <= x <= DOMAIN_WIDTH)
    )


def generate_particle_file():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    spacing = 1.0 / PARTICLES_PER_CELL
    psize = spacing
    volume = spacing * spacing
    rows = []
    nx = int(round(DOMAIN_WIDTH))
    ny = int(round(DOMAIN_HEIGHT))
    for ix in range(nx):
        for iy in range(ny):
            for sx in range(PARTICLES_PER_CELL):
                for sy in range(PARTICLES_PER_CELL):
                    x = ix + (sx + 0.5) * spacing
                    y = iy + (sy + 0.5) * spacing
                    if in_main_geometry(x, y) or in_free_field_geometry(x, y):
                        rows.append([x, y, volume, psize, psize, 0.0, 0.0, 0.0])
    particle_file = OUTPUT_PATH / "particles_example3_3.txt"
    np.savetxt(particle_file, np.array(rows, dtype=float), fmt="%.16e")
    return particle_file, len(rows)


@ti.kernel
def update_x_traction(constraints: ti.template(), nconstraints: int, value: float):
    for nboundary in range(nconstraints):
        if int(constraints[nboundary].dirs) == 0:
            constraints[nboundary].traction = value


@ti.kernel
def update_x_velocity(constraints: ti.template(), nconstraints: int, value: float):
    for nboundary in range(nconstraints):
        if int(constraints[nboundary].dirs) == 0:
            constraints[nboundary].velocity = value


def apply_dynamic_base_loading(model, time):
    if LOADING_MODE == "stress":
        nconstraints = int(model.scene.boundary.traction_list[0])
        update_x_traction(
            model.scene.boundary.traction_boundary,
            nconstraints,
            bottom_shear_stress(time),
        )
    elif LOADING_MODE == "velocity":
        nconstraints = int(model.scene.boundary.velocity_list[0])
        update_x_velocity(
            model.scene.boundary.velocity_boundary,
            nconstraints,
            bottom_impedance_velocity(time),
        )
    else:
        raise ValueError("EXAMPLE3_3_LOADING must be 'stress' or 'velocity'")


def mean_or_zero(values):
    return float(values.mean()) if values.size else 0.0


def mask_box(position, x0, x1, y0, y1):
    return (
        (position[:, 0] >= x0)
        & (position[:, 0] <= x1)
        & (position[:, 1] >= y0)
        & (position[:, 1] <= y1)
    )


def collect_histories(model, rows, previous, forced=False):
    time = float(model.sims.current_time)
    if (
        not forced
        and time - previous.get("_last_history_time", -1.0e30)
        < HISTORY_INTERVAL - 1.0e-12
    ):
        return

    particle_num = int(model.scene.particleNum[0])
    position = model.scene.particle.x.to_numpy()[:particle_num]
    velocity = model.scene.particle.v.to_numpy()[:particle_num]
    stress = model.scene.particle.stress.to_numpy()[:particle_num]

    top_y0 = DOMAIN_HEIGHT - 0.5
    masks = {
        "main_top": mask_box(position, shifted_x(1.5), shifted_x(4.5), top_y0, 4.1),
        "left_ff_top": mask_box(position, 0.0, FREE_FIELD_WIDTH, top_y0, 4.1),
        "right_ff_top": mask_box(position, MAIN_X0 + MAIN_WIDTH, DOMAIN_WIDTH, top_y0, 4.1),
        "base": mask_box(position, 0.0, DOMAIN_WIDTH, 0.0, 0.5),
    }

    dt_hist = max(time - previous.get("_last_history_time", time), float(model.sims.delta))
    row = {
        "time": time,
        "wave": wave_factor(time),
        "bottom_dstress": bottom_shear_stress(time),
        "bottom_impedance_velocity": bottom_impedance_velocity(time),
    }
    for key, mask in masks.items():
        vx = mean_or_zero(velocity[mask, 0])
        vy = mean_or_zero(velocity[mask, 1])
        tau_xy = mean_or_zero(stress[mask, 3]) if stress.size else 0.0
        previous_vx = previous.get(f"{key}_vx", vx)
        row[f"{key}_vx"] = vx
        row[f"{key}_vy"] = vy
        row[f"{key}_ax"] = (vx - previous_vx) / dt_hist
        row[f"{key}_tau_xy"] = tau_xy
        previous[f"{key}_vx"] = vx

    rows.append(row)
    previous["_last_history_time"] = time


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, particle_num, history_rows):
    lines = [
        "FLAC3D manual Example 3.3: Shear wave loading of a model with free-field boundaries",
        "source_pdf = C:/Users/Dell/Desktop/example3.3.pdf",
        "model_reduction = 2D x-z section of the 3D FLAC3D model; y is uniform in the source command block",
        f"loading_mode = {LOADING_MODE}",
        f"bulk_modulus = {BULK_MODULUS}",
        f"shear_modulus = {SHEAR_MODULUS}",
        f"density = {DENSITY}",
        f"young_modulus = {YOUNG_MODULUS:.12g}",
        f"poisson_ratio = {POISSON_RATIO:.12g}",
        f"shear_wave_velocity = {SHEAR_WAVE_VELOCITY:.12g}",
        f"p_wave_velocity = {P_WAVE_VELOCITY:.12g}",
        f"gravity = [0, {GRAVITY_Z}]",
        f"wave_period = {WAVE_PERIOD}",
        "wave = 0.5 * (1.0 - cos(2*pi*time/wave_period)) for 0 <= time <= wave_period",
        f"dstress_amplitude = {DSTRESS_AMPLITUDE}",
        f"impedance_velocity_amplitude = {IMPEDANCE_VELOCITY_AMPLITUDE:.12g}",
        f"simulation_time = {SIMULATION_TIME}",
        f"particle_number = {particle_num}",
        f"history_rows = {history_rows}",
        f"output_path = {OUTPUT_PATH.as_posix()}",
        "geoTaichi_note = FLAC3D apply ff and nquiet/squiet/dquiet are not command-level APIs in current GeoTaichi MPM",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


PARTICLE_FILE, PARTICLE_COUNT = generate_particle_file()

init(dim=2, device_memory_GB=2)

mpm = MPM(title="FLAC3D Example 3.3 free-field shear-wave benchmark")

mpm.set_configuration(
    domain=ti.Vector([DOMAIN_WIDTH, DOMAIN_HEIGHT]),
    dimension="2-Dimension",
    is_2DAxisy=False,
    boundary=["None", "None", "None"],
    gravity=ti.Vector([0.0, GRAVITY_Z]),
    background_damping=0.0,
    alphaPIC=0.0,
    mapping="USF",
    shape_function="Linear",
    stabilize=None,
    material_type="Solid",
)

mpm.set_solver(
    solver={
        "Timestep": INITIAL_TIMESTEP,
        "SimulationTime": SIMULATION_TIME,
        "SaveInterval": SAVE_INTERVAL,
        "SavePath": OUTPUT_PATH.as_posix(),
    }
)

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": PARTICLE_COUNT,
        "max_constraint_number": {
            "max_velocity_constraint": 200,
            "max_traction_constraint": 200,
        },
    }
)

mpm.add_material(
    model="LinearElastic",
    material={
        "MaterialID": 1,
        "Density": DENSITY,
        "YoungModulus": YOUNG_MODULUS,
        "PossionRatio": POISSON_RATIO,
    },
)

mpm.add_element(
    element={
        "ElementType": "Q4N2D",
        "ElementSize": ELEMENT_SIZE,
        "Contact": {},
    }
)

mpm.add_body_from_file(
    body={
        "FileType": "TXT",
        "Template": {
            "ParticleFile": PARTICLE_FILE.as_posix(),
            "ParticleNumber": PARTICLE_COUNT,
            "BodyID": 0,
            "MaterialID": 1,
            "ParticleStress": {
                "GravityField": True,
                "InternalStress": ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            "FixVelocity": ["Free", "Free"],
        },
    }
)

boundary = [
    # FLAC3D's static `fix z range z -0.1 0.1` mapped to vertical velocity.
    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [None, 0.0],
        "StartPoint": [0.0, 0.0],
        "EndPoint": [DOMAIN_WIDTH, 0.0],
    },
]

if LOADING_MODE == "stress":
    boundary.append(
        {
            "BoundaryType": "TractionConstraint",
            "ExternalForce": [bottom_shear_stress(0.0), None],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [DOMAIN_WIDTH, 0.0],
        }
    )
else:
    boundary.append(
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [bottom_impedance_velocity(0.0), None],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [DOMAIN_WIDTH, 0.0],
        }
    )

mpm.add_boundary_condition(boundary=boundary)
mpm.select_save_data(particle=False, grid=True)

histories = []
previous_history = {}


def per_step_update():
    time = float(mpm.sims.current_time + mpm.sims.delta)
    apply_dynamic_base_loading(mpm, time)
    collect_histories(mpm, histories, previous_history)


OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
mpm.run(function=per_step_update)
collect_histories(mpm, histories, previous_history, forced=True)

write_csv(OUTPUT_PATH / "histories.csv", histories)
write_summary(
    OUTPUT_PATH / "validation_summary.txt",
    int(mpm.scene.particleNum[0]),
    len(histories),
)

mpm.postprocessing(read_path=OUTPUT_PATH.as_posix(), write_background_grid=True)
