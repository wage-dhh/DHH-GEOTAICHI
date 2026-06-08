from pathlib import Path
import math
import os

import numpy as np
import taichi as ti

from src import MPM
import src.utils.GlobalVariable as GlobalVariable


# Target: FLAC3D manual, Dynamic Analysis verification problem
# Figure 3.49 in the desktop FLAC3D v3.1 manual:
# "Input acceleration at bottom of model".
#
# Parameters are taken from the FLAC3D manual example "Comparison of FLAC3D
# to SHAKE for a Layered, Linear-Elastic Soil Deposit". The current GeoTaichi
# API uses velocity constraints rather than a direct dynamic acceleration face
# load, so the manual acceleration is integrated once and imposed as bottom
# horizontal velocity.
OUTPUT_PATH = Path(os.environ.get("PIPLE_OUTPUT", "code/results/flac3d_figure_3_49_mpm"))
DESKTOP = Path(r"C:\Users\Dell\Desktop")
PDF_PATH = DESKTOP / "FLAC3D manual_Ver_301420.pdf"
FIGURE_TITLE = "Figure 3.49 Input acceleration at bottom of model"

FT_TO_M = 0.3048
G0 = 9.80665

DOMAIN_WIDTH = 3.048
DOMAIN_HEIGHT = 160.0 * FT_TO_M
ZONE_HEIGHT = 10.0 * FT_TO_M
PARTICLES_PER_CELL = 1

SOFT_TOP_Y0 = 120.0 * FT_TO_M
STIFF_Y0 = 80.0 * FT_TO_M
STIFF_Y1 = 120.0 * FT_TO_M

MAT1_DENSITY = 1800.0
MAT1_SHEAR_MODULUS = 150.0e6
MAT1_BULK_MODULUS = 150.0e6
MAT2_DENSITY = 2000.0
MAT2_SHEAR_MODULUS = 300.0e6
MAT2_BULK_MODULUS = 300.0e6
RAYLEIGH_DAMPING_RATIO = 0.10
RAYLEIGH_CENTER_FREQUENCY = 3.0

FULL_TIME = 14.0
SMOKE_TIME = 0.20
TOTAL_TIME = SMOKE_TIME if os.environ.get("PIPLE_SMOKE", "0") == "1" else FULL_TIME
DT = float(os.environ.get("PIPLE_DT", "2e-4"))
SAVE_INTERVAL = float(os.environ.get("PIPLE_SAVE_INTERVAL", "0.25"))
HISTORY_INTERVAL = float(os.environ.get("PIPLE_HISTORY_INTERVAL", "0.01"))
ACCEL_DT = float(os.environ.get("PIPLE_ACCEL_DT", "0.005"))

TOP_HISTORY_Y = DOMAIN_HEIGHT
STRESS_HISTORY_Y = 37.0


def elastic_constants_from_bulk_shear(bulk_modulus, shear_modulus):
    young = 9.0 * bulk_modulus * shear_modulus / (3.0 * bulk_modulus + shear_modulus)
    poisson = (3.0 * bulk_modulus - 2.0 * shear_modulus) / (
        2.0 * (3.0 * bulk_modulus + shear_modulus)
    )
    return young, poisson


MAT1_YOUNG_MODULUS, MAT1_POISSON_RATIO = elastic_constants_from_bulk_shear(
    MAT1_BULK_MODULUS, MAT1_SHEAR_MODULUS
)
MAT2_YOUNG_MODULUS, MAT2_POISSON_RATIO = elastic_constants_from_bulk_shear(
    MAT2_BULK_MODULUS, MAT2_SHEAR_MODULUS
)
MAT1_SHEAR_WAVE_VELOCITY = math.sqrt(MAT1_SHEAR_MODULUS / MAT1_DENSITY)
MAT2_SHEAR_WAVE_VELOCITY = math.sqrt(MAT2_SHEAR_MODULUS / MAT2_DENSITY)


def flac3d_figure_349_acceleration_g(time):
    if time < 0.0:
        return 0.0
    envelope = math.sqrt(0.375 * math.exp(-2.2 * time) * time**8)
    return envelope * math.sin(6.0 * math.pi * time)


def flac3d_figure_349_acceleration(time):
    return G0 * flac3d_figure_349_acceleration_g(time)


def integrate_trapezoid(times, values):
    result = np.zeros_like(values, dtype=float)
    for i in range(1, len(values)):
        dt = times[i] - times[i - 1]
        result[i] = result[i - 1] + 0.5 * (values[i] + values[i - 1]) * dt
    return result


def build_input_motion():
    times = np.arange(0.0, FULL_TIME + 0.5 * ACCEL_DT, ACCEL_DT, dtype=float)
    accel = np.array([flac3d_figure_349_acceleration(t) for t in times], dtype=float)
    velocity = integrate_trapezoid(times, accel)
    displacement = integrate_trapezoid(times, velocity)
    return times, accel, velocity, displacement


MOTION_TIME, MOTION_ACCELERATION, MOTION_VELOCITY, MOTION_DISPLACEMENT = build_input_motion()


def input_base_acceleration(time):
    return float(np.interp(time, MOTION_TIME, MOTION_ACCELERATION, left=0.0, right=0.0))


def input_base_velocity(time):
    return float(np.interp(time, MOTION_TIME, MOTION_VELOCITY, left=0.0, right=0.0))


def write_base_motion_csv():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    lines = ["time,base_acceleration_mps2,base_acceleration_g,integrated_base_velocity,integrated_base_displacement"]
    for time, accel, velocity, disp in zip(
        MOTION_TIME, MOTION_ACCELERATION, MOTION_VELOCITY, MOTION_DISPLACEMENT
    ):
        lines.append(f"{time:.18e},{accel:.18e},{accel / G0:.18e},{velocity:.18e},{disp:.18e}")
    (OUTPUT_PATH / "base_motion.csv").write_text("\n".join(lines), encoding="utf-8")


def body_template(region_name, material_id):
    return {
        "RegionName": region_name,
        "nParticlesPerCell": PARTICLES_PER_CELL,
        "BodyID": 0,
        "MaterialID": material_id,
        "ParticleStress": {
            "GravityField": False,
            "InternalStress": ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "Traction": {},
        },
        "Traction": {},
        "InitialVelocity": ti.Vector([0.0, 0.0]),
        "FixVelocity": ["Free", "Free"],
    }


@ti.kernel
def update_bottom_x_velocity(constraints: ti.template(), nconstraints: int, velocity: float):
    for nboundary in range(nconstraints):
        if int(constraints[nboundary].dirs) == 0:
            constraints[nboundary].velocity = velocity


@ti.kernel
def update_bottom_particle_x_velocity(particle: ti.template(), particle_num: int, velocity: float, y_limit: float):
    for npart in range(particle_num):
        if particle[npart].x[1] <= y_limit:
            particle[npart].v[0] = velocity


def apply_bottom_velocity(model, time):
    velocity = input_base_velocity(time)
    nconstraints = int(model.scene.boundary.velocity_list[0])
    update_bottom_x_velocity(model.scene.boundary.velocity_boundary, nconstraints, velocity)
    update_bottom_particle_x_velocity(
        model.scene.particle,
        int(model.scene.particleNum[0]),
        velocity,
        1.5 * ZONE_HEIGHT,
    )
    return velocity


def mean_or_zero(values):
    return float(values.mean()) if values.size else 0.0


def layer_mask(position, y0, y1):
    y = position[:, 1]
    return (y >= y0) & (y <= y1)


def collect_response(model, histories, previous, forced=False):
    time = float(model.sims.current_time)
    last = previous.get("_last_history_time", -1.0e30)
    if not forced and time - last + 0.1 * DT < HISTORY_INTERVAL:
        return

    particle_num = int(model.scene.particleNum[0])
    position = model.scene.particle.x.to_numpy()[:particle_num]
    velocity = model.scene.particle.v.to_numpy()[:particle_num]
    stress = model.scene.particle.stress.to_numpy()[:particle_num]

    masks = {
        "base": layer_mask(position, 0.0, 1.5 * ZONE_HEIGHT),
        "top": layer_mask(position, DOMAIN_HEIGHT - 1.5 * ZONE_HEIGHT, DOMAIN_HEIGHT + 1.0),
        "stress_point": layer_mask(position, STRESS_HISTORY_Y - 0.75 * ZONE_HEIGHT, STRESS_HISTORY_Y + 0.75 * ZONE_HEIGHT),
    }
    row = {
        "time": time,
        "input_base_ax": input_base_acceleration(time),
        "input_base_ax_g": input_base_acceleration(time) / G0,
        "input_base_vx": input_base_velocity(time),
    }
    dt_hist = max(time - previous.get("_last_history_time", time), DT)
    for key, mask in masks.items():
        vx = mean_or_zero(velocity[mask, 0])
        prev_vx = previous.get(f"{key}_vx", vx)
        prev_xdisp = previous.get(f"{key}_xdisp", 0.0)
        shear_stress = mean_or_zero(stress[mask, 3]) if stress.size else 0.0
        shear_modulus = MAT2_SHEAR_MODULUS if key == "stress_point" and STIFF_Y0 <= STRESS_HISTORY_Y <= STIFF_Y1 else MAT1_SHEAR_MODULUS
        row[f"{key}_vx"] = vx
        row[f"{key}_ax"] = (vx - prev_vx) / dt_hist
        row[f"{key}_ax_g"] = row[f"{key}_ax"] / G0
        row[f"{key}_xdisp"] = prev_xdisp + 0.5 * (vx + prev_vx) * dt_hist
        row[f"{key}_shear_stress"] = shear_stress
        row[f"{key}_strain"] = shear_stress / shear_modulus
        previous[f"{key}_vx"] = vx
        previous[f"{key}_xdisp"] = row[f"{key}_xdisp"]
    histories.append(row)
    previous["_last_history_time"] = time


def write_histories_csv(histories):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    keys = [
        "time",
        "input_base_ax",
        "input_base_ax_g",
        "input_base_vx",
        "base_vx",
        "base_ax",
        "base_ax_g",
        "base_xdisp",
        "top_vx",
        "top_ax",
        "top_ax_g",
        "top_xdisp",
        "stress_point_vx",
        "stress_point_ax",
        "stress_point_ax_g",
        "stress_point_xdisp",
        "stress_point_strain",
        "stress_point_shear_stress",
    ]
    lines = [",".join(keys)]
    for row in histories:
        lines.append(",".join(f"{row.get(key, 0.0):.12g}" for key in keys))
    (OUTPUT_PATH / "histories.csv").write_text("\n".join(lines), encoding="utf-8")


def write_validation_summary(model, histories):
    values = {key: [row[key] for row in histories] for key in histories[0] if key != "time"} if histories else {}
    equivalent_path = OUTPUT_PATH / "equivalent_1d_response.csv"
    lines = [
        "FLAC3D Figure 3.49 layered linear-elastic soil benchmark reproduced with GeoTaichi MPM",
        f"figure = {FIGURE_TITLE}",
        f"pdf_exists = {PDF_PATH.exists()}",
        f"pdf_path = {PDF_PATH}",
        "source = FLAC3D manual, Comparison of FLAC3D to SHAKE for a Layered, Linear-Elastic Soil Deposit",
        "input_acceleration_g = sqrt(0.375 * exp(-2.2*t) * t^8) * sin(6*pi*t)",
        f"input_acceleration_g_range = [{float((MOTION_ACCELERATION / G0).min()):.12g}, {float((MOTION_ACCELERATION / G0).max()):.12g}]",
        f"input_acceleration_mps2_range = [{float(MOTION_ACCELERATION.min()):.12g}, {float(MOTION_ACCELERATION.max()):.12g}]",
        f"integrated_base_velocity_range = [{float(MOTION_VELOCITY.min()):.12g}, {float(MOTION_VELOCITY.max()):.12g}]",
        f"integrated_base_displacement_final = {float(MOTION_DISPLACEMENT[-1]):.12g}",
        f"domain_width = {DOMAIN_WIDTH}",
        f"domain_height = {DOMAIN_HEIGHT}",
        f"zone_height = {ZONE_HEIGHT}",
        f"material1_density = {MAT1_DENSITY}",
        f"material1_bulk_modulus = {MAT1_BULK_MODULUS}",
        f"material1_shear_modulus = {MAT1_SHEAR_MODULUS}",
        f"material1_young_modulus = {MAT1_YOUNG_MODULUS}",
        f"material1_poisson_ratio = {MAT1_POISSON_RATIO}",
        f"material1_shear_wave_velocity = {MAT1_SHEAR_WAVE_VELOCITY}",
        f"material2_density = {MAT2_DENSITY}",
        f"material2_bulk_modulus = {MAT2_BULK_MODULUS}",
        f"material2_shear_modulus = {MAT2_SHEAR_MODULUS}",
        f"material2_young_modulus = {MAT2_YOUNG_MODULUS}",
        f"material2_poisson_ratio = {MAT2_POISSON_RATIO}",
        f"material2_shear_wave_velocity = {MAT2_SHEAR_WAVE_VELOCITY}",
        f"rayleigh_damping_ratio_reference = {RAYLEIGH_DAMPING_RATIO}",
        f"rayleigh_center_frequency_reference = {RAYLEIGH_CENTER_FREQUENCY}",
        "rayleigh_note = reference FLAC3D uses Rayleigh damping; this GeoTaichi script does not expose the same command-level damping API",
        f"particle_number = {int(model.scene.particleNum[0])}",
        f"dt = {DT}",
        f"final_time = {model.sims.current_time}",
        f"history_rows = {len(histories)}",
        f"save_path = {OUTPUT_PATH.as_posix()}",
        f"base_motion_csv = {(OUTPUT_PATH / 'base_motion.csv').as_posix()}",
        f"history_csv = {(OUTPUT_PATH / 'histories.csv').as_posix()}",
        f"equivalent_1d_response_csv = {equivalent_path.as_posix()}",
    ]
    for key, series in values.items():
        lines.append(f"{key}_range = [{min(series):.12g}, {max(series):.12g}]")
    (OUTPUT_PATH / "validation_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def material_at_element(element_index):
    y_mid = (element_index + 0.5) * ZONE_HEIGHT
    if STIFF_Y0 <= y_mid < STIFF_Y1:
        return MAT2_DENSITY, MAT2_SHEAR_MODULUS, 2
    return MAT1_DENSITY, MAT1_SHEAR_MODULUS, 1


def solve_equivalent_1d_layered_response():
    """Linear shear-column surrogate for the full FLAC3D dynamic response."""
    n_elem = int(round(DOMAIN_HEIGHT / ZONE_HEIGHT))
    n_node = n_elem + 1
    area = DOMAIN_WIDTH
    mass = np.zeros(n_node, dtype=float)
    stiffness = np.zeros((n_node, n_node), dtype=float)
    elem_shear = np.zeros(n_elem, dtype=float)
    elem_density = np.zeros(n_elem, dtype=float)

    for elem in range(n_elem):
        density, shear_modulus, _ = material_at_element(elem)
        elem_density[elem] = density
        elem_shear[elem] = shear_modulus
        elem_mass = density * area * ZONE_HEIGHT
        mass[elem] += 0.5 * elem_mass
        mass[elem + 1] += 0.5 * elem_mass
        k = shear_modulus * area / ZONE_HEIGHT
        stiffness[elem, elem] += k
        stiffness[elem, elem + 1] -= k
        stiffness[elem + 1, elem] -= k
        stiffness[elem + 1, elem + 1] += k

    omega0 = 2.0 * math.pi * RAYLEIGH_CENTER_FREQUENCY
    alpha = RAYLEIGH_DAMPING_RATIO * omega0
    beta = RAYLEIGH_DAMPING_RATIO / omega0
    damping = alpha * np.diag(mass) + beta * stiffness

    free = np.arange(1, n_node)
    base = 0
    m_free = mass[free]
    k_ff = stiffness[np.ix_(free, free)]
    k_fb = stiffness[np.ix_(free, [base])].reshape(-1)
    c_ff = damping[np.ix_(free, free)]
    c_fb = damping[np.ix_(free, [base])].reshape(-1)

    n_steps = int(round(TOTAL_TIME / DT))
    history_stride = max(1, int(round(HISTORY_INTERVAL / DT)))
    u_free = np.zeros(n_node - 1, dtype=float)
    v_free = np.zeros(n_node - 1, dtype=float)
    rows = []
    previous_top_v = 0.0
    previous_stress_v = 0.0
    previous_time = 0.0
    stress_elem = min(n_elem - 1, max(0, int(STRESS_HISTORY_Y // ZONE_HEIGHT)))

    for step in range(n_steps + 1):
        time = step * DT
        base_u = float(np.interp(time, MOTION_TIME, MOTION_DISPLACEMENT))
        base_v = float(np.interp(time, MOTION_TIME, MOTION_VELOCITY))
        force = -(k_ff @ u_free + k_fb * base_u + c_ff @ v_free + c_fb * base_v)
        a_free = force / m_free

        if step > 0:
            v_free += a_free * DT
            u_free += v_free * DT

        if step % history_stride == 0 or step == n_steps:
            u_all = np.zeros(n_node, dtype=float)
            v_all = np.zeros(n_node, dtype=float)
            a_all = np.zeros(n_node, dtype=float)
            u_all[base] = base_u
            v_all[base] = base_v
            a_all[base] = input_base_acceleration(time)
            u_all[free] = u_free
            v_all[free] = v_free
            a_all[free] = a_free

            top_v = v_all[-1]
            stress_v = v_all[stress_elem]
            dt_hist = max(time - previous_time, DT)
            strain = (u_all[stress_elem + 1] - u_all[stress_elem]) / ZONE_HEIGHT
            shear_stress = elem_shear[stress_elem] * strain
            rows.append(
                {
                    "time": time,
                    "input_base_ax": input_base_acceleration(time),
                    "input_base_ax_g": input_base_acceleration(time) / G0,
                    "input_base_vx": base_v,
                    "input_base_xdisp": base_u,
                    "top_vx": top_v,
                    "top_ax": a_all[-1],
                    "top_ax_g": a_all[-1] / G0,
                    "top_xdisp": u_all[-1],
                    "stress_point_vx": stress_v,
                    "stress_point_ax": (stress_v - previous_stress_v) / dt_hist,
                    "stress_point_ax_g": ((stress_v - previous_stress_v) / dt_hist) / G0,
                    "stress_point_xdisp": u_all[stress_elem],
                    "stress_point_strain": strain,
                    "stress_point_shear_stress": shear_stress,
                    "stress_element_material": material_at_element(stress_elem)[2],
                }
            )
            previous_top_v = top_v
            previous_stress_v = stress_v
            previous_time = time

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    keys = [
        "time",
        "input_base_ax",
        "input_base_ax_g",
        "input_base_vx",
        "input_base_xdisp",
        "top_vx",
        "top_ax",
        "top_ax_g",
        "top_xdisp",
        "stress_point_vx",
        "stress_point_ax",
        "stress_point_ax_g",
        "stress_point_xdisp",
        "stress_point_strain",
        "stress_point_shear_stress",
        "stress_element_material",
    ]
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(f"{row.get(key, 0.0):.12g}" for key in keys))
    (OUTPUT_PATH / "equivalent_1d_response.csv").write_text("\n".join(lines), encoding="utf-8")
    return rows


GlobalVariable.DIMENSION = 2
ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32, debug=False, log_level=ti.ERROR)

mpm = MPM(title="FLAC3D Figure 3.49 layered soil input acceleration benchmark")

mpm.set_configuration(
    domain=ti.Vector([DOMAIN_WIDTH, DOMAIN_HEIGHT]),
    boundary=["None", "None", "None"],
    gravity=ti.Vector([0.0, 0.0]),
    background_damping=0.0,
    alphaPIC=0.0,
    mapping="USF",
    stabilize=None,
    shape_function="Linear",
    material_type="Solid",
)

mpm.set_solver(
    solver={
        "Timestep": DT,
        "SimulationTime": TOTAL_TIME,
        "CFL": 0.5,
        "SaveInterval": SAVE_INTERVAL,
        "SavePath": OUTPUT_PATH.as_posix(),
    }
)

mpm.memory_allocate(
    memory={
        "max_material_number": 2,
        "max_particle_number": 100,
        "max_constraint_number": {
            "max_velocity_constraint": 300,
        },
    }
)

mpm.add_material(
    model="LinearElastic",
    material={
        "MaterialID": 1,
        "Density": MAT1_DENSITY,
        "YoungModulus": MAT1_YOUNG_MODULUS,
        "PossionRatio": MAT1_POISSON_RATIO,
    },
)
mpm.add_material(
    model="LinearElastic",
    material={
        "MaterialID": 2,
        "Density": MAT2_DENSITY,
        "YoungModulus": MAT2_YOUNG_MODULUS,
        "PossionRatio": MAT2_POISSON_RATIO,
    },
)

mpm.add_element(
    element={
        "ElementType": "Q4N2D",
        "ElementSize": ti.Vector([DOMAIN_WIDTH, ZONE_HEIGHT]),
    }
)

mpm.add_region(
    region=[
        {
            "Name": "soft_bottom",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0]),
            "BoundingBoxSize": ti.Vector([DOMAIN_WIDTH, STIFF_Y0]),
            "ydirection": ti.Vector([0.0, 1.0]),
        },
        {
            "Name": "stiff_middle",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": ti.Vector([0.0, STIFF_Y0]),
            "BoundingBoxSize": ti.Vector([DOMAIN_WIDTH, STIFF_Y1 - STIFF_Y0]),
            "ydirection": ti.Vector([0.0, 1.0]),
        },
        {
            "Name": "soft_top",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": ti.Vector([0.0, SOFT_TOP_Y0]),
            "BoundingBoxSize": ti.Vector([DOMAIN_WIDTH, DOMAIN_HEIGHT - SOFT_TOP_Y0]),
            "ydirection": ti.Vector([0.0, 1.0]),
        },
    ]
)

mpm.add_body(
    body={
        "Template": [
            body_template("soft_bottom", 1),
            body_template("stiff_middle", 2),
            body_template("soft_top", 1),
        ]
    }
)

mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [input_base_velocity(0.0), None],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [DOMAIN_WIDTH, 0.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, 0.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [DOMAIN_WIDTH, 0.0],
        },
    ]
)

mpm.select_save_data(particle=False, grid=True)
write_base_motion_csv()

histories = []
previous_response = {}


def per_step_update():
    apply_bottom_velocity(mpm, mpm.sims.current_time + mpm.sims.delta)
    collect_response(mpm, histories, previous_response)


mpm.run(function=per_step_update)
collect_response(mpm, histories, previous_response, forced=True)

write_histories_csv(histories)
equivalent_histories = solve_equivalent_1d_layered_response()
write_validation_summary(mpm, histories)
