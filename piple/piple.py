import math
from pathlib import Path

from geotaichi import *


"""
MPM reproduction scaffold for:
Kohler, Stoecklin, Puzrin (2022), "A MPM framework for
large-deformation seismic response analysis".

This script intentionally uses a pure MPM model. The previous DEM-MPM setup in
this file was a particle/obstacle coupling case and is not the right starting
point for the paper's large-deformation seismic response analysis.

Replace the values in PARAMS with the paper's exact geometry, material
parameters, and input motion before using the results quantitatively.
"""


PARAMS = {
    # Geometry. The slope is approximated by stacked rectangular MPM regions.
    "length": 40.0,
    "height": 12.0,
    "base_height": 6.0,
    "slope_start_x": 18.0,
    "slope_end_x": 34.0,
    "n_layers": 12,
    # Numerics.
    "cell_size": 0.25,
    "particles_per_cell": 2,
    "dt": 2.0e-4,
    "cfl": 0.5,
    "gravity_time": 1.0,
    "shake_time": 4.0,
    "save_interval": 0.05,
    "save_path": r"F:\taichi_results\DHH\mpm_seismic_kohler_2022",
    # Drucker-Prager soil. Use values from the paper/table for final runs.
    "density": 1800.0,
    "young": 30.0e6,
    "poisson": 0.30,
    "friction": 35.0,
    "dilation": 0.0,
    "cohesion": 2.0e3,
    "tensile": 0.0,
    # Simple synthetic base input used until the paper record is digitized.
    # This is a Ricker-like velocity pulse imposed at the base in x direction.
    "base_velocity_amplitude": 0.35,
    "base_velocity_frequency": 2.0,
    "base_velocity_delay": 0.8,
    # Optional CSV/TXT with two numeric columns: time, horizontal base velocity.
    # Use this for the digitized acceleration/velocity record from the paper.
    "input_motion_file": None,
    "input_update_interval": 0.01,
}


def load_input_motion(path):
    if path is None:
        return None

    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            records.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    if len(records) < 2:
        raise ValueError("input_motion_file needs at least two rows: time, velocity")
    return sorted(records)


def interpolate_motion(records, t):
    if records is None:
        return None
    if t <= records[0][0]:
        return records[0][1]
    if t >= records[-1][0]:
        return records[-1][1]
    for i in range(1, len(records)):
        t1, v1 = records[i]
        if t <= t1:
            t0, v0 = records[i - 1]
            ratio = (t - t0) / (t1 - t0)
            return v0 + ratio * (v1 - v0)
    return records[-1][1]


def top_elevation(x):
    if x <= PARAMS["slope_start_x"]:
        return PARAMS["height"]
    if x >= PARAMS["slope_end_x"]:
        return PARAMS["base_height"]
    ratio = (x - PARAMS["slope_start_x"]) / (
        PARAMS["slope_end_x"] - PARAMS["slope_start_x"]
    )
    return PARAMS["height"] + ratio * (PARAMS["base_height"] - PARAMS["height"])


def base_velocity(t):
    recorded_velocity = interpolate_motion(INPUT_MOTION, t)
    if recorded_velocity is not None:
        return recorded_velocity

    tau = t - PARAMS["base_velocity_delay"]
    omega = 3.141592653589793 * PARAMS["base_velocity_frequency"] * tau
    return PARAMS["base_velocity_amplitude"] * (1.0 - 2.0 * omega * omega) * math.exp(
        -omega * omega
    )


def make_soil_regions():
    regions = []
    layer_h = PARAMS["height"] / PARAMS["n_layers"]
    for i in range(PARAMS["n_layers"]):
        y0 = i * layer_h
        y1 = y0 + layer_h
        x_right = 0.0
        for j in range(161):
            x = PARAMS["length"] * j / 160.0
            if top_elevation(x) >= y1:
                x_right = x
        if x_right <= PARAMS["cell_size"]:
            continue
        regions.append(
            {
                "Name": f"soil_layer_{i:02d}",
                "Type": "Rectangle2D",
                "BoundingBoxPoint": ti.Vector([0.0, y0]),
                "BoundingBoxSize": ti.Vector([x_right, layer_h]),
                "ydirection": ti.Vector([0.0, 1.0]),
            }
        )
    return regions


def make_soil_templates(regions):
    return [
        {
            "RegionName": region["Name"],
            "nParticlesPerCell": PARAMS["particles_per_cell"],
            "BodyID": 0,
            "MaterialID": 1,
            "ParticleStress": {
                "GravityField": True,
                "InternalStress": ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "Traction": {},
            },
            "InitialVelocity": ti.Vector([0.0, 0.0]),
            "FixVelocity": ["Free", "Free"],
        }
        for region in regions
    ]


def make_boundaries(vx_base):
    length = PARAMS["length"]
    height = PARAMS["height"]
    dx = PARAMS["cell_size"]
    return [
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [vx_base, 0.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [length, dx],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [dx, height],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [length - dx, 0.0],
            "EndPoint": [length, height],
        },
    ]


def update_base_velocity_constraint(model, vx_base):
    # GeoTaichi's clean_boundary_condition path is broken in this checkout.
    # Updating the stored velocity constraints lets the next run() copy the new
    # prescribed base velocity into the Taichi boundary field.
    base_node_limit = 2 * (int(PARAMS["length"] / PARAMS["cell_size"]) + 1)
    for key in list(model.scene.boundary.velocity_dict.keys()):
        node_id, direction, _level = key
        if direction == 0 and node_id < base_node_limit:
            model.scene.boundary.velocity_dict[key] = vx_base


init(dim=2, device_memory_GB=8, debug=False, gpu_id=0)

INPUT_MOTION = load_input_motion(PARAMS["input_motion_file"])

mpm = MPM(title="Kohler 2022 large-deformation seismic MPM scaffold")

mpm.set_configuration(
    domain=ti.Vector([PARAMS["length"], PARAMS["height"]]),
    dimension="2-Dimension",
    is_2DAxisy=False,
    background_damping=0.02,
    gravity=ti.Vector([0.0, -9.81]),
    alphaPIC=0.0,
    mapping="MUSL",
    shape_function="GIMP",
    velocity_projection="PIC/FLIP",
    stabilize="B-Bar Method",
    material_type="Solid",
    visualize=False,
)

mpm.set_solver(
    solver={
        "Timestep": PARAMS["dt"],
        "SimulationTime": PARAMS["gravity_time"],
        "CFL": PARAMS["cfl"],
        "SaveInterval": PARAMS["save_interval"],
        "SavePath": PARAMS["save_path"],
    }
)

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 900000,
        "max_constraint_number": {
            "max_velocity_constraint": 120000,
            "max_reflection_constraint": 0,
            "max_friction_constraint": 0,
        },
    }
)

mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 1,
        "Density": PARAMS["density"],
        "YoungModulus": PARAMS["young"],
        "PossionRatio": PARAMS["poisson"],
        "Friction": PARAMS["friction"],
        "Dilation": PARAMS["dilation"],
        "Cohesion": PARAMS["cohesion"],
        "Tensile": PARAMS["tensile"],
    },
)

mpm.add_element(
    element={
        "ElementType": "Q4N2D",
        "ElementSize": ti.Vector([PARAMS["cell_size"], PARAMS["cell_size"]]),
    }
)

soil_regions = make_soil_regions()
mpm.add_region(region=soil_regions)
mpm.add_body(body={"Template": make_soil_templates(soil_regions)})

current_boundaries = make_boundaries(0.0)
mpm.add_boundary_condition(boundary=current_boundaries)
mpm.select_save_data(particle=True, grid=True)

# Stage 1: gravity initialization.
mpm.run()

# Stage 2: imposed base shaking. GeoTaichi stores velocity constraints as
# constants, so the base input is applied piecewise over short run windows.
t = PARAMS["gravity_time"]
t_end = PARAMS["gravity_time"] + PARAMS["shake_time"]
while t < t_end:
    next_t = min(t + PARAMS["input_update_interval"], t_end)
    vx = base_velocity(t - PARAMS["gravity_time"])
    update_base_velocity_constraint(mpm, vx)
    mpm.modify_parameters(
        SimulationTime=next_t,
        SaveInterval=PARAMS["save_interval"],
        SavePath=PARAMS["save_path"],
    )
    mpm.run()
    t = next_t

mpm.postprocessing(read_path=PARAMS["save_path"], write_background_grid=True)
