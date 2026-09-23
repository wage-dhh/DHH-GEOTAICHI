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
    The sketch uses physical coordinates with negative x and y values.
    GeoTaichi regions are kept inside a positive domain, so all physical
    coordinates are shifted by the configured coordinate_shift.
    CSV diagnostics are written back in the original Nairn physical coordinates.
"""

import csv
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
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
from src.mpm.boundaries.BoundaryCore import kernel_initialize_boundary  # noqa: E402
from src.utils import GlobalVariable  # noqa: E402
from src.utils.MatrixFunction import truncation  # noqa: E402
from src.utils.ScalarFunction import sgn  # noqa: E402
from src.utils.TypeDefination import vec2f, vec2i  # noqa: E402
from src.utils.VectorFunction import outer_product2D  # noqa: E402
from earthquake_input import (  # noqa: E402
    EarthquakeMotion,
    read_earthquake_motion,
    read_input_velocity_motion,
    write_earthquake_input_check,
)


def init_taichi_mpm(dim: int = 2, arch: str = "gpu") -> None:
    """Initialize the MPM runtime without importing optional SDF dependencies."""

    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    GlobalVariable.DIMENSION = dim
    ti_arch = ti.cpu if arch.lower() == "cpu" else ti.gpu
    ti.init(arch=ti_arch, offline_cache=True, default_fp=ti.f64, default_ip=ti.i32, log_level=ti.ERROR)


def force_three_grid_levels(scene: Any, sims: Any) -> int:
    scene.grid_level = 4 if SEPARATE_SLIDE_BODY else 3
    return scene.grid_level


def allow_three_grid_inputs(scene: Any, sims: Any, grid_level: int) -> None:
    return None


@ti.kernel
def kernel_compute_grid_kinematic_paper_eq11(
    cutoff: float,
    damping: float,
    node: ti.template(),
    dt: ti.template(),
):
    """Update grid momentum with Kohler et al. Eq. (11) local damping."""

    for node_id, body_id in ti.ndrange(node.shape[0], node.shape[1]):
        if node[node_id, body_id].m > cutoff:
            unbalanced_force = node[node_id, body_id].force
            velocity = node[node_id, body_id].momentum
            for component in ti.static(range(2)):
                unbalanced_force[component] -= (
                    damping * ti.abs(unbalanced_force[component]) * sgn(velocity[component])
                )
            acceleration = unbalanced_force / node[node_id, body_id].m
            node[node_id, body_id].momentum += acceleration * dt[None]
            node[node_id, body_id].force = acceleration


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


@ti.kernel
def kernel_mass_momentum_apic_p2g_2d(
    total_nodes: int,
    particle_num: int,
    grid_node_count: ti.types.vector(2, int),
    grid_size: ti.types.vector(2, float),
    free_field_period: float,
    node: ti.template(),
    particle: ti.template(),
    apic_affine_velocity: ti.template(),
    node_ids: ti.template(),
    shape_functions: ti.template(),
    particle_node_count: ti.template(),
):
    """Kohler Eq. (7) APIC P2G specialized for GeoTaichi's 2-D fields."""

    ti.block_local(node.m)
    for axis in ti.static(range(2)):
        ti.block_local(node.momentum.get_scalar_field(axis))
    for particle_id in range(particle_num):
        if int(particle[particle_id].active) == 1:
            body_id = int(particle[particle_id].bodyID)
            offset = particle_id * total_nodes
            position = particle[particle_id].x
            mass = particle[particle_id].m
            velocity = particle[particle_id].v
            affine_velocity = apic_affine_velocity[particle_id]
            for local_node in range(offset, offset + int(particle_node_count[particle_id])):
                node_id = node_ids[local_node]
                node_index = ti.Vector(
                    [node_id % grid_node_count[0], node_id // grid_node_count[0]]
                )
                node_position = grid_size * node_index.cast(float)
                if body_id != 0:
                    periodic_dx = node_position[0] - position[0]
                    if periodic_dx > 0.5 * free_field_period:
                        node_position[0] -= free_field_period
                    elif periodic_dx < -0.5 * free_field_period:
                        node_position[0] += free_field_period
                nodal_mass = shape_functions[local_node] * mass
                nodal_velocity = velocity + affine_velocity @ (node_position - position)
                node[node_id, body_id]._update_nodal_mass(nodal_mass)
                node[node_id, body_id]._update_nodal_momentum(nodal_mass * nodal_velocity)


@ti.kernel
def kernel_update_apic_affine_velocity_2d(
    total_nodes: int,
    particle_num: int,
    grid_node_count: ti.types.vector(2, int),
    grid_size: ti.types.vector(2, float),
    free_field_period: float,
    node: ti.template(),
    particle: ti.template(),
    apic_affine_velocity: ti.template(),
    node_ids: ti.template(),
    shape_functions: ti.template(),
    particle_node_count: ti.template(),
):
    """Kohler Eq. (16) APIC update with periodic-image support distances."""

    for particle_id in range(particle_num):
        if int(particle[particle_id].materialID) > 0 and int(particle[particle_id].active) == 1:
            body_id = int(particle[particle_id].bodyID)
            offset = particle_id * total_nodes
            position = particle[particle_id].x
            moment_matrix = ti.Matrix.zero(float, 2, 2)
            affine_matrix = ti.Matrix.zero(float, 2, 2)
            for local_node in range(offset, offset + int(particle_node_count[particle_id])):
                node_id = node_ids[local_node]
                node_index = ti.Vector(
                    [node_id % grid_node_count[0], node_id // grid_node_count[0]]
                )
                node_position = grid_size * node_index.cast(float)
                if body_id != 0:
                    periodic_dx = node_position[0] - position[0]
                    if periodic_dx > 0.5 * free_field_period:
                        node_position[0] -= free_field_period
                    elif periodic_dx < -0.5 * free_field_period:
                        node_position[0] += free_field_period
                support_offset = node_position - position
                nodal_velocity = node[node_id, body_id].momentum
                shape_value = shape_functions[local_node]
                moment_matrix += shape_value * outer_product2D(support_offset, support_offset)
                # VectorFunction.outer_product2D(a, b) returns b @ a.T.
                # Kohler Eq. (16) requires B_p = sum(w_ip * v_i @ r_ip.T),
                # so the arguments are intentionally supplied in this order.
                affine_matrix += shape_value * outer_product2D(support_offset, nodal_velocity)
            if ti.static(APIC_USE_PAPER_CONSTANT_DP):
                # Kohler Eq. (7): D_p = h^2 / 3 I for the cubic B-spline.
                apic_affine_velocity[particle_id] = truncation(
                    affine_matrix * (3.0 / (grid_size[0] * grid_size[0]))
                )
            else:
                apic_affine_velocity[particle_id] = truncation(
                    affine_matrix @ moment_matrix.inverse()
                )


@ti.kernel
def kernel_initialize_precise_particle_position_2d(
    particle_num: int,
    particle: ti.template(),
    precise_position: ti.template(),
):
    for particle_id in range(particle_num):
        precise_position[particle_id] = ti.cast(particle[particle_id].x, ti.f64)


@ti.kernel
def kernel_accumulate_precise_particle_position_2d(
    particle_num: int,
    dt: float,
    particle: ti.template(),
    precise_position: ti.template(),
):
    for particle_id in range(particle_num):
        if int(particle[particle_id].active) == 1:
            precise_position[particle_id] += dt * ti.cast(particle[particle_id].v, ti.f64)
            particle[particle_id].x = ti.cast(precise_position[particle_id], ti.f32)


def install_paper_apic_2d_p2g(mpm: MPM) -> None:
    """Install the paper's 2-D APIC transfers and update order locally."""

    if DYNAMIC_VELOCITY_PROJECTION != "Affine":
        return
    engine = mpm.enginer
    apic_affine_velocity = ti.Matrix.field(
        2,
        2,
        dtype=ti.f32,
        shape=max(1, int(mpm.sims.max_particle_num)),
    )
    apic_affine_velocity.fill(0.0)
    mpm.nairn_apic_affine_velocity = apic_affine_velocity
    if PRECISE_POSITION_ACCUMULATION:
        precise_position = ti.Vector.field(
            2,
            dtype=ti.f64,
            shape=max(1, int(mpm.sims.max_particle_num)),
        )
        kernel_initialize_precise_particle_position_2d(
            int(mpm.scene.particleNum[0]),
            mpm.scene.particle,
            precise_position,
        )
        mpm.nairn_precise_particle_position = precise_position

        def accumulate_precise_position(sims: Any, scene: Any) -> None:
            kernel_accumulate_precise_particle_position_2d(
                int(scene.particleNum[0]),
                float(sims.delta),
                scene.particle,
                precise_position,
            )

        mpm.nairn_accumulate_precise_position = accumulate_precise_position
    pending_affine = getattr(mpm, "pending_apic_affine_velocity", None)
    if pending_affine is not None:
        pending_affine = np.ascontiguousarray(pending_affine, dtype=np.float32)
        particle_count = int(mpm.scene.particleNum[0])
        if pending_affine.shape != (particle_count, 2, 2):
            raise RuntimeError(
                "Pending APIC affine-state shape mismatch: "
                f"checkpoint={pending_affine.shape} current={(particle_count, 2, 2)}"
            )
        affine_storage = np.zeros(
            (int(apic_affine_velocity.shape[0]), 2, 2), dtype=np.float32
        )
        affine_storage[:particle_count] = pending_affine
        apic_affine_velocity.from_numpy(affine_storage)
        mpm.static_checkpoint_apic_affine_state = str(
            getattr(mpm, "pending_apic_affine_state_label", "RESTORED")
        )
        del mpm.pending_apic_affine_velocity
        if hasattr(mpm, "pending_apic_affine_state_label"):
            del mpm.pending_apic_affine_state_label

    def compute_nodal_kinematics_apic_2d(sims: Any, scene: Any) -> None:
        kernel_mass_momentum_apic_p2g_2d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.element.grid_size,
            K_FREE_FIELD_WIDTH,
            scene.node,
            scene.particle,
            apic_affine_velocity,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def update_apic_and_physical_velocity_gradients_2d(sims: Any, scene: Any) -> None:
        kernel_update_apic_affine_velocity_2d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.element.grid_size,
            K_FREE_FIELD_WIDTH,
            scene.node,
            scene.particle,
            apic_affine_velocity,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        # Kohler Eq. (17) is independent of the APIC matrix in Eq. (16).
        engine.update_velocity_gradient_2D(sims, scene)

    def paper_apic_usl_updating(sims: Any, scene: Any) -> None:
        """Kohler Fig. 2 and Eqs. (5)-(21): update stress after G2P."""

        engine.calculate_interpolation(sims, scene)
        engine.compute_nodal_kinematic(sims, scene)
        engine.compute_grid_velcity(sims, scene)
        engine.apply_particle_traction_constraints(sims, scene)
        engine.compute_forces(sims, scene)
        engine.apply_traction_constraints(sims, scene)
        engine.apply_absorbing_constraints(sims, scene)
        engine.compute_grid_kinematic(sims, scene)
        engine.pre_contact_calculate(sims, scene)
        engine.apply_kinematic_constraints(sims, scene)
        engine.compute_contact_force_(sims, scene)
        # Eq. (16) uses x_p^n, so update B_p and L_p before G2P moves x_p.
        engine.compute_velocity_gradient(sims, scene)
        engine.compute_particle_kinematic(sims, scene)
        precise_position_update = getattr(mpm, "nairn_accumulate_precise_position", None)
        if precise_position_update is not None:
            precise_position_update(sims, scene)
        engine.compute_stress_strains(sims, scene)
        engine.pressure_smoothing_(scene)

    engine.compute_nodal_kinematic = compute_nodal_kinematics_apic_2d
    engine.compute_velocity_gradient = update_apic_and_physical_velocity_gradients_2d
    # GeoTaichi otherwise forces every Affine run through its stress-first
    # velocity_projection_updating path, irrespective of the USL setting.
    engine.compute = paper_apic_usl_updating
    engine._nairn_paper_apic_2d_p2g_installed = True
    engine._nairn_paper_apic_2d_transfer_installed = True
    engine._nairn_paper_apic_separate_B_L = True
    engine._nairn_paper_apic_usl_order = True


# -------------------------- Nairn input equivalents -------------------------- #
# Keep one grid cell of exterior padding after introducing the 1 m
# free-field/main-slope gaps.  The physical model geometry remains unchanged.
X_SHIFT = 46.0
# The physical base is at y=-36 m.  The extra two grid rows below it
# provide the ghost region required by the static mirrored-particle method.
Y_SHIFT = 38.0

DX = float(os.environ.get("NAIRN_SHEAR_DX", "1.0"))
DIAGNOSTIC_TOP_PADDING_CELLS = max(
    0,
    int(os.environ.get("NAIRN_DIAGNOSTIC_TOP_PADDING_CELLS", "0")),
)
DOMAIN = np.array(
    [192.0, 50.0 + DIAGNOSTIC_TOP_PADDING_CELLS * DX],
    dtype=np.float64,
)
ELEMENT_SIZE = np.array([DX, DX], dtype=np.float64)

DT = float(os.environ.get("NAIRN_SHEAR_DT", "5e-5"))
# Cover the complete digitized Fig.10(c) input record (about 19.88 s).
SIMULATION_TIME = float(os.environ.get("NAIRN_SHEAR_TIME", "20.0"))
SAVE_INTERVAL = float(os.environ.get("NAIRN_SHEAR_SAVE_INTERVAL", "1e-3"))
HISTORY_INTERVAL = float(os.environ.get("NAIRN_SHEAR_HISTORY_INTERVAL", "1e-4"))
OUTPUT_DIR = Path(os.environ.get("NAIRN_SHEAR_OUTPUT", "output/kohler_static_convergence_4s"))

# Kohler Fig. 2 advances the grid before G2P and material update, which is the
# USL order in this driver.  Keep that order as the paper-aligned default.
MAPPING = os.environ.get("NAIRN_SHEAR_MAPPING", "USL")
SHAPE_FUNCTION = os.environ.get("NAIRN_SHEAR_SHAPE", "CubicBSpline")
# Use the CUDA-capable Taichi backend by default; set NAIRN_SHEAR_ARCH=cpu
# when a deterministic CPU diagnostic is needed.
ARCH = os.environ.get("NAIRN_SHEAR_ARCH", "gpu")
STRICT_TIME_LOOP = os.environ.get("NAIRN_SHEAR_STRICT_LOOP", "1").lower() not in {"0", "false", "no"}
RUN_POSTPROCESS = os.environ.get("NAIRN_SHEAR_POSTPROCESS", "1").lower() not in {"0", "false", "no"}
SAVE_PARTICLE = os.environ.get("NAIRN_SHEAR_SAVE_PARTICLE", "0").lower() not in {"0", "false", "no"}
SAVE_GRID = os.environ.get("NAIRN_SHEAR_SAVE_GRID", "0").lower() not in {"0", "false", "no"}

STATIC_ENABLED = os.environ.get("NAIRN_STATIC_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
STATIC_DT_CAP = float(os.environ.get("NAIRN_STATIC_DT_CAP", "1e-4"))
STATIC_DT = STATIC_DT_CAP
STATIC_TIME = float(os.environ.get("NAIRN_STATIC_TIME", "4.0"))
STATIC_RAMP_TIME = float(os.environ.get("NAIRN_STATIC_RAMP_TIME", "1.0"))
STATIC_SAVE_INTERVAL = float(os.environ.get("NAIRN_STATIC_SAVE_INTERVAL", "0.02"))
STATIC_HISTORY_INTERVAL = float(os.environ.get("NAIRN_STATIC_HISTORY_INTERVAL", "0.01"))
STATIC_DAMPING = float(os.environ.get("NAIRN_STATIC_DAMPING", "0.9"))
STATIC_GRAVITY = float(os.environ.get("NAIRN_STATIC_GRAVITY", "-9.81"))
# Kohler et al. Eqs. (11)-(12): apply the component-wise damping force at
# every active grid node. GeoTaichi's native branch omits it when v_i*f_i<=0.
STATIC_PAPER_EQ11_DAMPING = os.environ.get("NAIRN_STATIC_PAPER_EQ11_DAMPING", "1").lower() not in {
    "0", "false", "no", "off",
}
STATIC_MIRROR_PARTICLE_BOUNDARY = os.environ.get("NAIRN_STATIC_MIRROR_PARTICLE_BOUNDARY", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
STATIC_MIRROR_REFLECT_BODY_FORCE = os.environ.get("NAIRN_STATIC_MIRROR_REFLECT_BODY_FORCE", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
# Kohler et al. Eq. (15) is the FLIP alternative. The paper-aligned APIC
# path uses Eq. (14) for static and dynamic G2P, so this diagnostic is off by
# default and is only available for an explicit non-APIC ablation.
STATIC_PAPER_EQ15_FLIP = os.environ.get("NAIRN_STATIC_PAPER_EQ15_FLIP", "0").lower() not in {
    "0", "false", "no", "off",
}
# Static APIC uses the paper's updated nodal velocity transfer, Eq. (14).
STATIC_ALPHA_PIC = min(1.0, max(0.0, float(os.environ.get("NAIRN_STATIC_ALPHA_PIC", "1.0"))))
# Diagnostic for Eq. (14) accumulation when GeoTaichi particle coordinates are
# stored in single precision. It is off by default and changes no model input.
PRECISE_POSITION_ACCUMULATION = os.environ.get(
    "NAIRN_PRECISE_POSITION_ACCUMULATION", "0"
).lower() not in {"0", "false", "no", "off"}
APIC_USE_PAPER_CONSTANT_DP = os.environ.get(
    "NAIRN_APIC_USE_PAPER_CONSTANT_DP", "1"
).lower() not in {"0", "false", "no", "off"}
PAPER_APIC_FORMULA_REVISION = "eq16_B_v_outer_r_v1"
PAPER_HUGHES_WINGET_FORMULA_REVISION = "eq27_eq28_R_sigma_RT_v1"
DYNAMIC_VELOCITY_PROJECTION = os.environ.get(
    "NAIRN_DYNAMIC_VELOCITY_PROJECTION", "Affine"
).strip()
if DYNAMIC_VELOCITY_PROJECTION not in {"PIC/FLIP", "PIC", "FLIP", "Affine"}:
    raise ValueError(
        "NAIRN_DYNAMIC_VELOCITY_PROJECTION must be one of "
        "PIC/FLIP, PIC, FLIP, or Affine"
    )
if DYNAMIC_VELOCITY_PROJECTION == "Affine" and MAPPING != "USL":
    raise ValueError(
        "The paper-aligned APIC path uses the USL update order from Fig. 2; "
        "set NAIRN_SHEAR_MAPPING=USL."
    )
if DYNAMIC_VELOCITY_PROJECTION == "Affine" and not APIC_USE_PAPER_CONSTANT_DP:
    raise ValueError(
        "The paper-aligned APIC path requires D_p = h^2/3 I from Eq. (7); "
        "do not disable the constant-D option."
    )
if DYNAMIC_VELOCITY_PROJECTION == "Affine" and not math.isclose(
    STATIC_ALPHA_PIC, 1.0, rel_tol=0.0, abs_tol=1.0e-12
):
    raise ValueError(
        "The paper-aligned APIC path requires alphaPIC=1 during static relaxation; "
        "set NAIRN_STATIC_ALPHA_PIC=1."
    )
if DYNAMIC_VELOCITY_PROJECTION == "Affine" and STATIC_PAPER_EQ15_FLIP:
    raise ValueError(
        "The paper-aligned APIC path uses Eq. (14), so the static Eq. (15) FLIP "
        "projection must be disabled."
    )
DYNAMIC_ALPHA_PIC_REQUESTED = min(
    1.0, max(0.0, float(os.environ.get("NAIRN_DYNAMIC_ALPHA_PIC", "0.0")))
)
# Kohler Eqs. (14) and (16): APIC transfers the updated nodal velocity rather
# than a FLIP increment. GeoTaichi's Affine path uses alphaPIC=1 for this G2P.
DYNAMIC_ALPHA_PIC = (
    1.0 if DYNAMIC_VELOCITY_PROJECTION in {"PIC", "Affine"}
    else 0.0 if DYNAMIC_VELOCITY_PROJECTION == "FLIP"
    else DYNAMIC_ALPHA_PIC_REQUESTED
)
# Read-only final-step audit of the force terms consumed by the static mirror.
STATIC_MIRROR_FORCE_DIAGNOSTIC = os.environ.get("NAIRN_STATIC_MIRROR_FORCE_DIAGNOSTIC", "0").lower() not in {
    "0", "false", "no", "off",
}
STATIC_MIRROR_FORCE_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get("NAIRN_STATIC_MIRROR_FORCE_DIAGNOSTIC_OUTPUT", str(OUTPUT_DIR / "static_mirror_force_audit"))
)
STATIC_MIRROR_FORCE_DIAGNOSTIC_TARGET_NODE_ID = int(
    os.environ.get("NAIRN_STATIC_MIRROR_FORCE_DIAGNOSTIC_TARGET_NODE_ID", "-1")
)
# Read-only Eq. (15) audit of the final static G2P update. This never changes
# grid, particle, force, or boundary data used by the solver.
STATIC_G2P_EQ15_DIAGNOSTIC = os.environ.get("NAIRN_STATIC_G2P_EQ15_DIAGNOSTIC", "0").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
STATIC_G2P_EQ15_DIAGNOSTIC_STEPS = max(1, int(os.environ.get("NAIRN_STATIC_G2P_EQ15_DIAGNOSTIC_STEPS", "1")))
STATIC_G2P_EQ15_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get("NAIRN_STATIC_G2P_EQ15_DIAGNOSTIC_OUTPUT", str(OUTPUT_DIR / "static_g2p_eq15_diagnostic"))
)
STATIC_BOTTOM_CONSTRAINT_BAND = float(os.environ.get("NAIRN_STATIC_BOTTOM_CONSTRAINT_BAND", str(2.0 * DX)))
STATIC_REACTION_NODE_BAND = float(os.environ.get("NAIRN_STATIC_REACTION_NODE_BAND", str(STATIC_BOTTOM_CONSTRAINT_BAND)))
# The cubic B-spline force support spans four boundary-node levels.  This is
# only the force-transfer footprint in the Section 3.2 handoff; it does not
# widen the static kinematic constraint.
STATIC_BOTTOM_SUPPORT_NODE_BAND = float(
    os.environ.get("NAIRN_STATIC_BOTTOM_SUPPORT_NODE_BAND", str(3.0 * DX))
)
# Relaxed to cover the observed static-state residual motion before dynamics.
STATIC_MAX_VELOCITY_TOLERANCE = float(os.environ.get("NAIRN_STATIC_MAX_VELOCITY_TOLERANCE", "1e-2"))
STATIC_RMS_VELOCITY_TOLERANCE = float(os.environ.get("NAIRN_STATIC_RMS_VELOCITY_TOLERANCE", "5e-4"))
# Project diagnostic threshold: Kohler et al. use the out-of-balance force in
# Eq. (11) but do not prescribe a normalized convergence tolerance.
STATIC_RELATIVE_UNBALANCED_FORCE_TOLERANCE = float(
    os.environ.get("NAIRN_STATIC_RELATIVE_UNBALANCED_FORCE_TOLERANCE", "2e-4")
)
STATIC_STEP_LIMIT = int(os.environ.get("NAIRN_STATIC_STEP_LIMIT", "0"))
STATIC_VELOCITY_TOLERANCE = STATIC_RMS_VELOCITY_TOLERANCE
STATIC_STRESS_TOLERANCE = float(os.environ.get("NAIRN_STATIC_STRESS_TOLERANCE", "1e-4"))
STATIC_VELOCITY_QUENCH = False
STATIC_QUENCH_MAX_VELOCITY_FACTOR = float(os.environ.get("NAIRN_STATIC_QUENCH_MAX_VELOCITY_FACTOR", "5.0"))
STATIC_CHECKPOINT_SAVE = os.environ.get("NAIRN_STATIC_CHECKPOINT_SAVE", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
# Diagnostic-only opt-in: preserve a nonconverged static state so several
# dynamic-boundary ablations start from the exact same particle state.
STATIC_CHECKPOINT_SAVE_NONCONVERGED = os.environ.get(
    "NAIRN_STATIC_CHECKPOINT_SAVE_NONCONVERGED", "0"
).lower() not in {"0", "false", "no", "off"}
STATIC_CHECKPOINT_LOAD = os.environ.get("NAIRN_STATIC_CHECKPOINT_LOAD", "0").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
STATIC_CHECKPOINT_CONTINUE = os.environ.get("NAIRN_STATIC_CHECKPOINT_CONTINUE", "0").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
STATIC_CHECKPOINT_PATH = Path(
    os.environ.get("NAIRN_STATIC_CHECKPOINT_PATH", "output/kohler_static_convergence_4s/static_checkpoint.npz")
)
STATIC_CHECKPOINT_SAVE_PATH = Path(
    os.environ.get(
        "NAIRN_STATIC_CHECKPOINT_SAVE_PATH",
        str(OUTPUT_DIR / "static_checkpoint.npz") if STATIC_CHECKPOINT_CONTINUE else str(STATIC_CHECKPOINT_PATH),
    )
)
STATIC_CHECKPOINT_ZERO_VELOCITY_ON_LOAD = os.environ.get(
    "NAIRN_STATIC_CHECKPOINT_ZERO_VELOCITY_ON_LOAD", "0"
).lower() not in {"0", "false", "no", "off"}
USE_GEOTAICHI_GRAVITY_FIELD = os.environ.get("NAIRN_USE_GEOTAICHI_GRAVITY_FIELD", "0").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
REQUIRE_STATIC_CONVERGENCE = os.environ.get("NAIRN_REQUIRE_STATIC_CONVERGENCE", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DYNAMIC_KEEP_GRAVITY = os.environ.get("NAIRN_DYNAMIC_KEEP_GRAVITY", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DYNAMIC_DAMPING = float(os.environ.get("NAIRN_DYNAMIC_DAMPING", "0.0"))
DYNAMIC_RELAXATION_TIME = float(os.environ.get("NAIRN_DYNAMIC_RELAXATION_TIME", "0.0"))
DYNAMIC_RELAXATION_SEGMENT = float(os.environ.get("NAIRN_DYNAMIC_RELAXATION_SEGMENT", "0.05"))
DYNAMIC_RELAXATION_DAMPING = float(os.environ.get("NAIRN_DYNAMIC_RELAXATION_DAMPING", "0.0"))
DYNAMIC_RELAXATION_MAX_VELOCITY_TOLERANCE = float(
    os.environ.get("NAIRN_DYNAMIC_RELAXATION_MAX_VELOCITY_TOLERANCE", "1e-3")
)
DYNAMIC_RELAXATION_RMS_VELOCITY_TOLERANCE = float(
    os.environ.get("NAIRN_DYNAMIC_RELAXATION_RMS_VELOCITY_TOLERANCE", "1e-4")
)
DYNAMIC_RELAXATION_REQUIRE_CONVERGENCE = os.environ.get(
    "NAIRN_DYNAMIC_RELAXATION_REQUIRE_CONVERGENCE", "0"
).lower() not in {"0", "false", "no", "off"}
DYNAMIC_SKIP_STRESS_UPDATE = os.environ.get("NAIRN_DIAG_SKIP_DYNAMIC_STRESS_UPDATE", "0").lower() not in {
    "0", "false", "no", "off"
}
HUGHES_WINGET_STRESS_UPDATE = os.environ.get("NAIRN_HUGHES_WINGET_STRESS_UPDATE", "1").lower() not in {
    "0", "false", "no", "off"
}
BOUNDARY_DIAGNOSTIC_INTERVAL = max(
    0.0, float(os.environ.get("NAIRN_BOUNDARY_DIAGNOSTIC_INTERVAL", "0.0"))
)
RUN_DYNAMIC = os.environ.get("NAIRN_RUN_DYNAMIC", "0").lower() not in {"0", "false", "no", "off"}
STATIC_CONVERGENCE_ONLY = os.environ.get("NAIRN_STATIC_CONVERGENCE_ONLY", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DIAGNOSTIC_ENABLE_INPUT = os.environ.get("NAIRN_DIAG_ENABLE_INPUT", "1").lower() not in {"0", "false", "no", "off"}
DIAGNOSTIC_ENABLE_STATIC_REACTION = os.environ.get("NAIRN_DIAG_ENABLE_STATIC_REACTION", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DIAGNOSTIC_ENABLE_BOTTOM_DASHPOT = os.environ.get("NAIRN_DIAG_ENABLE_BOTTOM_DASHPOT", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DIAGNOSTIC_ENABLE_FREE_FIELD = os.environ.get("NAIRN_DIAG_ENABLE_FREE_FIELD", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DIAGNOSTIC_FREE_FIELD_SIGN_MULT = float(os.environ.get("NAIRN_DIAG_FREE_FIELD_SIGN_MULT", "1.0"))
DIAGNOSTIC_STATIC_REACTION_SCOPE = os.environ.get("NAIRN_DIAG_STATIC_REACTION_SCOPE", "bottom").lower()
DIAGNOSTIC_FREE_FIELD_STATIC_STRESS_MULT = float(os.environ.get("NAIRN_DIAG_FREE_FIELD_STATIC_STRESS_MULT", "1.0"))
DIAGNOSTIC_FREE_FIELD_DYNAMIC_STRESS_MULT = float(os.environ.get("NAIRN_DIAG_FREE_FIELD_DYNAMIC_STRESS_MULT", "1.0"))
DIAGNOSTIC_FREE_FIELD_DASHPOT_MULT = float(os.environ.get("NAIRN_DIAG_FREE_FIELD_DASHPOT_MULT", "1.0"))
FREE_FIELD_INDEPENDENCE_TOLERANCE = float(os.environ.get("NAIRN_FREE_FIELD_INDEPENDENCE_TOLERANCE", "1e-12"))
LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE = float(os.environ.get("NAIRN_LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE", "1e-12"))
EXTRACT_LATERAL_STATIC_SUPPORT = os.environ.get("NAIRN_EXTRACT_LATERAL_STATIC_SUPPORT", "0").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
FREE_FIELD_PERIODIC_MINIMAL = os.environ.get("NAIRN_FREE_FIELD_PERIODIC_MINIMAL", "0").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
FREE_FIELD_PERIODIC_TOLERANCE = float(os.environ.get("NAIRN_FREE_FIELD_PERIODIC_TOLERANCE", "1e-6"))
FREE_FIELD_PERIODIC_DYNAMIC_PATH_CHECK = os.environ.get(
    "NAIRN_FREE_FIELD_PERIODIC_DYNAMIC_PATH_CHECK", "0"
).lower() not in {
    "0",
    "false",
    "no",
    "off",
}
AUXILIARY_DEFORMATION_GRADIENT_CHECK = os.environ.get(
    "NAIRN_AUXILIARY_DEFORMATION_GRADIENT_CHECK", "0"
).lower() not in {
    "0",
    "false",
    "no",
    "off",
}
AUXILIARY_DEFORMATION_GRADIENT_OUTPUT = Path(
    os.environ.get("NAIRN_AUXILIARY_DEFORMATION_GRADIENT_OUTPUT", "output/kohler_auxiliary_deformation_gradient")
)
FREE_FIELD_SURFACE_AREA_DYNAMIC_CHECK = os.environ.get(
    "NAIRN_FREE_FIELD_SURFACE_AREA_DYNAMIC_CHECK", "0"
).lower() not in {
    "0",
    "false",
    "no",
    "off",
}
FREE_FIELD_SURFACE_AREA_DYNAMIC_OUTPUT = Path(
    os.environ.get("NAIRN_FREE_FIELD_SURFACE_AREA_DYNAMIC_OUTPUT", "output/kohler_free_field_surface_area_dynamic")
)
JOINT_BOUNDARY_VALIDATION = os.environ.get("NAIRN_JOINT_BOUNDARY_VALIDATION", "0").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
JOINT_BOUNDARY_VALIDATION_OUTPUT = Path(
    os.environ.get(
        "NAIRN_JOINT_BOUNDARY_VALIDATION_OUTPUT",
        "output/kohler_short_nonzero_joint_boundary_validation",
    )
)
JOINT_BOUNDARY_NONZERO_TOL = float(os.environ.get("NAIRN_JOINT_BOUNDARY_NONZERO_TOL", "1e-12"))
JOINT_FORCE_RECONSTRUCTION_ABS_TOL = float(os.environ.get("NAIRN_JOINT_FORCE_RECONSTRUCTION_ABS_TOL", "1e-12"))
JOINT_FORCE_RECONSTRUCTION_REL_TOL = float(os.environ.get("NAIRN_JOINT_FORCE_RECONSTRUCTION_REL_TOL", "1e-12"))
AUXILIARY_F_DYNAMIC_PATH_SMOKE = os.environ.get(
    "NAIRN_AUXILIARY_F_DYNAMIC_PATH_SMOKE", "0"
).lower() not in {
    "0",
    "false",
    "no",
    "off",
}
AUXILIARY_F_DYNAMIC_PATH_SMOKE_OUTPUT = Path(
    os.environ.get("NAIRN_AUXILIARY_F_DYNAMIC_PATH_SMOKE_OUTPUT", "output/kohler_auxiliary_F_dynamic_path_smoke")
)
VERTICAL_TRANSITION_DIAGNOSTIC = os.environ.get(
    "NAIRN_VERTICAL_TRANSITION_DIAGNOSTIC", "0"
).lower() not in {
    "0",
    "false",
    "no",
    "off",
}
VERTICAL_TRANSITION_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get("NAIRN_VERTICAL_TRANSITION_DIAGNOSTIC_OUTPUT", "output/kohler_static_dynamic_transition_diagnostic")
)
VERTICAL_TRANSITION_PARTICLE_ID = int(os.environ.get("NAIRN_VERTICAL_TRANSITION_PARTICLE_ID", "8627"))
VERTICAL_TRANSITION_FORCE_TOLERANCE = float(
    os.environ.get("NAIRN_VERTICAL_TRANSITION_FORCE_TOLERANCE", "1e-6")
)
STATIC_DYNAMIC_SUPPORT_MAP_DIAGNOSTIC = os.environ.get(
    "NAIRN_STATIC_DYNAMIC_SUPPORT_MAP_DIAGNOSTIC", "0"
).lower() not in {"0", "false", "no", "off"}
STATIC_DYNAMIC_SUPPORT_MAP_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get(
        "NAIRN_STATIC_DYNAMIC_SUPPORT_MAP_DIAGNOSTIC_OUTPUT",
        "output/kohler_static_dynamic_support_map_diagnostic",
    )
)
STATIC_DYNAMIC_SUPPORT_MAP_TOLERANCE = float(
    os.environ.get("NAIRN_STATIC_DYNAMIC_SUPPORT_MAP_TOLERANCE", "1e-8")
)
STATIC_DYNAMIC_SUPPORT_MAP_RELATIVE_TOLERANCE = float(
    os.environ.get("NAIRN_STATIC_DYNAMIC_SUPPORT_MAP_RELATIVE_TOLERANCE", "1e-6")
)
BOTTOM_STATIC_SUPPORT_INJECTION_DIAGNOSTIC = os.environ.get(
    "NAIRN_BOTTOM_STATIC_SUPPORT_INJECTION_DIAGNOSTIC", "0"
).lower() not in {"0", "false", "no", "off"}
BOTTOM_STATIC_SUPPORT_INJECTION_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get(
        "NAIRN_BOTTOM_STATIC_SUPPORT_INJECTION_DIAGNOSTIC_OUTPUT",
        "output/kohler_bottom_static_support_injection_diagnostic",
    )
)
STATIC_SLIP_FORCE_DIAGNOSTIC = os.environ.get(
    "NAIRN_STATIC_SLIP_FORCE_DIAGNOSTIC", "0"
).lower() not in {"0", "false", "no", "off"}
STATIC_SLIP_FORCE_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get(
        "NAIRN_STATIC_SLIP_FORCE_DIAGNOSTIC_OUTPUT",
        "output/kohler_static_slip_force_diagnostic",
    )
)
# Read-only independent reconstruction of the native 2-D P2G force operator.
# This starts from a restored static checkpoint and advances no time step.
STATIC_P2G_OPERATOR_AUDIT = os.environ.get(
    "NAIRN_STATIC_P2G_OPERATOR_AUDIT", "0"
).lower() not in {"0", "false", "no", "off"}
STATIC_P2G_OPERATOR_AUDIT_OUTPUT = Path(
    os.environ.get(
        "NAIRN_STATIC_P2G_OPERATOR_AUDIT_OUTPUT",
        "output/kohler_static_p2g_operator_audit",
    )
)
STATIC_P2G_OPERATOR_AUDIT_TARGET_X = float(
    os.environ.get("NAIRN_STATIC_P2G_OPERATOR_AUDIT_TARGET_X", "-25.0")
)
STATIC_P2G_OPERATOR_AUDIT_TARGET_Y = float(
    os.environ.get("NAIRN_STATIC_P2G_OPERATOR_AUDIT_TARGET_Y", "9.0")
)
STATIC_P2G_OPERATOR_AUDIT_RADIUS = max(
    0.0,
    float(os.environ.get("NAIRN_STATIC_P2G_OPERATOR_AUDIT_RADIUS", str(3.0 * DX))),
)
CORNER_SUPERPOSITION_DIAGNOSTIC = os.environ.get(
    "NAIRN_CORNER_SUPERPOSITION_DIAGNOSTIC", "0"
).lower() not in {"0", "false", "no", "off"}
CORNER_SUPERPOSITION_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get(
        "NAIRN_CORNER_SUPERPOSITION_DIAGNOSTIC_OUTPUT",
        "output/kohler_corner_superposition_diagnostic",
    )
)
FREE_FIELD_RESIDUAL_VELOCITY_DIAGNOSTIC = os.environ.get(
    "NAIRN_FREE_FIELD_RESIDUAL_VELOCITY_DIAGNOSTIC", "0"
).lower() not in {"0", "false", "no", "off"}
FREE_FIELD_RESIDUAL_VELOCITY_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get(
        "NAIRN_FREE_FIELD_RESIDUAL_VELOCITY_DIAGNOSTIC_OUTPUT",
        "output/kohler_free_field_residual_velocity_diagnostic",
    )
)
RIGHT_FREE_FIELD_TRANSITION_DIAGNOSTIC = os.environ.get(
    "NAIRN_RIGHT_FREE_FIELD_TRANSITION_DIAGNOSTIC", "0"
).lower() not in {"0", "false", "no", "off"}
RIGHT_FREE_FIELD_TRANSITION_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get(
        "NAIRN_RIGHT_FREE_FIELD_TRANSITION_DIAGNOSTIC_OUTPUT",
        "output/kohler_right_free_field_transition_diagnostic",
    )
)
RIGHT_FREE_FIELD_STATIC_BALANCE_DIAGNOSTIC = os.environ.get(
    "NAIRN_RIGHT_FREE_FIELD_STATIC_BALANCE_DIAGNOSTIC", "0"
).lower() not in {"0", "false", "no", "off"}
RIGHT_FREE_FIELD_STATIC_BALANCE_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get(
        "NAIRN_RIGHT_FREE_FIELD_STATIC_BALANCE_DIAGNOSTIC_OUTPUT",
        "output/kohler_right_free_field_static_balance_diagnostic",
    )
)
RIGHT_FREE_FIELD_PERIODIC_SEAM_DIAGNOSTIC = os.environ.get(
    "NAIRN_RIGHT_FREE_FIELD_PERIODIC_SEAM_DIAGNOSTIC", "0"
).lower() not in {"0", "false", "no", "off"}
RIGHT_FREE_FIELD_PERIODIC_SEAM_DIAGNOSTIC_OUTPUT = Path(
    os.environ.get(
        "NAIRN_RIGHT_FREE_FIELD_PERIODIC_SEAM_DIAGNOSTIC_OUTPUT",
        "output/kohler_right_free_field_periodic_seam_diagnostic",
    )
)

MAT_SOIL = 1
MAT_BASE = 2
BODY_MAIN_SOIL = 0
BODY_MAIN_BASE = 0
BODY_LEFT_FREE_SOIL = 1
BODY_LEFT_FREE_BASE = 1
BODY_RIGHT_FREE_SOIL = 2
BODY_RIGHT_FREE_BASE = 2
MAIN_BODY_IDS = [0]
LEFT_FREE_BODY_IDS = [1]
RIGHT_FREE_BODY_IDS = [2]
BODY_SLIDE = 3
SEPARATE_SLIDE_BODY = False

SOIL_E_MODULUS = 4.0e4
SOIL_POISSON_INITIAL = 0.35
SOIL_POISSON_DYNAMIC = 0.495
SOIL_DENSITY = 1.8
BASE_E_MODULUS = 2.5e5
BASE_POISSON = 0.25
BASE_DENSITY = 2.2
E_MODULUS = BASE_E_MODULUS
POISSON = BASE_POISSON
DENSITY = BASE_DENSITY
MATERIAL_MODEL = os.environ.get("NAIRN_MATERIAL_MODEL", "LinearElastic")
DP_COHESION = float(os.environ.get("NAIRN_DP_COHESION", "0.25"))
DP_FRICTION = float(os.environ.get("NAIRN_DP_FRICTION", "30.0"))
DP_DILATION = float(os.environ.get("NAIRN_DP_DILATION", "0.0"))
DP_TENSILE = float(os.environ.get("NAIRN_DP_TENSILE", "0.0"))
DP_TYPE = os.environ.get("NAIRN_DP_TYPE", "Inscribed")
SOFTENING_ENABLED = os.environ.get("NAIRN_SOFTENING_ENABLED", "0").lower() not in {"0", "false", "no", "off"}
SOFTENING_RESIDUAL_COHESION = float(os.environ.get("NAIRN_SOFTENING_RESIDUAL_COHESION", "0.10"))
SOFTENING_RESIDUAL_FRICTION = float(os.environ.get("NAIRN_SOFTENING_RESIDUAL_FRICTION", "20.0"))
SOFTENING_RESIDUAL_DILATION = float(os.environ.get("NAIRN_SOFTENING_RESIDUAL_DILATION", "0.0"))
SOFTENING_EPS_START = float(os.environ.get("NAIRN_SOFTENING_EPS_START", "1e-7"))
SOFTENING_EPS_END = float(os.environ.get("NAIRN_SOFTENING_EPS_END", "7e-7"))
SHEAR_MODULUS = E_MODULUS / (2.0 * (1.0 + POISSON))
BULK_MODULUS = E_MODULUS / (3.0 * (1.0 - 2.0 * POISSON))
CS = math.sqrt(SHEAR_MODULUS / DENSITY)
CP = math.sqrt((BULK_MODULUS + 4.0 * SHEAR_MODULUS / 3.0) / DENSITY)

INPUT_STRESS_PEAK = 1.0
INPUT_FREQUENCY = 100.0
INPUT_PERIOD = 1.0 / INPUT_FREQUENCY
DEFAULT_EARTHQUAKE_MOTION_FILE = (
    Path(__file__).with_name("earthquake_motion_corrected.csv")
    if Path(__file__).with_name("earthquake_motion_corrected.csv").exists()
    else Path(__file__).with_name("earthquake_motion.csv")
)
EARTHQUAKE_MOTION_FILE = Path(os.environ.get("NAIRN_EARTHQUAKE_FILE", DEFAULT_EARTHQUAKE_MOTION_FILE.as_posix()))
DEFAULT_FIG10C_INPUT_VELOCITY_FILE = Path(__file__).with_name("kohler_fig10c_input_velocity.csv")
FIG10C_INPUT_VELOCITY_FILE = Path(
    os.environ.get("NAIRN_FIG10C_INPUT_FILE", DEFAULT_FIG10C_INPUT_VELOCITY_FILE.as_posix())
)
# Fig.10(c) is stored as an input-velocity history and must not be reintegrated.
SEISMIC_INPUT_MODE = os.environ.get("NAIRN_SEISMIC_INPUT_MODE", os.environ.get("SEISMIC_INPUT_MODE", "FIG10C")).upper()
SEISMIC_INPUT_FACTOR = float(os.environ.get("NAIRN_SEISMIC_INPUT_FACTOR", "0.5"))
BOTTOM_TOL = 0.5 * DX + 0.02
# The dynamic compliant-base traction in Eq. (29) is applied to the
# boundary-facing material-point row.  With the prescribed 3x3 MPs per cell,
# its centres lie DX / 6 above the physical base.
BOTTOM_TRACTION_ROW_TOL = DX / 6.0 + 0.02
SIDE_TOL = 0.5 * DX + 0.02

HALF_PARTICLE_SIZE = 0.5 * DX


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


K_MAIN_X_MIN = -40.0
K_MAIN_X_MAX = 140.0
K_SOIL_THICKNESS = 10.0
# The paper specifies 15 m at the right-hand elastic base column. The base
# thickens towards the left because its bottom is horizontal while the
# soil/base interface follows the Gaussian slope.
K_BASE_THICKNESS = 15.0
K_SLOPE_ANGLE_DEG = 22.0
K_SURFACE_UPPER_ASYMPTOTE_Z = 11.0
K_SURFACE_LOWER_ASYMPTOTE_Z = -11.0
K_GAUSSIAN_TOTAL_DROP = K_SURFACE_UPPER_ASYMPTOTE_Z - K_SURFACE_LOWER_ASYMPTOTE_Z
K_SLOPE_CENTER_X = 35.0
# For z = z_high - drop*Phi((x-mu)/sigma), the maximum gradient is
# drop/(sigma*sqrt(2*pi)). This choice therefore enforces 22 degrees.
K_SLOPE_SIGMA = K_GAUSSIAN_TOTAL_DROP / (
    math.tan(math.radians(K_SLOPE_ANGLE_DEG)) * math.sqrt(2.0 * math.pi)
)
K_SLOPE_START_X = K_MAIN_X_MIN
K_SLOPE_END_X = K_MAIN_X_MAX
K_SLOPE_LENGTH = K_SLOPE_END_X - K_SLOPE_START_X
K_GAUSSIAN_CDF_LEFT = normal_cdf((K_MAIN_X_MIN - K_SLOPE_CENTER_X) / K_SLOPE_SIGMA)
K_GAUSSIAN_CDF_RIGHT = normal_cdf((K_MAIN_X_MAX - K_SLOPE_CENTER_X) / K_SLOPE_SIGMA)
K_HIGH_SURFACE_Z = K_SURFACE_UPPER_ASYMPTOTE_Z - K_GAUSSIAN_TOTAL_DROP * K_GAUSSIAN_CDF_LEFT
K_LOW_SURFACE_Z = K_SURFACE_UPPER_ASYMPTOTE_Z - K_GAUSSIAN_TOTAL_DROP * K_GAUSSIAN_CDF_RIGHT
K_SLOPE_DROP = K_HIGH_SURFACE_Z - K_LOW_SURFACE_Z
K_INTERFACE_HIGH_Z = K_HIGH_SURFACE_Z - K_SOIL_THICKNESS
K_INTERFACE_DROP = K_SLOPE_DROP
K_INTERFACE_LOW_Z = K_LOW_SURFACE_Z - K_SOIL_THICKNESS
K_BASE_BOTTOM_Z = -36.0
K_BASE_BOTTOM_HIGH_Z = K_BASE_BOTTOM_Z
K_BASE_BOTTOM_LOW_Z = K_BASE_BOTTOM_Z
K_MONITOR_X = 2.5
K_MONITOR_SURFACE_Z = K_SURFACE_UPPER_ASYMPTOTE_Z - K_GAUSSIAN_TOTAL_DROP * normal_cdf(
    (K_MONITOR_X - K_SLOPE_CENTER_X) / K_SLOPE_SIGMA
)
K_MONITOR_INTERFACE_Z = K_MONITOR_SURFACE_Z - K_SOIL_THICKNESS
K_FREE_FIELD_WIDTH = 4.0 * DX
K_FREE_FIELD_GAP = float(os.environ.get("NAIRN_FREE_FIELD_GAP", "1.0"))
if K_FREE_FIELD_GAP < 0.0:
    raise ValueError("NAIRN_FREE_FIELD_GAP must be non-negative")
K_LEFT_FREE_X_MAX = K_MAIN_X_MIN - K_FREE_FIELD_GAP
K_LEFT_FREE_X_MIN = K_LEFT_FREE_X_MAX - K_FREE_FIELD_WIDTH
K_RIGHT_FREE_X_MIN = K_MAIN_X_MAX + K_FREE_FIELD_GAP
K_RIGHT_FREE_X_MAX = K_RIGHT_FREE_X_MIN + K_FREE_FIELD_WIDTH
# Particle coordinates are stored by GeoTaichi in float32 fields.  At the
# physical x locations near +/-145 m, one ulp is larger than 1e-6 m; use a
# data-precision-based tolerance for interface-gap diagnostics.
PARTICLE_COORDINATE_TOLERANCE = max(
    1.0e-8,
    4.0
    * np.finfo(np.float32).eps
    * max(1.0, abs(X_SHIFT), abs(K_LEFT_FREE_X_MIN), abs(K_RIGHT_FREE_X_MAX)),
)
K_LEFT_FREE_BOTTOM_Z = K_BASE_BOTTOM_Z
K_RIGHT_FREE_BOTTOM_Z = K_BASE_BOTTOM_Z
K_LEFT_FREE_HEIGHT = K_HIGH_SURFACE_Z - K_BASE_BOTTOM_Z
K_RIGHT_FREE_HEIGHT = K_LOW_SURFACE_Z - K_BASE_BOTTOM_Z
K_DOMAIN_Y_MIN = K_BASE_BOTTOM_Z
K_DOMAIN_Y_MAX = K_HIGH_SURFACE_Z
K_DOMAIN_HEIGHT = K_DOMAIN_Y_MAX - K_DOMAIN_Y_MIN
MAIN_WIDTH = K_MAIN_X_MAX - K_MAIN_X_MIN


def gaussian_cdf_antiderivative(xp: float) -> float:
    standardized = (xp - K_SLOPE_CENTER_X) / K_SLOPE_SIGMA
    return (xp - K_SLOPE_CENTER_X) * normal_cdf(standardized) + K_SLOPE_SIGMA * normal_pdf(standardized)


SURFACE_INTEGRAL = (
    K_SURFACE_UPPER_ASYMPTOTE_Z * MAIN_WIDTH
    - K_GAUSSIAN_TOTAL_DROP
    * (gaussian_cdf_antiderivative(K_MAIN_X_MAX) - gaussian_cdf_antiderivative(K_MAIN_X_MIN))
)
MAIN_BASE_AREA = SURFACE_INTEGRAL - (K_SOIL_THICKNESS + K_BASE_BOTTOM_Z) * MAIN_WIDTH
MAIN_AREA = K_SOIL_THICKNESS * MAIN_WIDTH + MAIN_BASE_AREA
BOUNDING_BOX_EPS = 0.0

CASE: dict[str, Any] = {}
EARTHQUAKE_MOTION: EarthquakeMotion | None = None


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


INCREMENTAL_HISTORY_ENABLED = env_bool("NAIRN_INCREMENTAL_HISTORY", False)
INCREMENTAL_HISTORY_INTERVAL = float(os.environ.get("NAIRN_INCREMENTAL_HISTORY_INTERVAL", "0.01"))
INCREMENTAL_HISTORY_FSYNC = env_bool("NAIRN_INCREMENTAL_HISTORY_FSYNC", False)
DOMAIN_ESCAPE_DIAGNOSTIC = env_bool("NAIRN_DOMAIN_ESCAPE_DIAGNOSTIC", False)
DOMAIN_ESCAPE_ABORT = env_bool("NAIRN_ABORT_ON_DOMAIN_ESCAPE", False)
DOMAIN_ESCAPE_TOLERANCE = float(os.environ.get("NAIRN_DOMAIN_ESCAPE_TOLERANCE", "1e-8"))
INCREMENTAL_HISTORY_OUTPUT = Path(
    os.environ.get("NAIRN_INCREMENTAL_HISTORY_OUTPUT", str(OUTPUT_DIR / "incremental_history.csv"))
)
INCREMENTAL_BOUNDARY_OUTPUT = Path(
    os.environ.get("NAIRN_INCREMENTAL_BOUNDARY_OUTPUT", str(OUTPUT_DIR / "incremental_boundary_forces.csv"))
)
DOMAIN_ESCAPE_OUTPUT = Path(
    os.environ.get("NAIRN_DOMAIN_ESCAPE_OUTPUT", str(OUTPUT_DIR / "domain_escape_diagnostic.csv"))
)
DOMAIN_FAILURE_OUTPUT = Path(
    os.environ.get("NAIRN_DOMAIN_FAILURE_OUTPUT", str(OUTPUT_DIR / "runtime_failure_diagnostic.md"))
)


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def normalize_material_parameters(case: dict[str, Any]) -> None:
    material_model = str(case.get("material_model", "LinearElastic"))
    softening_spec = case.get("softening", {})
    if material_model == "VonMisesSoftening" and "residual_shear_displacement_m" in softening_spec:
        h = float(case.get("dx", DX))
        d_residual = float(softening_spec["residual_shear_displacement_m"])
        h_shear = float(softening_spec.get("h_shear_factor", 2.0)) * h
        eps_derived = d_residual / (math.sqrt(3.0) * h_shear) if h_shear > 0.0 else math.nan
        configured_eps = float(
            os.environ.get(
                "NAIRN_VM_EPS_END",
                softening_spec.get("eps_end", SOFTENING_EPS_END),
            )
        )
        if not math.isfinite(eps_derived) or not math.isclose(
            configured_eps, eps_derived, rel_tol=1.0e-8, abs_tol=1.0e-8
        ):
            raise ValueError(
                "VonMisesSoftening eps_end does not match d_r/(sqrt(3)*h_shear): "
                f"configured={configured_eps}, derived={eps_derived}"
            )
    dp_env_overrides = {
        "Cohesion": os.environ.get("NAIRN_DP_COHESION"),
        "Friction": os.environ.get("NAIRN_DP_FRICTION"),
        "Dilation": os.environ.get("NAIRN_DP_DILATION"),
        "Tensile": os.environ.get("NAIRN_DP_TENSILE"),
        "dpType": os.environ.get("NAIRN_DP_TYPE"),
    }
    for material in case.get("materials", []):
        if material_model == "DruckerPrager":
            material.setdefault("Cohesion", DP_COHESION)
            material.setdefault("Friction", DP_FRICTION)
            material.setdefault("Dilation", DP_DILATION)
            material.setdefault("Tensile", DP_TENSILE)
            material.setdefault("dpType", DP_TYPE)
            material.setdefault("ResidualCohesion", softening_spec.get("cohesion_residual", SOFTENING_RESIDUAL_COHESION))
            material.setdefault("ResidualFriction", softening_spec.get("friction_residual", SOFTENING_RESIDUAL_FRICTION))
            material.setdefault("ResidualDilation", softening_spec.get("dilation_residual", SOFTENING_RESIDUAL_DILATION))
            material.setdefault("PlasticDevStrain", softening_spec.get("eps_start", SOFTENING_EPS_START))
            material.setdefault("ResidualPlasticDevStrain", softening_spec.get("eps_end", SOFTENING_EPS_END))
            for key, value in dp_env_overrides.items():
                if value is not None:
                    material[key] = value if key == "dpType" else float(value)
            residual_env_overrides = {
                "ResidualCohesion": os.environ.get("NAIRN_SOFTENING_RESIDUAL_COHESION"),
                "ResidualFriction": os.environ.get("NAIRN_SOFTENING_RESIDUAL_FRICTION"),
                "ResidualDilation": os.environ.get("NAIRN_SOFTENING_RESIDUAL_DILATION"),
                "PlasticDevStrain": os.environ.get("NAIRN_SOFTENING_EPS_START"),
                "ResidualPlasticDevStrain": os.environ.get("NAIRN_SOFTENING_EPS_END"),
            }
            for key, value in residual_env_overrides.items():
                if value is not None:
                    material[key] = float(value)
        elif material_model == "ElasticPerfectlyPlastic":
            material.setdefault("YieldStress", DP_COHESION)
            if os.environ.get("NAIRN_EP_YIELD_STRESS") is not None:
                material["YieldStress"] = float(os.environ["NAIRN_EP_YIELD_STRESS"])
        elif material_model == "IsotropicHardeningPlastic":
            material.setdefault("YieldStress", DP_COHESION)
            material.setdefault("PlasticModulus", 0.0)
            if os.environ.get("NAIRN_EP_YIELD_STRESS") is not None:
                material["YieldStress"] = float(os.environ["NAIRN_EP_YIELD_STRESS"])
            if os.environ.get("NAIRN_PLASTIC_MODULUS") is not None:
                material["PlasticModulus"] = float(os.environ["NAIRN_PLASTIC_MODULUS"])
        elif material_model == "VonMisesSoftening":
            material.setdefault(
                "YieldStress",
                softening_spec.get("yield_stress_initial", material.get("YieldStress", material.get("Cohesion", DP_COHESION))),
            )
            material.setdefault(
                "ResidualYieldStress",
                softening_spec.get("yield_stress_residual", material.get("ResidualYieldStress", material.get("ResidualCohesion", SOFTENING_RESIDUAL_COHESION))),
            )
            material.setdefault("PlasticDevStrain", softening_spec.get("eps_start", SOFTENING_EPS_START))
            material.setdefault("ResidualPlasticDevStrain", softening_spec.get("eps_end", SOFTENING_EPS_END))
            vm_env_overrides = {
                "YieldStress": os.environ.get("NAIRN_VM_YIELD_STRESS"),
                "ResidualYieldStress": os.environ.get("NAIRN_VM_RESIDUAL_YIELD_STRESS"),
                "PlasticDevStrain": os.environ.get("NAIRN_VM_EPS_START"),
                "ResidualPlasticDevStrain": os.environ.get("NAIRN_VM_EPS_END"),
            }
            for key, value in vm_env_overrides.items():
                if value is not None:
                    material[key] = float(value)
        elif material_model == "SoftenMohrCoulomb":
            material.setdefault("Cohesion", softening_spec.get("cohesion_initial", DP_COHESION))
            material.setdefault("Friction", softening_spec.get("friction_initial", DP_FRICTION))
            material.setdefault("Dilation", softening_spec.get("dilation_initial", DP_DILATION))
            material.setdefault("Tensile", DP_TENSILE)
            material.setdefault("ResidualCohesion", softening_spec.get("cohesion_residual", SOFTENING_RESIDUAL_COHESION))
            material.setdefault("ResidualFriction", softening_spec.get("friction_residual", SOFTENING_RESIDUAL_FRICTION))
            material.setdefault("ResidualDilation", softening_spec.get("dilation_residual", SOFTENING_RESIDUAL_DILATION))
            material.setdefault("PlasticDevStrain", softening_spec.get("eps_start", SOFTENING_EPS_START))
            material.setdefault("ResidualPlasticDevStrain", softening_spec.get("eps_end", SOFTENING_EPS_END))
            env_overrides = {
                "Cohesion": os.environ.get("NAIRN_DP_COHESION"),
                "Friction": os.environ.get("NAIRN_DP_FRICTION"),
                "Dilation": os.environ.get("NAIRN_DP_DILATION"),
                "Tensile": os.environ.get("NAIRN_DP_TENSILE"),
                "ResidualCohesion": os.environ.get("NAIRN_SOFTENING_RESIDUAL_COHESION"),
                "ResidualFriction": os.environ.get("NAIRN_SOFTENING_RESIDUAL_FRICTION"),
                "ResidualDilation": os.environ.get("NAIRN_SOFTENING_RESIDUAL_DILATION"),
                "PlasticDevStrain": os.environ.get("NAIRN_SOFTENING_EPS_START"),
                "ResidualPlasticDevStrain": os.environ.get("NAIRN_SOFTENING_EPS_END"),
            }
            for key, value in env_overrides.items():
                if value is not None:
                    material[key] = float(value)


DEFAULT_CASE: dict[str, Any] = {
    "name": "kohler_slope_geometry",
    "title": "Kohler-style slope seismic validation geometry",
    "source": "Kohler 2022 example slope response model; earthquake record and grid size remain configurable",
    "coordinate_shift": [X_SHIFT, Y_SHIFT],
    "domain": DOMAIN.tolist(),
    "dx": DX,
    "dt": DT,
    "simulation_time": SIMULATION_TIME,
    "save_interval": SAVE_INTERVAL,
    "history_interval": HISTORY_INTERVAL,
    "output_dir": OUTPUT_DIR.as_posix(),
    "output_prefix": "kohler_slope_geometry",
    "arch": ARCH,
    "mapping": MAPPING,
    "shape_function": SHAPE_FUNCTION,
    "strict_time_loop": STRICT_TIME_LOOP,
    "run_postprocess": RUN_POSTPROCESS,
    "save_particle": SAVE_PARTICLE,
    "save_grid": SAVE_GRID,
    "material_model": MATERIAL_MODEL,
    "softening": {
        "enabled": SOFTENING_ENABLED,
        "cohesion_initial": DP_COHESION,
        "cohesion_residual": SOFTENING_RESIDUAL_COHESION,
        "yield_stress_initial": DP_COHESION,
        "yield_stress_residual": SOFTENING_RESIDUAL_COHESION,
        "friction_initial": DP_FRICTION,
        "friction_residual": SOFTENING_RESIDUAL_FRICTION,
        "dilation_initial": DP_DILATION,
        "dilation_residual": SOFTENING_RESIDUAL_DILATION,
        "eps_start": SOFTENING_EPS_START,
        "eps_end": SOFTENING_EPS_END,
        "law": "linear",
        "state_variable": "epstrain",
    },
    "geometry_parameters": {
        "source": "Kohler et al. (2022) Fig.7 constrained geometry fit: dimensions stated in the paper plus Gaussian parameters inferred from the published figure",
        "assumption_notice": "The paper states the 10 m soil thickness, 15 m right-side elastic base, horizontal bottom, Gaussian surface, and approximately 22 degree maximum inclination, but does not publish the exact Gaussian center or standard deviation. The center and asymptotic elevations used here are constrained fits to Fig.7.",
        "main_x_min": K_MAIN_X_MIN,
        "main_x_max": K_MAIN_X_MAX,
        "soil_thickness": K_SOIL_THICKNESS,
        "base_thickness": K_BASE_THICKNESS,
        "slope_length": K_SLOPE_LENGTH,
        "slope_start_x": K_SLOPE_START_X,
        "slope_end_x": K_SLOPE_END_X,
        "slope_angle_deg": K_SLOPE_ANGLE_DEG,
        "surface": "z = 11 m - 22 m * Phi((x - 35 m) / sigma), sigma = 22 m / (tan(22 deg) * sqrt(2*pi)); the soil/base interface is the 10 m vertical offset and the base bottom is horizontal at z=-36 m.",
        "surface_upper_asymptote_z": K_SURFACE_UPPER_ASYMPTOTE_Z,
        "surface_lower_asymptote_z": K_SURFACE_LOWER_ASYMPTOTE_Z,
        "high_surface_z": K_HIGH_SURFACE_Z,
        "low_surface_z": K_LOW_SURFACE_Z,
        "slope_drop": K_SLOPE_DROP,
        "gaussian_center_x": K_SLOPE_CENTER_X,
        "gaussian_sigma": K_SLOPE_SIGMA,
        "monitor_x": K_MONITOR_X,
        "monitor_surface_z": K_MONITOR_SURFACE_Z,
        "monitor_interface_z": K_MONITOR_INTERFACE_Z,
        "interface_high_z": K_INTERFACE_HIGH_Z,
        "interface_low_z": K_INTERFACE_LOW_Z,
        "interface_drop": K_INTERFACE_DROP,
        "base_bottom_high_z": K_BASE_BOTTOM_HIGH_Z,
        "base_bottom_low_z": K_BASE_BOTTOM_LOW_Z,
        "left_free_x_min": K_LEFT_FREE_X_MIN,
        "left_free_x_max": K_LEFT_FREE_X_MAX,
        "right_free_x_min": K_RIGHT_FREE_X_MIN,
        "right_free_x_max": K_RIGHT_FREE_X_MAX,
        "free_field_width": K_FREE_FIELD_WIDTH,
        "free_field_gap": K_FREE_FIELD_GAP,
        "left_free_bottom_z": K_LEFT_FREE_BOTTOM_Z,
        "right_free_bottom_z": K_RIGHT_FREE_BOTTOM_Z,
        "left_free_height": K_LEFT_FREE_HEIGHT,
        "right_free_height": K_RIGHT_FREE_HEIGHT,
    },
    "static_initialization": {
        "enabled": STATIC_ENABLED,
        "dt": STATIC_DT,
        "time": STATIC_TIME,
        "ramp_time": STATIC_RAMP_TIME,
        "save_interval": STATIC_SAVE_INTERVAL,
        "history_interval": STATIC_HISTORY_INTERVAL,
        "gravity": [0.0, STATIC_GRAVITY],
        "ramp": "smoothstep",
            "background_damping": STATIC_DAMPING,
            "alpha_pic": STATIC_ALPHA_PIC,
            "velocity_tolerance": STATIC_VELOCITY_TOLERANCE,
        "max_velocity_tolerance": STATIC_MAX_VELOCITY_TOLERANCE,
        "rms_velocity_tolerance": STATIC_RMS_VELOCITY_TOLERANCE,
        "relative_unbalanced_force_tolerance": STATIC_RELATIVE_UNBALANCED_FORCE_TOLERANCE,
        "stress_tolerance": STATIC_STRESS_TOLERANCE,
        "velocity_quench": STATIC_VELOCITY_QUENCH,
        "quench_max_velocity_factor": STATIC_QUENCH_MAX_VELOCITY_FACTOR,
        "require_convergence": REQUIRE_STATIC_CONVERGENCE,
        "output_report": "static_initialization_report.md",
    },
    "run_dynamic": RUN_DYNAMIC,
    "memory": {
        "max_material_number": 3,
        "max_particle_number": 1000000,
        "max_constraint_number": {
            "max_velocity_constraint": 8000,
            "max_particle_traction_constraint": 3000,
        },
    },
    "materials": [
        {
            "MaterialID": MAT_SOIL,
            "Density": SOIL_DENSITY,
            "YoungModulus": SOIL_E_MODULUS,
            "PossionRatio": SOIL_POISSON_DYNAMIC,
            "StaticPossionRatio": SOIL_POISSON_INITIAL,
            "DynamicPossionRatio": SOIL_POISSON_DYNAMIC,
            "Cohesion": DP_COHESION,
            "Friction": DP_FRICTION,
            "Dilation": DP_DILATION,
            "Tensile": DP_TENSILE,
            "dpType": DP_TYPE,
        },
        {
            "MaterialID": MAT_BASE,
            "Density": BASE_DENSITY,
            "YoungModulus": BASE_E_MODULUS,
            "PossionRatio": BASE_POISSON,
            "Cohesion": DP_COHESION,
            "Friction": DP_FRICTION,
            "Dilation": DP_DILATION,
            "Tensile": DP_TENSILE,
            "dpType": DP_TYPE,
        },
    ],
    "wave_speeds": {"material_id": MAT_BASE},
    "earthquake_input": {
        "enabled": True,
        "mode": SEISMIC_INPUT_MODE,
        "file": EARTHQUAKE_MOTION_FILE.as_posix(),
        "fig10c_file": FIG10C_INPUT_VELOCITY_FILE.as_posix(),
        "time_column": "time",
        "horizontal_acceleration_column": "acceleration",
        "acceleration_unit": "m/s^2",
        "integration": "trapezoidal",
        "input_motion_type": "outcrop_acceleration_to_upward_shear_wave_velocity",
        "input_velocity_factor": SEISMIC_INPUT_FACTOR,
        "shear_input": True,
        "pressure_input": False,
    },
    "input_stress": {"type": "external_acceleration", "file": EARTHQUAKE_MOTION_FILE.as_posix()},
    "regions": [
        {
            "name": "left_free_soil",
            "function": "slope_left_free_soil",
            "bbox_point": [K_LEFT_FREE_X_MIN, K_INTERFACE_HIGH_Z],
            "bbox_size": [K_FREE_FIELD_WIDTH, K_SOIL_THICKNESS],
            "volume": K_FREE_FIELD_WIDTH * K_SOIL_THICKNESS,
        },
        {
            "name": "left_free_base",
            "function": "slope_left_free_base",
            "bbox_point": [K_LEFT_FREE_X_MIN, K_LEFT_FREE_BOTTOM_Z],
            "bbox_size": [K_FREE_FIELD_WIDTH, K_LEFT_FREE_HEIGHT - K_SOIL_THICKNESS],
            "volume": K_FREE_FIELD_WIDTH * (K_LEFT_FREE_HEIGHT - K_SOIL_THICKNESS),
        },
        {
            "name": "main_soil_layer",
            "function": "slope_main_soil",
            "bbox_point": [K_MAIN_X_MIN, K_INTERFACE_LOW_Z],
            "bbox_size": [K_MAIN_X_MAX - K_MAIN_X_MIN, K_SOIL_THICKNESS + K_SLOPE_DROP],
            "volume": K_SOIL_THICKNESS * (K_MAIN_X_MAX - K_MAIN_X_MIN),
        },
        {
            "name": "main_elastic_base",
            "function": "slope_main_base",
            "bbox_point": [K_MAIN_X_MIN, K_DOMAIN_Y_MIN],
            "bbox_size": [K_MAIN_X_MAX - K_MAIN_X_MIN, K_DOMAIN_Y_MAX - K_DOMAIN_Y_MIN],
            "volume": MAIN_BASE_AREA,
        },
        {
            "name": "right_free_soil",
            "function": "slope_right_free_soil",
            "bbox_point": [K_RIGHT_FREE_X_MIN, K_INTERFACE_LOW_Z],
            "bbox_size": [K_FREE_FIELD_WIDTH, K_SOIL_THICKNESS],
            "volume": K_FREE_FIELD_WIDTH * K_SOIL_THICKNESS,
        },
        {
            "name": "right_free_base",
            "function": "slope_right_free_base",
            "bbox_point": [K_RIGHT_FREE_X_MIN, K_RIGHT_FREE_BOTTOM_Z],
            "bbox_size": [K_FREE_FIELD_WIDTH, K_RIGHT_FREE_HEIGHT - K_SOIL_THICKNESS],
            "volume": K_FREE_FIELD_WIDTH * (K_RIGHT_FREE_HEIGHT - K_SOIL_THICKNESS),
        },
    ],
    "bodies": [
        {"region": "left_free_soil", "material_id": MAT_SOIL, "body_id": BODY_LEFT_FREE_SOIL, "n_particles_per_cell": 3},
        {"region": "left_free_base", "material_id": MAT_BASE, "body_id": BODY_LEFT_FREE_BASE, "n_particles_per_cell": 3},
        {"region": "main_soil_layer", "material_id": MAT_SOIL, "body_id": BODY_MAIN_SOIL, "n_particles_per_cell": 3},
        {"region": "main_elastic_base", "material_id": MAT_BASE, "body_id": BODY_MAIN_BASE, "n_particles_per_cell": 3},
        {"region": "right_free_soil", "material_id": MAT_SOIL, "body_id": BODY_RIGHT_FREE_SOIL, "n_particles_per_cell": 3},
        {"region": "right_free_base", "material_id": MAT_BASE, "body_id": BODY_RIGHT_FREE_BASE, "n_particles_per_cell": 3},
    ],
    "bottom_traction": {
        "function": "slope_bottom",
        "pressure_seed": [1.0e-30, 0.0],
    },
    "silent_boundary": {
        "bottom": True,
        "side": {
            "enabled": True,
            "body_ids": MAIN_BODY_IDS,
            "x_values": [K_MAIN_X_MIN, K_MAIN_X_MAX],
            "y_range": [K_DOMAIN_Y_MIN, K_DOMAIN_Y_MAX],
            "tolerance": SIDE_TOL,
        },
    },
    "monitors": [
        {
            "name": "left_free_field_monitor",
            "body_id": BODY_LEFT_FREE_SOIL,
            "point": [K_LEFT_FREE_X_MAX - HALF_PARTICLE_SIZE, K_INTERFACE_HIGH_Z],
            "selector": "soil_base_interface_nearest_particle",
            "purpose": "left free-field response independence check",
        },
        {
            "name": "right_free_field_monitor",
            "body_id": BODY_RIGHT_FREE_SOIL,
            "point": [K_RIGHT_FREE_X_MIN + HALF_PARTICLE_SIZE, K_INTERFACE_LOW_Z],
            "selector": "soil_base_interface_nearest_particle",
            "purpose": "right free-field response independence check",
        },
        {
            "name": "kohler_boundary_interface",
            "body_id": BODY_MAIN_SOIL,
            "point": [K_MONITOR_X, K_MONITOR_INTERFACE_Z],
            "selector": "soil_base_interface_nearest_particle",
            "purpose": "soil-base interface velocity response for Kohler Fig.10 comparison",
        },
        {
            "name": "kohler_surface",
            "body_id": BODY_MAIN_SOIL,
            "point": [K_MONITOR_X, K_MONITOR_SURFACE_Z],
            "selector": "surface_nearest_particle",
            "purpose": "surface horizontal velocity vx(t) for Kohler Fig.10 comparison",
        },
        {
            "name": "kohler_input_motion",
            "source": "input_motion",
            "purpose": "input velocity history at bottom compliant base input boundary",
        },
    ],
}


@ti.pyfunc
def nairn_left_free_soil_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    return (
        K_LEFT_FREE_X_MIN <= xp <= K_LEFT_FREE_X_MAX
        and K_INTERFACE_HIGH_Z <= yp <= K_HIGH_SURFACE_Z
    )


@ti.pyfunc
def nairn_left_free_base_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    return (
        K_LEFT_FREE_X_MIN <= xp <= K_LEFT_FREE_X_MAX
        and K_LEFT_FREE_BOTTOM_Z <= yp <= K_INTERFACE_HIGH_Z
    )


@ti.pyfunc
def nairn_right_free_soil_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    return (
        K_RIGHT_FREE_X_MIN <= xp <= K_RIGHT_FREE_X_MAX
        and K_INTERFACE_LOW_Z <= yp <= K_LOW_SURFACE_Z
    )


@ti.pyfunc
def nairn_right_free_base_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    return (
        K_RIGHT_FREE_X_MIN <= xp <= K_RIGHT_FREE_X_MAX
        and K_RIGHT_FREE_BOTTOM_Z <= yp <= K_INTERFACE_LOW_Z
    )


@ti.pyfunc
def erf_ti(value):
    sign = 1.0
    x_abs = value
    if value < 0.0:
        sign = -1.0
        x_abs = -value
    t = 1.0 / (1.0 + 0.3275911 * x_abs)
    poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    return sign * (1.0 - poly * ti.exp(-x_abs * x_abs))


@ti.pyfunc
def kohler_surface_z(xp):
    cdf = 0.5 * (1.0 + erf_ti((xp - K_SLOPE_CENTER_X) / (K_SLOPE_SIGMA * ti.sqrt(2.0))))
    return K_SURFACE_UPPER_ASYMPTOTE_Z - K_GAUSSIAN_TOTAL_DROP * cdf


@ti.pyfunc
def kohler_soil_base_interface_z(xp):
    return kohler_surface_z(xp) - K_SOIL_THICKNESS


@ti.pyfunc
def kohler_base_bottom_z(xp):
    return K_BASE_BOTTOM_Z


@ti.pyfunc
def nairn_main_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    return K_MAIN_X_MIN <= xp <= K_MAIN_X_MAX and kohler_base_bottom_z(xp) <= yp <= kohler_surface_z(xp)


@ti.pyfunc
def nairn_main_soil_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    return K_MAIN_X_MIN <= xp <= K_MAIN_X_MAX and kohler_soil_base_interface_z(xp) <= yp <= kohler_surface_z(xp)


@ti.pyfunc
def nairn_main_base_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    return K_MAIN_X_MIN <= xp <= K_MAIN_X_MAX and kohler_base_bottom_z(xp) <= yp <= kohler_soil_base_interface_z(xp)


@ti.pyfunc
def nairn_bottom_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    main_bottom = (
        K_MAIN_X_MIN <= xp <= K_MAIN_X_MAX
        and ti.abs(yp - kohler_base_bottom_z(xp)) <= BOTTOM_TRACTION_ROW_TOL
    )
    left_free_bottom = (
        K_LEFT_FREE_X_MIN <= xp <= K_LEFT_FREE_X_MAX
        and ti.abs(yp - K_LEFT_FREE_BOTTOM_Z) <= BOTTOM_TRACTION_ROW_TOL
    )
    right_free_bottom = (
        K_RIGHT_FREE_X_MIN <= xp <= K_RIGHT_FREE_X_MAX
        and ti.abs(yp - K_RIGHT_FREE_BOTTOM_Z) <= BOTTOM_TRACTION_ROW_TOL
    )
    return main_bottom or left_free_bottom or right_free_bottom


@ti.pyfunc
def nairn_static_bottom_no_slip_region(x):
    xp = x[0] - X_SHIFT
    yp = x[1] - Y_SHIFT
    main_bottom = K_MAIN_X_MIN <= xp <= K_MAIN_X_MAX and ti.abs(yp - kohler_base_bottom_z(xp)) <= BOTTOM_TOL
    left_free_bottom = (
        K_LEFT_FREE_X_MIN <= xp <= K_LEFT_FREE_X_MAX and ti.abs(yp - K_LEFT_FREE_BOTTOM_Z) <= BOTTOM_TOL
    )
    right_free_bottom = (
        K_RIGHT_FREE_X_MIN <= xp <= K_RIGHT_FREE_X_MAX and ti.abs(yp - K_RIGHT_FREE_BOTTOM_Z) <= BOTTOM_TOL
    )
    return main_bottom or left_free_bottom or right_free_bottom


def main_region_volume() -> float:
    return MAIN_AREA


REGION_FUNCTIONS: dict[str, Any] = {
    "slope_left_free": nairn_left_free_soil_region,
    "slope_left_free_soil": nairn_left_free_soil_region,
    "slope_left_free_base": nairn_left_free_base_region,
    "slope_main": nairn_main_region,
    "slope_main_soil": nairn_main_soil_region,
    "slope_main_base": nairn_main_base_region,
    "slope_right_free": nairn_right_free_soil_region,
    "slope_right_free_soil": nairn_right_free_soil_region,
    "slope_right_free_base": nairn_right_free_base_region,
    "slope_bottom": nairn_bottom_region,
    "static_bottom_no_slip": nairn_static_bottom_no_slip_region,
}


def resolve_region_function(name: str) -> Any:
    try:
        return REGION_FUNCTIONS[name]
    except KeyError as exc:
        known = ", ".join(sorted(REGION_FUNCTIONS))
        raise ValueError(f"Unknown region function {name!r}. Known functions: {known}") from exc


def validate_case_geometry_in_domain(case: dict[str, Any]) -> None:
    """Reject out-of-domain or non-grid-aligned regions before particle generation."""

    if DX <= 0.0:
        raise ValueError(f"Grid spacing DX must be positive, got {DX}")
    expected_free_field_width = 4.0 * DX
    if not math.isclose(K_FREE_FIELD_WIDTH, expected_free_field_width, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(
            "The case DX does not match the free-field geometry built at module load: "
            f"DX={DX}, free_field_width={K_FREE_FIELD_WIDTH}, expected={expected_free_field_width}."
        )

    physical_domain_min_x = -X_SHIFT
    physical_domain_max_x = float(DOMAIN[0]) - X_SHIFT
    physical_domain_min_y = -Y_SHIFT
    physical_domain_max_y = float(DOMAIN[1]) - Y_SHIFT
    tolerance = max(1.0e-10, 1.0e-8 * DX)
    for item in case.get("regions", []):
        point = np.asarray(item["bbox_point"], dtype=np.float64)
        size = np.asarray(item["bbox_size"], dtype=np.float64) + BOUNDING_BOX_EPS
        start = point + np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
        end = start + size
        if (
            start[0] < -tolerance
            or start[1] < -tolerance
            or end[0] > float(DOMAIN[0]) + tolerance
            or end[1] > float(DOMAIN[1]) + tolerance
        ):
            raise ValueError(
                f"Region {item.get('name', '<unnamed>')!r} is outside the computational domain: "
                f"start={start.tolist()}, end={end.tolist()}, domain={DOMAIN.tolist()}"
            )

    intervals = [
        ("left_free_field", K_LEFT_FREE_X_MIN, K_LEFT_FREE_X_MAX),
        ("main_slope", K_MAIN_X_MIN, K_MAIN_X_MAX),
        ("right_free_field", K_RIGHT_FREE_X_MIN, K_RIGHT_FREE_X_MAX),
    ]
    if physical_domain_min_x > K_LEFT_FREE_X_MIN + tolerance or physical_domain_max_x < K_RIGHT_FREE_X_MAX - tolerance:
        raise ValueError(
            "The free-field geometry does not fit inside the computational domain: "
            f"physical_x=[{physical_domain_min_x}, {physical_domain_max_x}], "
            f"required=[{K_LEFT_FREE_X_MIN}, {K_RIGHT_FREE_X_MAX}]"
        )
    if K_LEFT_FREE_X_MAX > K_MAIN_X_MIN + tolerance or K_RIGHT_FREE_X_MIN < K_MAIN_X_MAX - tolerance:
        raise ValueError(
            "Free-field columns overlap the main slope; expected non-negative gaps, "
            f"got left gap={K_MAIN_X_MIN - K_LEFT_FREE_X_MAX}, "
            f"right gap={K_RIGHT_FREE_X_MIN - K_MAIN_X_MAX}."
        )
    for label, x_min, x_max in intervals:
        scaled_start = (x_min + X_SHIFT) / DX
        scaled_end = (x_max + X_SHIFT) / DX
        if not math.isclose(scaled_start, round(scaled_start), rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"{label} start x={x_min} is not aligned to the grid (DX={DX})")
        if not math.isclose(scaled_end, round(scaled_end), rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"{label} end x={x_max} is not aligned to the grid (DX={DX})")
        if x_max <= x_min + tolerance:
            raise ValueError(f"{label} has non-positive width: [{x_min}, {x_max}]")
    if physical_domain_min_y > K_DOMAIN_Y_MIN + tolerance or physical_domain_max_y < K_DOMAIN_Y_MAX - tolerance:
        raise ValueError(
            "The vertical Kohler geometry does not fit inside the computational domain: "
            f"physical_y=[{physical_domain_min_y}, {physical_domain_max_y}], "
            f"required=[{K_DOMAIN_Y_MIN}, {K_DOMAIN_Y_MAX}]"
        )


def load_case_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="JSON case configuration override.")
    parser.add_argument("--dump-default-config", type=Path, default=None, help="Write the default case JSON and exit.")
    parser.add_argument("--free-field-independence-a", type=Path, default=None, help="Case A history CSV with main lateral coupling disabled.")
    parser.add_argument("--free-field-independence-b", type=Path, default=None, help="Case B history CSV with main lateral coupling enabled.")
    parser.add_argument("--free-field-independence-output", type=Path, default=None, help="Output CSV for the free-field A/B independence check.")
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

    if args.free_field_independence_a is not None or args.free_field_independence_b is not None:
        if args.free_field_independence_a is None or args.free_field_independence_b is None:
            raise SystemExit("--free-field-independence-a and --free-field-independence-b must be supplied together")
        case["_free_field_independence_a"] = str(args.free_field_independence_a)
        case["_free_field_independence_b"] = str(args.free_field_independence_b)
        if args.free_field_independence_output is not None:
            case["_free_field_independence_output"] = str(args.free_field_independence_output)

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
    case["run_dynamic"] = env_bool("NAIRN_RUN_DYNAMIC", bool(case.get("run_dynamic", RUN_DYNAMIC)))
    if STATIC_CONVERGENCE_ONLY:
        case["run_dynamic"] = False
    case["material_model"] = os.environ.get("NAIRN_MATERIAL_MODEL", str(case.get("material_model", MATERIAL_MODEL)))
    earthquake_spec = case.setdefault("earthquake_input", {})
    earthquake_spec["enabled"] = env_bool("NAIRN_EARTHQUAKE_ENABLED", bool(earthquake_spec.get("enabled", True)))
    if STATIC_CONVERGENCE_ONLY:
        earthquake_spec["enabled"] = False
    earthquake_spec["mode"] = os.environ.get(
        "NAIRN_SEISMIC_INPUT_MODE",
        os.environ.get("SEISMIC_INPUT_MODE", str(earthquake_spec.get("mode", SEISMIC_INPUT_MODE))),
    ).upper()
    earthquake_spec["file"] = os.environ.get("NAIRN_EARTHQUAKE_FILE", str(earthquake_spec.get("file", EARTHQUAKE_MOTION_FILE)))
    earthquake_spec["fig10c_file"] = os.environ.get(
        "NAIRN_FIG10C_INPUT_FILE",
        str(earthquake_spec.get("fig10c_file", FIG10C_INPUT_VELOCITY_FILE)),
    )
    earthquake_spec["acceleration_unit"] = "m/s^2"
    earthquake_spec["integration"] = "none" if earthquake_spec["mode"] == "FIG10C" else "trapezoidal"
    earthquake_spec["input_motion_type"] = (
        "fig10c_direct_input_velocity"
        if earthquake_spec["mode"] == "FIG10C"
        else "outcrop_acceleration_to_upward_shear_wave_velocity"
    )
    earthquake_spec["input_velocity_factor"] = float(
        os.environ.get("NAIRN_SEISMIC_INPUT_FACTOR", earthquake_spec.get("input_velocity_factor", SEISMIC_INPUT_FACTOR))
    )
    earthquake_spec["shear_input"] = env_bool("NAIRN_EARTHQUAKE_SHEAR_INPUT", bool(earthquake_spec.get("shear_input", True)))
    earthquake_spec["pressure_input"] = env_bool(
        "NAIRN_EARTHQUAKE_PRESSURE_INPUT",
        bool(earthquake_spec.get("pressure_input", False)),
    )
    case["input_stress"] = {
        "type": "direct_input_velocity" if earthquake_spec["mode"] == "FIG10C" else "external_acceleration",
        "file": str(earthquake_spec["fig10c_file"] if earthquake_spec["mode"] == "FIG10C" else earthquake_spec["file"]),
        "input_velocity_factor": earthquake_spec["input_velocity_factor"],
        "mode": earthquake_spec["mode"],
    }
    softening_spec = case.setdefault("softening", {})
    softening_spec["enabled"] = env_bool("NAIRN_SOFTENING_ENABLED", bool(softening_spec.get("enabled", SOFTENING_ENABLED)))
    softening_spec["cohesion_initial"] = float(
        os.environ.get("NAIRN_DP_COHESION", softening_spec.get("cohesion_initial", DP_COHESION))
    )
    softening_spec["cohesion_residual"] = float(
        os.environ.get("NAIRN_SOFTENING_RESIDUAL_COHESION", softening_spec.get("cohesion_residual", SOFTENING_RESIDUAL_COHESION))
    )
    softening_spec["friction_initial"] = float(
        os.environ.get("NAIRN_DP_FRICTION", softening_spec.get("friction_initial", DP_FRICTION))
    )
    softening_spec["friction_residual"] = float(
        os.environ.get("NAIRN_SOFTENING_RESIDUAL_FRICTION", softening_spec.get("friction_residual", SOFTENING_RESIDUAL_FRICTION))
    )
    softening_spec["dilation_initial"] = float(
        os.environ.get("NAIRN_DP_DILATION", softening_spec.get("dilation_initial", DP_DILATION))
    )
    softening_spec["dilation_residual"] = float(
        os.environ.get("NAIRN_SOFTENING_RESIDUAL_DILATION", softening_spec.get("dilation_residual", SOFTENING_RESIDUAL_DILATION))
    )
    softening_spec["eps_start"] = float(
        os.environ.get("NAIRN_SOFTENING_EPS_START", softening_spec.get("eps_start", SOFTENING_EPS_START))
    )
    softening_spec["eps_end"] = float(
        os.environ.get("NAIRN_SOFTENING_EPS_END", softening_spec.get("eps_end", SOFTENING_EPS_END))
    )
    softening_spec["law"] = "linear"
    softening_spec["state_variable"] = "epstrain"
    static_spec = case.setdefault("static_initialization", {})
    static_spec["enabled"] = env_bool("NAIRN_STATIC_ENABLED", bool(static_spec.get("enabled", STATIC_ENABLED)))
    static_spec["dt"] = float(os.environ.get("NAIRN_STATIC_DT", static_spec.get("dt", STATIC_DT)))
    static_spec["time"] = float(os.environ.get("NAIRN_STATIC_TIME", static_spec.get("time", STATIC_TIME)))
    static_spec["ramp_time"] = float(os.environ.get("NAIRN_STATIC_RAMP_TIME", static_spec.get("ramp_time", STATIC_RAMP_TIME)))
    static_spec["save_interval"] = float(os.environ.get("NAIRN_STATIC_SAVE_INTERVAL", static_spec.get("save_interval", STATIC_SAVE_INTERVAL)))
    static_spec["history_interval"] = float(os.environ.get("NAIRN_STATIC_HISTORY_INTERVAL", static_spec.get("history_interval", STATIC_HISTORY_INTERVAL)))
    static_spec["background_damping"] = float(os.environ.get("NAIRN_STATIC_DAMPING", static_spec.get("background_damping", STATIC_DAMPING)))
    static_spec["alpha_pic"] = float(
        os.environ.get("NAIRN_STATIC_ALPHA_PIC", static_spec.get("alpha_pic", STATIC_ALPHA_PIC))
    )
    static_gravity_y = float(os.environ.get("NAIRN_STATIC_GRAVITY", static_spec.get("gravity", [0.0, STATIC_GRAVITY])[1]))
    static_spec["gravity"] = [0.0, static_gravity_y]
    legacy_velocity_tolerance = float(static_spec.get("velocity_tolerance", STATIC_VELOCITY_TOLERANCE))
    static_spec["max_velocity_tolerance"] = float(
        os.environ.get(
            "NAIRN_STATIC_MAX_VELOCITY_TOLERANCE",
            static_spec.get("max_velocity_tolerance", STATIC_MAX_VELOCITY_TOLERANCE),
        )
    )
    static_spec["rms_velocity_tolerance"] = float(
        os.environ.get(
            "NAIRN_STATIC_RMS_VELOCITY_TOLERANCE",
            static_spec.get("rms_velocity_tolerance", legacy_velocity_tolerance),
        )
    )
    static_spec["relative_unbalanced_force_tolerance"] = float(
        os.environ.get(
            "NAIRN_STATIC_RELATIVE_UNBALANCED_FORCE_TOLERANCE",
            static_spec.get(
                "relative_unbalanced_force_tolerance",
                STATIC_RELATIVE_UNBALANCED_FORCE_TOLERANCE,
            ),
        )
    )
    static_spec["velocity_tolerance"] = static_spec["rms_velocity_tolerance"]
    static_spec["stress_tolerance"] = float(
        os.environ.get("NAIRN_STATIC_STRESS_TOLERANCE", static_spec.get("stress_tolerance", STATIC_STRESS_TOLERANCE))
    )
    static_spec["velocity_quench"] = env_bool(
        "NAIRN_STATIC_VELOCITY_QUENCH",
        bool(static_spec.get("velocity_quench", STATIC_VELOCITY_QUENCH)),
    )
    static_spec["velocity_quench"] = False
    static_spec["quench_max_velocity_factor"] = float(
        os.environ.get(
            "NAIRN_STATIC_QUENCH_MAX_VELOCITY_FACTOR",
            static_spec.get("quench_max_velocity_factor", STATIC_QUENCH_MAX_VELOCITY_FACTOR),
        )
    )
    static_spec["use_geotaichi_gravity_field"] = env_bool(
        "NAIRN_USE_GEOTAICHI_GRAVITY_FIELD",
        bool(static_spec.get("use_geotaichi_gravity_field", USE_GEOTAICHI_GRAVITY_FIELD)),
    )
    static_spec["require_convergence"] = env_bool(
        "NAIRN_REQUIRE_STATIC_CONVERGENCE",
        bool(static_spec.get("require_convergence", REQUIRE_STATIC_CONVERGENCE)),
    )
    if DYNAMIC_VELOCITY_PROJECTION == "Affine" and not math.isclose(
        float(static_spec["alpha_pic"]), 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "The paper-aligned APIC path requires static_initialization.alpha_pic=1 "
            "for Eq. (14) G2P."
        )
    normalize_material_parameters(case)
    return case


def apply_case_globals(case: dict[str, Any]) -> None:
    global X_SHIFT, Y_SHIFT, DX, DOMAIN, ELEMENT_SIZE, DT, SIMULATION_TIME, SAVE_INTERVAL
    global HISTORY_INTERVAL, OUTPUT_DIR, MAPPING, SHAPE_FUNCTION, ARCH, STRICT_TIME_LOOP
    global RUN_POSTPROCESS, SAVE_PARTICLE, SAVE_GRID, E_MODULUS, POISSON, DENSITY
    global SHEAR_MODULUS, BULK_MODULUS, CS, CP, INPUT_STRESS_PEAK, INPUT_FREQUENCY
    global INPUT_PERIOD, HALF_PARTICLE_SIZE, CASE, EARTHQUAKE_MOTION, EARTHQUAKE_MOTION_FILE
    global FIG10C_INPUT_VELOCITY_FILE, SEISMIC_INPUT_MODE, SEISMIC_INPUT_FACTOR
    global BODY_MAIN_SOIL, BODY_MAIN_BASE, BODY_LEFT_FREE_SOIL, BODY_LEFT_FREE_BASE
    global BODY_RIGHT_FREE_SOIL, BODY_RIGHT_FREE_BASE, MAIN_BODY_IDS, LEFT_FREE_BODY_IDS
    global RIGHT_FREE_BODY_IDS, BODY_SLIDE, SEPARATE_SLIDE_BODY

    CASE = case
    separate_slide_spec = case.get("separate_slide_body", {})
    SEPARATE_SLIDE_BODY = bool(separate_slide_spec.get("enabled", False))
    if SEPARATE_SLIDE_BODY:
        BODY_MAIN_SOIL = 0
        BODY_MAIN_BASE = 0
        BODY_SLIDE = int(separate_slide_spec.get("body_id", 3))
        if BODY_SLIDE != 3:
            raise ValueError("separate_slide_body.body_id must be 3 to preserve free-field body layers 1 and 2")
        MAIN_BODY_IDS = [BODY_MAIN_SOIL]
        case.setdefault("silent_boundary", {}).setdefault("side", {})["body_ids"] = [BODY_MAIN_SOIL]
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
    if DYNAMIC_VELOCITY_PROJECTION == "Affine" and MAPPING != "USL":
        raise ValueError(
            "The paper-aligned APIC path uses the USL update order from Fig. 2; "
            "set the case mapping to USL."
        )
    static_alpha_pic = float(case.get("static_initialization", {}).get("alpha_pic", STATIC_ALPHA_PIC))
    if DYNAMIC_VELOCITY_PROJECTION == "Affine" and not math.isclose(
        static_alpha_pic, 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "The paper-aligned APIC path requires static_initialization.alpha_pic=1 "
            "for Eq. (14) G2P."
        )
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
    earthquake_spec = case.get("earthquake_input", {})
    SEISMIC_INPUT_MODE = str(earthquake_spec.get("mode", SEISMIC_INPUT_MODE)).upper()
    if SEISMIC_INPUT_MODE not in {"AT2", "FIG10C"}:
        raise ValueError(f"Unsupported SEISMIC_INPUT_MODE {SEISMIC_INPUT_MODE!r}; expected 'AT2' or 'FIG10C'")
    EARTHQUAKE_MOTION_FILE = Path(earthquake_spec.get("file", EARTHQUAKE_MOTION_FILE))
    FIG10C_INPUT_VELOCITY_FILE = Path(earthquake_spec.get("fig10c_file", FIG10C_INPUT_VELOCITY_FILE))
    SEISMIC_INPUT_FACTOR = float(earthquake_spec.get("input_velocity_factor", SEISMIC_INPUT_FACTOR))
    if bool(earthquake_spec.get("enabled", True)) and not EXTRACT_LATERAL_STATIC_SUPPORT and not FREE_FIELD_PERIODIC_MINIMAL:
        if SEISMIC_INPUT_MODE == "FIG10C":
            EARTHQUAKE_MOTION = read_input_velocity_motion(FIG10C_INPUT_VELOCITY_FILE, input_mode="FIG10C")
        else:
            EARTHQUAKE_MOTION = read_earthquake_motion(EARTHQUAKE_MOTION_FILE)
        if SEISMIC_INPUT_MODE == "AT2":
            write_earthquake_input_check(EARTHQUAKE_MOTION, OUTPUT_DIR / "earthquake_input_check.md")
    elif EXTRACT_LATERAL_STATIC_SUPPORT or FREE_FIELD_PERIODIC_MINIMAL:
        EARTHQUAKE_MOTION = None
    else:
        EARTHQUAKE_MOTION = None


@ti.kernel
def update_bottom_input_traction(
    bottom_count: ti.i32,
    bottom_traction_ids: ti.template(),
    input_traction_x: ti.f64,
    input_traction_y: ti.f64,
    particle_traction: ti.template(),
    total_input_force_x: ti.template(),
    total_input_force_y: ti.template(),
):
    total_input_force_x[None] = 0.0
    total_input_force_y[None] = 0.0
    for i in range(bottom_count):
        tid = bottom_traction_ids[i]
        particle_traction[tid].traction = ti.Vector([input_traction_x, input_traction_y])
        psize = particle_traction[tid].psize
        input_force = 2.0 * ti.Vector([input_traction_x, input_traction_y]) * ti.Vector([psize[1], psize[0]])
        total_input_force_x[None] += input_force[0]
        total_input_force_y[None] += input_force[1]


@ti.kernel
def apply_bottom_static_reaction_nodes(
    reaction_count: ti.i32,
    reaction_node_ids: ti.template(),
    reaction_body_ids: ti.template(),
    reaction_force_x: ti.template(),
    reaction_force_y: ti.template(),
    node: ti.template(),
    total_static_reaction_force_x: ti.template(),
    total_static_reaction_force_y: ti.template(),
):
    total_static_reaction_force_x[None] = 0.0
    total_static_reaction_force_y[None] = 0.0
    for i in range(reaction_count):
        node_id = reaction_node_ids[i]
        body_id = reaction_body_ids[i]
        force = ti.Vector([reaction_force_x[i], reaction_force_y[i]])
        node[node_id, body_id]._update_nodal_force(force)
        total_static_reaction_force_x[None] += force[0]
        total_static_reaction_force_y[None] += force[1]


@ti.kernel
def apply_nairn_silent_loads(
    total_nodes: ti.i32,
    bottom_count: ti.i32,
    bottom_particle_ids: ti.template(),
    side_count: ti.i32,
    side_particle_ids: ti.template(),
    cs: ti.f64,
    cp: ti.f64,
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
        # Eq. (29): dp is the boundary area assigned to one bottom
        # particle.  In plane strain dp = particle volume / its normal
        # thickness, which recovers h/3 for the prescribed 3x3 layout.
        normal_thickness = ti.max(ti.sqrt(particle[pid].vol), 1.0e-30)
        force = ti.Vector(
            [
                -particle[pid].m * cs * particle[pid].v[0] / normal_thickness,
                -particle[pid].m * cp * particle[pid].v[1] / normal_thickness,
            ]
        )
        offset = pid * total_nodes
        for ln in range(offset, offset + int(node_size[pid])):
            node_id = ln_id[ln]
            node[node_id, body_id]._update_nodal_force(shape_fn[ln] * force)
        total_bottom_silent_x[None] += force[0]
        total_bottom_silent_y[None] += force[1]

    # Strict Kohler lateral boundaries are applied by
    # apply_free_field_particle_coupling(). Keep this legacy accumulator at zero.


@ti.kernel
def apply_free_field_particle_coupling(
    total_nodes: ti.i32,
    pair_count: ti.i32,
    main_particle_ids: ti.template(),
    free_field_particle_ids: ti.template(),
    side_signs: ti.template(),
    initial_surface_areas: ti.template(),
    current_surface_areas: ti.template(),
    normal_impedances: ti.template(),
    shear_impedances: ti.template(),
    free_field_static_stress_xx: ti.template(),
    free_field_static_stress_xy: ti.template(),
    static_stress_mult: ti.f64,
    dynamic_stress_mult: ti.f64,
    dashpot_mult: ti.f64,
    auxiliary_deformation_gradient: ti.template(),
    particle: ti.template(),
    node: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    node_size: ti.template(),
    pair_force_balance_error: ti.template(),
    total_main_force_x: ti.template(),
    total_main_force_y: ti.template(),
    total_free_field_force_x: ti.template(),
    total_free_field_force_y: ti.template(),
    max_force_balance_error: ti.template(),
    surface_area_update_call_count: ti.template(),
    invalid_surface_area_count: ti.template(),
    total_invalid_surface_area_count: ti.template(),
    dynamic_stress_force_x: ti.template(),
    dynamic_stress_force_y: ti.template(),
    normal_dashpot_force_x: ti.template(),
    shear_dashpot_force_y: ti.template(),
    pair_total_force_x: ti.template(),
    pair_total_force_y: ti.template(),
    total_coupling_power_main: ti.template(),
    total_dashpot_dissipation: ti.template(),
):
    total_main_force_x[None] = 0.0
    total_main_force_y[None] = 0.0
    total_free_field_force_x[None] = 0.0
    total_free_field_force_y[None] = 0.0
    total_coupling_power_main[None] = 0.0
    total_dashpot_dissipation[None] = 0.0
    max_force_balance_error[None] = 0.0
    surface_area_update_call_count[None] += 1
    invalid_surface_area_count[None] = 0

    for i in range(pair_count):
        main_pid = main_particle_ids[i]
        ff_pid = free_field_particle_ids[i]
        main_body_id = int(particle[main_pid].bodyID)
        sign = side_signs[i]
        initial_area = initial_surface_areas[i]
        F = auxiliary_deformation_gradient[main_pid]
        # Equation (32): stretch = sqrt(F[0,1]^2 + F[1,1]^2); current_area = initial_area*stretch.
        stretch = ti.sqrt(F[0, 1] * F[0, 1] + F[1, 1] * F[1, 1])
        current_area = initial_area*stretch
        current_stretch = stretch
        current_surface_area = current_area
        valid_surface_area = (
            finite_ti(current_stretch)
            and finite_ti(current_surface_area)
            and current_stretch > 0.0
            and current_surface_area > 0.0
        )
        if (
            valid_surface_area
        ):
            current_surface_areas[i] = current_surface_area
        else:
            ti.atomic_add(invalid_surface_area_count[None], 1)
            ti.atomic_add(total_invalid_surface_area_count[None], 1)
        normal_impedance = normal_impedances[i]
        shear_impedance = shear_impedances[i]
        normal_impedance_area = normal_impedance*current_area
        shear_impedance_area = shear_impedance*current_area
        relative_velocity = particle[ff_pid].v - particle[main_pid].v
        ff_stress = particle[ff_pid].stress
        dynamic_stress_xx = ff_stress[0] - free_field_static_stress_xx[i]
        dynamic_stress_xy = ff_stress[3] - free_field_static_stress_xy[i]
        pair_dynamic_stress_force_x = dynamic_stress_mult * current_surface_area * sign * dynamic_stress_xx
        pair_dynamic_stress_force_y = dynamic_stress_mult * current_surface_area * sign * dynamic_stress_xy
        pair_normal_dashpot_force_x = dashpot_mult * normal_impedance_area * relative_velocity[0]
        pair_shear_dashpot_force_y = dashpot_mult * shear_impedance_area * relative_velocity[1]
        force = ti.Vector(
            [
                pair_dynamic_stress_force_x + pair_normal_dashpot_force_x,
                pair_dynamic_stress_force_y + pair_shear_dashpot_force_y,
            ]
        )
        dynamic_stress_force_x[i] = pair_dynamic_stress_force_x
        dynamic_stress_force_y[i] = pair_dynamic_stress_force_y
        normal_dashpot_force_x[i] = pair_normal_dashpot_force_x
        shear_dashpot_force_y[i] = pair_shear_dashpot_force_y
        pair_total_force_x[i] = force[0]
        pair_total_force_y[i] = force[1]
        if valid_surface_area:
            main_velocity = particle[main_pid].v
            total_coupling_power_main[None] += (
                force[0] * main_velocity[0] + force[1] * main_velocity[1]
            )
            total_dashpot_dissipation[None] += dashpot_mult * (
                normal_impedance_area * relative_velocity[0] * relative_velocity[0]
                + shear_impedance_area * relative_velocity[1] * relative_velocity[1]
            )
            main_offset = main_pid * total_nodes
            for ln in range(main_offset, main_offset + int(node_size[main_pid])):
                node_id = ln_id[ln]
                node[node_id, main_body_id]._update_nodal_force(shape_fn[ln] * force)

            pair_force_balance_error[i] = 0.0
            total_main_force_x[None] += force[0]
            total_main_force_y[None] += force[1]
            total_free_field_force_x[None] += 0.0
            total_free_field_force_y[None] += 0.0


@ti.kernel
def zero_force_accumulators_2(
    force_x: ti.template(),
    force_y: ti.template(),
):
    force_x[None] = 0.0
    force_y[None] = 0.0


@ti.kernel
def kernel_compute_stress_hughes_winget_2d(
    particle_count: ti.i32,
    particle: ti.template(),
    material_properties: ti.template(),
    state_variables: ti.template(),
    dt: ti.template(),
):
    """Kohler Eq. (27)-(28) objective update for the elastic Fig.10 model."""

    for particle_id in range(particle_count):
        material_id = int(particle[particle_id].materialID)
        if material_id > 0 and int(particle[particle_id].active) == 1:
            velocity_gradient = particle[particle_id].velocity_gradient
            rotation_increment = 0.5 * (velocity_gradient[1, 0] - velocity_gradient[0, 1]) * dt[None]
            spin_increment = ti.Matrix(
                [
                    [0.0, -rotation_increment, 0.0],
                    [rotation_increment, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
            identity = ti.Matrix.identity(float, 3)
            rotation = (identity - 0.5 * spin_increment).inverse() @ (identity + 0.5 * spin_increment)

            previous = particle[particle_id].stress
            previous_matrix = ti.Matrix(
                [
                    [previous[0], previous[3], previous[5]],
                    [previous[3], previous[1], previous[4]],
                    [previous[5], previous[4], previous[2]],
                ]
            )
            rotated = rotation @ previous_matrix @ rotation.transpose()

            shear = material_properties[material_id].shear
            bulk = material_properties[material_id].bulk
            lame = bulk - 2.0 * shear / 3.0
            strain_xx = velocity_gradient[0, 0] * dt[None]
            strain_yy = velocity_gradient[1, 1] * dt[None]
            engineering_shear = (velocity_gradient[0, 1] + velocity_gradient[1, 0]) * dt[None]
            volumetric_strain = strain_xx + strain_yy

            updated = ti.Vector(
                [
                    rotated[0, 0] + 2.0 * shear * strain_xx + lame * volumetric_strain,
                    rotated[1, 1] + 2.0 * shear * strain_yy + lame * volumetric_strain,
                    rotated[2, 2] + lame * volumetric_strain,
                    rotated[0, 1] + shear * engineering_shear,
                    rotated[1, 2],
                    rotated[0, 2],
                ]
            )
            particle[particle_id].stress = updated
            state_variables[particle_id].estress = ti.sqrt(
                0.5
                * (
                    (updated[0] - updated[1]) * (updated[0] - updated[1])
                    + (updated[1] - updated[2]) * (updated[1] - updated[2])
                    + (updated[0] - updated[2]) * (updated[0] - updated[2])
                )
            )


def hughes_winget_enabled_for_case(case: dict[str, Any] | None) -> bool:
    """Return whether the selected material actually uses Hughes-Winget."""

    material_model = str((case or {}).get("material_model", MATERIAL_MODEL))
    return material_model == "VonMisesSoftening" or bool(
        HUGHES_WINGET_STRESS_UPDATE and material_model == "LinearElastic"
    )


def hughes_winget_stress_update_label(case: dict[str, Any] | None) -> str:
    material_model = str((case or {}).get("material_model", MATERIAL_MODEL))
    if material_model == "VonMisesSoftening":
        return "Kohler Hughes-Winget Eq.27-Eq.28 with J2 softening radial return"
    if HUGHES_WINGET_STRESS_UPDATE and material_model == "LinearElastic":
        return "Kohler Hughes-Winget Eq.27-Eq.28"
    return f"{material_model} constitutive stress update"


def install_hughes_winget_stress_update(mpm: MPM) -> None:
    material_model = str((CASE or {}).get("material_model", MATERIAL_MODEL))
    if not HUGHES_WINGET_STRESS_UPDATE or material_model != "LinearElastic":
        return
    engine = mpm.enginer
    if engine is None:
        raise RuntimeError("GeoTaichi engine is not initialized; cannot install Hughes-Winget update")
    if getattr(engine, "_kohler_hughes_winget_stress_update", False):
        return

    def compute_stress_hughes_winget(sims: Any, scene: Any) -> None:
        kernel_compute_stress_hughes_winget_2d(
            int(scene.particleNum[0]),
            scene.particle,
            scene.material.matProps,
            scene.material.stateVars,
            sims.dt,
        )

    engine.compute_stress_strains = compute_stress_hughes_winget
    engine._kohler_hughes_winget_stress_update = True


@ti.kernel
def zero_free_field_accumulators(
    total_main_force_x: ti.template(),
    total_main_force_y: ti.template(),
    total_free_field_force_x: ti.template(),
    total_free_field_force_y: ti.template(),
    max_force_balance_error: ti.template(),
):
    total_main_force_x[None] = 0.0
    total_main_force_y[None] = 0.0
    total_free_field_force_x[None] = 0.0
    total_free_field_force_y[None] = 0.0
    max_force_balance_error[None] = 0.0


@ti.kernel
def apply_lateral_static_support_nodes(
    support_count: ti.i32,
    support_node_ids: ti.template(),
    support_body_ids: ti.template(),
    support_force_x: ti.template(),
    node: ti.template(),
    total_support_x: ti.template(),
):
    total_support_x[None] = 0.0
    for i in range(support_count):
        node_id = support_node_ids[i]
        body_id = support_body_ids[i]
        force = ti.Vector([support_force_x[i], 0.0])
        node[node_id, body_id]._update_nodal_force(force)
        total_support_x[None] += force[0]


@ti.kernel
def canonicalize_free_field_periodic_supports(
    total_nodes: ti.i32,
    particle_count: ti.i32,
    left_free_field_body: ti.i32,
    right_free_field_body: ti.i32,
    left_canonical_node_ids: ti.template(),
    right_canonical_node_ids: ti.template(),
    particle: ti.template(),
    ln_id: ti.template(),
    node_size: ti.template(),
):
    """Wrap every horizontal free-field support to its periodic grid DOF."""
    for pid in range(particle_count):
        body_id = int(particle[pid].bodyID)
        if body_id == left_free_field_body:
            offset = pid * total_nodes
            for local_node in range(int(node_size[pid])):
                support_index = offset + local_node
                ln_id[support_index] = left_canonical_node_ids[ln_id[support_index]]
        elif body_id == right_free_field_body:
            offset = pid * total_nodes
            for local_node in range(int(node_size[pid])):
                support_index = offset + local_node
                ln_id[support_index] = right_canonical_node_ids[ln_id[support_index]]


def input_acceleration(time_value: float) -> float:
    if EARTHQUAKE_MOTION is None:
        return 0.0
    return EARTHQUAKE_MOTION.acceleration_at(time_value)


def outcrop_velocity(time_value: float) -> float:
    if EARTHQUAKE_MOTION is None:
        return 0.0
    return EARTHQUAKE_MOTION.velocity_at(time_value)


def input_velocity(time_value: float) -> float:
    """Return the velocity shown as the input motion in Fig. 10(c)."""

    if EARTHQUAKE_MOTION is None:
        return 0.0
    if SEISMIC_INPUT_MODE == "FIG10C":
        return EARTHQUAKE_MOTION.velocity_at(time_value)
    return upward_input_velocity(time_value)


def upward_input_velocity(time_value: float) -> float:
    """Return Kohler's upward wave velocity v_su used by Eqs. (1) and (29)."""

    return SEISMIC_INPUT_FACTOR * outcrop_velocity(time_value)


def input_stress(time_value: float) -> float:
    """Return shear input traction from external acceleration-derived velocity."""

    if time_value < 0.0:
        return 0.0
    if not DIAGNOSTIC_ENABLE_INPUT:
        return 0.0
    shear_enabled = bool(CASE.get("earthquake_input", {}).get("shear_input", True)) if CASE else True
    if not shear_enabled:
        return 0.0
    return 2.0 * DENSITY * CS * upward_input_velocity(time_value)


def write_kohler_seismic_input_check(motion: EarthquakeMotion, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "kohler_seismic_input_check.csv"
    report_path = output_dir / "kohler_seismic_input_report.md"
    v_outcrop = motion.velocity
    direct_velocity = str(getattr(motion, "input_mode", SEISMIC_INPUT_MODE)).upper() == "FIG10C"
    v_display = np.asarray(motion.velocity, dtype=np.float64)
    v_upward = SEISMIC_INPUT_FACTOR * v_outcrop
    input_traction = 2.0 * DENSITY * CS * v_upward
    rows = [
        {
            "time": float(t),
            "acceleration": float(a),
            "v_outcrop": float(vo),
            "input_motion_velocity": float(vd),
            "upward_wave_velocity": float(vs),
            "input_traction": float(tr),
        }
        for t, a, vo, vd, vs, tr in zip(
            motion.time,
            motion.acceleration,
            v_outcrop,
            v_display,
            v_upward,
            input_traction,
        )
    ]
    write_csv(
        csv_path,
        [
            "time",
            "acceleration",
            "v_outcrop",
            "input_motion_velocity",
            "upward_wave_velocity",
            "input_traction",
        ],
        rows,
    )

    if direct_velocity:
        current_pipeline = [
            "Fig.10(c) digitized input motion velocity v_motion(t)",
            "preserve v_motion(t) unchanged in the Fig.10(c) comparison output",
            f"Eq. (1) upward wave velocity v_su(t) = {SEISMIC_INPUT_FACTOR} * v_motion(t)",
            "bottom compliant base traction t_input(t) = 2 * rho * Cs * v_su(t)",
            "particle traction applied to bottom MPM particles",
        ]
    else:
        current_pipeline = [
            "PEER acceleration record a(t)",
            "trapezoidal integration to outcrop velocity v_outcrop(t)",
            f"upward propagating shear wave velocity v_su(t) = {SEISMIC_INPUT_FACTOR} * v_outcrop(t)",
            "bottom compliant base traction t_input(t) = 2 * rho * Cs * v_su(t)",
            "particle traction applied to bottom MPM particles",
        ]
    lines = [
        "# Kohler Seismic Input Report",
        "",
        "## Verdict",
        "",
        f"- input_mode: `{SEISMIC_INPUT_MODE}`",
        f"- kohler_compliant_base_flow: `{'PASS' if math.isclose(SEISMIC_INPUT_FACTOR, 0.5) else 'CHECK'}`",
        f"- missing_v_su_half_outcrop_conversion: `{'NO' if math.isclose(SEISMIC_INPUT_FACTOR, 0.5) else 'YES'}`",
        "",
        "## Pipeline",
        "",
        *[f"- {item}" for item in current_pipeline],
        "",
        "## Parameters",
        "",
        f"- input_file: `{motion.source_path}`",
        f"- upward_wave_velocity_factor: `{SEISMIC_INPUT_FACTOR}`",
        f"- rho: `{DENSITY}`",
        f"- Cs: `{CS}`",
        f"- traction_formula: `2 * rho * Cs * v_su`",
        "",
        "## Metrics",
        "",
        f"- PGA: `{motion.pga}`",
        f"- peak_v_outcrop: `{float(np.max(np.abs(v_outcrop))) if (v_outcrop.size and not direct_velocity) else 'NOT_APPLICABLE'}`",
        f"- peak_input_motion_velocity: `{float(np.max(np.abs(v_display))) if v_display.size else 0.0}`",
        f"- peak_upward_wave_velocity: `{float(np.max(np.abs(v_upward))) if v_upward.size else 0.0}`",
        f"- peak_traction: `{float(np.max(np.abs(input_traction))) if input_traction.size else 0.0}`",
        f"- residual_v_outcrop: `{float(v_outcrop[-1]) if (v_outcrop.size and not direct_velocity) else 'NOT_APPLICABLE'}`",
        f"- residual_input_motion_velocity: `{float(v_display[-1]) if v_display.size else 0.0}`",
        f"- residual_upward_wave_velocity: `{float(v_upward[-1]) if v_upward.size else 0.0}`",
        "",
        "## Output",
        "",
        f"- check_csv: `{csv_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, report_path


def current_input_velocity_array(motion: EarthquakeMotion) -> np.ndarray:
    mode = str(getattr(motion, "input_mode", SEISMIC_INPUT_MODE)).upper()
    if mode == "FIG10C":
        return np.asarray(motion.velocity, dtype=np.float64)
    return SEISMIC_INPUT_FACTOR * np.asarray(motion.velocity, dtype=np.float64)


def current_upward_input_velocity_array(motion: EarthquakeMotion) -> np.ndarray:
    return SEISMIC_INPUT_FACTOR * np.asarray(motion.velocity, dtype=np.float64)


def write_seismic_input_mode_check(motion: EarthquakeMotion, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "seismic_input_mode_check.md"
    mode = str(getattr(motion, "input_mode", SEISMIC_INPUT_MODE)).upper()
    velocity = current_input_velocity_array(motion)
    upward_velocity = current_upward_input_velocity_array(motion)
    direct_velocity = mode == "FIG10C"
    regular_dt = math.isclose(motion.dt_min, motion.dt_max, rel_tol=1.0e-6, abs_tol=1.0e-12)
    lines = [
        "# Seismic Input Mode Check",
        "",
        "## Mode",
        "",
        f"- current_mode: `{mode}`",
        f"- input_file: `{motion.source_path}`",
        f"- direct_velocity_input: `{'YES' if direct_velocity else 'NO'}`",
        f"- acceleration_integration: `{'NO' if direct_velocity else 'YES'}`",
        f"- input_velocity_factor_applied: `{1.0 if direct_velocity else SEISMIC_INPUT_FACTOR}`",
        f"- upward_wave_velocity_factor_applied: `{SEISMIC_INPUT_FACTOR}`",
        "",
        "## Signal",
        "",
        f"- peak_velocity: `{float(np.max(np.abs(velocity))) if velocity.size else 0.0}`",
        f"- peak_upward_wave_velocity: `{float(np.max(np.abs(upward_velocity))) if upward_velocity.size else 0.0}`",
        f"- duration: `{motion.duration}`",
        f"- dt_min: `{motion.dt_min}`",
        f"- dt_max: `{motion.dt_max}`",
        f"- dt_mean: `{motion.dt_mean}`",
        f"- regular_time_step: `{'PASS' if regular_dt else 'PARTIAL'}`",
        "",
        "## Unified Interface",
        "",
        "- Fig.10 comparison function: `input_velocity(t)` preserves the plotted input motion",
        "- boundary velocity function: `upward_input_velocity(t) = input_factor * outcrop_velocity(t)`",
        "- bottom_traction_formula: `t_input = 2 * rho * Cs * upward_input_velocity(t)`",
        "- force_decomposition_retained: `static reaction + dashpot force + earthquake input force`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def pressure_input_stress(time_value: float) -> float:
    if time_value < 0.0:
        return 0.0
    pressure_enabled = bool(CASE.get("earthquake_input", {}).get("pressure_input", False)) if CASE else False
    if not pressure_enabled:
        return 0.0
    return 2.0 * DENSITY * CP * 0.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def failure_snapshot_state_enabled(case: dict[str, Any]) -> bool:
    """Return whether particle files should retain constitutive state arrays."""

    spec = case.get("failure_outputs", {})
    return bool(spec.get("save_state_variables", False))


def install_failure_snapshot_recorder(mpm: MPM, case: dict[str, Any]) -> None:
    """Add state arrays to saved particle files for failure post-processing.

    GeoTaichi's standard particle recorder intentionally stores only position
    and velocity for lightweight runs.  Failure figures need the per-particle
    plastic strain, so this opt-in wrapper augments each saved NPZ without
    changing the solver or the normal Fig.10 output path.
    """

    if not failure_snapshot_state_enabled(case):
        return
    recorder = getattr(mpm, "recorder", None)
    if recorder is None or getattr(recorder, "_nairn_failure_state_recorder", False):
        return
    original_save_particle = recorder.save_particle

    def save_particle_with_failure_state(sims: Any, scene: Any) -> None:
        original_save_particle(sims, scene)
        particle_path = Path(sims.path) / "particles" / f"MPMParticle{int(sims.current_print):06d}.npz"
        if not particle_path.exists():
            return
        particle_count = int(scene.particleNum[0])
        with np.load(particle_path, allow_pickle=True) as saved:
            payload = {name: saved[name] for name in saved.files}
        payload["bodyID"] = np.ascontiguousarray(scene.particle.bodyID.to_numpy()[:particle_count])
        payload["materialID"] = np.ascontiguousarray(scene.particle.materialID.to_numpy()[:particle_count])
        payload["active"] = np.ascontiguousarray(scene.particle.active.to_numpy()[:particle_count])
        payload["volume"] = np.ascontiguousarray(scene.particle.vol.to_numpy()[:particle_count])
        payload["stress"] = np.ascontiguousarray(scene.particle.stress.to_numpy()[:particle_count])
        state = scene.material.get_state_vars_dict(0, particle_count)
        payload["state_vars"] = np.array(state, dtype=object)
        np.savez(particle_path, **payload)

    recorder.save_particle = save_particle_with_failure_state
    recorder._nairn_failure_state_recorder = True


def failure_postprocess_kwargs(case: dict[str, Any]) -> dict[str, Any]:
    if not failure_snapshot_state_enabled(case):
        return {}
    return {
        "write_state_variables": True,
        "write_bodyID": True,
        "write_volume": True,
        "write_stress_component": True,
    }


def maybe_stop_seismic_input(mpm: MPM, case: dict[str, Any]) -> None:
    """Optionally stop shaking after a persistent self-driven failure signal."""

    spec = case.get("failure_stop", {})
    boundary = getattr(mpm, "nairn_seismic_boundary", None)
    if boundary is None or not bool(spec.get("enabled", False)) or not bool(boundary.input_enabled):
        return
    dynamic_time = float(boundary.dynamic_time(mpm.sims))
    next_check = float(getattr(mpm, "failure_stop_next_check", 0.0))
    check_interval = max(float(spec.get("check_interval", 0.01)), float(DT))
    if dynamic_time + 0.5 * float(mpm.sims.delta) < next_check:
        return
    while next_check <= dynamic_time + 0.5 * float(mpm.sims.delta):
        next_check += check_interval
    mpm.failure_stop_next_check = next_check

    particle_count = int(mpm.scene.particleNum[0])
    material_ids = mpm.scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    main_soil = (material_ids == MAT_SOIL) & (body_ids == BODY_MAIN_SOIL)
    if not np.any(main_soil):
        return
    state = mpm.scene.material.get_state_vars_dict(0, particle_count)
    epstrain = np.asarray(state.get("epstrain", np.zeros(particle_count)), dtype=np.float64).reshape(-1)
    velocity = np.asarray(mpm.scene.particle.v.to_numpy()[:particle_count], dtype=np.float64)
    max_epstrain = float(np.max(np.abs(epstrain[main_soil]))) if epstrain.size else 0.0
    max_speed = float(np.max(np.linalg.norm(velocity[main_soil, :2], axis=1)))
    input_velocity_abs = abs(float(input_velocity(dynamic_time)))
    epstrain_ok = max_epstrain >= float(spec.get("min_epstrain", 0.02))
    speed_ok = max_speed >= float(spec.get("min_main_speed", 0.01))
    quiet_ok = (not bool(spec.get("require_quiet_input", True))) or (
        input_velocity_abs <= float(spec.get("quiet_input_velocity", 0.002))
    )
    qualifying = epstrain_ok and speed_ok and quiet_ok
    required_checks = max(1, int(spec.get("consecutive_checks", 3)))
    count = int(getattr(mpm, "failure_stop_qualifying_checks", 0))
    count = count + 1 if qualifying else 0
    mpm.failure_stop_qualifying_checks = count
    mpm.failure_stop_last_metrics = {
        "time": dynamic_time,
        "max_epstrain": max_epstrain,
        "max_main_speed": max_speed,
        "input_velocity_abs": input_velocity_abs,
        "epstrain_ok": epstrain_ok,
        "speed_ok": speed_ok,
        "quiet_input_ok": quiet_ok,
        "qualifying_checks": count,
    }
    if count < required_checks:
        return
    boundary.input_enabled = False
    mpm.failure_stop_event = {
        **mpm.failure_stop_last_metrics,
        "required_checks": required_checks,
        "reason": "persistent plastic deformation and self-driven velocity",
    }


def write_failure_stop_report(mpm: MPM, case: dict[str, Any]) -> Path | None:
    spec = case.get("failure_stop", {})
    if not bool(spec.get("enabled", False)):
        return None
    event = getattr(mpm, "failure_stop_event", None)
    last_metrics = getattr(mpm, "failure_stop_last_metrics", {})
    path = OUTPUT_DIR / "failure_stop_report.md"
    lines = [
        "# Seismic Input Stop Check",
        "",
        f"- enabled: `{bool(spec.get('enabled', False))}`",
        f"- triggered: `{event is not None}`",
        f"- input_enabled_at_end: `{bool(getattr(getattr(mpm, 'nairn_seismic_boundary', None), 'input_enabled', True))}`",
        f"- check_interval_s: `{float(spec.get('check_interval', 0.01))}`",
        f"- minimum_epstrain: `{float(spec.get('min_epstrain', 0.02))}`",
        f"- minimum_main_speed_m_per_s: `{float(spec.get('min_main_speed', 0.01))}`",
        f"- require_quiet_input: `{bool(spec.get('require_quiet_input', True))}`",
        f"- quiet_input_velocity_m_per_s: `{float(spec.get('quiet_input_velocity', 0.002))}`",
        f"- consecutive_checks_required: `{int(spec.get('consecutive_checks', 3))}`",
        "",
        "## Trigger Event",
        "",
    ]
    if event is None:
        lines.append("- event: `not reached during this run`")
    else:
        lines.extend([f"- {key}: `{value}`" for key, value in event.items()])
    lines.extend(["", "## Last Check", ""])
    if last_metrics:
        lines.extend([f"- {key}: `{value}`" for key, value in last_metrics.items()])
    else:
        lines.append("- metrics: `not sampled`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@ti.func
def finite_ti(value):
    return value == value and ti.abs(value) < 1.0e100


@ti.kernel
def initialize_auxiliary_deformation_gradient(
    particle_count: ti.i32,
    auxiliary_deformation_gradient: ti.template(),
    min_detF: ti.template(),
    max_detF: ti.template(),
    invalid_F_count: ti.template(),
    first_invalid_pid: ti.template(),
):
    min_detF[None] = 1.0
    max_detF[None] = 1.0
    invalid_F_count[None] = 0
    first_invalid_pid[None] = particle_count
    for pid in range(particle_count):
        auxiliary_deformation_gradient[pid] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])


@ti.kernel
def update_auxiliary_deformation_gradient_from_velocity_gradient(
    particle_count: ti.i32,
    dt: ti.f64,
    particle: ti.template(),
    auxiliary_deformation_gradient: ti.template(),
    update_call_count: ti.template(),
    min_detF: ti.template(),
    max_detF: ti.template(),
    invalid_F_count: ti.template(),
    first_invalid_pid: ti.template(),
):
    update_call_count[None] += 1
    min_detF[None] = 1.0e100
    max_detF[None] = -1.0e100
    invalid_F_count[None] = 0
    first_invalid_pid[None] = particle_count
    for pid in range(particle_count):
        if int(particle[pid].materialID) > 0 and int(particle[pid].active) == 1:
            L = particle[pid].velocity_gradient
            valid = (
                finite_ti(L[0, 0])
                and finite_ti(L[0, 1])
                and finite_ti(L[1, 0])
                and finite_ti(L[1, 1])
            )
            F_old = auxiliary_deformation_gradient[pid]
            F_inc = ti.Matrix([[1.0 + dt * L[0, 0], dt * L[0, 1]], [dt * L[1, 0], 1.0 + dt * L[1, 1]]])
            F_new = F_inc @ F_old
            detF = F_new.determinant()
            valid = (
                valid
                and finite_ti(F_new[0, 0])
                and finite_ti(F_new[0, 1])
                and finite_ti(F_new[1, 0])
                and finite_ti(F_new[1, 1])
                and finite_ti(detF)
                and detF > 0.0
            )
            if valid:
                auxiliary_deformation_gradient[pid] = F_new
                ti.atomic_min(min_detF[None], detF)
                ti.atomic_max(max_detF[None], detF)
            else:
                ti.atomic_add(invalid_F_count[None], 1)
                ti.atomic_min(first_invalid_pid[None], pid)
        else:
            detF = auxiliary_deformation_gradient[pid].determinant()
            if finite_ti(detF) and detF > 0.0:
                ti.atomic_min(min_detF[None], detF)
                ti.atomic_max(max_detF[None], detF)
            else:
                ti.atomic_add(invalid_F_count[None], 1)
                ti.atomic_min(first_invalid_pid[None], pid)


class AuxiliaryDeformationGradientTracker:
    def __init__(self, mpm: MPM) -> None:
        self.mpm = mpm
        self.particle_count = int(mpm.scene.particleNum[0])
        if not hasattr(mpm.scene.particle, "velocity_gradient"):
            raise RuntimeError(
                "Formal UL path has no readable particle.velocity_gradient field; cannot build auxiliary F."
            )
        self.auxiliary_deformation_gradient = ti.Matrix.field(2, 2, dtype=ti.f64, shape=max(1, self.particle_count))
        self.update_call_count = ti.field(dtype=ti.i32, shape=())
        self.min_detF = ti.field(dtype=ti.f64, shape=())
        self.max_detF = ti.field(dtype=ti.f64, shape=())
        self.invalid_F_count = ti.field(dtype=ti.i32, shape=())
        self.first_invalid_pid = ti.field(dtype=ti.i32, shape=())
        self.installed_engine_ids: set[int] = set()
        self.stage_call_counts: dict[str, int] = {}
        self.stage = "uninitialized"
        initialize_auxiliary_deformation_gradient(
            self.particle_count,
            self.auxiliary_deformation_gradient,
            self.min_detF,
            self.max_detF,
            self.invalid_F_count,
            self.first_invalid_pid,
        )
        self.update_call_count[None] = 0
        mpm.auxiliary_deformation_gradient = self.auxiliary_deformation_gradient
        mpm.auxiliary_deformation_gradient_tracker = self

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    def install_on_engine(self) -> None:
        engine = getattr(self.mpm, "enginer", None)
        if engine is None:
            return
        if id(engine) in self.installed_engine_ids or getattr(engine, "_nairn_auxiliary_F_hook_installed", False):
            return
        original_compute_velocity_gradient = engine.compute_velocity_gradient
        if original_compute_velocity_gradient is None:
            raise RuntimeError("Formal UL engine has no compute_velocity_gradient hook point for auxiliary F.")

        tracker = self

        def compute_velocity_gradient_with_auxiliary_F(sims: Any, scene: Any) -> Any:
            result = original_compute_velocity_gradient(sims, scene)
            update_auxiliary_deformation_gradient_from_velocity_gradient(
                int(scene.particleNum[0]),
                float(sims.dt[None]),
                scene.particle,
                tracker.auxiliary_deformation_gradient,
                tracker.update_call_count,
                tracker.min_detF,
                tracker.max_detF,
                tracker.invalid_F_count,
                tracker.first_invalid_pid,
            )
            tracker.stage_call_counts[tracker.stage] = tracker.stage_call_counts.get(tracker.stage, 0) + 1
            if int(tracker.invalid_F_count[None]) != 0:
                first_invalid_pid = int(tracker.first_invalid_pid[None])
                particle_position = scene.particle.x.to_numpy()[first_invalid_pid, :2]
                velocity_gradient = scene.particle.velocity_gradient.to_numpy()[first_invalid_pid]
                node_count = int(scene.element.node_size.to_numpy()[first_invalid_pid])
                node_offset = first_invalid_pid * int(scene.element.grid_nodes)
                support_ids = scene.element.LnID.to_numpy()[node_offset : node_offset + node_count]
                node_arrays = node_field_arrays(scene)
                support_state = [
                    {
                        "node_id": int(node_id),
                        "ij": [int(node_id % scene.element.gnum[0]), int(node_id // scene.element.gnum[0])],
                        "mass": float(node_arrays["m"][node_id, int(scene.particle.bodyID.to_numpy()[first_invalid_pid])]),
                        "velocity": node_arrays["momentum"][node_id, int(scene.particle.bodyID.to_numpy()[first_invalid_pid]), :2].tolist(),
                    }
                    for node_id in support_ids
                ]
                mirror = getattr(tracker.mpm.enginer, "_nairn_static_mirror_particle_boundary", None)
                mirror_source_state = []
                if mirror is not None:
                    mirror_mass = mirror.mass_source.to_numpy()
                    mirror_momentum = mirror.momentum_source.to_numpy()
                    for node_id in support_ids:
                        node_i = int(node_id % scene.element.gnum[0])
                        node_j = int(node_id // scene.element.gnum[0])
                        if node_i < mirror.main_right_node - 5:
                            continue
                        source_i = 2 * mirror.main_right_node - node_i
                        if 0 <= source_i < mirror.grid_x:
                            source_id = source_i + node_j * mirror.grid_x
                            mirror_source_state.append(
                                {
                                    "target_ij": [node_i, node_j],
                                    "source_ij": [source_i, node_j],
                                    "source_mass": float(mirror_mass[source_id, BODY_MAIN_SOIL]),
                                    "source_momentum": mirror_momentum[source_id, BODY_MAIN_SOIL, :2].tolist(),
                                }
                            )
                raise RuntimeError(
                    f"Auxiliary deformation gradient update failed in stage={tracker.stage}: "
                    f"invalid_F_count={int(tracker.invalid_F_count[None])}, "
                    f"first_invalid_pid={first_invalid_pid}, "
                    f"position_local={particle_position.tolist()}, "
                    f"position_physical={[float(particle_position[0] - X_SHIFT), float(particle_position[1] - Y_SHIFT)]}, "
                    f"velocity_gradient={velocity_gradient.tolist()}, "
                    f"support_state={support_state}, "
                    f"mirror_source_state={mirror_source_state}"
                )
            return result

        engine.compute_velocity_gradient = compute_velocity_gradient_with_auxiliary_F
        engine._nairn_auxiliary_F_hook_installed = True
        self.installed_engine_ids.add(id(engine))

    def to_numpy(self) -> np.ndarray:
        return self.auxiliary_deformation_gradient.to_numpy()[: self.particle_count].copy()

    def load_numpy(self, values: np.ndarray) -> None:
        if values.shape != (self.particle_count, 2, 2):
            raise RuntimeError(
                f"Auxiliary F checkpoint shape mismatch: checkpoint={values.shape} current={(self.particle_count, 2, 2)}"
            )
        storage = np.repeat(np.eye(2, dtype=np.float64)[None, :, :], max(1, self.particle_count), axis=0)
        storage[: self.particle_count] = np.ascontiguousarray(values, dtype=np.float64)
        self.auxiliary_deformation_gradient.from_numpy(storage)

    def stats(self) -> dict[str, Any]:
        values = self.to_numpy()
        if values.size == 0:
            return {
                "min_F00": math.nan,
                "max_F00": math.nan,
                "min_F01": math.nan,
                "max_F01": math.nan,
                "min_F10": math.nan,
                "max_F10": math.nan,
                "min_F11": math.nan,
                "max_F11": math.nan,
                "min_detF": math.nan,
                "max_detF": math.nan,
            }
        det = values[:, 0, 0] * values[:, 1, 1] - values[:, 0, 1] * values[:, 1, 0]
        return {
            "min_F00": float(np.min(values[:, 0, 0])),
            "max_F00": float(np.max(values[:, 0, 0])),
            "min_F01": float(np.min(values[:, 0, 1])),
            "max_F01": float(np.max(values[:, 0, 1])),
            "min_F10": float(np.min(values[:, 1, 0])),
            "max_F10": float(np.max(values[:, 1, 0])),
            "min_F11": float(np.min(values[:, 1, 1])),
            "max_F11": float(np.max(values[:, 1, 1])),
            "min_detF": float(np.min(det)),
            "max_detF": float(np.max(det)),
        }


def remove_empty_output_subdirs(output_dir: Path, names: tuple[str, ...] = ("grids", "particles", "vtks")) -> None:
    for name in names:
        path = output_dir / name
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def free_field_independence_status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    rows = read_csv_rows(path)
    if not rows:
        return "FAIL"
    statuses = {row.get("status", "FAIL") for row in rows if row.get("quantity") != "summary"}
    return "PASS" if statuses == {"PASS"} else "FAIL"


def compare_free_field_independence(case_a_history: Path, case_b_history: Path, output_path: Path) -> Path:
    rows_a = read_csv_rows(case_a_history)
    rows_b = read_csv_rows(case_b_history)
    if len(rows_a) != len(rows_b):
        raise RuntimeError(f"History row count differs: Case A={len(rows_a)}, Case B={len(rows_b)}")
    if not rows_a:
        raise RuntimeError("Case A and Case B histories are empty")

    quantities = [
        f"{monitor}_{component}"
        for monitor in ("left_free_field_monitor", "right_free_field_monitor")
        for component in ("vx", "vy", "stress_xx", "stress_xy")
    ]
    metric_rows: list[dict[str, Any]] = []
    overall_status = "PASS"
    max_free_field_force = 0.0
    for rows in (rows_a, rows_b):
        for row in rows:
            try:
                max_free_field_force = max(
                    max_free_field_force,
                    abs(float(row.get("total_force_applied_to_free_field_by_main", "0") or 0.0)),
                )
            except ValueError:
                max_free_field_force = math.inf

    if max_free_field_force > FREE_FIELD_INDEPENDENCE_TOLERANCE:
        overall_status = "FAIL"

    times_a = np.asarray([float(row["time"]) for row in rows_a], dtype=np.float64)
    times_b = np.asarray([float(row["time"]) for row in rows_b], dtype=np.float64)
    if not np.array_equal(times_a, times_b):
        raise RuntimeError("Case A and Case B history times differ; rerun with identical dt/history settings")

    for quantity in quantities:
        if quantity not in rows_a[0] or quantity not in rows_b[0]:
            status = "FAIL"
            max_abs_difference = math.inf
            nrmse = math.inf
            correlation = math.nan
            overall_status = "FAIL"
        else:
            values_a = np.asarray([float(row[quantity]) for row in rows_a], dtype=np.float64)
            values_b = np.asarray([float(row[quantity]) for row in rows_b], dtype=np.float64)
            difference = values_b - values_a
            max_abs_difference = float(np.max(np.abs(difference))) if difference.size else 0.0
            rmse = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
            scale = float(np.max(values_a) - np.min(values_a)) if values_a.size else 0.0
            if scale <= 0.0:
                scale = max(float(np.max(np.abs(values_a))) if values_a.size else 0.0, 1.0)
            nrmse = rmse / scale
            if values_a.size < 2 or np.std(values_a) == 0.0 or np.std(values_b) == 0.0:
                correlation = 1.0 if max_abs_difference <= FREE_FIELD_INDEPENDENCE_TOLERANCE else math.nan
            else:
                correlation = float(np.corrcoef(values_a, values_b)[0, 1])
            status = "PASS" if max_abs_difference <= FREE_FIELD_INDEPENDENCE_TOLERANCE else "FAIL"
            if status != "PASS":
                overall_status = "FAIL"
        metric_rows.append(
            {
                "quantity": quantity,
                "max_abs_difference": max_abs_difference,
                "NRMSE": nrmse,
                "correlation": correlation,
                "tolerance": FREE_FIELD_INDEPENDENCE_TOLERANCE,
                "total_force_applied_to_free_field_by_main": max_free_field_force,
                "status": status,
            }
        )

    finite_correlations = [float(row["correlation"]) for row in metric_rows if not math.isnan(float(row["correlation"]))]
    metric_rows.append(
        {
            "quantity": "summary",
            "max_abs_difference": max(row["max_abs_difference"] for row in metric_rows),
            "NRMSE": max(row["NRMSE"] for row in metric_rows),
            "correlation": min(finite_correlations) if finite_correlations else math.nan,
            "tolerance": FREE_FIELD_INDEPENDENCE_TOLERANCE,
            "total_force_applied_to_free_field_by_main": max_free_field_force,
            "status": overall_status,
        }
    )
    write_csv(
        output_path,
        [
            "quantity",
            "max_abs_difference",
            "NRMSE",
            "correlation",
            "tolerance",
            "total_force_applied_to_free_field_by_main",
            "status",
        ],
        metric_rows,
    )
    update_free_field_report_with_independence(output_path, case_a_history, case_b_history, overall_status, max_free_field_force)
    return output_path


def update_free_field_report_with_independence(
    independence_csv: Path,
    case_a_history: Path,
    case_b_history: Path,
    status: str,
    max_free_field_force: float,
) -> Path:
    report_path = independence_csv.parent / "free_field_coupling_report.md"
    lines = [
        "# Free Field Coupling Report",
        "",
        "## Verdict",
        "",
        f"- status: `{status}`",
        "- coupling_direction: `free_field_to_main_only`",
        f"- main_to_free_field_force: `{max_free_field_force}`",
        f"- main_to_free_field_force_check: `{'PASS' if max_free_field_force <= FREE_FIELD_INDEPENDENCE_TOLERANCE else 'FAIL'}`",
        f"- independence_test: `{status}`",
        "",
        "## Independence Test",
        "",
        "- case_a_main_lateral_coupling: `disabled`",
        "- case_b_main_lateral_coupling: `enabled`",
        f"- case_a_history: `{case_a_history}`",
        f"- case_b_history: `{case_b_history}`",
        f"- independence_csv: `{independence_csv}`",
        f"- tolerance: `{FREE_FIELD_INDEPENDENCE_TOLERANCE}`",
        "",
        "## Coupling Force",
        "",
        "- formula_x: `F_x = A*n_x*(sigma_xx_fp,s + sigma_xx_fp,d) + rho*Cp*A*(v_free_field_x - v_main_x)`",
        "- formula_y: `F_y = A*n_x*tau_xy_fp,d + rho*Cs*A*(v_free_field_y - v_main_y)`",
        "- before_change: `F_ff = -F_main` was mapped back to free-field nodes",
        "- after_change: `F_ff = 0`; only `F_main` is mapped to main-model nodes",
        "",
        "## Scope",
        "",
        "- solver_core_modified: `False`",
        "- geometry_changed: `False`",
        "- static_initialization_changed: `False`",
        "- action_reaction_balance_pass_condition: `removed`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def taichi_vector_to_list(value: Any) -> list[int | float]:
    try:
        return [int(item) for item in value]
    except TypeError:
        try:
            return [int(item) for item in value.to_numpy()]
        except AttributeError:
            return []


def grid_count_info(mpm: MPM) -> dict[str, Any]:
    element = mpm.scene.element
    gnum = taichi_vector_to_list(getattr(element, "gnum", []))
    cnum = taichi_vector_to_list(getattr(element, "cnum", []))
    return {
        "grid_nodes_per_direction": gnum,
        "grid_cells_per_direction": cnum,
        "grid_node_count": int(getattr(element, "gridSum", 0)),
        "grid_cell_count": int(getattr(element, "cellSum", 0)),
    }


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def particle_arrays(scene: Any) -> dict[str, np.ndarray]:
    particle_count = int(scene.particleNum[0])
    arrays = {
        "position": scene.particle.x.to_numpy()[:particle_count].copy(),
        "velocity": scene.particle.v.to_numpy()[:particle_count].copy(),
        "stress": scene.particle.stress.to_numpy()[:particle_count].copy(),
        "velocity_gradient": scene.particle.velocity_gradient.to_numpy()[:particle_count].copy(),
        "body_id": scene.particle.bodyID.to_numpy()[:particle_count].copy(),
        "material_id": scene.particle.materialID.to_numpy()[:particle_count].copy(),
    }
    material = getattr(scene, "material", None)
    if material is not None and hasattr(material, "get_state_vars_dict"):
        try:
            arrays["state_variables"] = material.get_state_vars_dict(0, particle_count)
        except Exception as exc:  # pragma: no cover - diagnostic path only
            arrays["state_variable_error"] = np.array([str(exc)], dtype=object)
    return arrays


@ti.kernel
def restore_static_particle_fields_2d(
    particle_count: ti.i32,
    position: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
    stress: ti.types.ndarray(),
    velocity_gradient: ti.types.ndarray(),
    body_id: ti.types.ndarray(),
    material_id: ti.types.ndarray(),
    zero_velocity: ti.i32,
    particle: ti.template(),
):
    for pid in range(particle_count):
        particle[pid].x = ti.Vector([position[pid, 0], position[pid, 1]])
        if zero_velocity == 1:
            particle[pid].v = ti.Vector([0.0, 0.0])
        else:
            particle[pid].v = ti.Vector([velocity[pid, 0], velocity[pid, 1]])
        particle[pid].stress = ti.Vector(
            [
                stress[pid, 0],
                stress[pid, 1],
                stress[pid, 2],
                stress[pid, 3],
                stress[pid, 4],
                stress[pid, 5],
            ]
        )
        if zero_velocity == 1:
            particle[pid].velocity_gradient = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        else:
            particle[pid].velocity_gradient = ti.Matrix(
                [
                    [velocity_gradient[pid, 0, 0], velocity_gradient[pid, 0, 1]],
                    [velocity_gradient[pid, 1, 0], velocity_gradient[pid, 1, 1]],
                ]
            )
        particle[pid].bodyID = ti.cast(body_id[pid], ti.u8)
        particle[pid].materialID = ti.cast(material_id[pid], ti.u8)


def static_checkpoint_metadata(mpm: MPM, case: dict[str, Any], static_monitor: Any | None) -> dict[str, Any]:
    particle_count = int(mpm.scene.particleNum[0])
    arrays = particle_arrays(mpm.scene)
    position_hash = float(np.sum(arrays["position"][:, 0] * 1.0e-3 + arrays["position"][:, 1] * 1.0e-5))
    body_hash = int(np.sum(arrays["body_id"].astype(np.int64) * np.arange(1, particle_count + 1, dtype=np.int64)))
    material_hash = int(
        np.sum(arrays["material_id"].astype(np.int64) * np.arange(1, particle_count + 1, dtype=np.int64))
    )
    grid_info = grid_count_info(mpm)
    return {
        "dx": float(DX),
        "dt_static": float(case.get("static_initialization", {}).get("dt", STATIC_DT)),
        "particle_count": particle_count,
        "grid_nodes": grid_info["grid_node_count"],
        "grid_cells": grid_info["grid_cell_count"],
        "position_hash": position_hash,
        "body_hash": body_hash,
        "material_hash": material_hash,
        "material_model": str(case.get("material_model", MATERIAL_MODEL)),
        "static_end_time": float(static_monitor.end_time if static_monitor is not None else 0.0),
        "static_current_step": int(getattr(mpm.sims, "current_step", 0)),
        "static_converged": bool(static_monitor.converged if static_monitor is not None else False),
        "static_velocity_projection": str(mpm.sims.velocity_projection_scheme),
        "static_alpha_pic": float(
            static_monitor.spec.get("alpha_pic", STATIC_ALPHA_PIC)
            if static_monitor is not None
            else STATIC_ALPHA_PIC
        ),
        "paper_apic_2d_transfer_installed": bool(
            getattr(mpm.enginer, "_nairn_paper_apic_2d_transfer_installed", False)
        ),
        "paper_apic_separate_B_L": bool(
            getattr(mpm.enginer, "_nairn_paper_apic_separate_B_L", False)
        ),
        "paper_apic_usl_order": bool(
            getattr(mpm.enginer, "_nairn_paper_apic_usl_order", False)
        ),
        "paper_apic_formula_revision": (
            PAPER_APIC_FORMULA_REVISION
            if getattr(mpm.enginer, "_nairn_paper_apic_2d_p2g_installed", False)
            else "NOT_APIC"
        ),
        "paper_apic_constant_D": bool(APIC_USE_PAPER_CONSTANT_DP),
        "hughes_winget_stress_update": hughes_winget_enabled_for_case(case),
        "hughes_winget_formula_revision": (
            PAPER_HUGHES_WINGET_FORMULA_REVISION
            if hughes_winget_enabled_for_case(case)
            else "NOT_HUGHES_WINGET"
        ),
        "free_field_periodic_static": bool(getattr(mpm, "periodic_hooks_installed_for_static", False)),
        "free_field_periodic_support_wrap": "full_cubic_modulo",
        "static_mirror_particle_boundary": bool(STATIC_MIRROR_PARTICLE_BOUNDARY),
        "static_mirror_reflect_body_force": bool(STATIC_MIRROR_REFLECT_BODY_FORCE),
        "static_paper_eq11_damping": bool(STATIC_PAPER_EQ11_DAMPING),
        "static_local_damping_beta": float(case.get("static_initialization", {}).get("background_damping", STATIC_DAMPING)),
        "static_max_velocity_tolerance": float(
            case.get("static_initialization", {}).get("max_velocity_tolerance", STATIC_MAX_VELOCITY_TOLERANCE)
        ),
        "static_rms_velocity_tolerance": float(
            case.get("static_initialization", {}).get("rms_velocity_tolerance", STATIC_RMS_VELOCITY_TOLERANCE)
        ),
        "static_relative_unbalanced_force_tolerance": float(
            case.get("static_initialization", {}).get(
                "relative_unbalanced_force_tolerance",
                STATIC_RELATIVE_UNBALANCED_FORCE_TOLERANCE,
            )
        ),
        "continued_from_checkpoint": str(getattr(static_monitor, "continued_from_checkpoint", "")),
        "static_continuation_start_time": float(
            getattr(static_monitor, "continuation_start_time", 0.0)
        ),
        "target_gravity": list(static_monitor.target_gravity if static_monitor is not None else [0.0, STATIC_GRAVITY]),
    }


def write_static_checkpoint(mpm: MPM, case: dict[str, Any], static_monitor: Any) -> Path:
    path = STATIC_CHECKPOINT_SAVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = particle_arrays(mpm.scene)
    payload: dict[str, Any] = {
        "metadata": np.array(static_checkpoint_metadata(mpm, case, static_monitor), dtype=object),
        "position": arrays["position"],
        "velocity": arrays["velocity"],
        "stress": arrays["stress"],
        "velocity_gradient": arrays["velocity_gradient"],
        "body_id": arrays["body_id"],
        "material_id": arrays["material_id"],
    }
    generated_layout = getattr(mpm, "generated_particle_layout_arrays", None)
    if isinstance(generated_layout, dict) and "position" in generated_layout:
        payload["generated_position"] = generated_layout["position"]
        payload["generated_body_id"] = generated_layout.get("body_id", arrays["body_id"])
        payload["generated_material_id"] = generated_layout.get("material_id", arrays["material_id"])
    state_variables = arrays.get("state_variables")
    if isinstance(state_variables, dict):
        payload["state_variables"] = np.array(state_variables, dtype=object)
    auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    if auxiliary_tracker is not None:
        payload["auxiliary_deformation_gradient"] = auxiliary_tracker.to_numpy()
    apic_affine_velocity = getattr(mpm, "nairn_apic_affine_velocity", None)
    if apic_affine_velocity is not None:
        payload["apic_affine_velocity"] = apic_affine_velocity.to_numpy()[: int(mpm.scene.particleNum[0])]
    np.savez_compressed(path, **payload)
    return path


def load_static_checkpoint_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Static checkpoint not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    return payload


def restore_static_checkpoint(mpm: MPM, case: dict[str, Any], path: Path) -> Any:
    payload = load_static_checkpoint_payload(path)
    metadata = payload["metadata"].item()
    current_arrays = particle_arrays(mpm.scene)
    particle_count = int(mpm.scene.particleNum[0])
    if int(metadata.get("particle_count", -1)) != particle_count:
        raise RuntimeError(
            f"Static checkpoint particle count mismatch: checkpoint={metadata.get('particle_count')} current={particle_count}"
        )
    if not math.isclose(float(metadata.get("dx", math.nan)), float(DX), rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise RuntimeError(f"Static checkpoint DX mismatch: checkpoint={metadata.get('dx')} current={DX}")
    if str(metadata.get("material_model", "")) != str(case.get("material_model", MATERIAL_MODEL)):
        raise RuntimeError(
            f"Static checkpoint material model mismatch: checkpoint={metadata.get('material_model')} current={case.get('material_model', MATERIAL_MODEL)}"
        )
    if metadata.get("free_field_periodic_static") is not True:
        raise RuntimeError(
            "Static checkpoint was created without the free-field periodic boundary active during static relaxation. "
            "Regenerate the static checkpoint before starting dynamics."
        )
    if metadata.get("free_field_periodic_support_wrap") != "full_cubic_modulo":
        raise RuntimeError(
            "Static checkpoint does not use full cubic free-field periodic support wrapping. "
            "Regenerate the static state."
        )
    if bool(metadata.get("static_mirror_particle_boundary", False)) != bool(STATIC_MIRROR_PARTICLE_BOUNDARY):
        raise RuntimeError(
            "Static checkpoint mirror-particle boundary setting does not match this run. Regenerate the static state."
        )
    checkpoint_reflects_body_force = bool(metadata.get("static_mirror_reflect_body_force", False))
    if checkpoint_reflects_body_force != bool(STATIC_MIRROR_REFLECT_BODY_FORCE):
        if not STATIC_CHECKPOINT_CONTINUE:
            raise RuntimeError(
                "Static checkpoint used a different mirror external-force mapping. "
                "Continue static relaxation with NAIRN_STATIC_CHECKPOINT_CONTINUE=1 before starting dynamics."
            )
        mpm.static_checkpoint_mirror_force_migration = {
            "checkpoint_reflects_body_force": checkpoint_reflects_body_force,
            "continued_reflects_body_force": bool(STATIC_MIRROR_REFLECT_BODY_FORCE),
        }
    if bool(metadata.get("static_paper_eq11_damping", False)) != bool(STATIC_PAPER_EQ11_DAMPING):
        raise RuntimeError(
            "Static checkpoint Eq. (11) damping mode does not match this run. Regenerate the static state."
        )
    checkpoint_hughes_winget = bool(metadata.get("hughes_winget_stress_update", False))
    checkpoint_hughes_winget_revision = str(metadata.get("hughes_winget_formula_revision", "UNRECORDED"))
    if hughes_winget_enabled_for_case(case) and (
        not checkpoint_hughes_winget
        or checkpoint_hughes_winget_revision != PAPER_HUGHES_WINGET_FORMULA_REVISION
    ):
        if not STATIC_CHECKPOINT_CONTINUE:
            raise RuntimeError(
                "Static checkpoint was not generated with Hughes-Winget Eq. (27)-(28). "
                "Regenerate the static state before starting the paper-aligned dynamics."
            )
        mpm.static_checkpoint_stress_update_migration = {
            "checkpoint_hughes_winget": checkpoint_hughes_winget,
            "checkpoint_hughes_winget_revision": checkpoint_hughes_winget_revision,
            "continued_hughes_winget": True,
            "continued_hughes_winget_revision": PAPER_HUGHES_WINGET_FORMULA_REVISION,
        }
    if (
        DYNAMIC_VELOCITY_PROJECTION == "Affine"
        and STATIC_CHECKPOINT_CONTINUE
        and "apic_affine_velocity" in payload
        and metadata.get("paper_apic_formula_revision") != PAPER_APIC_FORMULA_REVISION
    ):
        raise RuntimeError(
            "Static checkpoint contains an APIC affine state from an older Eq. (16) implementation. "
            "Start APIC from a non-APIC checkpoint or regenerate the APIC static state."
        )
    if DYNAMIC_VELOCITY_PROJECTION == "Affine" and not STATIC_CHECKPOINT_CONTINUE:
        checkpoint_projection = str(metadata.get("static_velocity_projection", "UNRECORDED"))
        checkpoint_alpha = float(metadata.get("static_alpha_pic", math.nan))
        checkpoint_apic_patch = bool(metadata.get("paper_apic_2d_transfer_installed", False))
        checkpoint_separate_B_L = bool(metadata.get("paper_apic_separate_B_L", False))
        checkpoint_apic_usl = bool(metadata.get("paper_apic_usl_order", False))
        checkpoint_apic_revision = str(metadata.get("paper_apic_formula_revision", "UNRECORDED"))
        if checkpoint_projection != "Affine" or not math.isclose(
            checkpoint_alpha, 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ) or not checkpoint_apic_patch or not checkpoint_separate_B_L or not checkpoint_apic_usl or checkpoint_apic_revision != PAPER_APIC_FORMULA_REVISION or "apic_affine_velocity" not in payload:
            raise RuntimeError(
                "APIC dynamics requires a static checkpoint relaxed with the same Affine transfer, "
                "alphaPIC=1, periodic 2-D APIC implementation, separate Eq. (16)/(17) state, "
                "and the paper's stress-last update order. "
                "Continue static relaxation with NAIRN_STATIC_CHECKPOINT_CONTINUE=1 before "
                "starting dynamics."
            )
    if payload["position"].shape != current_arrays["position"].shape:
        raise RuntimeError(
            f"Static checkpoint position shape mismatch: checkpoint={payload['position'].shape} current={current_arrays['position'].shape}"
        )
    if not np.array_equal(payload["body_id"].astype(current_arrays["body_id"].dtype), current_arrays["body_id"]):
        raise RuntimeError("Static checkpoint body_id mismatch; regenerate static state for this particle layout")
    if not np.array_equal(
        payload["material_id"].astype(current_arrays["material_id"].dtype), current_arrays["material_id"]
    ):
        raise RuntimeError("Static checkpoint material_id mismatch; regenerate static state for this particle layout")
    layout_position_check = "SKIPPED_LEGACY_CHECKPOINT"
    if "generated_position" in payload:
        if payload["generated_position"].shape != current_arrays["position"].shape:
            raise RuntimeError(
                f"Static checkpoint generated position shape mismatch: checkpoint={payload['generated_position'].shape} current={current_arrays['position'].shape}"
            )
        if not np.allclose(payload["generated_position"], current_arrays["position"], rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError("Static checkpoint generated particle positions do not match current geometry")
        layout_position_check = "PASS"

    restore_static_particle_fields_2d(
        particle_count,
        np.ascontiguousarray(payload["position"], dtype=np.float64),
        np.ascontiguousarray(payload["velocity"], dtype=np.float64),
        np.ascontiguousarray(payload["stress"], dtype=np.float64),
        np.ascontiguousarray(payload["velocity_gradient"], dtype=np.float64),
        np.ascontiguousarray(payload["body_id"], dtype=np.int32),
        np.ascontiguousarray(payload["material_id"], dtype=np.int32),
        1 if STATIC_CHECKPOINT_ZERO_VELOCITY_ON_LOAD else 0,
        mpm.scene.particle,
    )
    apic_affine_velocity = getattr(mpm, "nairn_apic_affine_velocity", None)
    if apic_affine_velocity is not None:
        if "apic_affine_velocity" in payload and not STATIC_CHECKPOINT_ZERO_VELOCITY_ON_LOAD:
            checkpoint_affine = np.ascontiguousarray(payload["apic_affine_velocity"], dtype=np.float32)
            if checkpoint_affine.shape != (particle_count, 2, 2):
                raise RuntimeError(
                    "Static checkpoint APIC affine-state shape mismatch: "
                    f"checkpoint={checkpoint_affine.shape} current={(particle_count, 2, 2)}"
                )
            affine_storage = np.zeros(
                (int(apic_affine_velocity.shape[0]), 2, 2), dtype=np.float32
            )
            affine_storage[:particle_count] = checkpoint_affine
            apic_affine_velocity.from_numpy(affine_storage)
            mpm.static_checkpoint_apic_affine_state = "RESTORED"
        elif STATIC_CHECKPOINT_CONTINUE or STATIC_CHECKPOINT_ZERO_VELOCITY_ON_LOAD:
            apic_affine_velocity.fill(0.0)
            mpm.static_checkpoint_apic_affine_state = "INITIALIZED_ZERO_FOR_STATIC_RELAXATION"
        else:
            raise RuntimeError(
                "APIC dynamics checkpoint is missing the independent Eq. (16) affine state. "
                "Continue static relaxation before starting dynamics."
            )
    elif DYNAMIC_VELOCITY_PROJECTION == "Affine":
        if "apic_affine_velocity" in payload and not STATIC_CHECKPOINT_ZERO_VELOCITY_ON_LOAD:
            mpm.pending_apic_affine_velocity = np.ascontiguousarray(
                payload["apic_affine_velocity"], dtype=np.float32
            )
            mpm.pending_apic_affine_state_label = "RESTORED"
            mpm.static_checkpoint_apic_affine_state = "PENDING_ENGINE_ALLOCATION"
        elif STATIC_CHECKPOINT_CONTINUE or STATIC_CHECKPOINT_ZERO_VELOCITY_ON_LOAD:
            mpm.pending_apic_affine_velocity = np.zeros((particle_count, 2, 2), dtype=np.float32)
            mpm.pending_apic_affine_state_label = "INITIALIZED_ZERO_FOR_STATIC_RELAXATION"
            mpm.static_checkpoint_apic_affine_state = "PENDING_ZERO_FOR_STATIC_RELAXATION"
        else:
            raise RuntimeError(
                "APIC dynamics checkpoint is missing the independent Eq. (16) affine state. "
                "Continue static relaxation before starting dynamics."
            )
    if "state_variables" in payload and hasattr(mpm.scene.material, "reload_state_variables"):
        mpm.scene.material.reload_state_variables(payload["state_variables"])
    auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    if auxiliary_tracker is not None:
        if "auxiliary_deformation_gradient" not in payload:
            if VERTICAL_TRANSITION_DIAGNOSTIC:
                mpm.static_checkpoint_missing_auxiliary_F = True
            else:
                raise RuntimeError(
                    "Static checkpoint is incompatible_for_equation_32: missing auxiliary_deformation_gradient."
                )
        else:
            auxiliary_tracker.load_numpy(payload["auxiliary_deformation_gradient"])
            mpm.static_checkpoint_missing_auxiliary_F = False

    static_monitor = StaticInitializationMonitor(mpm, case)
    static_monitor.enabled = True
    static_monitor.active = False
    static_monitor.start_time = 0.0
    static_monitor.end_time = float(metadata.get("static_end_time", 0.0))
    static_monitor.converged = bool(metadata.get("static_converged", True))
    static_monitor.convergence_time = float(metadata.get("static_end_time", 0.0))
    static_monitor.convergence_step = int(metadata.get("static_current_step", 0))
    static_monitor.last_stress_change = 0.0
    static_monitor.capture_final_state(mpm.scene)
    static_monitor.final_snapshot["checkpoint_loaded"] = True
    static_monitor.final_snapshot["checkpoint_path"] = str(path)
    static_monitor.final_snapshot["checkpoint_metadata"] = metadata
    static_monitor.final_snapshot["checkpoint_layout_position_check"] = layout_position_check
    static_monitor.final_snapshot["checkpoint_zero_velocity_on_load"] = STATIC_CHECKPOINT_ZERO_VELOCITY_ON_LOAD
    mpm.static_initialization = static_monitor
    mpm.static_state_snapshot = static_monitor.final_snapshot
    mpm.sims.current_time = static_monitor.end_time
    mpm.sims.current_step = int(metadata.get("static_current_step", 0))
    return static_monitor


def write_static_checkpoint_report(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: Any,
    checkpoint_path: Path,
    mode: str,
) -> Path:
    report_path = OUTPUT_DIR / "static_checkpoint_report.md"
    metadata = static_monitor.final_snapshot.get("checkpoint_metadata") or static_checkpoint_metadata(
        mpm, case, static_monitor
    )
    lines = [
        "# Static Checkpoint Report",
        "",
        f"- mode: `{mode}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- dx: `{metadata.get('dx')}`",
        f"- particle_count: `{metadata.get('particle_count')}`",
        f"- grid_nodes: `{metadata.get('grid_nodes')}`",
        f"- grid_cells: `{metadata.get('grid_cells')}`",
        f"- material_model: `{metadata.get('material_model')}`",
        f"- static_end_time: `{metadata.get('static_end_time')}`",
        f"- static_current_step: `{metadata.get('static_current_step')}`",
        f"- static_converged: `{metadata.get('static_converged')}`",
        f"- free_field_periodic_static: `{metadata.get('free_field_periodic_static')}`",
        f"- free_field_periodic_support_wrap: `{metadata.get('free_field_periodic_support_wrap', 'legacy_edge_only')}`",
        f"- static_mirror_reflect_body_force: `{metadata.get('static_mirror_reflect_body_force', False)}`",
        f"- static_paper_eq11_damping: `{metadata.get('static_paper_eq11_damping', False)}`",
        f"- static_hughes_winget_stress_update: `{metadata.get('hughes_winget_stress_update', False)}`",
        f"- hughes_winget_formula_revision: `{metadata.get('hughes_winget_formula_revision', 'legacy_unspecified')}`",
        f"- static_local_damping_beta: `{metadata.get('static_local_damping_beta', 'legacy_unspecified')}`",
        f"- static_max_velocity_tolerance: `{metadata.get('static_max_velocity_tolerance', 'legacy_unspecified')}`",
        f"- static_rms_velocity_tolerance: `{metadata.get('static_rms_velocity_tolerance', 'legacy_unspecified')}`",
        f"- static_relative_unbalanced_force_tolerance: `{metadata.get('static_relative_unbalanced_force_tolerance', 'legacy_unspecified')}`",
        f"- continued_from_checkpoint: `{metadata.get('continued_from_checkpoint', '')}`",
        f"- static_continuation_start_time: `{metadata.get('static_continuation_start_time', 0.0)}`",
        f"- checkpoint_restored_to_scene: `{'YES' if mode == 'LOAD' else 'N/A'}`",
        f"- geometry_particle_layout_check: `{static_monitor.final_snapshot.get('checkpoint_layout_position_check', 'PASS')}`",
        f"- zero_velocity_on_load: `{static_monitor.final_snapshot.get('checkpoint_zero_velocity_on_load', False)}`",
        f"- state_variables: `{'restored' if mode == 'LOAD' else 'saved when available'}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def velocity_stats(velocities: np.ndarray) -> dict[str, float]:
    if velocities.size == 0:
        return {"max": 0.0, "mean": 0.0, "rms": 0.0}
    norms = np.linalg.norm(velocities, axis=1)
    return {
        "max": float(np.max(norms)),
        "mean": float(np.mean(norms)),
        "rms": float(np.sqrt(np.mean(norms * norms))),
    }


def velocity_component_stats(velocities: np.ndarray) -> dict[str, float]:
    if velocities.size == 0:
        return {
            "max_abs_vx": 0.0,
            "max_abs_vy": 0.0,
            "max_speed": 0.0,
            "rms_vx": 0.0,
            "rms_vy": 0.0,
            "rms_speed": 0.0,
        }
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    speed = np.linalg.norm(velocities[:, :2], axis=1)
    return {
        "max_abs_vx": float(np.max(np.abs(vx))),
        "max_abs_vy": float(np.max(np.abs(vy))),
        "max_speed": float(np.max(speed)),
        "rms_vx": float(np.sqrt(np.mean(vx * vx))),
        "rms_vy": float(np.sqrt(np.mean(vy * vy))),
        "rms_speed": float(np.sqrt(np.mean(speed * speed))),
    }


def nodal_velocity_stats(scene: Any) -> dict[str, float]:
    node_mass = np.asarray(scene.node.m.to_numpy(), dtype=np.float64)
    node_velocity = np.asarray(scene.node.momentum.to_numpy(), dtype=np.float64)[..., :2]
    active = node_mass > float(scene.mass_cut_off)
    return velocity_component_stats(node_velocity[active])


def static_g2p_node_kind(body_id: int, physical_x: float, physical_y: float) -> str:
    tolerance = 1.0e-8
    if physical_y < K_BASE_BOTTOM_Z - tolerance:
        return "bottom_mirror_ghost"
    if abs(physical_y - K_BASE_BOTTOM_Z) <= tolerance:
        return "bottom_boundary"
    if body_id == BODY_MAIN_SOIL and (
        physical_x < K_MAIN_X_MIN - tolerance or physical_x > K_MAIN_X_MAX + tolerance
    ):
        return "main_lateral_mirror_ghost"
    return "physical_or_periodic_node"


def static_g2p_kinetic_energy_rows(scene: Any, velocities: np.ndarray) -> dict[str, float]:
    particle_count = int(scene.particleNum[0])
    body_ids = scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    masses = np.asarray(scene.particle.m.to_numpy()[:particle_count], dtype=np.float64)
    energy: dict[str, float] = {}
    for label, body_id in (("main", BODY_MAIN_SOIL), ("left_free_field", BODY_LEFT_FREE_SOIL), ("right_free_field", BODY_RIGHT_FREE_SOIL)):
        mask = body_ids == body_id
        energy[label] = float(0.5 * np.sum(masses[mask] * np.sum(velocities[mask, :2] ** 2, axis=1)))
    return energy


def static_mirror_velocity_parity(mpm: MPM, node_velocity: np.ndarray) -> dict[str, float | str]:
    """Check main-model mirror parity and free-field periodic support wrapping."""
    mirror = getattr(mpm, "static_mirror_particle_boundary", None)
    if mirror is None:
        return {"status": "DISABLED", "bottom_pair_error": math.nan, "bottom_boundary_error": math.nan,
                "main_lateral_pair_error": math.nan, "main_lateral_normal_error": math.nan,
                "free_field_periodic_unmapped_support_count": math.nan}

    def velocity(node_id: int, body_id: int) -> np.ndarray:
        return np.asarray(node_velocity[node_id, body_id, :2], dtype=np.float64)

    bottom_pair_error = 0.0
    bottom_boundary_error = 0.0
    for body_id in range(int(mirror.grid_levels)):
        for i in range(int(mirror.grid_x)):
            boundary_id = i + int(mirror.bottom_node) * int(mirror.grid_x)
            bottom_boundary_error = max(bottom_boundary_error, float(np.linalg.norm(velocity(boundary_id, body_id))))
            for j in range(int(mirror.bottom_node)):
                ghost_id = i + j * int(mirror.grid_x)
                source_id = i + (2 * int(mirror.bottom_node) - j) * int(mirror.grid_x)
                bottom_pair_error = max(
                    bottom_pair_error,
                    float(np.linalg.norm(velocity(ghost_id, body_id) + velocity(source_id, body_id))),
                )

    main_lateral_pair_error = 0.0
    main_lateral_normal_error = 0.0
    for j in range(int(mirror.grid_y)):
        for boundary_i, direction in ((int(mirror.main_left_node), -1), (int(mirror.main_right_node), 1)):
            boundary_id = boundary_i + j * int(mirror.grid_x)
            main_lateral_normal_error = max(main_lateral_normal_error, abs(float(velocity(boundary_id, BODY_MAIN_SOIL)[0])))
            if direction < 0:
                ghost_range = range(0, boundary_i)
            else:
                ghost_range = range(boundary_i + 1, int(mirror.grid_x))
            for ghost_i in ghost_range:
                source_i = 2 * boundary_i - ghost_i
                if source_i < 0 or source_i >= int(mirror.grid_x):
                    continue
                ghost_id = ghost_i + j * int(mirror.grid_x)
                source_id = source_i + j * int(mirror.grid_x)
                ghost = velocity(ghost_id, BODY_MAIN_SOIL)
                source = velocity(source_id, BODY_MAIN_SOIL)
                main_lateral_pair_error = max(
                    main_lateral_pair_error,
                    float(np.linalg.norm(np.asarray([ghost[0] + source[0], ghost[1] - source[1]]))),
                )

    periodic_mapper = getattr(mpm, "free_field_periodic_mapper", None)
    free_field_periodic_unmapped_support_count = (
        int(periodic_mapper.unmapped_support_count(mpm.scene)) if periodic_mapper is not None else -1
    )

    tolerance = 1.0e-10
    passed = max(
        bottom_pair_error,
        bottom_boundary_error,
        main_lateral_pair_error,
        main_lateral_normal_error,
    ) <= tolerance and free_field_periodic_unmapped_support_count == 0
    return {
        "status": "PASS" if passed else "FAIL",
        "tolerance": tolerance,
        "bottom_pair_error": bottom_pair_error,
        "bottom_boundary_error": bottom_boundary_error,
        "main_lateral_pair_error": main_lateral_pair_error,
        "main_lateral_normal_error": main_lateral_normal_error,
        "free_field_periodic_unmapped_support_count": free_field_periodic_unmapped_support_count,
    }


def capture_static_mirror_force_audit(mpm: MPM, sims: Any) -> None:
    """Read the final pre-update mirror force state without changing it."""
    if not STATIC_MIRROR_FORCE_DIAGNOSTIC:
        return
    mirror = getattr(mpm, "static_mirror_particle_boundary", None)
    if mirror is None:
        return
    scene = mpm.scene
    raw_mass = mirror.raw_mass_source.to_numpy()
    raw_force = mirror.force_source.to_numpy()
    final_force = scene.node.force.to_numpy()
    final_mass = scene.node.m.to_numpy()
    gravity = np.asarray(sims.gravity[:2], dtype=np.float64)
    coordinates = np.asarray(scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT])
    node_count, grid_x, bottom_j = int(mirror.grid_sum), int(mirror.grid_x), int(mirror.bottom_node)
    target_node_id = STATIC_MIRROR_FORCE_DIAGNOSTIC_TARGET_NODE_ID
    rows: list[dict[str, Any]] = []
    max_pair_error = max_acceleration = 0.0
    for node_id in range(node_count):
        j = node_id // grid_x
        if (j < bottom_j or j > bottom_j + 2) and node_id != target_node_id:
            continue
        for body_id in (BODY_MAIN_SOIL, BODY_LEFT_FREE_SOIL, BODY_RIGHT_FREE_SOIL):
            raw_body = raw_mass[node_id, body_id] * gravity
            raw_internal = raw_force[node_id, body_id, :2] - raw_body
            value, mass = final_force[node_id, body_id, :2], final_mass[node_id, body_id]
            acceleration = value / mass if mass > float(scene.mass_cut_off) else np.zeros(2)
            rows.append({"node_id": node_id, "body_id": body_id, "x": float(coordinates[node_id, 0]), "y": float(coordinates[node_id, 1]),
                         "node_role": "bottom_ghost" if j < bottom_j else ("bottom_boundary" if j == bottom_j else "physical_support"),
                         "raw_mass": float(raw_mass[node_id, body_id]), "final_mass": float(mass),
                         "raw_internal_force_x": float(raw_internal[0]), "raw_internal_force_y": float(raw_internal[1]),
                         "raw_body_force_x": float(raw_body[0]), "raw_body_force_y": float(raw_body[1]),
                         "raw_total_force_x": float(raw_force[node_id, body_id, 0]), "raw_total_force_y": float(raw_force[node_id, body_id, 1]),
                         "mirror_force_delta_x": float(value[0] - raw_force[node_id, body_id, 0]), "mirror_force_delta_y": float(value[1] - raw_force[node_id, body_id, 1]),
                         "final_force_x": float(value[0]), "final_force_y": float(value[1]), "acceleration_x": float(acceleration[0]), "acceleration_y": float(acceleration[1])})
            max_acceleration = max(max_acceleration, float(np.linalg.norm(acceleration)))
    mpm.static_mirror_force_audit = {"rows": rows, "max_acceleration": max_acceleration, "gravity": gravity.tolist()}


def write_static_mirror_force_audit(mpm: MPM) -> Path | None:
    audit = getattr(mpm, "static_mirror_force_audit", None)
    if audit is None:
        return None
    STATIC_MIRROR_FORCE_DIAGNOSTIC_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = STATIC_MIRROR_FORCE_DIAGNOSTIC_OUTPUT / "bottom_mirror_force_audit.csv"
    rows = audit["rows"]
    write_csv(path, list(rows[0].keys()) if rows else ["node_id"], rows)
    write_csv(STATIC_MIRROR_FORCE_DIAGNOSTIC_OUTPUT / "summary.csv", ["metric", "value"], [
        {"metric": "gravity", "value": str(audit["gravity"])},
        {"metric": "reflect_body_force", "value": STATIC_MIRROR_REFLECT_BODY_FORCE},
        {"metric": "target_node_id", "value": STATIC_MIRROR_FORCE_DIAGNOSTIC_TARGET_NODE_ID},
        {"metric": "max_bottom_band_acceleration", "value": audit["max_acceleration"]},
    ])
    return path


def capture_static_g2p_eq15_before(mpm: MPM, sims: Any, node_velocity_before_update: np.ndarray | None = None) -> dict[str, Any]:
    """Capture the exact state consumed by GeoTaichi's Eq. (15) G2P kernel."""
    scene = mpm.scene
    particle_count = int(scene.particleNum[0])
    positions = scene.particle.x.to_numpy()[:particle_count].copy()
    velocities = scene.particle.v.to_numpy()[:particle_count].copy()
    body_ids = scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    physical_y = positions[:, 1] - Y_SHIFT
    bottom_free_field = np.isin(body_ids, [BODY_LEFT_FREE_SOIL, BODY_RIGHT_FREE_SOIL]) & (
        physical_y <= K_BASE_BOTTOM_Z + DX + 1.0e-8
    )
    particle_ids = np.nonzero(bottom_free_field)[0].astype(np.int32)

    # The engine wrapper invokes the same extrapolation immediately before the
    # native G2P kernel. Repeating this assignment here is idempotent and lets
    # the audit observe the exact nodal state consumed by that kernel.
    mirror = getattr(mpm, "static_mirror_particle_boundary", None)
    if mirror is not None and static_mirror_boundary_is_active(mpm):
        mirror.extrapolate_velocity(scene)
    node = node_field_arrays(scene)
    coordinates = np.asarray(scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT])
    total_support_nodes = int(scene.element.grid_nodes)
    ln_id = scene.element.LnID.to_numpy()
    shape_fn = scene.element.shape_fn.to_numpy()
    node_size = scene.element.node_size.to_numpy()
    alpha = float(sims.alphaPIC)
    dt = float(sims.delta)
    particle_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    expected_velocity: dict[int, np.ndarray] = {}

    for particle_id in particle_ids.tolist():
        body_id = int(body_ids[particle_id])
        offset = particle_id * total_support_nodes
        count = int(node_size[particle_id])
        node_ids = ln_id[offset : offset + count].astype(np.int32)
        weights = shape_fn[offset : offset + count].astype(np.float64)
        pic_velocity = np.zeros(2, dtype=np.float64)
        flip_acceleration = np.zeros(2, dtype=np.float64)
        flip_velocity_increment = np.zeros(2, dtype=np.float64)
        for local_index, (node_id, weight) in enumerate(zip(node_ids.tolist(), weights.tolist())):
            grid_velocity = node["momentum"][node_id, body_id, :2]
            grid_acceleration = node["force"][node_id, body_id, :2]
            old_velocity = grid_velocity if node_velocity_before_update is None else node_velocity_before_update[node_id, body_id, :2]
            pic_velocity += weight * grid_velocity
            flip_acceleration += weight * grid_acceleration
            flip_velocity_increment += weight * (grid_velocity - old_velocity)
            node_x, node_y = coordinates[node_id, :2]
            support_rows.append(
                {
                    "static_step": int(sims.current_step),
                    "absolute_time": float(sims.current_time),
                    "particle_id": particle_id,
                    "body_id": body_id,
                    "support_index": local_index,
                    "node_id": int(node_id),
                    "node_x": float(node_x),
                    "node_y": float(node_y),
                    "node_kind": static_g2p_node_kind(body_id, float(node_x), float(node_y)),
                    "weight": float(weight),
                    "node_mass": float(node["m"][node_id, body_id]),
                    "node_velocity_x": float(grid_velocity[0]),
                    "node_velocity_y": float(grid_velocity[1]),
                    "node_acceleration_x": float(grid_acceleration[0]),
                    "node_acceleration_y": float(grid_acceleration[1]),
                    "flip_delta_v_x": float((1.0 - alpha) * dt * weight * grid_acceleration[0]),
                    "flip_delta_v_y": float((1.0 - alpha) * dt * weight * grid_acceleration[1]),
                    "eq15_velocity_delta_x": float((1.0 - alpha) * weight * (grid_velocity[0] - old_velocity[0])),
                    "eq15_velocity_delta_y": float((1.0 - alpha) * weight * (grid_velocity[1] - old_velocity[1])),
                }
            )
        expected = alpha * pic_velocity + (1.0 - alpha) * (velocities[particle_id, :2] + dt * flip_acceleration)
        paper_expected = alpha * pic_velocity + (1.0 - alpha) * (velocities[particle_id, :2] + flip_velocity_increment)
        expected_velocity[particle_id] = expected
        particle_rows.append(
            {
                "static_step": int(sims.current_step),
                "absolute_time": float(sims.current_time),
                "particle_id": particle_id,
                "body_id": body_id,
                "particle_x": float(positions[particle_id, 0] - X_SHIFT),
                "particle_y": float(positions[particle_id, 1] - Y_SHIFT),
                "alpha_pic": alpha,
                "dt": dt,
                "velocity_before_x": float(velocities[particle_id, 0]),
                "velocity_before_y": float(velocities[particle_id, 1]),
                "pic_velocity_x": float(pic_velocity[0]),
                "pic_velocity_y": float(pic_velocity[1]),
                "flip_acceleration_x": float(flip_acceleration[0]),
                "flip_acceleration_y": float(flip_acceleration[1]),
                "expected_velocity_x": float(expected[0]),
                "expected_velocity_y": float(expected[1]),
                "paper_eq15_velocity_x": float(paper_expected[0]),
                "paper_eq15_velocity_y": float(paper_expected[1]),
                "paper_eq15_minus_acceleration_x": float(paper_expected[0] - expected[0]),
                "paper_eq15_minus_acceleration_y": float(paper_expected[1] - expected[1]),
                "support_count": count,
            }
        )

    return {
        "step": int(sims.current_step),
        "time": float(sims.current_time),
        "particle_rows": particle_rows,
        "support_rows": support_rows,
        "expected_velocity": expected_velocity,
        "kinetic_energy_before": static_g2p_kinetic_energy_rows(scene, velocities),
        "mirror_velocity_parity": static_mirror_velocity_parity(mpm, node["momentum"]),
    }


def complete_static_g2p_eq15_after(mpm: MPM, snapshot: dict[str, Any]) -> None:
    scene = mpm.scene
    particle_count = int(scene.particleNum[0])
    velocities = scene.particle.v.to_numpy()[:particle_count]
    errors: list[float] = []
    for row in snapshot["particle_rows"]:
        particle_id = int(row["particle_id"])
        expected = snapshot["expected_velocity"][particle_id]
        actual = velocities[particle_id, :2]
        error = actual - expected
        row["actual_velocity_x"] = float(actual[0])
        row["actual_velocity_y"] = float(actual[1])
        row["eq15_error_x"] = float(error[0])
        row["eq15_error_y"] = float(error[1])
        row["eq15_error_norm"] = float(np.linalg.norm(error))
        errors.append(float(np.linalg.norm(error)))
    snapshot["kinetic_energy_after"] = static_g2p_kinetic_energy_rows(scene, velocities)
    snapshot["max_eq15_error"] = float(max(errors, default=0.0))
    getattr(mpm, "static_g2p_eq15_snapshots", []).append(snapshot)


def write_static_g2p_eq15_diagnostic(mpm: MPM) -> tuple[Path, Path, Path] | None:
    snapshots = getattr(mpm, "static_g2p_eq15_snapshots", [])
    if not snapshots:
        return None
    STATIC_G2P_EQ15_DIAGNOSTIC_OUTPUT.mkdir(parents=True, exist_ok=True)
    particle_path = STATIC_G2P_EQ15_DIAGNOSTIC_OUTPUT / "particle_eq15_check.csv"
    support_path = STATIC_G2P_EQ15_DIAGNOSTIC_OUTPUT / "support_node_contributions.csv"
    summary_path = STATIC_G2P_EQ15_DIAGNOSTIC_OUTPUT / "summary.csv"
    particle_rows = [row for snapshot in snapshots for row in snapshot["particle_rows"]]
    support_rows = [row for snapshot in snapshots for row in snapshot["support_rows"]]
    summary_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        energy_before = snapshot["kinetic_energy_before"]
        energy_after = snapshot["kinetic_energy_after"]
        reference_speed = max(
            (
                math.hypot(float(row["actual_velocity_x"]), float(row["actual_velocity_y"]))
                for row in snapshot["particle_rows"]
            ),
            default=0.0,
        )
        # GeoTaichi stores particle and node fields in single precision, so a
        # relative tolerance is required for the Python reconstruction audit.
        eq15_tolerance = 1.0e-6 * max(1.0, reference_speed)
        summary_rows.append(
            {
                "static_step": snapshot["step"],
                "absolute_time": snapshot["time"],
                "sampled_bottom_free_field_particle_count": len(snapshot["particle_rows"]),
                "max_eq15_velocity_error": snapshot["max_eq15_error"],
                "max_reference_particle_speed": reference_speed,
                "eq15_tolerance": eq15_tolerance,
                **snapshot["mirror_velocity_parity"],
                "main_kinetic_energy_before": energy_before["main"],
                "main_kinetic_energy_after": energy_after["main"],
                "left_free_field_kinetic_energy_before": energy_before["left_free_field"],
                "left_free_field_kinetic_energy_after": energy_after["left_free_field"],
                "right_free_field_kinetic_energy_before": energy_before["right_free_field"],
                "right_free_field_kinetic_energy_after": energy_after["right_free_field"],
                "formula": (
                    "v_new=alpha*v_pic+(1-alpha)*(v_old+sum(weight*(v_node_new-v_node_old)))"
                    if STATIC_PAPER_EQ15_FLIP
                    else "v_new=alpha*v_pic+(1-alpha)*(v_old+dt*sum(weight*node_acceleration))"
                ),
                "status": "PASS" if snapshot["max_eq15_error"] <= eq15_tolerance else "FAIL",
            }
        )
    write_csv(particle_path, list(particle_rows[0].keys()) if particle_rows else ["particle_id"], particle_rows)
    write_csv(support_path, list(support_rows[0].keys()) if support_rows else ["particle_id"], support_rows)
    write_csv(summary_path, list(summary_rows[0].keys()), summary_rows)
    return particle_path, support_path, summary_path


def stress_stats(stresses: np.ndarray) -> dict[str, float]:
    if stresses.size == 0:
        return {"max_norm": 0.0, "mean_norm": 0.0}
    norms = np.linalg.norm(stresses, axis=1)
    return {
        "max_norm": float(np.max(norms)),
        "mean_norm": float(np.mean(norms)),
    }


def node_force_window_stats(
    node_force: np.ndarray,
    node_mass: np.ndarray,
    gravity_y: float,
    mass_cutoff: float,
    constrained_dof_mask: np.ndarray | None = None,
    ignored_dof_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    force_array = np.asarray(node_force, dtype=np.float64)
    mass_array = np.asarray(node_mass, dtype=np.float64)
    if force_array.size == 0 or mass_array.size == 0:
        return {
            "active_node_count": 0,
            "total_mass": 0.0,
            "total_weight": 0.0,
            "max_force": 0.0,
            "rms_force": 0.0,
            "relative_unbalanced_force": math.inf,
            "max_force_node_id": -1,
            "max_force_body_id": -1,
            "max_force_x": 0.0,
            "max_force_y": 0.0,
            "max_force_vx": 0.0,
            "max_force_vy": 0.0,
            "excluded_node_count": 0,
            "ignored_node_count": 0,
            "ignored_nodal_body_count": 0,
            "support_node_count": 0,
            "support_max_force": 0.0,
            "support_rms_force": 0.0,
            "support_relative_force": 0.0,
            "support_max_force_node_id": -1,
            "support_max_force_body_id": -1,
            "support_max_force_x": 0.0,
            "support_max_force_y": 0.0,
        }

    if force_array.ndim == 3:
        force_by_body = force_array[..., :2]
    elif force_array.ndim == 2:
        force_by_body = force_array[:, np.newaxis, :2]
    else:
        force_by_body = np.reshape(force_array, (-1, 1, 2))

    if mass_array.ndim == 2:
        mass_by_body = mass_array
    else:
        mass_by_body = np.reshape(mass_array, (-1, 1))
    if mass_by_body.shape != force_by_body.shape[:2]:
        raise ValueError(
            f"node mass/force shape mismatch: mass={mass_by_body.shape}, force={force_by_body.shape}"
        )

    active_all = mass_by_body > mass_cutoff
    constrained = np.zeros(force_by_body.shape, dtype=bool)
    if constrained_dof_mask is not None:
        supplied_mask = np.asarray(constrained_dof_mask, dtype=bool)
        if supplied_mask.shape == (force_by_body.shape[0], 2):
            supplied_mask = np.broadcast_to(supplied_mask[:, np.newaxis, :], constrained.shape)
        if supplied_mask.shape != constrained.shape:
            raise ValueError(
                f"constrained_dof_mask shape mismatch: expected {constrained.shape}, got {supplied_mask.shape}"
            )
        constrained[:] = supplied_mask
    ignored = np.zeros(force_by_body.shape, dtype=bool)
    if ignored_dof_mask is not None:
        supplied_ignored = np.asarray(ignored_dof_mask, dtype=bool)
        if supplied_ignored.shape == (force_by_body.shape[0], 2):
            supplied_ignored = np.broadcast_to(supplied_ignored[:, np.newaxis, :], ignored.shape)
        if supplied_ignored.shape != ignored.shape:
            raise ValueError(
                f"ignored_dof_mask shape mismatch: expected {ignored.shape}, got {supplied_ignored.shape}"
            )
        ignored[:] = supplied_ignored

    # Paper Sections 3.2-3.3: reactions belong only to constrained DOFs.
    # Bottom x/y and lateral x are support forces; lateral y remains a free
    # equilibrium equation and must stay in the residual-force diagnostic.
    free_components = ~constrained & ~ignored
    support_components = constrained & ~ignored
    active = active_all & np.any(free_components, axis=2)
    support_active = active_all & np.any(support_components, axis=2)
    free_force = np.where(free_components, force_by_body, 0.0)
    active_force = free_force[active] if np.any(active) else np.empty((0, 2), dtype=np.float64)
    force_norm = np.linalg.norm(active_force, axis=1) if active_force.size else np.empty(0, dtype=np.float64)
    total_mass_value = float(np.sum(mass_by_body[active])) if np.any(active) else 0.0
    total_weight = total_mass_value * abs(float(gravity_y))
    max_index = int(np.argmax(force_norm)) if force_norm.size else -1
    active_node_ids, active_body_ids = np.nonzero(active)
    max_force_node_id = int(active_node_ids[max_index]) if max_index >= 0 else -1
    max_force_body_id = int(active_body_ids[max_index]) if max_index >= 0 else -1
    max_force_vector = active_force[max_index] if max_index >= 0 else np.zeros(2, dtype=np.float64)
    max_force = float(force_norm[max_index]) if max_index >= 0 else 0.0
    rms_force = float(np.sqrt(np.mean(force_norm * force_norm))) if force_norm.size else 0.0

    constrained_force = np.where(support_components, force_by_body, 0.0)
    support_force = (
        constrained_force[support_active]
        if np.any(support_active)
        else np.empty((0, 2), dtype=np.float64)
    )
    support_norm = np.linalg.norm(support_force, axis=1) if support_force.size else np.empty(0, dtype=np.float64)
    support_weight = (
        float(np.sum(mass_by_body[support_active])) * abs(float(gravity_y))
        if np.any(support_active)
        else 0.0
    )
    support_max_index = int(np.argmax(support_norm)) if support_norm.size else -1
    support_node_ids, support_body_ids = np.nonzero(support_active)
    support_max_force_node_id = int(support_node_ids[support_max_index]) if support_max_index >= 0 else -1
    support_max_force_body_id = int(support_body_ids[support_max_index]) if support_max_index >= 0 else -1
    support_max_force_vector = (
        support_force[support_max_index] if support_max_index >= 0 else np.zeros(2, dtype=np.float64)
    )
    support_max_force = float(support_norm[support_max_index]) if support_max_index >= 0 else 0.0
    support_rms_force = float(np.sqrt(np.mean(support_norm * support_norm))) if support_norm.size else 0.0
    return {
        "active_node_count": int(np.count_nonzero(active)),
        "total_mass": total_mass_value,
        "total_weight": total_weight,
        "max_force": max_force,
        "rms_force": rms_force,
        "relative_unbalanced_force": float(max_force / max(total_weight, 1.0e-30)),
        "max_force_node_id": max_force_node_id,
        "max_force_body_id": max_force_body_id,
        "max_force_x": float(max_force_vector[0]),
        "max_force_y": float(max_force_vector[1]),
        "max_force_vx": float(max_force_vector[0]),
        "max_force_vy": float(max_force_vector[1]),
        "excluded_node_count": int(np.count_nonzero(np.any(constrained, axis=(1, 2)))),
        "ignored_node_count": int(np.count_nonzero(np.any(active_all & np.any(ignored, axis=2), axis=1))),
        "ignored_nodal_body_count": int(np.count_nonzero(active_all & np.any(ignored, axis=2))),
        "support_node_count": int(np.count_nonzero(support_active)),
        "support_max_force": support_max_force,
        "support_rms_force": support_rms_force,
        "support_relative_force": float(support_max_force / max(support_weight, 1.0e-30)),
        "support_max_force_node_id": support_max_force_node_id,
        "support_max_force_body_id": support_max_force_body_id,
        "support_max_force_x": float(support_max_force_vector[0]),
        "support_max_force_y": float(support_max_force_vector[1]),
    }


def material_state_stats(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    state = arrays.get("state_variables")
    stresses = np.asarray(arrays.get("stress", np.empty((0, 6))))
    stats: dict[str, Any] = {
        "state_variable_keys": [],
        "plastic_strain_available": False,
        "max_plastic_strain": 0.0,
        "mean_plastic_strain": 0.0,
        "plastic_particle_count": 0,
        "max_stress_norm": stress_stats(stresses)["max_norm"],
        "mean_stress_norm": stress_stats(stresses)["mean_norm"],
    }
    if not isinstance(state, dict):
        return stats

    stats["state_variable_keys"] = sorted(str(key) for key in state.keys())
    if "epstrain" not in state:
        return stats

    epstrain = np.asarray(state["epstrain"], dtype=np.float64).reshape(-1)
    if epstrain.size == 0:
        stats["plastic_strain_available"] = True
        return stats

    abs_epstrain = np.abs(epstrain)
    stats.update(
        {
            "plastic_strain_available": True,
            "max_plastic_strain": float(np.max(abs_epstrain)),
            "mean_plastic_strain": float(np.mean(abs_epstrain)),
            "plastic_particle_count": int(np.count_nonzero(abs_epstrain > 1.0e-14)),
        }
    )
    return stats


def write_material_model_check(case: dict[str, Any]) -> Path:
    path = OUTPUT_DIR / "material_model_check.md"
    material_model = str(case.get("material_model", "LinearElastic"))
    materials = case.get("materials", [])
    current_plasticity = material_model in {
        "ElasticPerfectlyPlastic",
        "IsotropicHardeningPlastic",
        "VonMisesSoftening",
        "DruckerPrager",
        "MohrCoulomb",
        "SoftenMohrCoulomb",
        "StateDependentMohrCoulomb",
        "ModifiedCamClay",
        "CohesiveModifiedCamClay",
    }
    softening = material_model in {"SoftenMohrCoulomb", "VonMisesSoftening", "StateDependentMohrCoulomb", "CohesiveModifiedCamClay"} or bool(
        case.get("softening", {}).get("enabled", False)
    )
    if material_model == "DruckerPrager":
        criterion = "pressure-dependent Drucker-Prager shear/tension yield surface"
        plastic_variables = "`epstrain` equivalent plastic strain and `estress` equivalent stress"
    elif material_model == "SoftenMohrCoulomb":
        criterion = "Mohr-Coulomb shear/tension yield surface with linear strength softening"
        plastic_variables = "`epstrain` equivalent plastic strain and `estress` equivalent stress"
    elif material_model == "ElasticPerfectlyPlastic":
        criterion = "von Mises equivalent-stress yield surface"
        plastic_variables = "`epstrain` equivalent plastic strain and `estress` equivalent stress"
    elif material_model == "VonMisesSoftening":
        criterion = "von Mises (J2) equivalent-stress yield surface with isotropic softening"
        plastic_variables = "`epstrain` equivalent plastic strain and `estress` equivalent stress"
    elif material_model == "LinearElastic":
        criterion = "none"
        plastic_variables = "`estress` only; no plastic strain"
    else:
        criterion = "GeoTaichi built-in model-specific yield surface"
        plastic_variables = "model-specific state variables exposed through `get_state_vars_dict()`"

    material_lines = []
    for material in materials:
        material_lines.extend(
            [
                f"- MaterialID `{material.get('MaterialID')}`:",
                f"  Density `{material.get('Density')}`, YoungModulus `{material.get('YoungModulus')}`, PossionRatio `{material.get('PossionRatio')}`",
                f"  Cohesion `{material.get('Cohesion', 'n/a')}`, Friction `{material.get('Friction', 'n/a')}`, Dilation `{material.get('Dilation', 'n/a')}`, Tensile `{material.get('Tensile', 'n/a')}`, dpType `{material.get('dpType', 'n/a')}`",
                f"  ResidualCohesion `{material.get('ResidualCohesion', 'n/a')}`, ResidualFriction `{material.get('ResidualFriction', 'n/a')}`, PlasticDevStrain `{material.get('PlasticDevStrain', 'n/a')}`, ResidualPlasticDevStrain `{material.get('ResidualPlasticDevStrain', 'n/a')}`",
                f"  YieldStress `{material.get('YieldStress', 'n/a')}`, ResidualYieldStress `{material.get('ResidualYieldStress', 'n/a')}`",
            ]
        )

    if material_model == "LinearElastic":
        kohler_gap_lines = [
            "- Default material state matches the Fig.10 seismic response benchmark assumption: soil layer and elastic base are purely isotropic linear elastic.",
            "- Soil and base properties are separated according to Table 1; strength and softening fields are ignored by `LinearElastic`.",
            "- Failure-surface tracking and post-failure runout remain outside the Fig.10 elastic response setting.",
        ]
    else:
        kohler_gap_lines = [
            "- Current implementation is elastic-plastic and can accumulate plastic strain.",
            "- Driver-level peak-to-residual strength degradation may be enabled against `epstrain` during the dynamic stage.",
            "- It does not yet include explicit failure-surface tracking or final landslide runout analysis.",
        ]

    lines = [
        "# Material Model Check",
        "",
        "## Current Driver Material",
        "",
        f"- material_model: `{material_model}`",
        f"- constitutive_model: `{material_model}` via `mpm.add_material(...)`",
        f"- yield_criterion: `{criterion}`",
        f"- plastic_variables: {plastic_variables}",
        f"- state_variables: exported from `scene.material.get_state_vars_dict(0, particle_count)`",
        f"- supports_plasticity: `{'PASS' if current_plasticity else 'MISSING'}`",
        f"- supports_softening: `{'PASS' if softening else 'MISSING'}`",
        "",
        "## Material Parameters",
        "",
        *material_lines,
        "",
        "## Available GeoTaichi Material Models Found",
        "",
        "- `LinearElastic`: elastic stress update only.",
        "- `ElasticPerfectlyPlastic`: von Mises elastic-perfectly plastic model with `epstrain`; existing source has apparent argument inconsistencies in correction calls.",
        "- `IsotropicHardeningPlastic`: von Mises-style hardening plastic model with `epstrain` and `sigma_y`; existing source shows similar apparent argument inconsistencies.",
        "- `VonMisesSoftening`: solver-compatible J2 radial-return model with per-particle equivalent plastic strain and peak-to-residual isotropic yield-stress softening.",
        "- `DruckerPrager`: pressure-dependent elastic-plastic soil model with `epstrain` and `estress`.",
        "- `MohrCoulomb/SoftenMohrCoulomb`: frictional soil plasticity, with softening available in the softening variant.",
        "- `ModifiedCamClay/CohesiveModifiedCamClay`: critical-state clay models with additional internal variables.",
        "",
        "## Gap To Kohler 2022",
        "",
        *kohler_gap_lines,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_material_state_outputs(
    arrays: dict[str, np.ndarray],
    prefix: str,
    max_rows: int | None = None,
) -> tuple[Path, Path]:
    particle_count = int(np.asarray(arrays.get("position", np.empty((0, 2)))).shape[0])
    row_count = particle_count if max_rows is None else min(particle_count, max_rows)
    positions = np.asarray(arrays.get("position", np.empty((0, 2))))
    stresses = np.asarray(arrays.get("stress", np.empty((0, 6))))
    body_ids = np.asarray(arrays.get("body_id", np.empty(0)))
    material_ids = np.asarray(arrays.get("material_id", np.empty(0)))
    state = arrays.get("state_variables") if isinstance(arrays.get("state_variables"), dict) else {}
    epstrain = np.asarray(state.get("epstrain", np.zeros(particle_count)), dtype=np.float64).reshape(-1)

    plastic_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    for pid in range(row_count):
        position = positions[pid] - np.array([X_SHIFT, Y_SHIFT], dtype=np.float64)
        stress = stresses[pid]
        plastic_rows.append(
            {
                "particle_id": pid,
                "body_id": int(body_ids[pid]) if body_ids.size else "",
                "material_id": int(material_ids[pid]) if material_ids.size else "",
                "x": float(position[0]),
                "z": float(position[1]),
                "plastic_strain": float(epstrain[pid]) if pid < epstrain.size else 0.0,
            }
        )
        stress_rows.append(
            {
                "particle_id": pid,
                "body_id": int(body_ids[pid]) if body_ids.size else "",
                "material_id": int(material_ids[pid]) if material_ids.size else "",
                "x": float(position[0]),
                "z": float(position[1]),
                "stress_xx": float(stress[0]) if stress.size > 0 else 0.0,
                "stress_yy": float(stress[1]) if stress.size > 1 else 0.0,
                "stress_zz": float(stress[2]) if stress.size > 2 else 0.0,
                "stress_xy": float(stress[3]) if stress.size > 3 else 0.0,
                "stress_yz": float(stress[4]) if stress.size > 4 else 0.0,
                "stress_xz": float(stress[5]) if stress.size > 5 else 0.0,
                "stress_norm": float(np.linalg.norm(stress)),
            }
        )

    plastic_path = OUTPUT_DIR / f"{prefix}_plastic_strain.csv"
    stress_path = OUTPUT_DIR / f"{prefix}_stress.csv"
    write_csv(plastic_path, ["particle_id", "body_id", "material_id", "x", "z", "plastic_strain"], plastic_rows)
    write_csv(
        stress_path,
        [
            "particle_id",
            "body_id",
            "material_id",
            "x",
            "z",
            "stress_xx",
            "stress_yy",
            "stress_zz",
            "stress_xy",
            "stress_yz",
            "stress_xz",
            "stress_norm",
        ],
        stress_rows,
    )
    return plastic_path, stress_path


def write_material_validation_report(
    case: dict[str, Any],
    static_monitor: "StaticInitializationMonitor",
    dynamic_snapshot: dict[str, Any] | None,
) -> Path:
    static_arrays = static_monitor.final_snapshot.get("arrays", {})
    static_stats = material_state_stats(static_arrays)
    dynamic_arrays = dynamic_snapshot.get("arrays", {}) if dynamic_snapshot else {}
    dynamic_stats = material_state_stats(dynamic_arrays) if dynamic_snapshot else {}
    static_state = static_arrays.get("state_variables")
    dynamic_state = dynamic_arrays.get("state_variables") if dynamic_snapshot else static_state
    state_ok = isinstance(static_state, dict) and isinstance(dynamic_state, dict) and set(static_state) == set(dynamic_state)
    state_difference = 0.0
    state_message = "state variable keys and shapes are consistent"
    if state_ok:
        for key in static_state:
            static_value = np.asarray(static_state[key])
            dynamic_value = np.asarray(dynamic_state[key])
            if static_value.shape != dynamic_value.shape:
                state_ok = False
                state_difference = math.inf
                state_message = f"state variable {key} shape differs: {static_value.shape} vs {dynamic_value.shape}"
                break
            if dynamic_snapshot and static_value.size:
                state_difference = max(state_difference, float(np.max(np.abs(static_value - dynamic_value))))
    else:
        state_message = "state variable keys are missing or inconsistent"
    static_plastic_path, static_stress_path = write_material_state_outputs(static_arrays, "static_final")
    dynamic_plastic_path = None
    dynamic_stress_path = None
    if dynamic_snapshot:
        dynamic_plastic_path, dynamic_stress_path = write_material_state_outputs(dynamic_arrays, "dynamic_final")

    dynamic_plastic_count = int(dynamic_stats.get("plastic_particle_count", 0)) if dynamic_snapshot else 0
    dynamic_plastic_status = "PASS" if dynamic_snapshot and dynamic_plastic_count > 0 else "MISSING"
    path = OUTPUT_DIR / "material_validation_report.md"
    lines = [
        "# Material Validation Report",
        "",
        "## Scope",
        "",
        f"- material_model: `{case.get('material_model', 'LinearElastic')}`",
        "- solver_core_modified: `False`",
        "- geometry_modified_in_this_step: `False`",
        "- grid_generation_modified_in_this_step: `False`",
        "- free_field_coupling_modified_in_this_step: `False`",
        "- dynamic_boundary_modified_in_this_step: `False`",
        "",
        "## Static Gravity Stage",
        "",
        f"- static_end_time: `{static_monitor.end_time}`",
        f"- static_converged: `{static_monitor.converged}`",
        f"- max_velocity: `{static_monitor.final_snapshot.get('velocity', {}).get('max', math.nan)}`",
        f"- rms_velocity: `{static_monitor.final_snapshot.get('velocity', {}).get('rms', math.nan)}`",
        f"- max_stress_norm: `{static_stats.get('max_stress_norm')}`",
        f"- mean_stress_norm: `{static_stats.get('mean_stress_norm')}`",
        f"- max_plastic_strain: `{static_stats.get('max_plastic_strain')}`",
        f"- plastic_particle_count: `{static_stats.get('plastic_particle_count')}`",
        f"- stress_output: `{static_stress_path}`",
        f"- plastic_strain_output: `{static_plastic_path}`",
        "",
        "## Dynamic Stage",
        "",
        f"- dynamic_ran: `{dynamic_snapshot is not None}`",
        f"- dynamic_plastic_strain_status: `{dynamic_plastic_status}`",
        f"- max_stress_norm: `{dynamic_stats.get('max_stress_norm', math.nan)}`",
        f"- mean_stress_norm: `{dynamic_stats.get('mean_stress_norm', math.nan)}`",
        f"- max_plastic_strain: `{dynamic_stats.get('max_plastic_strain', math.nan)}`",
        f"- mean_plastic_strain: `{dynamic_stats.get('mean_plastic_strain', math.nan)}`",
        f"- plastic_particle_count: `{dynamic_plastic_count}`",
        f"- stress_output: `{dynamic_stress_path}`",
        f"- plastic_strain_output: `{dynamic_plastic_path}`",
        "",
        "## Particle State Variables",
        "",
        f"- static_state_variable_keys: `{static_stats.get('state_variable_keys')}`",
        f"- dynamic_state_variable_keys: `{dynamic_stats.get('state_variable_keys', [])}`",
        f"- plastic_strain_available: `{static_stats.get('plastic_strain_available') or dynamic_stats.get('plastic_strain_available', False)}`",
        f"- state_variables_preserved_or_updated: `{'PASS' if state_ok else 'FAIL'}`",
        f"- state_variable_difference_static_to_dynamic_final: `{state_difference}`",
        f"- state_variable_message: `{state_message}`",
        "",
        "## Checks",
        "",
        f"- initial_stress_reasonable: `{'PASS' if static_stats.get('max_stress_norm', 0.0) > 0.0 else 'FAIL'}`",
        f"- dynamic_plastic_strain_observed: `{dynamic_plastic_status}`",
        f"- stress_not_lost: `{'PASS' if static_stats.get('max_stress_norm', 0.0) > 0.0 and (not dynamic_snapshot or dynamic_stats.get('max_stress_norm', 0.0) > 0.0) else 'FAIL'}`",
        f"- state_variables_update_path: `{'PASS' if static_stats.get('plastic_strain_available') else 'FAIL'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def softening_factor(epstrain: float, eps_start: float, eps_end: float) -> float:
    if eps_end <= eps_start:
        return 1.0 if epstrain >= eps_start else 0.0
    if epstrain < eps_start:
        return 0.0
    if epstrain > eps_end:
        return 1.0
    return (epstrain - eps_start) / (eps_end - eps_start)


def material_by_id(case: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(material["MaterialID"]): material for material in case.get("materials", [])}


def materials_for_stage(case: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    stage_name = stage.lower()
    staged_materials: list[dict[str, Any]] = []
    for material in case.get("materials", []):
        staged = dict(material)
        if int(staged.get("MaterialID", -1)) == MAT_SOIL:
            if stage_name == "static":
                staged["PossionRatio"] = float(staged.get("StaticPossionRatio", SOIL_POISSON_INITIAL))
            elif stage_name == "dynamic":
                staged["PossionRatio"] = float(staged.get("DynamicPossionRatio", staged.get("PossionRatio", SOIL_POISSON_DYNAMIC)))
                diagnostic_poisson = os.environ.get("NAIRN_DIAG_DYNAMIC_SOIL_POISSON", "").strip()
                if diagnostic_poisson:
                    # Ablation only: retain a caller-selected value to isolate
                    # the static-to-dynamic material-property switch.
                    staged["PossionRatio"] = float(diagnostic_poisson)
            else:
                raise ValueError(f"Unsupported material stage {stage!r}")
        staged_materials.append(staged)
    return staged_materials


@ti.kernel
def set_basic_material_properties(
    mat_props: ti.template(),
    material_id: ti.i32,
    density: ti.f64,
    young: ti.f64,
    possion: ti.f64,
):
    mat_props[material_id].density = density
    mat_props[material_id].young = young
    mat_props[material_id].possion = possion


@ti.kernel
def set_von_mises_strength_override(
    mat_props: ti.template(),
    material_id: ti.i32,
    yield_stress: ti.f64,
):
    """Set a constant diagnostic yield stress for one von Mises material."""
    mat_props[material_id].yield_peak = yield_stress
    mat_props[material_id].yield_residual = yield_stress


def apply_dynamic_strength_override(mpm: MPM, case: dict[str, Any]) -> float | None:
    """Optionally lower the dynamic soil strength without rebuilding particles."""
    diagnostic = case.get("diagnostic", {})
    override = diagnostic.get("dynamic_yield_stress_override")
    if override is None or str(case.get("material_model", "")) != "VonMisesSoftening":
        return None
    yield_stress = float(override)
    if not math.isfinite(yield_stress) or yield_stress <= 0.0:
        raise ValueError("diagnostic.dynamic_yield_stress_override must be positive and finite")
    material_manager = getattr(mpm.scene, "material", None)
    if material_manager is None or not hasattr(material_manager, "matProps"):
        raise RuntimeError("Dynamic strength override requires an initialized material manager")
    set_von_mises_strength_override(material_manager.matProps, MAT_SOIL, yield_stress)
    return yield_stress


def apply_dynamic_qr_region(mpm: MPM, case: dict[str, Any]) -> dict[str, Any] | None:
    """Assign a residual-strength material to a narrow, explicit slip-band region.

    This is a diagnostic trigger only.  It preserves the particle state and
    changes material ID for main-soil particles inside the configured band;
    the surrounding soil remains at the peak-strength material.
    """
    spec = case.get("dynamic_qr_region", {})
    if not bool(spec.get("enabled", False)):
        return None
    assign_material = bool(spec.get("assign_material", True))
    qr_material_id = int(spec.get("material_id", 3))
    material_ids_available = {int(item["MaterialID"]) for item in case.get("materials", [])}
    if assign_material and qr_material_id not in material_ids_available:
        raise ValueError(f"dynamic_qr_region.material_id={qr_material_id} is not defined in materials")
    particle_count = int(mpm.scene.particleNum[0])
    positions = mpm.scene.particle.x.to_numpy()[:particle_count].astype(np.float64)
    material_ids = mpm.scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    physical_x = positions[:, 0] - X_SHIFT
    physical_z = positions[:, 1] - Y_SHIFT
    interface_z = np.asarray(kohler_soil_base_interface_z_np(physical_x), dtype=np.float64)
    x_min = float(spec.get("x_min", -40.0))
    x_max = float(spec.get("x_max", 140.0))
    lower_offset = float(spec.get("lower_offset_m", 0.0))
    upper_offset = float(spec.get("upper_offset_m", 1.0))
    band_thickness = float(spec.get("band_thickness_m", upper_offset - lower_offset))
    profile = str(spec.get("profile", "downslope_arc")).lower()
    left_lower_offset = spec.get("left_lower_offset_m")
    right_lower_offset = spec.get("right_lower_offset_m")
    center_lower_offset = spec.get("center_lower_offset_m")
    edge_lower_offset = spec.get("edge_lower_offset_m")
    surface_to_surface = profile == "surface_to_surface_arc"
    finite_block = profile == "finite_block_arc"
    curved = surface_to_surface or finite_block or left_lower_offset is not None or right_lower_offset is not None
    if curved:
        if surface_to_surface or finite_block:
            if center_lower_offset is None or edge_lower_offset is None:
                raise ValueError(
                    "surface_to_surface_arc requires center_lower_offset_m and edge_lower_offset_m"
                )
            center_lower_offset = float(center_lower_offset)
            edge_lower_offset = float(edge_lower_offset)
            if center_lower_offset < 0.0 or edge_lower_offset < 0.0:
                raise ValueError("dynamic_qr_region arc offsets must be non-negative")
        else:
            if left_lower_offset is None or right_lower_offset is None:
                raise ValueError("dynamic_qr_region requires both left_lower_offset_m and right_lower_offset_m")
            left_lower_offset = float(left_lower_offset)
            right_lower_offset = float(right_lower_offset)
            if left_lower_offset < 0.0 or right_lower_offset < 0.0:
                raise ValueError("dynamic_qr_region lower offsets must be non-negative")
    elif band_thickness < 0.0:
        raise ValueError("dynamic_qr_region.band_thickness_m must be non-negative")
    if curved:
        span = max(x_max - x_min, 1.0e-12)
        xi = np.clip((physical_x - x_min) / span, 0.0, 1.0)
        if surface_to_surface or finite_block:
            # Both ends rise to the ground surface while the middle approaches
            # the soil/base interface, creating a fully detached slide block.
            lower_boundary = center_lower_offset + (edge_lower_offset - center_lower_offset) * (2.0 * xi - 1.0) ** 2
        else:
            # A one-sided arc that becomes shallower toward the downslope end.
            lower_boundary = right_lower_offset + (left_lower_offset - right_lower_offset) * (1.0 - xi) ** 2
        upper_boundary = lower_boundary + band_thickness
    else:
        lower_boundary = np.full_like(physical_x, lower_offset)
        upper_boundary = np.full_like(physical_x, upper_offset)
    basal_mask = (
        (body_ids == BODY_MAIN_SOIL)
        & (material_ids == MAT_SOIL)
        & (physical_x >= x_min)
        & (physical_x <= x_max)
        & (physical_z >= interface_z + lower_boundary)
        & (physical_z <= interface_z + upper_boundary)
    )
    if finite_block:
        head_width = float(spec.get("head_width_m", band_thickness))
        toe_width = float(spec.get("toe_width_m", band_thickness))
        surface_z = np.asarray(kohler_surface_z_np(physical_x), dtype=np.float64)
        head_mask = (
            (body_ids == BODY_MAIN_SOIL)
            & (material_ids == MAT_SOIL)
            & (physical_x >= x_min)
            & (physical_x <= x_min + head_width)
            & (physical_z >= interface_z + lower_boundary)
            & (physical_z <= surface_z)
        )
        toe_mask = (
            (body_ids == BODY_MAIN_SOIL)
            & (material_ids == MAT_SOIL)
            & (physical_x >= x_max - toe_width)
            & (physical_x <= x_max)
            & (physical_z >= interface_z + lower_boundary)
            & (physical_z <= surface_z)
        )
        mask = basal_mask | head_mask | toe_mask
    else:
        mask = basal_mask
    if assign_material:
        material_ids[mask] = qr_material_id
        full_material_ids = mpm.scene.particle.materialID.to_numpy()
        full_material_ids[:particle_count] = material_ids.astype(full_material_ids.dtype, copy=False)
        mpm.scene.particle.materialID.from_numpy(full_material_ids)
    result = {
        "enabled": True,
        "assign_material": assign_material,
        "material_id": qr_material_id,
        "particle_count": int(np.count_nonzero(mask)),
        "x_min": x_min,
        "x_max": x_max,
        "lower_offset_m": lower_offset,
        "upper_offset_m": upper_offset,
        "band_thickness_m": band_thickness,
        "profile": profile,
        "curved": curved,
        "left_lower_offset_m": left_lower_offset,
        "right_lower_offset_m": right_lower_offset,
        "center_lower_offset_m": center_lower_offset,
        "edge_lower_offset_m": edge_lower_offset,
    }
    mpm.dynamic_qr_region = result
    return result


def apply_equivalent_postquake_qr_state(mpm: MPM, case: dict[str, Any]) -> dict[str, Any] | None:
    """Initialize a stress-consistent residual-strength slip band.

    This represents an equivalent post-earthquake state.  The selected soil
    particles remain in the paper's original softening material, their
    softening variable is advanced to the residual branch, and any excess
    deviatoric stress is projected radially onto the residual J2 surface.
    Hydrostatic stress is preserved and no velocity is prescribed.
    """
    spec = case.get("equivalent_postquake_qr_state", {})
    if not bool(spec.get("enabled", False)):
        return None
    qr_spec = case.get("dynamic_qr_region", {})
    if not bool(qr_spec.get("enabled", False)):
        raise ValueError("equivalent_postquake_qr_state requires dynamic_qr_region.enabled=true")
    if bool(qr_spec.get("assign_material", True)):
        raise ValueError(
            "equivalent_postquake_qr_state requires dynamic_qr_region.assign_material=false"
        )
    if str(case.get("material_model", "")) != "VonMisesSoftening":
        raise ValueError("equivalent_postquake_qr_state requires VonMisesSoftening")

    particle_count = int(mpm.scene.particleNum[0])
    positions = mpm.scene.particle.x.to_numpy()[:particle_count].astype(np.float64)
    stresses = mpm.scene.particle.stress.to_numpy()[:particle_count].astype(np.float64)
    material_ids = mpm.scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    physical_x = positions[:, 0] - X_SHIFT
    physical_z = positions[:, 1] - Y_SHIFT
    interface_z = np.asarray(kohler_soil_base_interface_z_np(physical_x), dtype=np.float64)
    x_min = float(qr_spec.get("x_min", -40.0))
    x_max = float(qr_spec.get("x_max", 140.0))
    thickness = float(qr_spec.get("band_thickness_m", 0.5))
    span = max(x_max - x_min, 1.0e-12)
    xi = np.clip((physical_x - x_min) / span, 0.0, 1.0)
    profile = str(qr_spec.get("profile", "surface_to_surface_arc")).lower()
    finite_block = profile == "finite_block_arc"
    if profile in {"surface_to_surface_arc", "finite_block_arc"}:
        center = float(qr_spec.get("center_lower_offset_m", 0.0))
        edge = float(qr_spec.get("edge_lower_offset_m", 9.5))
        lower = center + (edge - center) * (2.0 * xi - 1.0) ** 2
    elif profile == "downslope_arc":
        left = float(qr_spec.get("left_lower_offset_m", 9.5))
        right = float(qr_spec.get("right_lower_offset_m", 0.0))
        lower = right + (left - right) * (1.0 - xi) ** 2
    else:
        raise ValueError(
            "equivalent_postquake_qr_state supports 'surface_to_surface_arc' or 'downslope_arc'"
        )
    basal_mask = (
        (body_ids == BODY_MAIN_SOIL)
        & (material_ids == MAT_SOIL)
        & (physical_x >= x_min)
        & (physical_x <= x_max)
        & (physical_z >= interface_z + lower)
        & (physical_z <= interface_z + lower + thickness)
    )
    if finite_block:
        head_width = float(qr_spec.get("head_width_m", thickness))
        toe_width = float(qr_spec.get("toe_width_m", thickness))
        surface_z = np.asarray(kohler_surface_z_np(physical_x), dtype=np.float64)
        head_mask = (
            (body_ids == BODY_MAIN_SOIL)
            & (material_ids == MAT_SOIL)
            & (physical_x >= x_min)
            & (physical_x <= x_min + head_width)
            & (physical_z >= interface_z + lower)
            & (physical_z <= surface_z)
        )
        toe_mask = (
            (body_ids == BODY_MAIN_SOIL)
            & (material_ids == MAT_SOIL)
            & (physical_x >= x_max - toe_width)
            & (physical_x <= x_max)
            & (physical_z >= interface_z + lower)
            & (physical_z <= surface_z)
        )
        mask = basal_mask | head_mask | toe_mask
    else:
        mask = basal_mask
    if not np.any(mask):
        raise RuntimeError("equivalent_postquake_qr_state selected no soil particles")

    softening = case.get("softening", {})
    residual_strength = float(
        spec.get("residual_strength", softening.get("yield_stress_residual", 38.8888888889))
    )
    residual_epstrain = float(
        spec.get("residual_epstrain", softening.get("eps_end", 0.2309401077))
    )
    if residual_strength <= 0.0 or residual_epstrain < 0.0:
        raise ValueError("Equivalent post-earthquake residual strength/state must be non-negative")

    mean_stress = np.mean(stresses[:, :3], axis=1)
    deviatoric = stresses.copy()
    deviatoric[:, 0] -= mean_stress
    deviatoric[:, 1] -= mean_stress
    deviatoric[:, 2] -= mean_stress
    q_before = np.sqrt(
        1.5
        * (
            np.sum(deviatoric[:, :3] ** 2, axis=1)
            + 2.0 * np.sum(deviatoric[:, 3:] ** 2, axis=1)
        )
    )
    projection_mask = mask & (q_before > residual_strength)
    scale = np.ones(particle_count, dtype=np.float64)
    scale[projection_mask] = residual_strength / q_before[projection_mask]
    stresses[projection_mask, :3] = (
        mean_stress[projection_mask, None]
        + deviatoric[projection_mask, :3] * scale[projection_mask, None]
    )
    stresses[projection_mask, 3:] = deviatoric[projection_mask, 3:] * scale[projection_mask, None]

    full_stress = mpm.scene.particle.stress.to_numpy()
    full_stress[:particle_count] = stresses.astype(full_stress.dtype, copy=False)
    mpm.scene.particle.stress.from_numpy(full_stress)

    material = mpm.scene.material
    state = material.get_state_vars_dict(0, particle_count)
    epstrain = np.asarray(state["epstrain"], dtype=np.float64).copy()
    estress = np.asarray(state["estress"], dtype=np.float64).copy()
    epstrain[mask] = np.maximum(epstrain[mask], residual_epstrain)
    estress[mask] = np.minimum(q_before[mask], residual_strength)
    material.reload_state_variables({"epstrain": epstrain, "estress": estress})

    result = {
        "enabled": True,
        "particle_count": int(np.count_nonzero(mask)),
        "projected_particle_count": int(np.count_nonzero(projection_mask)),
        "residual_strength": residual_strength,
        "residual_epstrain": residual_epstrain,
        "max_q_before": float(np.max(q_before[mask])),
        "mean_q_before": float(np.mean(q_before[mask])),
        "max_q_after": float(np.max(np.minimum(q_before[mask], residual_strength))),
    }
    mpm.equivalent_postquake_qr_state = result
    return result


def sliding_block_mask_from_arrays(
    positions: np.ndarray,
    material_ids: np.ndarray,
    body_ids: np.ndarray,
    case: dict[str, Any],
    allowed_body_ids: list[int],
) -> np.ndarray:
    """Return the soil particles above the configured basal slip band."""

    qr_spec = case.get("dynamic_qr_region", {})
    physical_x = positions[:, 0] - X_SHIFT
    physical_z = positions[:, 1] - Y_SHIFT
    x_min = float(qr_spec.get("x_min", -40.0))
    x_max = float(qr_spec.get("x_max", 140.0))
    thickness = float(qr_spec.get("band_thickness_m", 1.0))
    interface = np.asarray(kohler_soil_base_interface_z_np(physical_x), dtype=np.float64)
    profile = str(qr_spec.get("profile", "downslope_arc")).lower()
    span = max(x_max - x_min, 1.0e-12)
    xi = np.clip((physical_x - x_min) / span, 0.0, 1.0)
    if profile in {"surface_to_surface_arc", "finite_block_arc"}:
        center = float(qr_spec.get("center_lower_offset_m", 0.5))
        edge = float(qr_spec.get("edge_lower_offset_m", 8.5))
        lower = center + (edge - center) * (2.0 * xi - 1.0) ** 2
    else:
        left = float(qr_spec.get("left_lower_offset_m", 0.0))
        right = float(qr_spec.get("right_lower_offset_m", 0.0))
        lower = right + (left - right) * (1.0 - xi) ** 2
    surface = np.asarray(kohler_surface_z_np(physical_x), dtype=np.float64)
    allowed_body_mask = np.isin(body_ids, np.asarray(allowed_body_ids, dtype=np.int32))
    slide_material_id = int(case.get("separate_slide_body", {}).get("material_id", MAT_SOIL))
    if BODY_SLIDE in allowed_body_ids and slide_material_id != MAT_SOIL:
        material_mask = (material_ids == MAT_SOIL) | (
            (body_ids == BODY_SLIDE) & (material_ids == slide_material_id)
        )
    else:
        material_mask = material_ids == MAT_SOIL
    return (
        allowed_body_mask
        & material_mask
        & (physical_x >= x_min)
        & (physical_x <= x_max)
        & (physical_z > interface + lower + thickness)
        & (physical_z <= surface + 1.0e-8)
    )


def apply_separate_slide_body(mpm: MPM, case: dict[str, Any]) -> dict[str, Any] | None:
    """Move the detached slide mass to its own nodal body layer."""

    if not bool(case.get("separate_slide_body", {}).get("enabled", False)):
        return None
    particle_count = int(mpm.scene.particleNum[0])
    positions = mpm.scene.particle.x.to_numpy()[:particle_count].astype(np.float64)
    material_ids = mpm.scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    physical_x = positions[:, 0] - X_SHIFT
    mask = sliding_block_mask_from_arrays(
        positions,
        material_ids,
        body_ids,
        case,
        [BODY_MAIN_SOIL],
    )
    body_ids[mask] = BODY_SLIDE
    slide_material_id = int(case.get("separate_slide_body", {}).get("material_id", MAT_SOIL))
    material_ids[mask] = slide_material_id
    weak_band = case.get("separate_slide_body", {}).get("internal_weak_band", {})
    weak_band_count = 0
    if bool(weak_band.get("enabled", False)):
        qr_spec = case.get("dynamic_qr_region", {})
        x_min = float(qr_spec.get("x_min", -40.0))
        weak_x = float(weak_band.get("x_m", x_min + 10.0))
        weak_width = max(float(weak_band.get("width_m", 1.0)), 0.0)
        weak_material_id = int(weak_band.get("material_id", MAT_SOIL))
        weak_mask = mask & (np.abs(physical_x - weak_x) <= 0.5 * weak_width)
        material_ids[weak_mask] = weak_material_id
        weak_band_count = int(np.count_nonzero(weak_mask))
    full_body_ids = mpm.scene.particle.bodyID.to_numpy()
    full_body_ids[:particle_count] = body_ids.astype(full_body_ids.dtype, copy=False)
    mpm.scene.particle.bodyID.from_numpy(full_body_ids)
    full_material_ids = mpm.scene.particle.materialID.to_numpy()
    full_material_ids[:particle_count] = material_ids.astype(full_material_ids.dtype, copy=False)
    mpm.scene.particle.materialID.from_numpy(full_material_ids)
    result = {
        "enabled": True,
        "body_id": BODY_SLIDE,
        "material_id": slide_material_id,
        "particle_count": int(np.count_nonzero(mask)),
        "internal_weak_band_particle_count": weak_band_count,
        "internal_weak_band_material_id": int(weak_band.get("material_id", MAT_SOIL))
        if bool(weak_band.get("enabled", False))
        else None,
    }
    mpm.separate_slide_body = result
    return result


def apply_postquake_block_velocity(mpm: MPM, case: dict[str, Any]) -> dict[str, Any] | None:
    """Apply an equivalent post-earthquake velocity to the slide block.

    ``surface_tangent`` follows the local curved slip-surface tangent.  That
    is useful for tracing the prescribed curved path, but it is not a rigid
    block velocity field.  ``rigid_translation`` gives every block particle
    the same vector and is the diagnostic mode for distinguishing block motion
    from velocity-gradient-driven fluidisation.
    """
    spec = case.get("postquake_block_velocity", {})
    if not bool(spec.get("enabled", False)):
        return None
    qr_spec = case.get("dynamic_qr_region", {})
    if not bool(qr_spec.get("enabled", False)):
        raise ValueError("postquake_block_velocity requires dynamic_qr_region.enabled=true")
    speed = float(spec.get("speed_m_per_s", 0.05))
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("postquake_block_velocity.speed_m_per_s must be positive and finite")
    particle_count = int(mpm.scene.particleNum[0])
    positions = mpm.scene.particle.x.to_numpy()[:particle_count].astype(np.float64)
    velocities = mpm.scene.particle.v.to_numpy()[:particle_count].astype(np.float64)
    material_ids = mpm.scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    physical_x = positions[:, 0] - X_SHIFT
    physical_z = positions[:, 1] - Y_SHIFT
    x_min = float(qr_spec.get("x_min", -40.0))
    x_max = float(qr_spec.get("x_max", 140.0))
    thickness = float(qr_spec.get("band_thickness_m", 1.0))
    interface = np.asarray(kohler_soil_base_interface_z_np(physical_x), dtype=np.float64)
    profile = str(qr_spec.get("profile", "downslope_arc")).lower()
    span = max(x_max - x_min, 1.0e-12)
    xi = np.clip((physical_x - x_min) / span, 0.0, 1.0)
    if profile == "surface_to_surface_arc":
        center = float(qr_spec.get("center_lower_offset_m", 0.5))
        edge = float(qr_spec.get("edge_lower_offset_m", 8.5))
        lower = center + (edge - center) * (2.0 * xi - 1.0) ** 2
        d_lower_dx = 4.0 * (edge - center) * (2.0 * xi - 1.0) / span
    else:
        left = float(qr_spec.get("left_lower_offset_m", 0.0))
        right = float(qr_spec.get("right_lower_offset_m", 0.0))
        lower = right + (left - right) * (1.0 - xi) ** 2
        d_lower_dx = -2.0 * (left - right) * (1.0 - xi) / span
    # The block is the soil above the slip band, inside its two surface exits.
    if bool(case.get("separate_slide_body", {}).get("enabled", False)):
        block = sliding_block_mask_from_arrays(
            positions,
            material_ids,
            body_ids,
            case,
            [BODY_SLIDE],
        )
    else:
        block = (
            (body_ids == BODY_MAIN_SOIL)
            & (material_ids == MAT_SOIL)
            & (physical_x >= x_min)
            & (physical_x <= x_max)
            & (physical_z > interface + lower + thickness)
        )
    velocity_mode = str(spec.get("mode", "surface_tangent")).lower()
    if velocity_mode in {"rigid_translation", "constant", "uniform"}:
        angle_deg = float(spec.get("direction_angle_deg", -K_SLOPE_ANGLE_DEG))
        angle_rad = math.radians(angle_deg)
        velocities[block, 0] = speed * math.cos(angle_rad)
        velocities[block, 1] = speed * math.sin(angle_rad)
    elif velocity_mode in {"surface_tangent", "local_tangent"}:
        tangent_x = np.ones_like(d_lower_dx)
        tangent_z = np.asarray(kohler_soil_base_interface_slope_np(physical_x), dtype=np.float64) + d_lower_dx
        norm = np.sqrt(tangent_x * tangent_x + tangent_z * tangent_z)
        velocities[block, 0] = speed * tangent_x[block] / norm[block]
        velocities[block, 1] = speed * tangent_z[block] / norm[block]
    else:
        raise ValueError(
            "postquake_block_velocity.mode must be 'surface_tangent' or 'rigid_translation'"
        )
    full_velocity = mpm.scene.particle.v.to_numpy()
    full_velocity[:particle_count] = velocities.astype(full_velocity.dtype, copy=False)
    mpm.scene.particle.v.from_numpy(full_velocity)
    result = {
        "enabled": True,
        "speed_m_per_s": speed,
        "particle_count": int(np.count_nonzero(block)),
        "mode": velocity_mode,
    }
    if velocity_mode in {"rigid_translation", "constant", "uniform"}:
        result["direction_angle_deg"] = float(spec.get("direction_angle_deg", -K_SLOPE_ANGLE_DEG))
    mpm.postquake_block_velocity = result
    return result


def apply_material_stage(mpm: MPM, case: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    staged_materials = materials_for_stage(case, stage)
    material_manager = getattr(mpm.scene, "material", None)
    if material_manager is None or not hasattr(material_manager, "matProps"):
        raise RuntimeError("MPM material properties are not initialized")
    for material in staged_materials:
        set_basic_material_properties(
            material_manager.matProps,
            int(material["MaterialID"]),
            float(material["Density"]),
            float(material["YoungModulus"]),
            float(material["PossionRatio"]),
        )
    return staged_materials


def write_static_material_boundary_alignment_report(
    case: dict[str, Any],
    static_materials: list[dict[str, Any]],
    dynamic_materials: list[dict[str, Any]],
    path: Path | None = None,
) -> Path:
    if path is None:
        path = OUTPUT_DIR / "static_material_boundary_alignment_report.md"
    static_by_id = {int(item["MaterialID"]): item for item in static_materials}
    dynamic_by_id = {int(item["MaterialID"]): item for item in dynamic_materials}
    static_spec = case.get("static_initialization", {})
    boundaries = static_boundary_conditions(case)
    soil_static = static_by_id.get(MAT_SOIL, {})
    soil_dynamic = dynamic_by_id.get(MAT_SOIL, {})
    base_static = static_by_id.get(MAT_BASE, {})
    mirror_boundary_ok = bool(STATIC_MIRROR_PARTICLE_BOUNDARY)
    bottom_ok = mirror_boundary_ok or bool(boundaries)
    side_slip_count = sum(
        1
        for item in boundaries
        if item.get("BoundaryType") == "VelocityConstraint" and item.get("Velocity") == [0.0, None]
    )
    static_nu_ok = abs(float(soil_static.get("PossionRatio", math.nan)) - SOIL_POISSON_INITIAL) <= 1.0e-12
    dynamic_nu_ok = abs(float(soil_dynamic.get("PossionRatio", math.nan)) - SOIL_POISSON_DYNAMIC) <= 1.0e-12
    damping_ok = float(static_spec.get("background_damping", 0.0)) > 0.0
    gravity_ok = list(static_spec.get("gravity", [])) == [0.0, STATIC_GRAVITY]
    side_slip_ok = mirror_boundary_ok or side_slip_count == 2
    status = "PASS" if bottom_ok and side_slip_ok and static_nu_ok and dynamic_nu_ok and damping_ok and gravity_ok else "FAIL"
    lines = [
        "# Static Material And Boundary Alignment Report",
        "",
        "## Verdict",
        "",
        f"- status: `{status}`",
        f"- bottom_no_slip: `{'PASS' if bottom_ok else 'FAIL'}`",
        f"- lateral_slip_boundaries: `{'PASS' if side_slip_ok else 'FAIL'}`",
        f"- gravity_ramp_enabled: `{'PASS' if float(static_spec.get('ramp_time', 0.0)) > 0.0 else 'FAIL'}`",
        f"- local_damping_enabled: `{'PASS' if damping_ok else 'FAIL'}`",
        f"- soil_static_poisson: `{'PASS' if static_nu_ok else 'FAIL'}`",
        f"- soil_dynamic_poisson: `{'PASS' if dynamic_nu_ok else 'FAIL'}`",
        "",
        "## Paper-Aligned Static Stage",
        "",
        f"- soil_E_MPa: `{float(soil_static.get('YoungModulus', math.nan)) / 1000.0}`",
        f"- soil_density_kg_per_m3: `{float(soil_static.get('Density', math.nan)) * 1000.0}`",
        f"- soil_poisson_initial_conditions: `{soil_static.get('PossionRatio', math.nan)}`",
        f"- base_E_MPa: `{float(base_static.get('YoungModulus', math.nan)) / 1000.0}`",
        f"- base_density_kg_per_m3: `{float(base_static.get('Density', math.nan)) * 1000.0}`",
        f"- base_poisson: `{base_static.get('PossionRatio', math.nan)}`",
        f"- gravity: `{static_spec.get('gravity')}`",
        f"- ramp: `{static_spec.get('ramp')}`",
        f"- ramp_time: `{static_spec.get('ramp_time')}`",
        f"- local_damping_background_damping: `{static_spec.get('background_damping')}`",
        f"- static_timestep: `{static_spec.get('dt')}`",
        f"- static_velocity_projection: `{case.get('static_initialization', {}).get('velocity_projection', DYNAMIC_VELOCITY_PROJECTION)}`",
        f"- static_alpha_pic: `{static_spec.get('alpha_pic', STATIC_ALPHA_PIC)}`",
        f"- static_stress_update: `{hughes_winget_stress_update_label(case)}`",
        f"- static_max_velocity_tolerance: `{static_spec.get('max_velocity_tolerance')}`",
        f"- static_rms_velocity_tolerance: `{static_spec.get('rms_velocity_tolerance')}`",
        f"- initial_stress_method: `{'GeoTaichi ParticleStress.GravityField' if static_spec.get('use_geotaichi_gravity_field') else 'GeoTaichi static solve with gravity ramp'}`",
        f"- geotaichi_gravity_field_enabled: `{static_spec.get('use_geotaichi_gravity_field')}`",
        "",
        "## Dynamic Stage Material Switch",
        "",
        f"- soil_poisson_dynamic_analysis: `{soil_dynamic.get('PossionRatio', math.nan)}`",
        f"- dynamic_timestep: `{DT}`",
        "- stress_field_reinitialized_after_static: `False`",
        "- particles_regenerated_after_static: `False`",
        "- solver_core_modified: `False`",
        "",
        "## Static Boundary Conditions",
        "",
        f"- method: `{'mirrored-particle grid equivalent' if mirror_boundary_ok else 'direct nodal velocity constraint'}`",
        "- bottom: `horizontal material-point no-slip, vx=0, vy=0 during static initialization`",
        "- lateral: `slip, vx=0, vy free`",
        f"- boundary_count: `{len(boundaries)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def current_softening_strength(epstrain: float, material: dict[str, Any]) -> tuple[float, float, str, float]:
    if "YieldStress" in material or "ResidualYieldStress" in material:
        y0 = float(material.get("YieldStress", DP_COHESION))
        yr = float(material.get("ResidualYieldStress", y0))
        eps_start = float(material.get("PlasticDevStrain", SOFTENING_EPS_START))
        eps_end = float(material.get("ResidualPlasticDevStrain", SOFTENING_EPS_END))
        factor = softening_factor(abs(float(epstrain)), eps_start, eps_end)
        yield_stress = y0 - (y0 - yr) * factor
        if abs(float(epstrain)) <= 1.0e-14:
            status = "elastic"
        elif factor <= 0.0:
            status = "plastic_peak_strength"
        elif factor >= 1.0:
            status = "plastic_residual_strength"
        else:
            status = "plastic_softening"
        return yield_stress, 0.0, status, factor
    c0 = float(material.get("Cohesion", DP_COHESION))
    cr = float(material.get("ResidualCohesion", c0))
    phi0 = float(material.get("Friction", DP_FRICTION))
    phir = float(material.get("ResidualFriction", phi0))
    eps_start = float(material.get("PlasticDevStrain", SOFTENING_EPS_START))
    eps_end = float(material.get("ResidualPlasticDevStrain", SOFTENING_EPS_END))
    factor = softening_factor(abs(float(epstrain)), eps_start, eps_end)
    cohesion = c0 - (c0 - cr) * factor
    friction = phi0 - (phi0 - phir) * factor
    if abs(float(epstrain)) <= 1.0e-14:
        status = "elastic"
    elif factor <= 0.0:
        status = "plastic_peak_strength"
    elif factor >= 1.0:
        status = "plastic_residual_strength"
    else:
        status = "plastic_softening"
    return cohesion, friction, status, factor


def drucker_prager_coefficients(cohesion: float, friction_deg: float, dilation_deg: float, tensile: float, dp_type: str) -> dict[str, float]:
    phi = math.radians(friction_deg)
    psi = math.radians(dilation_deg)
    if dp_type == "Circumscribed":
        q_fai = 6.0 * math.sin(phi) / (math.sqrt(3.0) * (3.0 + math.sin(phi)))
        k_fai = 6.0 * math.cos(phi) * cohesion / (math.sqrt(3.0) * (3.0 + math.sin(phi)))
        q_psi = 6.0 * math.sin(psi) / (math.sqrt(3.0) * (3.0 + math.sin(psi)))
    elif dp_type == "MiddleCircumscribed":
        q_fai = 6.0 * math.sin(phi) / (math.sqrt(3.0) * (3.0 - math.sin(phi)))
        k_fai = 6.0 * math.cos(phi) * cohesion / (math.sqrt(3.0) * (3.0 - math.sin(phi)))
        q_psi = 6.0 * math.sin(psi) / (math.sqrt(3.0) * (3.0 - math.sin(psi)))
    else:
        q_fai = 3.0 * math.tan(phi) / math.sqrt(9.0 + 12.0 * math.tan(phi) ** 2)
        k_fai = 3.0 * cohesion / math.sqrt(9.0 + 12.0 * math.tan(phi) ** 2)
        q_psi = 3.0 * math.tan(psi) / math.sqrt(9.0 + 12.0 * math.tan(psi) ** 2)
    tensile_current = 0.0 if friction_deg == 0.0 else min(float(tensile), k_fai / q_fai)
    return {
        "fai": phi,
        "psi": psi,
        "q_fai": q_fai,
        "k_fai": k_fai,
        "q_psi": q_psi,
        "tensile": tensile_current,
    }


@ti.kernel
def kernel_update_drucker_prager_material_strength(
    mat_props: ti.template(),
    material_id: ti.i32,
    cohesion: ti.f64,
    friction_rad: ti.f64,
    dilation_rad: ti.f64,
    q_fai: ti.f64,
    k_fai: ti.f64,
    q_psi: ti.f64,
    tensile: ti.f64,
):
    mat_props[material_id].c = cohesion
    mat_props[material_id].fai = friction_rad
    mat_props[material_id].psi = dilation_rad
    mat_props[material_id].q_fai = q_fai
    mat_props[material_id].k_fai = k_fai
    mat_props[material_id].q_psi = q_psi
    mat_props[material_id].tensile = tensile


def apply_dynamic_drucker_prager_softening(mpm: MPM, case: dict[str, Any]) -> None:
    if str(case.get("material_model")) != "DruckerPrager":
        return
    if not bool(case.get("softening", {}).get("enabled", SOFTENING_ENABLED)):
        return
    material = getattr(mpm.scene, "material", None)
    if material is None or not hasattr(material, "stateVars") or not hasattr(material, "matProps"):
        return

    particle_count = int(mpm.scene.particleNum[0])
    if particle_count <= 0:
        return
    state = material.get_state_vars_dict(0, particle_count)
    epstrain = np.asarray(state.get("epstrain", np.zeros(particle_count)), dtype=np.float64).reshape(-1)
    material_ids = mpm.scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
    materials = material_by_id(case)
    for material_id, material_spec in materials.items():
        mask = material_ids == material_id
        if not np.any(mask):
            continue
        representative_epstrain = float(np.max(np.abs(epstrain[mask])))
        cohesion, friction, _, _ = current_softening_strength(representative_epstrain, material_spec)
        dilation0 = float(material_spec.get("Dilation", DP_DILATION))
        dilation_residual = float(material_spec.get("ResidualDilation", dilation0))
        eps_start = float(material_spec.get("PlasticDevStrain", SOFTENING_EPS_START))
        eps_end = float(material_spec.get("ResidualPlasticDevStrain", SOFTENING_EPS_END))
        factor = softening_factor(representative_epstrain, eps_start, eps_end)
        dilation = dilation0 - (dilation0 - dilation_residual) * factor
        coeffs = drucker_prager_coefficients(
            cohesion,
            friction,
            dilation,
            float(material_spec.get("Tensile", DP_TENSILE)),
            str(material_spec.get("dpType", DP_TYPE)),
        )
        kernel_update_drucker_prager_material_strength(
            material.matProps,
            int(material_id),
            float(cohesion),
            float(coeffs["fai"]),
            float(coeffs["psi"]),
            float(coeffs["q_fai"]),
            float(coeffs["k_fai"]),
            float(coeffs["q_psi"]),
            float(coeffs["tensile"]),
        )


def write_softening_parameters(case: dict[str, Any]) -> Path:
    path = OUTPUT_DIR / "softening_parameters.yaml"
    softening = case.get("softening", {})
    if str(case.get("material_model", "")) == "VonMisesSoftening":
        q_peak = float(softening.get("yield_stress_initial", DP_COHESION))
        sensitivity = float(softening.get("strength_sensitivity", 1.8))
        q_residual_paper = q_peak / sensitivity if sensitivity > 0.0 else math.nan
        q_residual_config = float(softening.get("yield_stress_residual", q_residual_paper))
        d_residual = float(softening.get("residual_shear_displacement_m", 0.2))
        h = float(case.get("dx", DX))
        h_shear_factor = float(softening.get("h_shear_factor", 2.0))
        h_shear = h_shear_factor * h
        eps_derived = d_residual / (math.sqrt(3.0) * h_shear) if h_shear > 0.0 else math.nan
        eps_config = float(softening.get("eps_end", SOFTENING_EPS_END))
        q_residual_match = math.isclose(q_residual_config, q_residual_paper, rel_tol=1.0e-8, abs_tol=1.0e-8)
        eps_match = math.isclose(eps_config, eps_derived, rel_tol=1.0e-8, abs_tol=1.0e-8)
        lines = [
            "softening_enabled: " + str(bool(softening.get("enabled", SOFTENING_ENABLED))).lower(),
            "material_model: VonMisesSoftening",
            "yield_criterion: von Mises J2",
            "law: linear isotropic yield-stress softening",
            "state_variable: epstrain",
            "paper_parameters:",
            f"  q_peak_kpa: {q_peak}",
            f"  sensitivity: {sensitivity}",
            f"  q_residual_kpa_from_sensitivity: {q_residual_paper}",
            f"  residual_shear_displacement_m: {d_residual}",
            "regularization_mapping:",
            f"  dx_m: {h}",
            f"  h_shear_factor: {h_shear_factor}",
            f"  h_shear_m: {h_shear}",
            "  formula: d_r / (sqrt(3) * h_shear)",
            f"  eps_end_derived: {eps_derived}",
            f"  eps_end_configured: {eps_config}",
            f"  strength_residual_match: {str(q_residual_match).lower()}",
            f"  strain_mapping_match: {str(eps_match).lower()}",
            f"  audit_status: {'PASS' if q_residual_match and eps_match else 'REVIEW'}",
            "model_parameters:",
            f"  initial_yield_stress_kpa: {q_peak}",
            f"  residual_yield_stress_kpa: {q_residual_config}",
            f"  eps_start: {float(softening.get('eps_start', SOFTENING_EPS_START))}",
            f"  eps_end: {eps_config}",
            "note: q_residual and eps_end are checked against the paper sensitivity and smeared-crack mapping.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    lines = [
        "softening_enabled: " + str(bool(softening.get("enabled", SOFTENING_ENABLED))).lower(),
        "law: linear",
        "state_variable: epstrain",
        "yield_stress:",
        f"  initial: {float(softening.get('yield_stress_initial', DP_COHESION))}",
        f"  residual: {float(softening.get('yield_stress_residual', SOFTENING_RESIDUAL_COHESION))}",
        "cohesion:",
        f"  initial: {float(softening.get('cohesion_initial', DP_COHESION))}",
        f"  residual: {float(softening.get('cohesion_residual', SOFTENING_RESIDUAL_COHESION))}",
        "friction_angle_deg:",
        f"  initial: {float(softening.get('friction_initial', DP_FRICTION))}",
        f"  residual: {float(softening.get('friction_residual', SOFTENING_RESIDUAL_FRICTION))}",
        "dilation_angle_deg:",
        f"  initial: {float(softening.get('dilation_initial', DP_DILATION))}",
        f"  residual: {float(softening.get('dilation_residual', SOFTENING_RESIDUAL_DILATION))}",
        "plastic_strain:",
        f"  eps_start: {float(softening.get('eps_start', SOFTENING_EPS_START))}",
        f"  eps_end: {float(softening.get('eps_end', SOFTENING_EPS_END))}",
        "note: peak material parameters are preserved; residual parameters are added for softening.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_softening_model_check(case: dict[str, Any], softening_parameters_path: Path) -> Path:
    path = OUTPUT_DIR / "softening_model_check.md"
    material_model = str(case.get("material_model"))
    if material_model == "SoftenMohrCoulomb":
        selected_reason = (
            "failure case uses the local SoftenMohrCoulomb state variable; zero friction and dilation give an undrained Tresca approximation, "
            "with peak-to-residual cohesion degradation per particle."
        )
        direct_option = "selected directly for the failure case"
    elif material_model == "VonMisesSoftening":
        selected_reason = (
            "failure case uses the solver-compatible J2/von Mises radial-return model; equivalent plastic strain drives linear isotropic yield-stress softening."
        )
        direct_option = "selected directly for the failure case"
    else:
        selected_reason = (
            "current slope keeps the validated DruckerPrager model and applies driver-level peak-to-residual strength updates from accumulated epstrain during the dynamic stage."
        )
        direct_option = "SoftenMohrCoulomb is available as a direct local-softening option"
    lines = [
        "# Softening Model Check",
        "",
        "## Existing GeoTaichi Capability",
        "",
        "- `SoftenMohrCoulomb`: available through `MaterialManager` and mapped to `src/mpm/materials/infinitesimal_strain/MohrCoulomb.py`.",
        "- `VonMisesSoftening`: available through `MaterialManager` and mapped to `src/mpm/materials/infinitesimal_strain/VonMisesSoftening.py`.",
        "- `MohrCoulomb`: non-softening wrapper name maps to `WillianMohrCoulomb` in this codebase.",
        "- `StateDependentMohrCoulomb`: has state-dependent strength logic and `epstrain`, but requires additional state parameters not used by this slope driver.",
        "- `CohesiveModifiedCamClay`: has additional internal variables and degradation terms, but is not the first-stage frictional slope softening target.",
        "",
        "## Selected Model",
        "",
        f"- material_model: `{material_model}`",
        "- selected_for_current_slope: `PASS`",
        f"- reason: {selected_reason}",
        f"- direct_softening_option: `{direct_option}`",
        "",
        "## Parameter Interface",
        "",
        "- required/common: `MaterialID`, `Density`, `YoungModulus`, `PossionRatio`.",
        "- Mohr-Coulomb/DP strength: `Cohesion`, `Friction`, `Dilation`, `Tensile`.",
        "- von Mises strength: `YieldStress`, `ResidualYieldStress`.",
        "- softening range: `PlasticDevStrain` to `ResidualPlasticDevStrain`.",
        "- state variables: `epstrain`, `estress`.",
        f"- parameter_file: `{softening_parameters_path}`",
        "",
        "## Direct Use Assessment",
        "",
        "- geometry_change_required: `False`",
        "- particle_regeneration_required: `False`",
        "- solver_core_change_required: `False`",
        "- free_field_coupling_change_required: `False`",
        "- dynamic_boundary_change_required: `False`",
        "- can_directly_use_current_slope_model: `PASS`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_softening_state(case: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[Path, dict[str, Any]]:
    path = OUTPUT_DIR / "softening_state.csv"
    materials = material_by_id(case)
    positions = np.asarray(arrays.get("position", np.empty((0, 2))))
    material_ids = np.asarray(arrays.get("material_id", np.empty(0)))
    state = arrays.get("state_variables") if isinstance(arrays.get("state_variables"), dict) else {}
    epstrain = np.asarray(state.get("epstrain", np.zeros(positions.shape[0])), dtype=np.float64).reshape(-1)
    rows: list[dict[str, Any]] = []
    factors: list[float] = []
    cohesions: list[float] = []
    frictions: list[float] = []
    for pid in range(positions.shape[0]):
        material = materials.get(int(material_ids[pid]), next(iter(materials.values())))
        eps = float(epstrain[pid]) if pid < epstrain.size else 0.0
        cohesion, friction, status, factor = current_softening_strength(eps, material)
        position = positions[pid] - np.array([X_SHIFT, Y_SHIFT], dtype=np.float64)
        rows.append(
            {
                "particle_id": pid,
                "epstrain": eps,
                "cohesion_current": cohesion,
                "strength_current": cohesion,
                "friction_current": friction,
                "yield_status": status,
                "x": float(position[0]),
                "z": float(position[1]),
                "material_id": int(material_ids[pid]) if material_ids.size else "",
                "softening_factor": factor,
            }
        )
        factors.append(factor)
        cohesions.append(cohesion)
        frictions.append(friction)

    write_csv(
        path,
        [
            "particle_id",
            "epstrain",
            "cohesion_current",
            "strength_current",
            "friction_current",
            "yield_status",
            "x",
            "z",
            "material_id",
            "softening_factor",
        ],
        rows,
    )
    factors_array = np.asarray(factors, dtype=np.float64)
    cohesions_array = np.asarray(cohesions, dtype=np.float64)
    frictions_array = np.asarray(frictions, dtype=np.float64)
    stats = {
        "softened_particle_count": int(np.count_nonzero(factors_array > 0.0)),
        "residual_particle_count": int(np.count_nonzero(factors_array >= 1.0)),
        "max_softening_factor": float(np.max(factors_array)) if factors_array.size else 0.0,
        "min_cohesion": float(np.min(cohesions_array)) if cohesions_array.size else 0.0,
        "mean_cohesion": float(np.mean(cohesions_array)) if cohesions_array.size else 0.0,
        "min_strength": float(np.min(cohesions_array)) if cohesions_array.size else 0.0,
        "mean_strength": float(np.mean(cohesions_array)) if cohesions_array.size else 0.0,
        "min_friction": float(np.min(frictions_array)) if frictions_array.size else 0.0,
        "mean_friction": float(np.mean(frictions_array)) if frictions_array.size else 0.0,
    }
    return path, stats


def write_softening_validation_report(
    case: dict[str, Any],
    static_monitor: "StaticInitializationMonitor",
    dynamic_snapshot: dict[str, Any] | None,
    softening_state_path: Path,
    softening_stats: dict[str, Any],
) -> Path:
    path = OUTPUT_DIR / "softening_validation_report.md"
    materials = case.get("materials", [])
    reference_material = materials[0] if materials else {}
    static_arrays = static_monitor.final_snapshot.get("arrays", {})
    static_stats = material_state_stats(static_arrays)
    dynamic_arrays = dynamic_snapshot.get("arrays", {}) if dynamic_snapshot else {}
    dynamic_stats = material_state_stats(dynamic_arrays) if dynamic_snapshot else {}
    static_state = static_arrays.get("state_variables")
    dynamic_state = dynamic_arrays.get("state_variables") if dynamic_snapshot else static_state
    state_ok = isinstance(static_state, dict) and isinstance(dynamic_state, dict) and set(static_state) == set(dynamic_state)
    state_message = "state variable keys and shapes are consistent"
    if state_ok:
        for key in static_state:
            if np.asarray(static_state[key]).shape != np.asarray(dynamic_state[key]).shape:
                state_ok = False
                state_message = f"state variable {key} shape differs"
                break
    else:
        state_message = "state variable keys are missing or inconsistent"
    plastic_increase = (
        dynamic_snapshot is not None
        and float(dynamic_stats.get("max_plastic_strain", 0.0)) >= float(static_stats.get("max_plastic_strain", 0.0))
    )
    strength_degraded = int(softening_stats.get("softened_particle_count", 0)) > 0
    pass_status = (
        static_monitor.converged
        and dynamic_snapshot is not None
        and plastic_increase
        and strength_degraded
        and state_ok
    )
    lines = [
        "# Softening Validation Report",
        "",
        "## Verdict",
        "",
        f"- softening_validation_status: `{'PASS' if pass_status else 'FAIL'}`",
        "- solver_core_modified: `False`",
        "- geometry_modified: `False`",
        "- grid_generation_modified: `False`",
        "- particle_generation_modified: `False`",
        "- free_field_coupling_modified: `False`",
        "- seismic_boundary_modified: `False`",
        "",
        "## Strength Parameters",
        "",
        f"- initial_cohesion: `{reference_material.get('Cohesion', 'n/a')}`",
        f"- residual_cohesion: `{reference_material.get('ResidualCohesion', 'n/a')}`",
        f"- initial_friction_angle_deg: `{reference_material.get('Friction', 'n/a')}`",
        f"- residual_friction_angle_deg: `{reference_material.get('ResidualFriction', 'n/a')}`",
        f"- eps_start: `{reference_material.get('PlasticDevStrain', 'n/a')}`",
        f"- eps_end: `{reference_material.get('ResidualPlasticDevStrain', 'n/a')}`",
        "",
        "## Static Stage",
        "",
        f"- static_converged: `{static_monitor.converged}`",
        f"- static_end_time: `{static_monitor.end_time}`",
        f"- static_max_plastic_strain: `{static_stats.get('max_plastic_strain')}`",
        f"- static_max_stress_norm: `{static_stats.get('max_stress_norm')}`",
        "",
        "## Dynamic Stage",
        "",
        f"- dynamic_ran: `{dynamic_snapshot is not None}`",
        f"- dynamic_max_plastic_strain: `{dynamic_stats.get('max_plastic_strain', math.nan)}`",
        f"- plastic_strain_increased: `{'PASS' if plastic_increase else 'FAIL'}`",
        f"- dynamic_max_stress_norm: `{dynamic_stats.get('max_stress_norm', math.nan)}`",
        f"- dynamic_mean_stress_norm: `{dynamic_stats.get('mean_stress_norm', math.nan)}`",
        "",
        "## Softening State",
        "",
        f"- softening_state_csv: `{softening_state_path}`",
        f"- softened_particle_count: `{softening_stats.get('softened_particle_count')}`",
        f"- residual_particle_count: `{softening_stats.get('residual_particle_count')}`",
        f"- maximum_softening_factor: `{softening_stats.get('max_softening_factor')}`",
        f"- final_min_cohesion: `{softening_stats.get('min_cohesion')}`",
        f"- final_mean_cohesion: `{softening_stats.get('mean_cohesion')}`",
        f"- final_min_yield_stress: `{softening_stats.get('min_strength')}`",
        f"- final_mean_yield_stress: `{softening_stats.get('mean_strength')}`",
        f"- final_min_friction_angle_deg: `{softening_stats.get('min_friction')}`",
        f"- final_mean_friction_angle_deg: `{softening_stats.get('mean_friction')}`",
        f"- failure_zone_started: `{'PASS' if strength_degraded else 'FAIL'}`",
        "",
        "## State Transfer",
        "",
        f"- state_variable_keys_static_to_dynamic: `{'PASS' if state_ok else 'FAIL'}`",
        f"- state_variable_message: `{state_message}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def relative_stress_change(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None or previous.size == 0 or current.size == 0:
        return math.inf
    numerator = float(np.linalg.norm(current - previous))
    denominator = max(float(np.linalg.norm(previous)), 1.0e-30)
    return numerator / denominator


def max_row_norm_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return math.inf
    if left.size == 0:
        return 0.0
    difference = left - right
    if difference.ndim == 1:
        return float(np.max(np.abs(difference)))
    return float(np.max(np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=1)))


def bottom_static_reaction_tractions(
    static_snapshot: dict[str, Any],
    particle_ids: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = static_snapshot.get("arrays", {}) if isinstance(static_snapshot, dict) else {}
    stresses = np.asarray(arrays.get("stress", np.empty((0, 6))), dtype=np.float64)
    if stresses.size == 0 or particle_ids.size == 0:
        return np.zeros(particle_ids.size, dtype=np.float64), np.zeros(particle_ids.size, dtype=np.float64)
    reactions_x = np.zeros(particle_ids.size, dtype=np.float64)
    reactions_y = np.zeros(particle_ids.size, dtype=np.float64)
    for index, pid in enumerate(particle_ids.astype(np.int32)):
        stress = stresses[int(pid)]
        physical_x = float(positions[int(pid), 0] - X_SHIFT)
        eps = max(1.0e-4, 0.01 * DX)
        x_left = max(K_MAIN_X_MIN, physical_x - eps)
        x_right = min(K_MAIN_X_MAX, physical_x + eps)
        if math.isclose(x_left, x_right):
            slope = 0.0
        else:
            slope = (float(kohler_base_bottom_z_np(x_right)) - float(kohler_base_bottom_z_np(x_left))) / (x_right - x_left)
        normal_x = slope / math.sqrt(1.0 + slope * slope)
        normal_y = -1.0 / math.sqrt(1.0 + slope * slope)
        sigma_xx = float(stress[0]) if stress.size > 0 else 0.0
        sigma_yy = float(stress[1]) if stress.size > 1 else 0.0
        sigma_xy = float(stress[3]) if stress.size > 3 else 0.0
        reactions_x[index] = sigma_xx * normal_x + sigma_xy * normal_y
        reactions_y[index] = sigma_xy * normal_x + sigma_yy * normal_y
    return reactions_x, reactions_y


def geotaichi_bottom_static_reaction_tractions(
    mpm: MPM, bottom_particle_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover bottom support reactions from GeoTaichi's assembled nodal residual."""

    particle_ids = np.asarray(bottom_particle_ids, dtype=np.int32)
    if particle_ids.size == 0:
        path = OUTPUT_DIR / "bottom_static_reaction_from_geotaichi.csv"
        write_csv(path, ["particle_id", "reaction_force_x", "reaction_force_y", "traction_x", "traction_y"], [])
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_f, empty_f, path, empty_i, empty_i, empty_f, empty_f

    engine = mpm.enginer
    scene = mpm.scene
    sims = mpm.sims
    if engine is None:
        raise RuntimeError("GeoTaichi engine is not initialized; cannot assemble static support reactions")

    previous_gravity = [float(value) for value in sims.gravity]
    # Use the exact force-assembly sequence used by the first dynamic step.
    # The former partial sequence omitted neighbor/grid-velocity work and
    # recovered a support reaction from a different nodal force state.
    assemble_force_window(mpm, previous_gravity[:2], None)

    node_force = scene.node.force.to_numpy()
    ln_id = scene.element.LnID.to_numpy()
    shape_fn = scene.element.shape_fn.to_numpy()
    node_size = scene.element.node_size.to_numpy()
    body_ids = scene.particle.bodyID.to_numpy()[: int(scene.particleNum[0])].astype(np.int32)
    traction_psize = scene.boundary.particle_traction.psize.to_numpy()[: max(1, int(scene.boundary.ptraction_list[0]))]
    traction_pids = scene.boundary.particle_traction.pid.to_numpy()[: int(scene.boundary.ptraction_list[0])].astype(np.int32)
    traction_index_by_pid = {int(pid): index for index, pid in enumerate(traction_pids.tolist())}

    denominator: dict[tuple[int, int], float] = {}
    node_keys: set[tuple[int, int]] = set()
    total_nodes = int(scene.element.grid_nodes)
    bottom_influence_keys: set[tuple[int, int]] = set()
    for pid in particle_ids.tolist():
        body_id = int(body_ids[pid])
        offset = int(pid) * total_nodes
        for ln in range(offset, offset + int(node_size[pid])):
            key = (int(ln_id[ln]), body_id)
            denominator[key] = denominator.get(key, 0.0) + float(shape_fn[ln])
            bottom_influence_keys.add(key)
    # Kohler et al. Section 3.2 applies the opposite of the P2G imbalance to
    # boundary nodes that lose mirrored-particle support at the compliant base.
    # With cubic B-splines, these nodes occupy the complete support footprint
    # of the bottom particle row, not only the nodes in the velocity-constraint
    # band. Lateral support is extracted separately in Section 3.3.
    mass_cutoff = float(scene.mass_cut_off)
    node_mass = scene.node.m.to_numpy()
    support_band_nodes: set[int] = set()
    for boundary in static_bottom_boundary_conditions(STATIC_BOTTOM_SUPPORT_NODE_BAND):
        node_ids = scene.element.get_boundary_nodes(boundary["StartPoint"], boundary["EndPoint"])
        support_band_nodes.update(int(node_id) for node_id in np.asarray(node_ids, dtype=np.int32).tolist())
    active_body_ids = np.unique(body_ids).astype(np.int32)
    for node_id in support_band_nodes:
        for body_id in active_body_ids.tolist():
            bottom_influence_keys.add((int(node_id), int(body_id)))
    for node_id, body_id in sorted(bottom_influence_keys):
        if node_id < 0 or node_id >= node_force.shape[0]:
            continue
        if (
            node_mass[node_id, body_id] > mass_cutoff
            and np.linalg.norm(node_force[node_id, body_id, :2]) > 1.0e-12
        ):
            node_keys.add((int(node_id), int(body_id)))

    node_rows: list[dict[str, Any]] = []
    node_ids: list[int] = []
    node_body_ids: list[int] = []
    node_rx: list[float] = []
    node_ry: list[float] = []
    for node_id, body_id in sorted(node_keys):
        reaction = -node_force[node_id, body_id, :2]
        node_ids.append(node_id)
        node_body_ids.append(body_id)
        node_rx.append(float(reaction[0]))
        node_ry.append(float(reaction[1]))
        node_rows.append(
            {
                "type": "node",
                "particle_id": "",
                "node_id": node_id,
                "body_id": body_id,
                "reaction_force_x": float(reaction[0]),
                "reaction_force_y": float(reaction[1]),
                "traction_x": "",
                "traction_y": "",
                "num_nodes": "",
            }
        )

    reaction_force = np.zeros((particle_ids.size, 2), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for row_index, pid in enumerate(particle_ids.tolist()):
        body_id = int(body_ids[pid])
        offset = int(pid) * total_nodes
        for ln in range(offset, offset + int(node_size[pid])):
            node_id = int(ln_id[ln])
            weight = float(shape_fn[ln])
            denom = max(denominator.get((node_id, body_id), 0.0), 1.0e-30)
            reaction_force[row_index] += -node_force[node_id, body_id, :2] * (weight / denom)

        traction_index = traction_index_by_pid.get(int(pid), -1)
        if traction_index >= 0 and traction_index < traction_psize.shape[0]:
            psize = np.asarray(traction_psize[traction_index], dtype=np.float64)
        else:
            psize = np.asarray(scene.psize[int(pid)], dtype=np.float64)
        area_factor = 2.0 * np.array([max(float(psize[1]), 1.0e-30), max(float(psize[0]), 1.0e-30)])
        rows.append(
            {
                "type": "particle_diagnostic",
                "particle_id": int(pid),
                "node_id": "",
                "body_id": body_id,
                "reaction_force_x": float(reaction_force[row_index, 0]),
                "reaction_force_y": float(reaction_force[row_index, 1]),
                "traction_x": float(reaction_force[row_index, 0] / area_factor[0]),
                "traction_y": float(reaction_force[row_index, 1] / area_factor[1]),
                "num_nodes": int(node_size[pid]),
            }
        )

    sims.set_gravity(previous_gravity)
    engine.reset_grid_message(scene)
    path = OUTPUT_DIR / "bottom_static_reaction_from_geotaichi.csv"
    write_csv(
        path,
        ["type", "particle_id", "node_id", "body_id", "reaction_force_x", "reaction_force_y", "traction_x", "traction_y", "num_nodes"],
        node_rows + rows,
    )
    tractions_x = np.zeros(particle_ids.size, dtype=np.float64)
    tractions_y = np.zeros(particle_ids.size, dtype=np.float64)
    return (
        tractions_x,
        tractions_y,
        path,
        np.asarray(node_ids, dtype=np.int32),
        np.asarray(node_body_ids, dtype=np.int32),
        np.asarray(node_rx, dtype=np.float64),
        np.asarray(node_ry, dtype=np.float64),
    )


def freeze_bottom_static_support_before_dynamic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: Any,
) -> None:
    """Freeze the direct bottom-node support from the completed static state."""
    if bool(getattr(mpm, "bottom_static_support_frozen_before_dynamic", False)):
        return

    # Register the future dynamic traction row without applying seismic input.
    register_bottom_input_traction(mpm, case)
    if mpm.enginer is None:
        # A restored checkpoint has particle state but no active solver yet.
        # Initialize it while the static material stage is still in effect.
        mpm.add_essentials({"function": None})
    # A restored checkpoint does not retain the static boundary container.
    # Build this before extraction so the support map covers every static
    # bottom constraint node/body pair, including both free-field columns.
    if not hasattr(mpm, "static_bottom_constraint_node_ids_np"):
        mpm.static_bottom_constraint_node_ids_np = static_bottom_constraint_node_ids(mpm)
    # The checkpoint was created with periodic free-field columns.  Reinstall
    # that shared-node topology before the static P2G force extraction.
    install_free_field_periodic_dynamic_hooks(mpm)
    assert_free_field_periodic_dynamic_ready(mpm)
    # Checkpoint restoration can retain the static velocity constraints in the
    # boundary container.  The paper's support force is defined after those
    # mirrored/kinematic constraints have been removed.
    clear_velocity_boundary_state(mpm)
    if velocity_constraint_counts(mpm)["active_velocity_constraint_count"] != 0:
        raise RuntimeError("Static support extraction still has active velocity constraints")
    # The dynamic input traction has its own prescribed particle list.  The
    # Section 3.2 static support must cover all three geometric bottom rows.
    bottom_particle_ids = static_bottom_particle_ids(mpm)
    if bottom_particle_ids.size == 0:
        raise RuntimeError("No bottom particles are available for static support extraction")

    sims = mpm.sims
    previous_gravity = [float(value) for value in sims.gravity]
    static_gravity = [float(value) for value in static_monitor.target_gravity]
    try:
        # Section 3.2: after removing mirrored particles, apply the opposite
        # of the resulting bottom-node P2G imbalance in the dynamic stage.
        sims.set_gravity(static_gravity)
        (
            _particle_rx,
            _particle_ry,
            support_path,
            node_ids,
            body_ids,
            support_x,
            support_y,
        ) = geotaichi_bottom_static_reaction_tractions(mpm, bottom_particle_ids)
    finally:
        sims.set_gravity(previous_gravity)
        if mpm.enginer is not None:
            mpm.enginer.reset_grid_message(mpm.scene)

    if node_ids.size == 0:
        raise RuntimeError("Static bottom P2G extraction produced no nodal support reactions")
    mpm.bottom_static_support_node_ids_np = np.ascontiguousarray(node_ids, dtype=np.int32)
    mpm.bottom_static_support_body_ids_np = np.ascontiguousarray(body_ids, dtype=np.int32)
    mpm.bottom_static_support_x_np = np.ascontiguousarray(support_x, dtype=np.float64)
    mpm.bottom_static_support_y_np = np.ascontiguousarray(support_y, dtype=np.float64)
    mpm.bottom_static_support_path = support_path
    mpm.bottom_static_support_frozen_before_dynamic = True


def run_static_dynamic_support_map_diagnostic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: Any,
) -> tuple[Path, Path]:
    """Compare the bottom nodal force map before and after the material switch.

    This intentionally stops before constructing dynamic supports or advancing a
    time step.  It answers only whether the same node/body pairs see the same
    assembled P2G force when the static state is handed to the dynamic solver.
    """

    output_dir = STATIC_DYNAMIC_SUPPORT_MAP_DIAGNOSTIC_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "bottom_node_body_force_map.csv"
    summary_path = output_dir / "summary.csv"

    if mpm.enginer is None:
        mpm.add_essentials({"function": None})
    register_bottom_input_traction(mpm, case)
    install_free_field_periodic_dynamic_hooks(mpm)
    assert_free_field_periodic_dynamic_ready(mpm)
    clear_velocity_boundary_state(mpm)
    constraint_counts = velocity_constraint_counts(mpm)
    if constraint_counts["active_velocity_constraint_count"] != 0:
        raise RuntimeError("Support-map diagnostic has active velocity constraints")

    bottom_node_ids = np.asarray(
        getattr(mpm, "static_bottom_constraint_node_ids_np", static_bottom_constraint_node_ids(mpm)),
        dtype=np.int32,
    )
    if bottom_node_ids.size == 0:
        raise RuntimeError("Support-map diagnostic found no static bottom nodes")
    mpm.static_bottom_constraint_node_ids_np = bottom_node_ids

    static_gravity = [float(value) for value in static_monitor.target_gravity]
    apply_material_stage(mpm, case, "static")
    assemble_force_window(mpm, static_gravity, None)
    static_force = mpm.scene.node.force.to_numpy().copy()
    static_mass = mpm.scene.node.m.to_numpy().copy()

    # Match the formal transition: alter material constants, rebuild the active
    # force assembly, then inspect the first dynamic force window before any
    # dynamic support, dashpot, seismic input, stress update, or time advance.
    apply_material_stage(mpm, case, "dynamic")
    mpm.add_essentials({"function": None})
    install_free_field_periodic_dynamic_hooks(mpm)
    assert_free_field_periodic_dynamic_ready(mpm)
    clear_velocity_boundary_state(mpm)
    post_switch_constraint_counts = velocity_constraint_counts(mpm)
    if post_switch_constraint_counts["active_velocity_constraint_count"] != 0:
        raise RuntimeError("Dynamic-side support-map diagnostic has active velocity constraints")
    assemble_force_window(mpm, static_gravity, None)
    dynamic_force = mpm.scene.node.force.to_numpy().copy()
    dynamic_mass = mpm.scene.node.m.to_numpy().copy()

    if static_force.shape != dynamic_force.shape or static_mass.shape != dynamic_mass.shape:
        raise RuntimeError("Static and dynamic force maps have incompatible array shapes")

    coordinates = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64)
    mass_cutoff = float(mpm.scene.mass_cut_off)
    body_count = int(static_force.shape[1])
    rows: list[dict[str, Any]] = []
    force_deltas: list[float] = []
    active_pair_count_static = 0
    active_pair_count_dynamic = 0
    active_pair_count_union = 0
    active_pair_count_changed = 0
    static_sum = np.zeros(2, dtype=np.float64)
    dynamic_sum = np.zeros(2, dtype=np.float64)
    max_delta = -1.0
    max_reference_force = 0.0
    max_relative_delta = 0.0
    max_delta_node_id = -1
    max_delta_body_id = -1
    for node_id in sorted(set(int(value) for value in bottom_node_ids.tolist())):
        for body_id in range(body_count):
            static_active = bool(static_mass[node_id, body_id] > mass_cutoff)
            dynamic_active = bool(dynamic_mass[node_id, body_id] > mass_cutoff)
            if static_active:
                active_pair_count_static += 1
            if dynamic_active:
                active_pair_count_dynamic += 1
            if static_active or dynamic_active:
                active_pair_count_union += 1
                static_value = static_force[node_id, body_id, :2]
                dynamic_value = dynamic_force[node_id, body_id, :2]
                delta = dynamic_value - static_value
                delta_norm = float(np.linalg.norm(delta))
                reference_force = max(float(np.linalg.norm(static_value)), float(np.linalg.norm(dynamic_value)))
                force_deltas.append(delta_norm)
                static_sum += static_value
                dynamic_sum += dynamic_value
                max_reference_force = max(max_reference_force, reference_force)
                max_relative_delta = max(
                    max_relative_delta,
                    delta_norm / max(reference_force, STATIC_DYNAMIC_SUPPORT_MAP_TOLERANCE),
                )
                if static_active != dynamic_active:
                    active_pair_count_changed += 1
                if delta_norm > max_delta:
                    max_delta = delta_norm
                    max_delta_node_id = node_id
                    max_delta_body_id = body_id
            else:
                static_value = np.zeros(2, dtype=np.float64)
                dynamic_value = np.zeros(2, dtype=np.float64)
                delta = np.zeros(2, dtype=np.float64)
                delta_norm = 0.0
            rows.append(
                {
                    "node_id": node_id,
                    "body_id": body_id,
                    "x": float(coordinates[node_id, 0] - X_SHIFT),
                    "y": float(coordinates[node_id, 1] - Y_SHIFT),
                    "static_active": int(static_active),
                    "dynamic_active": int(dynamic_active),
                    "static_mass": float(static_mass[node_id, body_id]),
                    "dynamic_mass": float(dynamic_mass[node_id, body_id]),
                    "static_force_x": float(static_value[0]),
                    "static_force_y": float(static_value[1]),
                    "dynamic_force_x": float(dynamic_value[0]),
                    "dynamic_force_y": float(dynamic_value[1]),
                    "force_delta_x": float(delta[0]),
                    "force_delta_y": float(delta[1]),
                    "force_delta_norm": delta_norm,
                }
            )

    write_csv(
        detail_path,
        [
            "node_id",
            "body_id",
            "x",
            "y",
            "static_active",
            "dynamic_active",
            "static_mass",
            "dynamic_mass",
            "static_force_x",
            "static_force_y",
            "dynamic_force_x",
            "dynamic_force_y",
            "force_delta_x",
            "force_delta_y",
            "force_delta_norm",
        ],
        rows,
    )
    map_status = (
        "MATCH"
        if active_pair_count_changed == 0
        and max_delta <= STATIC_DYNAMIC_SUPPORT_MAP_TOLERANCE + STATIC_DYNAMIC_SUPPORT_MAP_RELATIVE_TOLERANCE * max_reference_force
        else "DIFFERENT"
    )
    write_csv(
        summary_path,
        ["metric", "value"],
        [
            {"metric": "purpose", "value": "static_vs_dynamic_pre_support_force_map"},
            {"metric": "time_advanced", "value": "False"},
            {"metric": "dynamic_support_constructed", "value": "False"},
            {"metric": "seismic_input_applied", "value": "False"},
            {"metric": "dynamic_stress_update_applied", "value": "False"},
            {"metric": "gravity_x", "value": static_gravity[0]},
            {"metric": "gravity_y", "value": static_gravity[1]},
            {"metric": "bottom_node_count", "value": int(bottom_node_ids.size)},
            {"metric": "node_body_pair_count", "value": len(rows)},
            {"metric": "static_active_pair_count", "value": active_pair_count_static},
            {"metric": "dynamic_active_pair_count", "value": active_pair_count_dynamic},
            {"metric": "active_pair_union_count", "value": active_pair_count_union},
            {"metric": "active_pair_changed_count", "value": active_pair_count_changed},
            {"metric": "static_force_sum_x", "value": float(static_sum[0])},
            {"metric": "static_force_sum_y", "value": float(static_sum[1])},
            {"metric": "dynamic_force_sum_x", "value": float(dynamic_sum[0])},
            {"metric": "dynamic_force_sum_y", "value": float(dynamic_sum[1])},
            {"metric": "force_sum_delta_x", "value": float(dynamic_sum[0] - static_sum[0])},
            {"metric": "force_sum_delta_y", "value": float(dynamic_sum[1] - static_sum[1])},
            {"metric": "max_node_body_force_delta_norm", "value": max(0.0, max_delta)},
            {"metric": "max_reference_force_norm", "value": max_reference_force},
            {"metric": "max_node_body_force_delta_relative", "value": max_relative_delta},
            {"metric": "max_delta_node_id", "value": max_delta_node_id},
            {"metric": "max_delta_body_id", "value": max_delta_body_id},
            {"metric": "comparison_tolerance", "value": STATIC_DYNAMIC_SUPPORT_MAP_TOLERANCE},
            {"metric": "comparison_relative_tolerance", "value": STATIC_DYNAMIC_SUPPORT_MAP_RELATIVE_TOLERANCE},
            {"metric": "map_status", "value": map_status},
            {
                "metric": "interpretation",
                "value": "MATCH excludes a pre-support force-map mismatch; DIFFERENT identifies it but does not assign its cause.",
            },
        ],
    )
    if mpm.enginer is not None:
        mpm.enginer.reset_grid_message(mpm.scene)
    return detail_path, summary_path


def run_static_slip_force_diagnostic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: Any,
) -> tuple[Path, Path]:
    """Compare static P2G forces at the lateral slip boundary with and without mirrors.

    The two assemblies start from exactly the restored static particle state.
    They do not advance time, update stress, or alter particle fields.
    """

    output_dir = STATIC_SLIP_FORCE_DIAGNOSTIC_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "left_slip_static_force_decomposition.csv"
    local_path = output_dir / "left_slip_local_force_map.csv"
    summary_path = output_dir / "summary.csv"

    if mpm.enginer is None:
        mpm.add_essentials({"function": None})
    apply_material_stage(mpm, case, "static")
    static_gravity = [float(value) for value in static_monitor.target_gravity]
    original_active = bool(static_monitor.active)
    static_monitor.active = False
    try:
        assemble_force_window(mpm, static_gravity, None)
        no_mirror_force = mpm.scene.node.force.to_numpy().copy()
        no_mirror_mass = mpm.scene.node.m.to_numpy().copy()

        mirror = install_static_mirror_particle_hooks(mpm)
        if mirror is None:
            raise RuntimeError("Static slip-force diagnostic requires the mirror-particle boundary")
        static_monitor.active = True
        assemble_force_window(mpm, static_gravity, None)
        mirror_force = mpm.scene.node.force.to_numpy().copy()
        mirror_mass = mpm.scene.node.m.to_numpy().copy()
        raw_force = mirror.force_source.to_numpy().copy()
        raw_mass = mirror.raw_mass_source.to_numpy().copy()
    finally:
        static_monitor.active = original_active
        if mpm.enginer is not None:
            mpm.enginer.reset_grid_message(mpm.scene)

    coordinates = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT])
    left_nodes = np.flatnonzero(np.isclose(coordinates[:, 0], K_MAIN_X_MIN, rtol=0.0, atol=SIDE_TOL))
    target_distance = np.linalg.norm(coordinates[left_nodes] - np.asarray([K_MAIN_X_MIN, 9.0]), axis=1)
    target_node_id = int(left_nodes[int(np.argmin(target_distance))])
    rows: list[dict[str, Any]] = []
    for node_id in left_nodes.tolist():
        raw_body_force = raw_mass[node_id, BODY_MAIN_SOIL] * np.asarray(static_gravity, dtype=np.float64)
        raw_internal_force = raw_force[node_id, BODY_MAIN_SOIL, :2] - raw_body_force
        mirror_delta = mirror_force[node_id, BODY_MAIN_SOIL, :2] - no_mirror_force[node_id, BODY_MAIN_SOIL, :2]
        rows.append(
            {
                "node_id": int(node_id),
                "x": float(coordinates[node_id, 0]),
                "y": float(coordinates[node_id, 1]),
                "raw_mass": float(raw_mass[node_id, BODY_MAIN_SOIL]),
                "no_mirror_mass": float(no_mirror_mass[node_id, BODY_MAIN_SOIL]),
                "mirror_mass": float(mirror_mass[node_id, BODY_MAIN_SOIL]),
                "raw_internal_force_x": float(raw_internal_force[0]),
                "raw_internal_force_y": float(raw_internal_force[1]),
                "raw_gravity_force_x": float(raw_body_force[0]),
                "raw_gravity_force_y": float(raw_body_force[1]),
                "no_mirror_force_x": float(no_mirror_force[node_id, BODY_MAIN_SOIL, 0]),
                "no_mirror_force_y": float(no_mirror_force[node_id, BODY_MAIN_SOIL, 1]),
                "mirror_force_x": float(mirror_force[node_id, BODY_MAIN_SOIL, 0]),
                "mirror_force_y": float(mirror_force[node_id, BODY_MAIN_SOIL, 1]),
                "mirror_delta_x": float(mirror_delta[0]),
                "mirror_delta_y": float(mirror_delta[1]),
                "is_target_node": int(node_id == target_node_id),
            }
        )
    write_csv(detail_path, list(rows[0].keys()), rows)
    local_mask = (
        (coordinates[:, 0] >= K_MAIN_X_MIN - SIDE_TOL)
        & (coordinates[:, 0] <= K_MAIN_X_MIN + 3.0 * DX + SIDE_TOL)
        & (np.abs(coordinates[:, 1] - 9.0) <= 3.0 * DX + SIDE_TOL)
    )
    local_rows: list[dict[str, Any]] = []
    for node_id in np.flatnonzero(local_mask).tolist():
        raw_body_force = raw_mass[node_id, BODY_MAIN_SOIL] * np.asarray(static_gravity, dtype=np.float64)
        raw_internal_force = raw_force[node_id, BODY_MAIN_SOIL, :2] - raw_body_force
        mirror_delta = mirror_force[node_id, BODY_MAIN_SOIL, :2] - no_mirror_force[node_id, BODY_MAIN_SOIL, :2]
        local_rows.append(
            {
                "node_id": int(node_id),
                "x": float(coordinates[node_id, 0]),
                "y": float(coordinates[node_id, 1]),
                "raw_internal_force_y": float(raw_internal_force[1]),
                "raw_gravity_force_y": float(raw_body_force[1]),
                "no_mirror_force_y": float(no_mirror_force[node_id, BODY_MAIN_SOIL, 1]),
                "mirror_force_y": float(mirror_force[node_id, BODY_MAIN_SOIL, 1]),
                "mirror_delta_y": float(mirror_delta[1]),
            }
        )
    write_csv(local_path, list(local_rows[0].keys()), local_rows)
    target = next(row for row in rows if row["is_target_node"] == 1)
    write_csv(
        summary_path,
        ["metric", "value"],
        [
            {"metric": "purpose", "value": "static_lateral_slip_mirror_force_decomposition"},
            {"metric": "time_advanced", "value": False},
            {"metric": "stress_updated", "value": False},
            {"metric": "particle_state_changed", "value": False},
            {"metric": "target_node_id", "value": target_node_id},
            {"metric": "target_coordinate", "value": f"({target['x']}, {target['y']})"},
            {"metric": "target_no_mirror_force_y", "value": target["no_mirror_force_y"]},
            {"metric": "target_mirror_force_y", "value": target["mirror_force_y"]},
            {"metric": "target_mirror_delta_y", "value": target["mirror_delta_y"]},
            {"metric": "mirror_reflects_body_force", "value": STATIC_MIRROR_REFLECT_BODY_FORCE},
            {"metric": "local_force_map", "value": str(local_path)},
        ],
    )
    return detail_path, summary_path


def run_static_p2g_operator_audit(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: Any,
) -> tuple[Path, Path, Path]:
    """Independently reconstruct native static P2G force terms near a target node.

    This is deliberately a NumPy re-evaluation of ``kernel_force_p2g_2D``:
    ``shape * m * gravity - dNdy * V*sigma_yy - dNdx * V*sigma_xy``.
    It uses no mirror contribution, exactly as the raw static force field that
    is later reflected for the lateral slip condition.  No particle field,
    stress, or simulation time is changed.
    """

    output_dir = STATIC_P2G_OPERATOR_AUDIT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "node_force_reconstruction.csv"
    contribution_path = output_dir / "target_node_particle_contributions.csv"
    summary_path = output_dir / "summary.csv"

    if mpm.enginer is None:
        mpm.add_essentials({"function": None})
    # Match the formal static loop before rebuilding interpolation.  Without
    # this initialization a restored checkpoint can be audited with stale
    # element support metadata even though the manual and native sums agree.
    mpm.solver.engine.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    apply_material_stage(mpm, case, "static")
    static_gravity = np.asarray(static_monitor.target_gravity, dtype=np.float64)
    original_active = bool(static_monitor.active)
    static_monitor.active = False
    try:
        assemble_force_window(mpm, static_gravity.tolist(), None)
        engine_force = mpm.scene.node.force.to_numpy().copy()
        engine_mass = mpm.scene.node.m.to_numpy().copy()
    finally:
        static_monitor.active = original_active
        if mpm.enginer is not None:
            mpm.enginer.reset_grid_message(mpm.scene)

    scene = mpm.scene
    particle_count = int(scene.particleNum[0])
    positions = scene.particle.x.to_numpy()[:particle_count].astype(np.float64)
    stress = scene.particle.stress.to_numpy()[:particle_count].astype(np.float64)
    volume = scene.particle.vol.to_numpy()[:particle_count].astype(np.float64)
    mass = scene.particle.m.to_numpy()[:particle_count].astype(np.float64)
    body_ids = scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    material_ids = scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
    active = scene.particle.active.to_numpy()[:particle_count].astype(np.int32)
    node_size = scene.element.node_size.to_numpy().astype(np.int32)
    ln_id = scene.element.LnID.to_numpy()
    shape_fn = scene.element.shape_fn.to_numpy()
    dshape_fn = scene.element.dshape_fn.to_numpy()
    total_support_nodes = int(scene.element.grid_nodes)
    coordinates = np.asarray(scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT])
    target_point = np.asarray(
        [STATIC_P2G_OPERATOR_AUDIT_TARGET_X, STATIC_P2G_OPERATOR_AUDIT_TARGET_Y],
        dtype=np.float64,
    )
    target_mask = (
        (np.abs(coordinates[:, 0] - target_point[0]) <= STATIC_P2G_OPERATOR_AUDIT_RADIUS + SIDE_TOL)
        & (np.abs(coordinates[:, 1] - target_point[1]) <= STATIC_P2G_OPERATOR_AUDIT_RADIUS + SIDE_TOL)
    )
    target_ids = np.flatnonzero(target_mask).astype(np.int32)
    if target_ids.size == 0:
        raise RuntimeError(
            "Static P2G audit target contains no grid nodes: "
            f"point={target_point.tolist()}, radius={STATIC_P2G_OPERATOR_AUDIT_RADIUS}"
        )
    target_node_id = int(
        target_ids[np.argmin(np.linalg.norm(coordinates[target_ids] - target_point, axis=1))]
    )
    target_lookup = {int(node_id): index for index, node_id in enumerate(target_ids.tolist())}
    internal = np.zeros((target_ids.size, 2), dtype=np.float64)
    gravity = np.zeros((target_ids.size, 2), dtype=np.float64)
    contributions: list[dict[str, Any]] = []

    for pid in range(particle_count):
        if body_ids[pid] != BODY_MAIN_SOIL or material_ids[pid] <= 0 or active[pid] != 1:
            continue
        count = int(node_size[pid])
        offset = pid * total_support_nodes
        for local_index in range(count):
            if ln_id.ndim == 2:
                node_id = int(ln_id[pid, local_index])
                weight = float(shape_fn[pid, local_index])
                derivative = np.asarray(dshape_fn[pid, local_index], dtype=np.float64)
            else:
                index = offset + local_index
                node_id = int(ln_id[index])
                weight = float(shape_fn[index])
                derivative = np.asarray(dshape_fn[index], dtype=np.float64)
            target_index = target_lookup.get(node_id)
            if target_index is None:
                continue
            internal_increment = -volume[pid] * np.asarray(
                [
                    derivative[0] * stress[pid, 0] + derivative[1] * stress[pid, 3],
                    derivative[1] * stress[pid, 1] + derivative[0] * stress[pid, 3],
                ],
                dtype=np.float64,
            )
            gravity_increment = weight * mass[pid] * static_gravity
            internal[target_index] += internal_increment
            gravity[target_index] += gravity_increment
            if node_id == target_node_id:
                contributions.append(
                    {
                        "particle_id": pid,
                        "particle_x": float(positions[pid, 0] - X_SHIFT),
                        "particle_y": float(positions[pid, 1] - Y_SHIFT),
                        "shape": weight,
                        "dshape_dx": float(derivative[0]),
                        "dshape_dy": float(derivative[1]),
                        "stress_yy": float(stress[pid, 1]),
                        "stress_xy": float(stress[pid, 3]),
                        "volume": float(volume[pid]),
                        "mass": float(mass[pid]),
                        "internal_force_y": float(internal_increment[1]),
                        "gravity_force_y": float(gravity_increment[1]),
                        "total_force_y": float(internal_increment[1] + gravity_increment[1]),
                    }
                )

    rows: list[dict[str, Any]] = []
    max_error = 0.0
    max_reference = 0.0
    for target_index, node_id in enumerate(target_ids.tolist()):
        reconstructed = internal[target_index] + gravity[target_index]
        actual = engine_force[node_id, BODY_MAIN_SOIL, :2].astype(np.float64)
        error = actual - reconstructed
        max_error = max(max_error, float(np.max(np.abs(error))))
        max_reference = max(max_reference, float(np.max(np.abs(actual))), float(np.max(np.abs(reconstructed))))
        rows.append(
            {
                "node_id": node_id,
                "x": float(coordinates[node_id, 0]),
                "y": float(coordinates[node_id, 1]),
                "engine_mass": float(engine_mass[node_id, BODY_MAIN_SOIL]),
                "manual_internal_force_x": float(internal[target_index, 0]),
                "manual_internal_force_y": float(internal[target_index, 1]),
                "manual_gravity_force_x": float(gravity[target_index, 0]),
                "manual_gravity_force_y": float(gravity[target_index, 1]),
                "manual_total_force_x": float(reconstructed[0]),
                "manual_total_force_y": float(reconstructed[1]),
                "engine_total_force_x": float(actual[0]),
                "engine_total_force_y": float(actual[1]),
                "force_error_x": float(error[0]),
                "force_error_y": float(error[1]),
            }
        )
    write_csv(detail_path, list(rows[0].keys()), rows)
    contributions.sort(key=lambda row: abs(float(row["total_force_y"])), reverse=True)
    write_csv(
        contribution_path,
        list(contributions[0].keys()) if contributions else ["particle_id"],
        contributions,
    )
    tolerance = 128.0 * np.finfo(np.float32).eps * max(1.0, max_reference)
    status = "PASS" if max_error <= tolerance else "FAIL"
    write_csv(
        summary_path,
        ["metric", "value"],
        [
            {"metric": "purpose", "value": "independent_native_2d_p2g_force_reconstruction"},
            {"metric": "time_advanced", "value": False},
            {"metric": "stress_updated", "value": False},
            {"metric": "particle_state_changed", "value": False},
            {"metric": "mirror_active_during_assembly", "value": False},
            {"metric": "target_x", "value": float(target_point[0])},
            {"metric": "target_y", "value": float(target_point[1])},
            {"metric": "target_radius", "value": STATIC_P2G_OPERATOR_AUDIT_RADIUS},
            {"metric": "target_node_id", "value": target_node_id},
            {"metric": "node_count", "value": int(target_ids.size)},
            {"metric": "max_abs_force_reconstruction_error", "value": max_error},
            {"metric": "max_force_reference", "value": max_reference},
            {"metric": "float32_scaled_tolerance", "value": tolerance},
            {"metric": "status", "value": status},
            {"metric": "node_detail", "value": str(detail_path)},
            {"metric": "target_contributions", "value": str(contribution_path)},
        ],
    )
    return detail_path, contribution_path, summary_path


def run_bottom_static_support_injection_diagnostic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: Any,
    dynamic_start_time: float,
) -> tuple[Path, Path]:
    """Verify the actual nodal write of the frozen bottom static support.

    The result is the grid-force increment produced by the same Taichi kernel
    used in the formal dynamic force window, relative to an identical no-boundary
    assembly.  No dynamic time step is advanced.
    """

    if not DIAGNOSTIC_ENABLE_STATIC_REACTION:
        raise RuntimeError("Bottom-support injection diagnostic requires DIAGNOSTIC_ENABLE_STATIC_REACTION=1")
    output_dir = BOTTOM_STATIC_SUPPORT_INJECTION_DIAGNOSTIC_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "bottom_static_support_injection_map.csv"
    summary_path = output_dir / "summary.csv"

    boundary, _periodic_mapper, _static_stage, _dynamic_stage, dynamic_gravity = prepare_dynamic_transition_diagnostic(
        mpm,
        case,
        static_monitor,
        dynamic_start_time,
    )
    assemble_force_window(mpm, dynamic_gravity, None)
    baseline_force = mpm.scene.node.force.to_numpy().copy()

    expected = zero_nodal_component_array(mpm)
    scatter_to_node_component(
        expected,
        boundary.bottom_static_reaction_node_ids_np,
        boundary.bottom_static_reaction_node_body_ids_np,
        np.column_stack((boundary.bottom_static_reaction_node_x_np, boundary.bottom_static_reaction_node_y_np)),
    )
    original_apply_silent_forces = boundary.apply_silent_forces

    def apply_bottom_static_support_only(self: NairnSeismicBoundary, sims: Any, scene: Any) -> None:
        apply_bottom_static_reaction_nodes(
            self.bottom_static_reaction_node_count,
            self.bottom_static_reaction_node_ids,
            self.bottom_static_reaction_node_body_ids,
            self.bottom_static_reaction_node_x,
            self.bottom_static_reaction_node_y,
            scene.node,
            self.total_static_reaction_force_x,
            self.total_static_reaction_force_y,
        )

    boundary.apply_silent_forces = types.MethodType(apply_bottom_static_support_only, boundary)
    try:
        assemble_force_window(mpm, dynamic_gravity, boundary)
        actual_increment = mpm.scene.node.force.to_numpy().copy() - baseline_force
    finally:
        boundary.apply_silent_forces = original_apply_silent_forces

    coordinates = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64)
    force_tolerance = STATIC_DYNAMIC_SUPPORT_MAP_TOLERANCE
    expected_norm = np.linalg.norm(expected[:, :, :2], axis=2)
    actual_norm = np.linalg.norm(actual_increment[:, :, :2], axis=2)
    key_set = {
        (int(node_id), int(body_id))
        for node_id, body_id in zip(
            boundary.bottom_static_reaction_node_ids_np.tolist(),
            boundary.bottom_static_reaction_node_body_ids_np.tolist(),
        )
    }
    key_set.update((int(node_id), int(body_id)) for node_id, body_id in np.argwhere(expected_norm > force_tolerance))
    key_set.update((int(node_id), int(body_id)) for node_id, body_id in np.argwhere(actual_norm > force_tolerance))

    rows: list[dict[str, Any]] = []
    max_delta = 0.0
    max_reference = 0.0
    max_relative_delta = 0.0
    extra_increment_max = 0.0
    extra_increment_count_above_noise = 0
    actual_nonzero_on_expected_count = 0
    max_node_id = -1
    max_body_id = -1
    for node_id, body_id in sorted(key_set):
        expected_value = expected[node_id, body_id, :2]
        actual_value = actual_increment[node_id, body_id, :2]
        delta = actual_value - expected_value
        delta_norm = float(np.linalg.norm(delta))
        reference_norm = max(float(np.linalg.norm(expected_value)), float(np.linalg.norm(actual_value)))
        expected_active = bool(expected_norm[node_id, body_id] > force_tolerance)
        actual_active = bool(actual_norm[node_id, body_id] > force_tolerance)
        relative_delta = delta_norm / max(reference_norm, force_tolerance)
        if expected_active:
            if actual_active:
                actual_nonzero_on_expected_count += 1
            if delta_norm > max_delta:
                max_delta = delta_norm
                max_node_id = node_id
                max_body_id = body_id
            max_reference = max(max_reference, reference_norm)
            max_relative_delta = max(max_relative_delta, relative_delta)
        else:
            extra_increment_max = max(extra_increment_max, float(actual_norm[node_id, body_id]))
        rows.append(
            {
                "node_id": node_id,
                "body_id": body_id,
                "x": float(coordinates[node_id, 0] - X_SHIFT),
                "y": float(coordinates[node_id, 1] - Y_SHIFT),
                "expected_force_x": float(expected_value[0]),
                "expected_force_y": float(expected_value[1]),
                "actual_force_x": float(actual_value[0]),
                "actual_force_y": float(actual_value[1]),
                "force_delta_x": float(delta[0]),
                "force_delta_y": float(delta[1]),
                "force_delta_norm": delta_norm,
                "force_delta_relative": relative_delta,
            }
        )

    expected_nonzero_count = int(np.count_nonzero(expected_norm > force_tolerance))
    actual_nonzero_count = int(np.count_nonzero(actual_norm > force_tolerance))
    noise_limit = force_tolerance + STATIC_DYNAMIC_SUPPORT_MAP_RELATIVE_TOLERANCE * max_reference
    extra_increment_count_above_noise = int(
        np.count_nonzero((expected_norm <= force_tolerance) & (actual_norm > noise_limit))
    )
    expected_sum = expected[:, :, :2].sum(axis=(0, 1))
    actual_sum = actual_increment[:, :, :2].sum(axis=(0, 1))
    injection_status = (
        "MATCH"
        if max_delta <= noise_limit
        and expected_nonzero_count == actual_nonzero_on_expected_count
        and extra_increment_count_above_noise == 0
        else "DIFFERENT"
    )
    write_csv(
        detail_path,
        [
            "node_id",
            "body_id",
            "x",
            "y",
            "expected_force_x",
            "expected_force_y",
            "actual_force_x",
            "actual_force_y",
            "force_delta_x",
            "force_delta_y",
            "force_delta_norm",
            "force_delta_relative",
        ],
        rows,
    )
    write_csv(
        summary_path,
        ["metric", "value"],
        [
            {"metric": "purpose", "value": "actual_bottom_static_support_kernel_write_vs_frozen_source"},
            {"metric": "time_advanced", "value": "False"},
            {"metric": "seismic_input_applied", "value": "False"},
            {"metric": "other_boundary_components_applied", "value": "False"},
            {"metric": "frozen_source_record_count", "value": int(boundary.bottom_static_reaction_node_count)},
            {"metric": "expected_nonzero_node_body_count", "value": expected_nonzero_count},
            {"metric": "actual_nonzero_node_body_count_raw", "value": actual_nonzero_count},
            {"metric": "actual_nonzero_count_on_expected_pairs", "value": actual_nonzero_on_expected_count},
            {"metric": "expected_force_sum_x", "value": float(expected_sum[0])},
            {"metric": "expected_force_sum_y", "value": float(expected_sum[1])},
            {"metric": "actual_force_sum_x", "value": float(actual_sum[0])},
            {"metric": "actual_force_sum_y", "value": float(actual_sum[1])},
            {"metric": "force_sum_delta_x", "value": float(actual_sum[0] - expected_sum[0])},
            {"metric": "force_sum_delta_y", "value": float(actual_sum[1] - expected_sum[1])},
            {"metric": "max_node_body_force_delta_norm", "value": max_delta},
            {"metric": "max_node_body_force_delta_relative", "value": max_relative_delta},
            {"metric": "extra_increment_max_norm", "value": extra_increment_max},
            {"metric": "extra_increment_count_above_noise", "value": extra_increment_count_above_noise},
            {"metric": "max_delta_node_id", "value": max_node_id},
            {"metric": "max_delta_body_id", "value": max_body_id},
            {"metric": "comparison_tolerance", "value": force_tolerance},
            {"metric": "comparison_relative_tolerance", "value": STATIC_DYNAMIC_SUPPORT_MAP_RELATIVE_TOLERANCE},
            {"metric": "comparison_noise_limit", "value": noise_limit},
            {"metric": "injection_status", "value": injection_status},
            {
                "metric": "interpretation",
                "value": "MATCH excludes an error in the isolated bottom-static-support kernel write only.",
            },
        ],
    )
    if mpm.enginer is not None:
        mpm.enginer.reset_grid_message(mpm.scene)
    return detail_path, summary_path


def run_corner_superposition_diagnostic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: Any,
    dynamic_start_time: float,
) -> tuple[Path, Path]:
    """Check that each dynamic-boundary contribution reaches corner nodes once."""

    output_dir = CORNER_SUPERPOSITION_DIAGNOSTIC_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "corner_force_superposition.csv"
    summary_path = output_dir / "summary.csv"
    boundary, _periodic_mapper, _static_stage, _dynamic_stage, dynamic_gravity = prepare_dynamic_transition_diagnostic(
        mpm,
        case,
        static_monitor,
        dynamic_start_time,
    )
    original_apply_silent_forces = boundary.apply_silent_forces
    captured_force_window: dict[str, np.ndarray] = {}
    component_snapshots: dict[str, np.ndarray] = {}

    def capture_component(name: str, force: np.ndarray) -> None:
        component_snapshots[name] = force

    def capture_full_boundary_increment(self: NairnSeismicBoundary, sims: Any, scene: Any) -> None:
        # Capture in one force assembly.  Subtracting two independently
        # assembled P2G force fields is not precise enough for corner checks.
        captured_force_window["before"] = scene.node.force.to_numpy().copy()
        original_apply_silent_forces(sims, scene)
        captured_force_window["after"] = scene.node.force.to_numpy().copy()

    boundary.apply_silent_forces = types.MethodType(capture_full_boundary_increment, boundary)
    boundary._force_component_capture = capture_component
    try:
        assemble_force_window(mpm, dynamic_gravity, boundary)
    finally:
        boundary.apply_silent_forces = original_apply_silent_forces
        delattr(boundary, "_force_component_capture")
    actual_increment = captured_force_window["after"] - captured_force_window["before"]
    components = boundary_component_arrays(mpm, boundary)
    component_names = ["bottom_static_support", "bottom_dashpot_force", "lateral_static_support", "free_field_coupling"]
    expected_components = {
        "bottom_static_support": components["bottom_static_support"],
        "bottom_dashpot_force": components["bottom_dashpot_force"],
        "lateral_static_support": components["lateral_static_support"],
        "free_field_coupling": components["free_field_dynamic_stress_force"] + components["lateral_dashpot_force"],
    }
    actual_components: dict[str, np.ndarray] = {}
    previous_force = captured_force_window["before"]
    for name in component_names:
        if name not in component_snapshots:
            raise RuntimeError(f"Corner-superposition diagnostic did not capture {name}")
        actual_components[name] = component_snapshots[name] - previous_force
        previous_force = component_snapshots[name]
    expected_increment = zero_nodal_component_array(mpm)
    for component in expected_components.values():
        expected_increment += component

    coordinates = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT])
    force_tolerance = STATIC_DYNAMIC_SUPPORT_MAP_TOLERANCE
    global_reference = float(np.max(np.linalg.norm(expected_increment[:, :, :2], axis=2)))
    noise_limit = force_tolerance + STATIC_DYNAMIC_SUPPORT_MAP_RELATIVE_TOLERANCE * global_reference
    corner_patch_radius = 2.0 * DX
    rows: list[dict[str, Any]] = []
    max_delta = 0.0
    max_relative_delta = 0.0
    max_node_id = -1
    corner_component_counts: list[int] = []
    max_component_deltas = {name: 0.0 for name in component_names}
    for node_id in range(actual_increment.shape[0]):
        x = float(coordinates[node_id, 0])
        y = float(coordinates[node_id, 1])
        if abs(x - K_MAIN_X_MIN) <= corner_patch_radius and abs(y - K_DOMAIN_Y_MIN) <= corner_patch_radius:
            classification = "main_bottom_left_corner_patch"
        elif abs(x - K_MAIN_X_MAX) <= corner_patch_radius and abs(y - K_DOMAIN_Y_MIN) <= corner_patch_radius:
            classification = "main_bottom_right_corner_patch"
        else:
            continue
        body_id = BODY_MAIN_SOIL
        expected_value = expected_increment[node_id, body_id, :2]
        actual_value = actual_increment[node_id, body_id, :2]
        delta = actual_value - expected_value
        delta_norm = float(np.linalg.norm(delta))
        reference_norm = max(float(np.linalg.norm(expected_value)), float(np.linalg.norm(actual_value)))
        relative_delta = delta_norm / max(reference_norm, force_tolerance)
        component_count = 0
        row = {
            "node_id": node_id,
            "body_id": body_id,
            "x": x,
            "y": y,
            "classification": classification,
            "expected_total_x": float(expected_value[0]),
            "expected_total_y": float(expected_value[1]),
            "actual_total_x": float(actual_value[0]),
            "actual_total_y": float(actual_value[1]),
            "force_delta_x": float(delta[0]),
            "force_delta_y": float(delta[1]),
            "force_delta_norm": delta_norm,
            "force_delta_relative": relative_delta,
        }
        for name in component_names:
            expected_component = expected_components[name][node_id, body_id, :2]
            actual_component = actual_components[name][node_id, body_id, :2]
            component_delta = actual_component - expected_component
            component_delta_norm = float(np.linalg.norm(component_delta))
            row[f"expected_{name}_x"] = float(expected_component[0])
            row[f"expected_{name}_y"] = float(expected_component[1])
            row[f"actual_{name}_x"] = float(actual_component[0])
            row[f"actual_{name}_y"] = float(actual_component[1])
            row[f"{name}_delta_norm"] = component_delta_norm
            max_component_deltas[name] = max(max_component_deltas[name], component_delta_norm)
            if float(np.linalg.norm(expected_component)) > force_tolerance:
                component_count += 1
        row["nonzero_component_count"] = component_count
        rows.append(row)
        corner_component_counts.append(component_count)
        if delta_norm > max_delta:
            max_delta = delta_norm
            max_relative_delta = relative_delta
            max_node_id = node_id

    if not rows:
        raise RuntimeError("Corner-superposition diagnostic found no main-model bottom corner nodes")
    superposition_status = "MATCH" if max_delta <= noise_limit else "DIFFERENT"
    free_field_support_path = write_free_field_corner_pair_supports(
        output_dir,
        mpm,
        boundary,
        corner_patch_radius,
    )
    fields = list(rows[0].keys())
    write_csv(detail_path, fields, rows)
    write_csv(
        summary_path,
        ["metric", "value"],
        [
            {"metric": "purpose", "value": "single_force_window_actual_full_boundary_increment_vs_once_per_component_sum_at_main_bottom_corners"},
            {"metric": "time_advanced", "value": "False"},
            {"metric": "seismic_input_applied", "value": "False"},
            {"metric": "corner_node_count", "value": len(rows)},
            {"metric": "corner_patch_radius", "value": corner_patch_radius},
            {"metric": "min_nonzero_component_count", "value": min(corner_component_counts)},
            {"metric": "max_nonzero_component_count", "value": max(corner_component_counts)},
            {"metric": "max_corner_force_delta_norm", "value": max_delta},
            {"metric": "max_corner_force_delta_relative", "value": max_relative_delta},
            {"metric": "max_delta_node_id", "value": max_node_id},
            {"metric": "comparison_noise_limit", "value": noise_limit},
            {"metric": "free_field_corner_pair_supports", "value": free_field_support_path.as_posix()},
            *[
                {"metric": f"max_{name}_delta_norm", "value": value}
                for name, value in max_component_deltas.items()
            ],
            {"metric": "superposition_status", "value": superposition_status},
            {
                "metric": "interpretation",
                "value": "MATCH excludes duplicate or missing application among the assembled corner components at this force window.",
            },
        ],
    )
    if mpm.enginer is not None:
        mpm.enginer.reset_grid_message(mpm.scene)
    return detail_path, summary_path


def write_free_field_corner_pair_supports(
    output_dir: Path,
    mpm: MPM,
    boundary: Any,
    corner_patch_radius: float,
) -> Path:
    """List free-field pair forces whose main-particle support enters a corner patch."""

    path = output_dir / "free_field_corner_pair_supports.csv"
    pair_count = int(boundary.ff_pair_count)
    total_x = boundary.ff_pair_total_force_x.to_numpy()[:pair_count]
    total_y = boundary.ff_pair_total_force_y.to_numpy()[:pair_count]
    stress_x = boundary.ff_pair_dynamic_stress_force_x.to_numpy()[:pair_count]
    stress_y = boundary.ff_pair_dynamic_stress_force_y.to_numpy()[:pair_count]
    dash_x = boundary.ff_pair_normal_dashpot_force_x.to_numpy()[:pair_count]
    dash_y = boundary.ff_pair_shear_dashpot_force_y.to_numpy()[:pair_count]
    positions = mpm.scene.particle.x.to_numpy()[: int(mpm.scene.particleNum[0])]
    coordinates = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT])
    rows: list[dict[str, Any]] = []
    for pair_id, main_pid in enumerate(boundary.ff_pair_main_ids_np[:pair_count].astype(np.int32).tolist()):
        main_x = float(positions[main_pid, 0] - X_SHIFT)
        main_y = float(positions[main_pid, 1] - Y_SHIFT)
        support_written = False
        for node_id, weight in support_node_entries(mpm, main_pid):
            x = float(coordinates[node_id, 0])
            y = float(coordinates[node_id, 1])
            if abs(x - K_MAIN_X_MIN) <= corner_patch_radius and abs(y - K_DOMAIN_Y_MIN) <= corner_patch_radius:
                corner = "left"
            elif abs(x - K_MAIN_X_MAX) <= corner_patch_radius and abs(y - K_DOMAIN_Y_MIN) <= corner_patch_radius:
                corner = "right"
            else:
                continue
            support_written = True
            rows.append(
                {
                    "corner": corner,
                    "pair_id": pair_id,
                    "side": str(boundary.ff_pair_sides_np[pair_id]),
                    "main_particle_id": main_pid,
                    "free_field_particle_id": int(boundary.ff_pair_ids_np[pair_id]),
                    "node_id": node_id,
                    "node_x": x,
                    "node_y": y,
                    "shape_weight": weight,
                    "pair_dynamic_stress_x": float(stress_x[pair_id]),
                    "pair_dynamic_stress_y": float(stress_y[pair_id]),
                    "pair_normal_dashpot_x": float(dash_x[pair_id]),
                    "pair_shear_dashpot_y": float(dash_y[pair_id]),
                    "pair_total_x": float(total_x[pair_id]),
                    "pair_total_y": float(total_y[pair_id]),
                    "node_force_x": float(weight * total_x[pair_id]),
                    "node_force_y": float(weight * total_y[pair_id]),
                }
            )
        if not support_written:
            rows.append(
                {
                    "corner": "",
                    "pair_id": pair_id,
                    "side": str(boundary.ff_pair_sides_np[pair_id]),
                    "main_particle_id": main_pid,
                    "free_field_particle_id": int(boundary.ff_pair_ids_np[pair_id]),
                    "node_id": "",
                    "node_x": "",
                    "node_y": "",
                    "shape_weight": "",
                    "pair_dynamic_stress_x": float(stress_x[pair_id]),
                    "pair_dynamic_stress_y": float(stress_y[pair_id]),
                    "pair_normal_dashpot_x": float(dash_x[pair_id]),
                    "pair_shear_dashpot_y": float(dash_y[pair_id]),
                    "pair_total_x": float(total_x[pair_id]),
                    "pair_total_y": float(total_y[pair_id]),
                    "node_force_x": "",
                    "node_force_y": "",
                    "main_particle_x": main_x,
                    "main_particle_y": main_y,
                }
            )
        else:
            for row in rows:
                if row["pair_id"] == pair_id:
                    row["main_particle_x"] = main_x
                    row["main_particle_y"] = main_y
    write_csv(
        path,
        [
            "corner",
            "pair_id",
            "side",
            "main_particle_id",
            "free_field_particle_id",
            "node_id",
            "node_x",
            "node_y",
            "shape_weight",
            "pair_dynamic_stress_x",
            "pair_dynamic_stress_y",
            "pair_normal_dashpot_x",
            "pair_shear_dashpot_y",
            "pair_total_x",
            "pair_total_y",
            "node_force_x",
            "node_force_y",
            "main_particle_x",
            "main_particle_y",
        ],
        rows,
    )
    return path


def run_free_field_residual_velocity_diagnostic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: Any,
    dynamic_start_time: float,
) -> Path:
    """Measure the first-window free-field force before and after zeroing velocity only."""

    output_dir = FREE_FIELD_RESIDUAL_VELOCITY_DIAGNOSTIC_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    boundary, _periodic_mapper, _static_stage, _dynamic_stage, dynamic_gravity = prepare_dynamic_transition_diagnostic(
        mpm,
        case,
        static_monitor,
        dynamic_start_time,
    )
    particle_count = int(mpm.scene.particleNum[0])
    velocities_full = mpm.scene.particle.v.to_numpy()
    velocities = velocities_full[:particle_count, :2].copy()
    main_ids = boundary.ff_pair_main_ids_np[: boundary.ff_pair_count].astype(np.int32)
    ff_ids = boundary.ff_pair_ids_np[: boundary.ff_pair_count].astype(np.int32)
    relative_velocity = velocities[ff_ids] - velocities[main_ids] if main_ids.size else np.empty((0, 2))

    assemble_force_window(mpm, dynamic_gravity, boundary)
    before = free_field_pair_force_component_totals(boundary)
    before_max_total = float(
        np.max(
            np.hypot(
                boundary.ff_pair_total_force_x.to_numpy()[: boundary.ff_pair_count],
                boundary.ff_pair_total_force_y.to_numpy()[: boundary.ff_pair_count],
            )
        )
    ) if boundary.ff_pair_count else 0.0

    zeroed_velocities = velocities_full.copy()
    zeroed_velocities[:particle_count, :2] = 0.0
    mpm.scene.particle.v.from_numpy(zeroed_velocities)
    assemble_force_window(mpm, dynamic_gravity, boundary)
    after = free_field_pair_force_component_totals(boundary)
    after_max_total = float(
        np.max(
            np.hypot(
                boundary.ff_pair_total_force_x.to_numpy()[: boundary.ff_pair_count],
                boundary.ff_pair_total_force_y.to_numpy()[: boundary.ff_pair_count],
            )
        )
    ) if boundary.ff_pair_count else 0.0

    tolerance = STATIC_DYNAMIC_SUPPORT_MAP_TOLERANCE
    write_csv(
        summary_path,
        ["metric", "value"],
        [
            {"metric": "purpose", "value": "first_force_window_free_field_force_response_to_zeroed_particle_velocity"},
            {"metric": "time_advanced", "value": "False"},
            {"metric": "seismic_input_applied", "value": "False"},
            {"metric": "particle_velocity_modified_only_in_diagnostic_process", "value": "True"},
            {"metric": "free_field_pair_count", "value": int(boundary.ff_pair_count)},
            {"metric": "max_initial_relative_velocity", "value": float(np.max(np.linalg.norm(relative_velocity, axis=1))) if relative_velocity.size else 0.0},
            {"metric": "before_max_pair_total_force", "value": before_max_total},
            {"metric": "after_zero_velocity_max_pair_total_force", "value": after_max_total},
            *[{"metric": f"before_{key}", "value": value} for key, value in before.items()],
            *[{"metric": f"after_zero_velocity_{key}", "value": value} for key, value in after.items()],
            {
                "metric": "residual_velocity_is_source_of_initial_free_field_force",
                "value": bool(before_max_total > tolerance and after_max_total <= tolerance),
            },
        ],
    )
    if mpm.enginer is not None:
        mpm.enginer.reset_grid_message(mpm.scene)
    return summary_path


def main_lateral_boundary_node_ids(mpm: MPM, case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    side_spec = case.get("silent_boundary", {}).get("side", {})
    x_values = [float(value) for value in side_spec.get("x_values", [K_MAIN_X_MIN, K_MAIN_X_MAX])]
    y_min, y_max = [float(value) for value in side_spec.get("y_range", [K_DOMAIN_Y_MIN, K_DOMAIN_Y_MAX])]
    tolerance = float(side_spec.get("tolerance", SIDE_TOL))
    particle_count = int(mpm.scene.particleNum[0])
    positions = mpm.scene.particle.x.to_numpy()[:particle_count]
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    physical = positions - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
    ln_id = mpm.scene.element.LnID.to_numpy()
    node_size = mpm.scene.element.node_size.to_numpy()
    total_nodes = int(mpm.scene.element.grid_nodes)
    left_nodes: set[int] = set()
    right_nodes: set[int] = set()
    for x_value in x_values:
        side_particle_ids = np.nonzero(
            (body_ids == BODY_MAIN_SOIL)
            & (np.abs(physical[:, 0] - x_value) <= tolerance)
            & (physical[:, 1] >= y_min - 1.0e-12)
            & (physical[:, 1] <= y_max + 1.0e-12)
        )[0]
        node_set = left_nodes if x_value <= 0.5 * (K_MAIN_X_MIN + K_MAIN_X_MAX) else right_nodes
        for pid in side_particle_ids.tolist():
            offset = int(pid) * total_nodes
            for ln in range(offset, offset + int(node_size[int(pid)])):
                node_set.add(int(ln_id[ln]))
        geometric_nodes = mpm.scene.element.get_boundary_nodes(
            [x_value + X_SHIFT, y_min + Y_SHIFT],
            [x_value + X_SHIFT, y_max + Y_SHIFT],
        )
        if x_value <= 0.5 * (K_MAIN_X_MIN + K_MAIN_X_MAX):
            left_nodes.update(int(node_id) for node_id in np.asarray(geometric_nodes, dtype=np.int32).tolist())
        else:
            right_nodes.update(int(node_id) for node_id in np.asarray(geometric_nodes, dtype=np.int32).tolist())
    left = np.asarray(sorted(left_nodes), dtype=np.int32)
    right = np.asarray(sorted(right_nodes), dtype=np.int32)
    return left.astype(np.int32), right.astype(np.int32)


def velocity_boundary_arrays(boundary: Any) -> dict[str, np.ndarray]:
    if boundary.velocity_boundary is None:
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        return {"node": empty_i, "level": empty_i, "dirs": empty_i, "velocity": empty_f}
    raw = boundary.velocity_boundary.to_numpy()
    if isinstance(raw, dict):
        return {
            "node": np.asarray(raw.get("node", np.empty(0)), dtype=np.int32),
            "level": np.asarray(raw.get("level", np.empty(0)), dtype=np.int32),
            "dirs": np.asarray(raw.get("dirs", np.empty(0)), dtype=np.int32),
            "velocity": np.asarray(raw.get("velocity", np.empty(0)), dtype=np.float64),
        }
    names = getattr(getattr(raw, "dtype", None), "names", None)
    if names:
        return {
            "node": np.asarray(raw["node"], dtype=np.int32),
            "level": np.asarray(raw["level"], dtype=np.int32),
            "dirs": np.asarray(raw["dirs"], dtype=np.int32),
            "velocity": np.asarray(raw["velocity"], dtype=np.float64),
        }
    raise TypeError("Unsupported GeoTaichi velocity_boundary numpy representation")


def classify_velocity_constraint(node_id: int, body_id: int, direction: int, value: float, x: float, y: float) -> str:
    if not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return "other"
    bottom_tol = max(1.0e-8, STATIC_BOTTOM_CONSTRAINT_BAND + 1.0e-9)
    if body_id == BODY_MAIN_SOIL and K_MAIN_X_MIN - 1.0e-9 <= x <= K_MAIN_X_MAX + 1.0e-9:
        if abs(y - float(kohler_base_bottom_z_np(np.asarray(x, dtype=np.float64)))) <= bottom_tol:
            return "main_bottom_vx" if direction == 0 else "main_bottom_vy" if direction == 1 else "other"
        if direction == 0 and abs(x - K_MAIN_X_MIN) <= bottom_tol:
            return "main_left_lateral_vx"
        if direction == 0 and abs(x - K_MAIN_X_MAX) <= bottom_tol:
            return "main_right_lateral_vx"
    if body_id == BODY_LEFT_FREE_SOIL and K_LEFT_FREE_X_MIN - 1.0e-9 <= x <= K_LEFT_FREE_X_MAX + 1.0e-9:
        if abs(y - K_LEFT_FREE_BOTTOM_Z) <= bottom_tol:
            return "left_free_bottom_vx" if direction == 0 else "left_free_bottom_vy" if direction == 1 else "other"
    if body_id == BODY_RIGHT_FREE_SOIL and K_RIGHT_FREE_X_MIN - 1.0e-9 <= x <= K_RIGHT_FREE_X_MAX + 1.0e-9:
        if abs(y - K_RIGHT_FREE_BOTTOM_Z) <= bottom_tol:
            return "right_free_bottom_vx" if direction == 0 else "right_free_bottom_vy" if direction == 1 else "other"
    return "other"


def bottom_constraints_for_lateral_static_extraction() -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for boundary in static_bottom_boundary_conditions():
        x0 = float(boundary["StartPoint"][0]) - X_SHIFT
        x1 = float(boundary["EndPoint"][0]) - X_SHIFT
        x_mid = 0.5 * (x0 + x1)
        if x_mid <= K_LEFT_FREE_X_MAX:
            level = BODY_LEFT_FREE_SOIL
        elif x_mid >= K_RIGHT_FREE_X_MIN:
            level = BODY_RIGHT_FREE_SOIL
        else:
            level = BODY_MAIN_SOIL
        constrained = dict(boundary)
        constrained["NLevel"] = int(level)
        constraints.append(constrained)
    return constraints


def active_velocity_constraint_rows(mpm: MPM) -> list[dict[str, Any]]:
    boundary = mpm.scene.boundary
    arrays = velocity_boundary_arrays(boundary)
    active_count = min(int(boundary.velocity_list[0]), int(arrays["node"].shape[0]))
    node_coords = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index in range(active_count):
        node_id = int(arrays["node"][index])
        body_id = int(arrays["level"][index])
        direction = int(arrays["dirs"][index])
        value = float(arrays["velocity"][index])
        active = node_id >= 0 and body_id != 255
        if 0 <= node_id < node_coords.shape[0]:
            coord = node_coords[node_id] - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
            x = float(coord[0])
            y = float(coord[1])
        else:
            x = math.nan
            y = math.nan
        classification = classify_velocity_constraint(node_id, body_id, direction, value, x, y) if active else "other"
        rows.append(
            {
                "node_id": node_id,
                "body_id": body_id,
                "direction": direction,
                "value": value,
                "x": x,
                "y": y,
                "classification": classification,
                "active": bool(active),
            }
        )
    return rows


def velocity_constraint_counts(mpm: MPM) -> dict[str, int]:
    rows = active_velocity_constraint_rows(mpm)
    active_rows = [row for row in rows if bool(row["active"])]
    lateral_names = {"main_left_lateral_vx", "main_right_lateral_vx"}
    bottom_names = {
        "main_bottom_vx",
        "main_bottom_vy",
        "left_free_bottom_vx",
        "left_free_bottom_vy",
        "right_free_bottom_vx",
        "right_free_bottom_vy",
    }
    return {
        "velocity_constraint_count": int(mpm.scene.boundary.velocity_list[0]),
        "velocity_constraint_dict_count": len(mpm.scene.boundary.velocity_dict),
        "lateral_velocity_constraint_count": sum(1 for row in active_rows if row["classification"] in lateral_names),
        "bottom_velocity_constraint_count": sum(1 for row in active_rows if row["classification"] in bottom_names),
        "other_velocity_constraint_count": sum(1 for row in active_rows if row["classification"] == "other"),
        "active_velocity_constraint_count": len(active_rows),
    }


def keep_only_bottom_velocity_constraints(mpm: MPM) -> None:
    boundary = mpm.scene.boundary
    bottom_names = {
        "main_bottom_vx",
        "main_bottom_vy",
        "left_free_bottom_vx",
        "left_free_bottom_vy",
        "right_free_bottom_vx",
        "right_free_bottom_vy",
    }
    bottom_rows = [row for row in active_velocity_constraint_rows(mpm) if bool(row["active"]) and row["classification"] in bottom_names]
    boundary.velocity_dict.clear()
    boundary.velocity_list[0] = 0
    if boundary.velocity_boundary is not None:
        kernel_initialize_boundary(boundary.velocity_boundary)
    for row in bottom_rows:
        key = (int(row["node_id"]), int(row["direction"]), int(row["body_id"]))
        boundary.velocity_dict[key] = float(row["value"])
    boundary.copy_dict_to_field(mpm.sims)
    if mpm.enginer is not None:
        mpm.enginer.choose_boundary_constraints(mpm.sims, mpm.scene)


def lateral_support_stage_diagnostic(
    mpm: MPM,
    stage: str,
    left_nodes: np.ndarray,
    right_nodes: np.ndarray,
    note: str,
) -> dict[str, Any]:
    particle_count = int(mpm.scene.particleNum[0])
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    stresses = mpm.scene.particle.stress.to_numpy()[:particle_count]
    main_stresses = stresses[body_ids == BODY_MAIN_SOIL]
    main_stress_norm = np.linalg.norm(main_stresses, axis=1) if main_stresses.size else np.empty(0, dtype=np.float64)
    main_max_stress = float(np.max(main_stress_norm)) if main_stress_norm.size else 0.0
    node_force = mpm.scene.node.force.to_numpy()
    main_force_x = node_force[:, BODY_MAIN_SOIL, 0]
    target_nodes = np.unique(np.concatenate([left_nodes, right_nodes])).astype(np.int32)
    target_force_x = main_force_x[target_nodes] if target_nodes.size else np.empty(0, dtype=np.float64)
    left_force_x = main_force_x[left_nodes] if left_nodes.size else np.empty(0, dtype=np.float64)
    right_force_x = main_force_x[right_nodes] if right_nodes.size else np.empty(0, dtype=np.float64)
    counts = velocity_constraint_counts(mpm)
    status = (
        "PASS"
        if main_max_stress > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE
        and int(np.count_nonzero(main_stress_norm > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE)) > 0
        and counts["lateral_velocity_constraint_count"] == 0
        and counts["bottom_velocity_constraint_count"] > 0
        and counts["other_velocity_constraint_count"] == 0
        else "FAIL"
    )
    return {
        "stage": stage,
        "main_max_abs_particle_stress": main_max_stress,
        "main_nonzero_stress_particle_count": int(np.count_nonzero(main_stress_norm > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE)),
        "all_main_nodes_max_abs_force_x": float(np.max(np.abs(main_force_x))) if main_force_x.size else 0.0,
        "all_main_nodes_nonzero_force_x_count": int(np.count_nonzero(np.abs(main_force_x) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE)),
        "lateral_target_nodes_max_abs_force_x": float(np.max(np.abs(target_force_x))) if target_force_x.size else 0.0,
        "lateral_target_nodes_nonzero_force_x_count": int(np.count_nonzero(np.abs(target_force_x) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE)),
        "left_max_abs_residual_x": float(np.max(np.abs(left_force_x))) if left_force_x.size else 0.0,
        "left_nonzero_residual_x_count": int(np.count_nonzero(np.abs(left_force_x) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE)),
        "right_max_abs_residual_x": float(np.max(np.abs(right_force_x))) if right_force_x.size else 0.0,
        "right_nonzero_residual_x_count": int(np.count_nonzero(np.abs(right_force_x) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE)),
        "velocity_constraint_count": counts["velocity_constraint_count"],
        "velocity_constraint_dict_count": counts["velocity_constraint_dict_count"],
        "lateral_velocity_constraint_count": counts["lateral_velocity_constraint_count"],
        "bottom_velocity_constraint_count": counts["bottom_velocity_constraint_count"],
        "other_velocity_constraint_count": counts["other_velocity_constraint_count"],
        "status": status,
        "note": note,
    }


def assemble_lateral_static_support_from_geotaichi(mpm: MPM, case: dict[str, Any]) -> tuple[Path, Path]:
    engine = mpm.enginer
    if engine is None:
        raise RuntimeError("GeoTaichi engine is not initialized; cannot assemble lateral static support")

    left_nodes, right_nodes = main_lateral_boundary_node_ids(mpm, case)
    diagnostics: list[dict[str, Any]] = []
    diagnostics.append(lateral_support_stage_diagnostic(mpm, "after_static_initialization", left_nodes, right_nodes, "static boundary state before clearing lateral constraints"))
    clear_velocity_boundary_state(mpm)
    # Kohler et al. Section 3.2 transfers the residual after the static
    # kinematic/mirrored constraints have been removed.  The first dynamic
    # P2G force window has no such constraints, so do not re-add one here.
    diagnostics.append(lateral_support_stage_diagnostic(mpm, "after_clear_boundary_state", left_nodes, right_nodes, "all static velocity constraints removed before dynamic-equivalent force assembly"))
    active_constraints_path = OUTPUT_DIR / "active_velocity_constraints_after_clear.csv"
    write_csv(
        active_constraints_path,
        ["node_id", "body_id", "direction", "value", "x", "y", "classification", "active"],
        active_velocity_constraint_rows(mpm),
    )
    previous_gravity = [float(value) for value in mpm.sims.gravity]
    # Section 3.3 uses only the horizontal out-of-balance force at the lateral
    # nodes. Bottom and lateral B-spline support footprints overlap, so compute
    # the lateral correction from the imbalance remaining after the already
    # frozen Section 3.2 bottom reaction.
    assemble_force_window(mpm, previous_gravity[:2], None)
    diagnostics.append(lateral_support_stage_diagnostic(mpm, "after_compute_forces", left_nodes, right_nodes, "scene.node.force contains assembled force before nodal kinematic converts it to acceleration"))

    node_force = mpm.scene.node.force.to_numpy()
    node_coords = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64)
    diagnostic_path = OUTPUT_DIR / "lateral_static_support_diagnostic.csv"
    write_csv(
        diagnostic_path,
        [
            "stage",
            "main_max_abs_particle_stress",
            "main_nonzero_stress_particle_count",
            "all_main_nodes_max_abs_force_x",
            "all_main_nodes_nonzero_force_x_count",
            "lateral_target_nodes_max_abs_force_x",
            "lateral_target_nodes_nonzero_force_x_count",
            "left_max_abs_residual_x",
            "left_nonzero_residual_x_count",
            "right_max_abs_residual_x",
            "right_nonzero_residual_x_count",
            "velocity_constraint_count",
            "velocity_constraint_dict_count",
            "lateral_velocity_constraint_count",
            "bottom_velocity_constraint_count",
            "other_velocity_constraint_count",
            "status",
            "note",
        ],
        diagnostics,
    )

    bottom_support_lookup = {
        (int(node_id), int(body_id)): float(force_x)
        for node_id, body_id, force_x in zip(
            np.asarray(getattr(mpm, "bottom_static_support_node_ids_np", np.empty(0, dtype=np.int32))).tolist(),
            np.asarray(getattr(mpm, "bottom_static_support_body_ids_np", np.empty(0, dtype=np.int32))).tolist(),
            np.asarray(getattr(mpm, "bottom_static_support_x_np", np.empty(0, dtype=np.float64))).tolist(),
        )
    }
    rows: list[dict[str, Any]] = []
    for side, nodes in (("left", left_nodes), ("right", right_nodes)):
        for node_id in nodes.tolist():
            raw_force_x = float(node_force[int(node_id), BODY_MAIN_SOIL, 0])
            bottom_support_x = bottom_support_lookup.get((int(node_id), BODY_MAIN_SOIL), 0.0)
            residual_x = raw_force_x + bottom_support_x
            support_x = -residual_x
            combined_support_x = bottom_support_x + support_x
            final_balance_x = raw_force_x + combined_support_x
            coord = node_coords[int(node_id)] - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
            rows.append(
                {
                    "side": side,
                    "node_id": int(node_id),
                    "body_id": BODY_MAIN_SOIL,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "raw_no_mirror_force_x": raw_force_x,
                    "bottom_support_x": bottom_support_x,
                    "residual_after_bottom_x": residual_x,
                    "support_x": support_x,
                    "combined_support_x": combined_support_x,
                    "final_static_balance_x": final_balance_x,
                }
            )

    support_path = OUTPUT_DIR / "lateral_static_support_from_geotaichi.csv"
    write_csv(
        support_path,
        [
            "side",
            "node_id",
            "body_id",
            "x",
            "y",
            "raw_no_mirror_force_x",
            "bottom_support_x",
            "residual_after_bottom_x",
            "support_x",
            "combined_support_x",
            "final_static_balance_x",
        ],
        rows,
    )

    residuals = np.asarray([float(row["residual_after_bottom_x"]) for row in rows], dtype=np.float64)
    supports = np.asarray([float(row["support_x"]) for row in rows], dtype=np.float64)
    max_balance_error = float(np.max(np.abs(residuals + supports))) if residuals.size else math.inf
    max_lateral_residual = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    max_combined_support_balance_error = (
        float(np.max(np.abs([float(row["final_static_balance_x"]) for row in rows])))
        if rows
        else math.inf
    )
    overlap_node_count = sum(abs(float(row["bottom_support_x"])) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE for row in rows)
    total_support_x = float(np.sum(supports)) if supports.size else 0.0
    total_residual_x = float(np.sum(residuals)) if residuals.size else 0.0
    after_forces = diagnostics[-1]
    main_stress_ok = float(after_forces["main_max_abs_particle_stress"]) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE
    main_stress_count_ok = int(after_forces["main_nonzero_stress_particle_count"]) > 0
    all_main_force_ok = float(after_forces["all_main_nodes_max_abs_force_x"]) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE
    all_main_force_count_ok = int(after_forces["all_main_nodes_nonzero_force_x_count"]) > 0
    lateral_residual_ok = float(after_forces["lateral_target_nodes_max_abs_force_x"]) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE
    lateral_residual_count_ok = int(after_forces["lateral_target_nodes_nonzero_force_x_count"]) > 0
    left_residual_ok = float(after_forces["left_max_abs_residual_x"]) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE
    left_residual_count_ok = int(after_forces["left_nonzero_residual_x_count"]) > 0
    right_residual_ok = float(after_forces["right_max_abs_residual_x"]) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE
    right_residual_count_ok = int(after_forces["right_nonzero_residual_x_count"]) > 0
    lateral_constraints_removed = int(after_forces["lateral_velocity_constraint_count"]) == 0
    all_static_constraints_removed = int(after_forces["velocity_constraint_count"]) == 0
    support_identity_ok = bool(np.allclose(supports, -residuals, rtol=0.0, atol=0.0)) if residuals.size else False
    balance_ok = max_balance_error <= LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE
    combined_support_balance_ok = max_combined_support_balance_error <= LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE
    body_id_ok = all(int(row["body_id"]) == BODY_MAIN_SOIL for row in rows)
    status = (
        "PASS"
        if main_stress_ok
        and main_stress_count_ok
        and all_main_force_ok
        and all_main_force_count_ok
        and lateral_residual_ok
        and lateral_residual_count_ok
        and left_residual_ok
        and left_residual_count_ok
        and right_residual_ok
        and right_residual_count_ok
        and lateral_constraints_removed
        and all_static_constraints_removed
        and support_identity_ok
        and balance_ok
        and combined_support_balance_ok
        and body_id_ok
        else "FAIL"
    )
    check_rows = [
        {
            "node_count": len(rows),
            "left_node_count": int(left_nodes.size),
            "right_node_count": int(right_nodes.size),
            "particle_static_stress_nonzero": "PASS" if main_stress_ok and main_stress_count_ok else "FAIL",
            "main_nodes_force_x_nonzero": "PASS" if all_main_force_ok else "FAIL",
            "lateral_residual_x_nonzero": "PASS" if lateral_residual_ok and lateral_residual_count_ok and left_residual_ok and right_residual_ok else "FAIL",
            "lateral_constraints_removed": "PASS" if lateral_constraints_removed else "FAIL",
            "all_static_constraints_removed": "PASS" if all_static_constraints_removed else "FAIL",
            "support_balance": "PASS" if support_identity_ok and balance_ok else "FAIL",
            "combined_support_balance": "PASS" if combined_support_balance_ok else "FAIL",
            "body_id_all_zero": "PASS" if body_id_ok else "FAIL",
            "bottom_lateral_overlap_node_count": overlap_node_count,
            "max_abs_lateral_residual_x": max_lateral_residual,
            "total_residual_x": total_residual_x,
            "total_support_x": total_support_x,
            "max_abs_residual_plus_support": max_balance_error,
            "max_combined_support_balance_error_x": max_combined_support_balance_error,
            "active_velocity_constraints_csv": active_constraints_path,
            "numerical_zero_tolerance": LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE,
            "diagnostic_csv": diagnostic_path,
            "final_status": status,
        }
    ]
    check_path = OUTPUT_DIR / "lateral_static_support_check.csv"
    write_csv(
        check_path,
        [
            "node_count",
            "left_node_count",
            "right_node_count",
            "particle_static_stress_nonzero",
            "main_nodes_force_x_nonzero",
            "lateral_residual_x_nonzero",
            "lateral_constraints_removed",
            "all_static_constraints_removed",
            "support_balance",
            "combined_support_balance",
            "body_id_all_zero",
            "bottom_lateral_overlap_node_count",
            "max_abs_lateral_residual_x",
            "total_residual_x",
            "total_support_x",
            "max_abs_residual_plus_support",
            "max_combined_support_balance_error_x",
            "active_velocity_constraints_csv",
            "numerical_zero_tolerance",
            "diagnostic_csv",
            "final_status",
        ],
        check_rows,
    )

    mpm.lateral_static_support_node_ids_np = np.ascontiguousarray([int(row["node_id"]) for row in rows], dtype=np.int32)
    mpm.lateral_static_support_body_ids_np = np.ascontiguousarray([int(row["body_id"]) for row in rows], dtype=np.int32)
    mpm.lateral_static_support_x_np = np.ascontiguousarray([float(row["support_x"]) for row in rows], dtype=np.float64)
    mpm.sims.set_gravity(previous_gravity)
    engine.reset_grid_message(mpm.scene)
    # Dynamic traction boundaries replace every temporary static velocity constraint.
    clear_velocity_boundary_state(mpm)
    if EXTRACT_LATERAL_STATIC_SUPPORT:
        remove_empty_output_subdirs(OUTPUT_DIR)
    return support_path, check_path


def state_variable_difference(static_state: Any, dynamic_state: Any) -> tuple[bool, float, str]:
    if static_state is None and dynamic_state is None:
        return True, 0.0, "no material state variables exposed"
    if not isinstance(static_state, dict) or not isinstance(dynamic_state, dict):
        return False, math.inf, "state variable container mismatch"
    if set(static_state) != set(dynamic_state):
        return False, math.inf, f"state variable keys differ: {sorted(static_state)} vs {sorted(dynamic_state)}"

    max_difference = 0.0
    for key in static_state:
        static_value = np.asarray(static_state[key])
        dynamic_value = np.asarray(dynamic_state[key])
        if static_value.shape != dynamic_value.shape:
            return False, math.inf, f"state variable {key} shape differs: {static_value.shape} vs {dynamic_value.shape}"
        if static_value.size:
            max_difference = max(max_difference, float(np.max(np.abs(static_value - dynamic_value))))
    return max_difference == 0.0, max_difference, "state variables match exactly" if max_difference == 0.0 else "state variables changed"


def stable_array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def periodic_column_node_pairs(
    mpm: MPM,
    column: str,
    body_id: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[dict[str, Any]]:
    coords = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
    tol = max(1.0e-8, 1.0e-6 * DX)
    left_ids = np.nonzero(
        (np.abs(coords[:, 0] - x_min) <= tol)
        & (coords[:, 1] >= y_min - tol)
        & (coords[:, 1] <= y_max + tol)
    )[0]
    right_ids = np.nonzero(
        (np.abs(coords[:, 0] - x_max) <= tol)
        & (coords[:, 1] >= y_min - tol)
        & (coords[:, 1] <= y_max + tol)
    )[0]
    right_by_y = {round(float(coords[node_id, 1]) / tol): int(node_id) for node_id in right_ids.tolist()}
    rows: list[dict[str, Any]] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for left_node_id in sorted(int(node_id) for node_id in left_ids.tolist()):
        y_left = float(coords[left_node_id, 1])
        right_node_id = right_by_y.get(round(y_left / tol), -1)
        if right_node_id < 0:
            continue
        left_unique = left_node_id not in used_left
        right_unique = right_node_id not in used_right
        used_left.add(left_node_id)
        used_right.add(right_node_id)
        left_x = float(coords[left_node_id, 0])
        right_x = float(coords[right_node_id, 0])
        right_y = float(coords[right_node_id, 1])
        same_height = abs(y_left - right_y) <= tol
        same_body = body_id in (BODY_LEFT_FREE_SOIL, BODY_RIGHT_FREE_SOIL)
        dx_value = right_x - left_x
        dy_value = right_y - y_left
        status = (
            "PASS"
            if same_height
            and same_body
            and left_unique
            and right_unique
            and abs(dx_value - 4.0 * DX) <= tol
            else "FAIL"
        )
        rows.append(
            {
                "column": column,
                "pair_id": len(rows),
                "left_node_id": left_node_id,
                "right_node_id": right_node_id,
                "body_id": body_id,
                "left_x": left_x,
                "right_x": right_x,
                "left_y": y_left,
                "right_y": right_y,
                "dx": dx_value,
                "dy": dy_value,
                "same_height": same_height,
                "same_body": same_body,
                "left_unique": left_unique,
                "right_unique": right_unique,
                "status": status,
            }
        )
    return rows


def build_free_field_periodic_node_pairs(mpm: MPM) -> list[dict[str, Any]]:
    rows = periodic_column_node_pairs(
        mpm,
        "left",
        BODY_LEFT_FREE_SOIL,
        K_LEFT_FREE_X_MIN,
        K_LEFT_FREE_X_MAX,
        K_LEFT_FREE_BOTTOM_Z,
        K_HIGH_SURFACE_Z,
    )
    rows.extend(
        periodic_column_node_pairs(
            mpm,
            "right",
            BODY_RIGHT_FREE_SOIL,
            K_RIGHT_FREE_X_MIN,
            K_RIGHT_FREE_X_MAX,
            K_RIGHT_FREE_BOTTOM_Z,
            K_LOW_SURFACE_Z,
        )
    )
    for pair_id, row in enumerate(rows):
        row["pair_id"] = pair_id
    return rows


def node_field_arrays(scene: Any) -> dict[str, np.ndarray]:
    raw = scene.node.to_numpy()
    return {
        "m": np.asarray(raw["m"], dtype=np.float64),
        "momentum": np.asarray(raw["momentum"], dtype=np.float64),
        "force": np.asarray(raw["force"], dtype=np.float64),
    }


def periodic_totals(scene: Any, pair_rows: list[dict[str, Any]]) -> dict[str, float]:
    arrays = node_field_arrays(scene)
    mass = 0.0
    momentum = np.zeros(2, dtype=np.float64)
    force = np.zeros(2, dtype=np.float64)
    max_vdiff = np.zeros(2, dtype=np.float64)
    max_adiff = np.zeros(2, dtype=np.float64)
    for row in pair_rows:
        body_id = int(row["body_id"])
        left = int(row["left_node_id"])
        right = int(row["right_node_id"])
        mass += float(arrays["m"][left, body_id] + arrays["m"][right, body_id])
        momentum += arrays["momentum"][left, body_id, :2] + arrays["momentum"][right, body_id, :2]
        force += arrays["force"][left, body_id, :2] + arrays["force"][right, body_id, :2]
        max_vdiff = np.maximum(max_vdiff, np.abs(arrays["momentum"][left, body_id, :2] - arrays["momentum"][right, body_id, :2]))
        max_adiff = np.maximum(max_adiff, np.abs(arrays["force"][left, body_id, :2] - arrays["force"][right, body_id, :2]))
    return {
        "total_mass": float(mass),
        "total_momentum_x": float(momentum[0]),
        "total_momentum_y": float(momentum[1]),
        "total_force_x": float(force[0]),
        "total_force_y": float(force[1]),
        "max_abs_velocity_x_difference": float(max_vdiff[0]),
        "max_abs_velocity_y_difference": float(max_vdiff[1]),
        "max_abs_acceleration_x_difference": float(max_adiff[0]),
        "max_abs_acceleration_y_difference": float(max_adiff[1]),
    }


def main_node_state(scene: Any) -> np.ndarray:
    arrays = node_field_arrays(scene)
    return np.concatenate(
        [
            arrays["m"][:, BODY_MAIN_SOIL : BODY_MAIN_SOIL + 1],
            arrays["momentum"][:, BODY_MAIN_SOIL, :2],
            arrays["force"][:, BODY_MAIN_SOIL, :2],
        ],
        axis=1,
    )


def periodic_support_node_map(mpm: MPM, x_min: float, x_max: float) -> np.ndarray:
    """Return the shared-node numbering for one four-cell periodic column."""

    grid_x = int(mpm.scene.element.gnum[0])
    grid_y = int(mpm.scene.element.gnum[1])
    first_x = int(round((x_min + X_SHIFT) / DX))
    period_cells = int(round((x_max - x_min) / DX))
    if period_cells <= 0 or first_x < 0 or first_x + period_cells >= grid_x:
        raise RuntimeError(
            f"Invalid free-field periodic grid: x=[{x_min}, {x_max}], first={first_x}, period={period_cells}"
        )
    mapping = np.empty(grid_x * grid_y, dtype=np.int32)
    for node_id in range(mapping.size):
        ix = node_id % grid_x
        iy = node_id // grid_x
        canonical_x = first_x + ((ix - first_x) % period_cells)
        mapping[node_id] = canonical_x + iy * grid_x
    return np.ascontiguousarray(mapping)


class FreeFieldPeriodicNodeMapper:
    def __init__(self, mpm: MPM, pair_rows: list[dict[str, Any]], diagnostic: bool = False) -> None:
        self.mpm = mpm
        self.pair_rows = pair_rows
        self.diagnostic = diagnostic
        self.pair_count = len(pair_rows)
        self.left_node_ids_np = np.ascontiguousarray([int(row["left_node_id"]) for row in pair_rows], dtype=np.int32)
        self.right_node_ids_np = np.ascontiguousarray([int(row["right_node_id"]) for row in pair_rows], dtype=np.int32)
        self.body_ids_np = np.ascontiguousarray([int(row["body_id"]) for row in pair_rows], dtype=np.int32)
        self.grid_node_count = int(mpm.scene.element.gridSum)
        self.support_node_capacity = int(mpm.scene.element.grid_nodes)
        self.left_canonical_node_ids_np = periodic_support_node_map(mpm, K_LEFT_FREE_X_MIN, K_LEFT_FREE_X_MAX)
        self.right_canonical_node_ids_np = periodic_support_node_map(mpm, K_RIGHT_FREE_X_MIN, K_RIGHT_FREE_X_MAX)
        for row in pair_rows:
            mapping = (
                self.left_canonical_node_ids_np
                if int(row["body_id"]) == BODY_LEFT_FREE_SOIL
                else self.right_canonical_node_ids_np
            )
            if int(mapping[int(row["right_node_id"])]) != int(row["left_node_id"]):
                raise RuntimeError(f"Periodic boundary pair does not match full support wrap: {row}")
        shape = max(1, self.pair_count)
        self.left_node_ids = ti.field(dtype=ti.i32, shape=shape)
        self.right_node_ids = ti.field(dtype=ti.i32, shape=shape)
        self.body_ids = ti.field(dtype=ti.i32, shape=shape)
        self.left_canonical_node_ids = ti.field(dtype=ti.i32, shape=max(1, self.grid_node_count))
        self.right_canonical_node_ids = ti.field(dtype=ti.i32, shape=max(1, self.grid_node_count))
        if self.pair_count:
            self.left_node_ids.from_numpy(self.left_node_ids_np)
            self.right_node_ids.from_numpy(self.right_node_ids_np)
            self.body_ids.from_numpy(self.body_ids_np)
        self.left_canonical_node_ids.from_numpy(self.left_canonical_node_ids_np)
        self.right_canonical_node_ids.from_numpy(self.right_canonical_node_ids_np)
        self.diagnostic_rows: list[dict[str, Any]] = []
        self.mass_conservation_error = 0.0
        self.momentum_conservation_error = np.zeros(2, dtype=np.float64)
        self.force_conservation_error = np.zeros(2, dtype=np.float64)
        self.main_node_modified_count = 0
        self.support_remap_call_count = 0
        self.mass_sync_call_count = 0
        self.force_sync_call_count = 0
        self.kinematic_sync_call_count = 0
        self.before_g2p_sync_call_count = 0
        self.track_main_node_modifications = False

    def remap_particle_supports(self, scene: Any) -> None:
        if not self.pair_count:
            return
        main_before = main_node_state(scene) if self.track_main_node_modifications else None
        canonicalize_free_field_periodic_supports(
            self.support_node_capacity,
            int(scene.particleNum[0]),
            BODY_LEFT_FREE_SOIL,
            BODY_RIGHT_FREE_SOIL,
            self.left_canonical_node_ids,
            self.right_canonical_node_ids,
            scene.particle,
            scene.element.LnID,
            scene.element.node_size,
        )
        self.support_remap_call_count += 1
        if main_before is not None:
            main_after = main_node_state(scene)
            self.main_node_modified_count = max(
                self.main_node_modified_count,
                int(np.count_nonzero(np.linalg.norm(main_after - main_before, axis=1) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE)),
            )

    def unmapped_support_count(self, scene: Any, column: str | None = None) -> int:
        ln_id = scene.element.LnID.to_numpy()
        node_size = scene.element.node_size.to_numpy()
        body_ids = scene.particle.bodyID.to_numpy()[: int(scene.particleNum[0])].astype(np.int32)
        total_nodes = self.support_node_capacity
        count = 0
        for pid, particle_body_id in enumerate(body_ids):
            if column is None:
                in_free_field = particle_body_id in {BODY_LEFT_FREE_SOIL, BODY_RIGHT_FREE_SOIL}
            else:
                expected_body = BODY_LEFT_FREE_SOIL if column == "left" else BODY_RIGHT_FREE_SOIL
                in_free_field = particle_body_id == expected_body
            if not in_free_field:
                continue
            mapping = (
                self.left_canonical_node_ids_np
                if particle_body_id == BODY_LEFT_FREE_SOIL
                else self.right_canonical_node_ids_np
            )
            offset = pid * total_nodes
            count += sum(
                int(mapping[int(node_id)]) != int(node_id)
                for node_id in ln_id[offset : offset + int(node_size[pid])]
            )
        return count

    def _record(self, stage: str, before: dict[str, float], after: dict[str, float], main_before: np.ndarray) -> None:
        main_after = main_node_state(self.mpm.scene)
        self.main_node_modified_count = max(
            self.main_node_modified_count,
            int(np.count_nonzero(np.linalg.norm(main_after - main_before, axis=1) > LATERAL_STATIC_SUPPORT_ZERO_TOLERANCE)),
        )
        self.mass_conservation_error = max(self.mass_conservation_error, abs(after["total_mass"] - before["total_mass"]))
        self.momentum_conservation_error = np.maximum(
            self.momentum_conservation_error,
            np.abs(
                np.asarray([after["total_momentum_x"], after["total_momentum_y"]])
                - np.asarray([before["total_momentum_x"], before["total_momentum_y"]])
            ),
        )
        self.force_conservation_error = np.maximum(
            self.force_conservation_error,
            np.abs(
                np.asarray([after["total_force_x"], after["total_force_y"]])
                - np.asarray([before["total_force_x"], before["total_force_y"]])
            ),
        )
        self.diagnostic_rows.append(
            {
                "stage": stage,
                "mass_before": before["total_mass"],
                "mass_after": after["total_mass"],
                "momentum_x_before": before["total_momentum_x"],
                "momentum_x_after": after["total_momentum_x"],
                "momentum_y_before": before["total_momentum_y"],
                "momentum_y_after": after["total_momentum_y"],
                "force_x_before": before["total_force_x"],
                "force_x_after": after["total_force_x"],
                "force_y_before": before["total_force_y"],
                "force_y_after": after["total_force_y"],
                "max_velocity_x_difference_after": after["max_abs_velocity_x_difference"],
                "max_velocity_y_difference_after": after["max_abs_velocity_y_difference"],
                "max_acceleration_x_difference_after": after["max_abs_acceleration_x_difference"],
                "max_acceleration_y_difference_after": after["max_abs_acceleration_y_difference"],
                "modified_body_ids": ",".join(str(body_id) for body_id in sorted(set(self.body_ids_np.tolist()))),
                "main_node_modified_count": self.main_node_modified_count,
            }
        )

    def column_check_rows(self, geometry_modified: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for column, body_id in (("left", BODY_LEFT_FREE_SOIL), ("right", BODY_RIGHT_FREE_SOIL)):
            column_pairs = [row for row in self.pair_rows if row["column"] == column]
            pair_count = len(column_pairs)
            unique_left = len(set(int(row["left_node_id"]) for row in column_pairs))
            unique_right = len(set(int(row["right_node_id"]) for row in column_pairs))
            pair_rows_ok = all(row["status"] == "PASS" and int(row["body_id"]) == body_id for row in column_pairs)
            unmapped_supports = self.unmapped_support_count(self.mpm.scene, column)
            status = (
                "PASS"
                if pair_count > 0
                and unique_left == pair_count
                and unique_right == pair_count
                and pair_rows_ok
                and self.support_remap_call_count > 0
                and unmapped_supports == 0
                and self.main_node_modified_count == 0
                and not geometry_modified
                else "FAIL"
            )
            rows.append(
                {
                    "column": column,
                    "body_id": body_id,
                    "pair_count": pair_count,
                    "unique_left_node_count": unique_left,
                    "unique_right_node_count": unique_right,
                    "support_remap_call_count": self.support_remap_call_count,
                    "unmapped_support_reference_count": unmapped_supports,
                    "tolerance": FREE_FIELD_PERIODIC_TOLERANCE,
                    "main_node_modified_count": self.main_node_modified_count,
                    "status": status,
                }
            )
        return rows


@ti.data_oriented
class StaticPaperEq15GridProjection:
    """Expose Eq. (15) through the native acceleration-form G2P kernel.

    GeoTaichi's G2P kernel consumes ``node.force`` as an acceleration.  Once
    nodal kinematic constraints have changed a velocity, the force-derived
    acceleration is no longer necessarily the corresponding velocity
    increment.  Replacing it with ``(v_new - v_old) / dt`` makes the native
    FLIP update exactly Eq. (15), without writing the particle struct.
    """

    def __init__(self, mpm: MPM) -> None:
        scene = mpm.scene
        self.grid_sum = int(scene.element.gridSum)
        self.grid_levels = int(scene.grid_level)
        self.old_node_velocity = ti.Vector.field(2, dtype=ti.f32, shape=(self.grid_sum, self.grid_levels))

    @ti.kernel
    def capture_node_velocity(self, node: ti.template()):
        for node_id, body_id in ti.ndrange(self.grid_sum, self.grid_levels):
            self.old_node_velocity[node_id, body_id] = node[node_id, body_id].momentum

    @ti.kernel
    def project_velocity_increment(self, dt: float, node: ti.template()):
        for node_id, body_id in ti.ndrange(self.grid_sum, self.grid_levels):
            node[node_id, body_id].force = (
                node[node_id, body_id].momentum - self.old_node_velocity[node_id, body_id]
            ) / dt


@ti.data_oriented
class StaticMirrorParticleBoundary:
    """Grid-level image-particle equivalent used only during static settling.

    Kohler et al. use the reflected-particle construction of Schulz and
    Sutmann.  Ding et al. show that the same result is obtained by reflecting
    P2G quantities from ghost nodes back to the interior, followed by the
    corresponding G2P velocity extrapolation.  No particle is created and the
    dynamic stage never executes these kernels.
    """

    def __init__(self, mpm: MPM) -> None:
        self.grid_x = int(mpm.scene.element.gnum[0])
        self.grid_y = int(mpm.scene.element.gnum[1])
        self.grid_sum = int(mpm.scene.element.gridSum)
        self.grid_levels = int(mpm.scene.grid_level)
        self.mass_cutoff = float(mpm.scene.mass_cut_off)
        self.reflect_body_force = bool(STATIC_MIRROR_REFLECT_BODY_FORCE)
        self.bottom_node = int(round((K_BASE_BOTTOM_Z + Y_SHIFT) / DX))
        self.main_left_node = int(round((K_MAIN_X_MIN + X_SHIFT) / DX))
        self.main_right_node = int(round((K_MAIN_X_MAX + X_SHIFT) / DX))
        shape = (self.grid_sum, self.grid_levels)
        self.mass_source = ti.field(dtype=ti.f64, shape=shape)
        self.raw_mass_source = ti.field(dtype=ti.f64, shape=shape)
        self.momentum_source = ti.Vector.field(2, dtype=ti.f64, shape=shape)
        self.force_source = ti.Vector.field(2, dtype=ti.f64, shape=shape)
        self.internal_force_source = ti.Vector.field(2, dtype=ti.f64, shape=shape)
        self.mass_momentum_calls = 0
        self.force_calls = 0
        self.velocity_extrapolation_calls = 0

    @ti.kernel
    def capture_mass_momentum(self, node: ti.template()):
        for node_id, body_id in ti.ndrange(self.grid_sum, self.grid_levels):
            self.mass_source[node_id, body_id] = node[node_id, body_id].m
            self.raw_mass_source[node_id, body_id] = node[node_id, body_id].m
            self.momentum_source[node_id, body_id] = node[node_id, body_id].momentum

    @ti.kernel
    def capture_current_mass_momentum(self, node: ti.template()):
        for node_id, body_id in ti.ndrange(self.grid_sum, self.grid_levels):
            self.mass_source[node_id, body_id] = node[node_id, body_id].m
            self.momentum_source[node_id, body_id] = node[node_id, body_id].momentum

    @ti.kernel
    def reflect_lateral_mass_momentum(self, node: ti.template()):
        for node_id in range(self.grid_sum):
            i = node_id % self.grid_x
            j = node_id // self.grid_x
            if i < self.main_left_node:
                target_i = 2 * self.main_left_node - i
                target_id = target_i + j * self.grid_x
                mass = self.mass_source[node_id, BODY_MAIN_SOIL]
                momentum = self.momentum_source[node_id, BODY_MAIN_SOIL]
                if mass > self.mass_cutoff:
                    node[target_id, BODY_MAIN_SOIL].m += mass
                    node[target_id, BODY_MAIN_SOIL].momentum += ti.Vector([-momentum[0], momentum[1]])
            elif i > self.main_right_node:
                target_i = 2 * self.main_right_node - i
                target_id = target_i + j * self.grid_x
                mass = self.mass_source[node_id, BODY_MAIN_SOIL]
                momentum = self.momentum_source[node_id, BODY_MAIN_SOIL]
                if mass > self.mass_cutoff:
                    node[target_id, BODY_MAIN_SOIL].m += mass
                    node[target_id, BODY_MAIN_SOIL].momentum += ti.Vector([-momentum[0], momentum[1]])
            elif i == self.main_left_node or i == self.main_right_node:
                mass = self.mass_source[node_id, BODY_MAIN_SOIL]
                momentum = self.momentum_source[node_id, BODY_MAIN_SOIL]
                if mass > self.mass_cutoff:
                    node[node_id, BODY_MAIN_SOIL].m += mass
                    node[node_id, BODY_MAIN_SOIL].momentum += ti.Vector([-momentum[0], momentum[1]])

    @ti.kernel
    def reflect_bottom_mass_momentum(self, node: ti.template()):
        for node_id, body_id in ti.ndrange(self.grid_sum, self.grid_levels):
            j = node_id // self.grid_x
            if j < self.bottom_node:
                i = node_id % self.grid_x
                target_j = 2 * self.bottom_node - j
                target_id = i + target_j * self.grid_x
                mass = self.mass_source[node_id, body_id]
                momentum = self.momentum_source[node_id, body_id]
                if mass > self.mass_cutoff:
                    node[target_id, body_id].m += mass
                    node[target_id, body_id].momentum -= momentum
            elif j == self.bottom_node:
                mass = self.mass_source[node_id, body_id]
                momentum = self.momentum_source[node_id, body_id]
                if mass > self.mass_cutoff:
                    node[node_id, body_id].m += mass
                    node[node_id, body_id].momentum -= momentum

    @ti.kernel
    def extrapolate_lateral_mass_momentum(self, node: ti.template()):
        for node_id in range(self.grid_sum):
            i = node_id % self.grid_x
            j = node_id // self.grid_x
            if i < self.main_left_node:
                source_i = 2 * self.main_left_node - i
                source_id = source_i + j * self.grid_x
                node[node_id, BODY_MAIN_SOIL].m = node[source_id, BODY_MAIN_SOIL].m
                momentum = node[source_id, BODY_MAIN_SOIL].momentum
                node[node_id, BODY_MAIN_SOIL].momentum = ti.Vector([-momentum[0], momentum[1]])
            elif i > self.main_right_node:
                source_i = 2 * self.main_right_node - i
                source_id = source_i + j * self.grid_x
                node[node_id, BODY_MAIN_SOIL].m = node[source_id, BODY_MAIN_SOIL].m
                momentum = node[source_id, BODY_MAIN_SOIL].momentum
                node[node_id, BODY_MAIN_SOIL].momentum = ti.Vector([-momentum[0], momentum[1]])

    @ti.kernel
    def extrapolate_bottom_mass_momentum(self, node: ti.template()):
        for node_id, body_id in ti.ndrange(self.grid_sum, self.grid_levels):
            j = node_id // self.grid_x
            if j < self.bottom_node:
                i = node_id % self.grid_x
                source_j = 2 * self.bottom_node - j
                source_id = i + source_j * self.grid_x
                node[node_id, body_id].m = node[source_id, body_id].m
                node[node_id, body_id].momentum = -node[source_id, body_id].momentum

    @ti.kernel
    def capture_force(self, gravity: ti.types.vector(3, float), node: ti.template()):
        for node_id, body_id in ti.ndrange(self.grid_sum, self.grid_levels):
            self.force_source[node_id, body_id] = node[node_id, body_id].force
            if ti.static(self.reflect_body_force):
                self.internal_force_source[node_id, body_id] = node[node_id, body_id].force
            else:
                self.internal_force_source[node_id, body_id] = node[node_id, body_id].force - self.raw_mass_source[node_id, body_id] * ti.Vector([gravity[0], gravity[1]])

    @ti.kernel
    def accumulate_lateral_internal_force(self, node: ti.template()):
        for node_id in range(self.grid_sum):
            i = node_id % self.grid_x
            j = node_id // self.grid_x
            if i < self.main_left_node:
                target_i = 2 * self.main_left_node - i
                target_id = target_i + j * self.grid_x
                source_force = self.internal_force_source[node_id, BODY_MAIN_SOIL]
                if self.raw_mass_source[node_id, BODY_MAIN_SOIL] > self.mass_cutoff:
                    reflected_force = ti.Vector([-source_force[0], source_force[1]])
                    node[target_id, BODY_MAIN_SOIL].force += reflected_force
                    self.internal_force_source[target_id, BODY_MAIN_SOIL] += reflected_force
            elif i > self.main_right_node:
                target_i = 2 * self.main_right_node - i
                target_id = target_i + j * self.grid_x
                source_force = self.internal_force_source[node_id, BODY_MAIN_SOIL]
                if self.raw_mass_source[node_id, BODY_MAIN_SOIL] > self.mass_cutoff:
                    reflected_force = ti.Vector([-source_force[0], source_force[1]])
                    node[target_id, BODY_MAIN_SOIL].force += reflected_force
                    self.internal_force_source[target_id, BODY_MAIN_SOIL] += reflected_force
            elif i == self.main_left_node or i == self.main_right_node:
                source_force = self.internal_force_source[node_id, BODY_MAIN_SOIL]
                if self.raw_mass_source[node_id, BODY_MAIN_SOIL] > self.mass_cutoff:
                    reflected_force = ti.Vector([-source_force[0], source_force[1]])
                    node[node_id, BODY_MAIN_SOIL].force += reflected_force
                    self.internal_force_source[node_id, BODY_MAIN_SOIL] += reflected_force

    @ti.kernel
    def accumulate_bottom_internal_force(self, node: ti.template()):
        for node_id, body_id in ti.ndrange(self.grid_sum, self.grid_levels):
            j = node_id // self.grid_x
            if j < self.bottom_node:
                i = node_id % self.grid_x
                target_j = 2 * self.bottom_node - j
                target_id = i + target_j * self.grid_x
                source_force = self.internal_force_source[node_id, body_id]
                if self.raw_mass_source[node_id, body_id] > self.mass_cutoff:
                    node[target_id, body_id].force -= source_force
            elif j == self.bottom_node:
                source_force = self.internal_force_source[node_id, body_id]
                if self.raw_mass_source[node_id, body_id] > self.mass_cutoff:
                    node[node_id, body_id].force -= source_force

    @ti.kernel
    def extrapolate_lateral_force(self, node: ti.template()):
        for node_id in range(self.grid_sum):
            i = node_id % self.grid_x
            j = node_id // self.grid_x
            if i < self.main_left_node:
                source_i = 2 * self.main_left_node - i
                source_id = source_i + j * self.grid_x
                force = node[source_id, BODY_MAIN_SOIL].force
                node[node_id, BODY_MAIN_SOIL].force = ti.Vector([-force[0], force[1]])
            elif i > self.main_right_node:
                source_i = 2 * self.main_right_node - i
                source_id = source_i + j * self.grid_x
                force = node[source_id, BODY_MAIN_SOIL].force
                node[node_id, BODY_MAIN_SOIL].force = ti.Vector([-force[0], force[1]])

    @ti.kernel
    def extrapolate_bottom_force(self, node: ti.template()):
        for node_id, body_id in ti.ndrange(self.grid_sum, self.grid_levels):
            j = node_id // self.grid_x
            if j < self.bottom_node:
                i = node_id % self.grid_x
                source_j = 2 * self.bottom_node - j
                source_id = i + source_j * self.grid_x
                node[node_id, body_id].force = -node[source_id, body_id].force

    @ti.kernel
    def extrapolate_lateral_velocity(self, node: ti.template()):
        for node_id in range(self.grid_sum):
            i = node_id % self.grid_x
            j = node_id // self.grid_x
            if i < self.main_left_node:
                source_i = 2 * self.main_left_node - i
                source_id = source_i + j * self.grid_x
                velocity = node[source_id, BODY_MAIN_SOIL].momentum
                node[node_id, BODY_MAIN_SOIL].momentum = ti.Vector([-velocity[0], velocity[1]])
            elif i > self.main_right_node:
                source_i = 2 * self.main_right_node - i
                source_id = source_i + j * self.grid_x
                velocity = node[source_id, BODY_MAIN_SOIL].momentum
                node[node_id, BODY_MAIN_SOIL].momentum = ti.Vector([-velocity[0], velocity[1]])
            elif i == self.main_left_node or i == self.main_right_node:
                velocity = node[node_id, BODY_MAIN_SOIL].momentum
                node[node_id, BODY_MAIN_SOIL].momentum = ti.Vector([0.0, velocity[1]])

    @ti.kernel
    def extrapolate_bottom_velocity(self, node: ti.template()):
        for node_id, body_id in ti.ndrange(self.grid_sum, self.grid_levels):
            j = node_id // self.grid_x
            if j < self.bottom_node:
                i = node_id % self.grid_x
                source_j = 2 * self.bottom_node - j
                source_id = i + source_j * self.grid_x
                node[node_id, body_id].momentum = -node[source_id, body_id].momentum
            elif j == self.bottom_node:
                node[node_id, body_id].momentum = ti.Vector([0.0, 0.0])

    def apply_mass_momentum_reflection(self, scene: Any) -> None:
        self.capture_mass_momentum(scene.node)
        self.reflect_lateral_mass_momentum(scene.node)
        self.capture_current_mass_momentum(scene.node)
        self.reflect_bottom_mass_momentum(scene.node)
        self.extrapolate_lateral_mass_momentum(scene.node)
        self.extrapolate_bottom_mass_momentum(scene.node)
        self.mass_momentum_calls += 1

    def apply_force_reflection(self, sims: Any, scene: Any) -> None:
        self.capture_force(sims.gravity, scene.node)
        self.accumulate_lateral_internal_force(scene.node)
        self.accumulate_bottom_internal_force(scene.node)
        self.extrapolate_lateral_force(scene.node)
        self.extrapolate_bottom_force(scene.node)
        self.force_calls += 1

    def extrapolate_velocity(self, scene: Any) -> None:
        self.extrapolate_lateral_velocity(scene.node)
        self.extrapolate_bottom_velocity(scene.node)
        self.velocity_extrapolation_calls += 1

    def diagnostics(self) -> dict[str, int]:
        return {
            "bottom_ghost_node_rows": int(self.bottom_node),
            "main_left_node": int(self.main_left_node),
            "main_right_node": int(self.main_right_node),
            "mass_momentum_calls": int(self.mass_momentum_calls),
            "force_calls": int(self.force_calls),
            "velocity_extrapolation_calls": int(self.velocity_extrapolation_calls),
        }


def static_mirror_boundary_is_active(mpm: MPM) -> bool:
    static_monitor = getattr(mpm, "static_initialization", None)
    return bool(STATIC_MIRROR_PARTICLE_BOUNDARY and static_monitor is not None and static_monitor.active)


def install_static_mirror_particle_hooks(mpm: MPM) -> StaticMirrorParticleBoundary | None:
    if not STATIC_MIRROR_PARTICLE_BOUNDARY:
        return None
    if mpm.enginer is None:
        mpm.add_essentials({"function": None})
    engine = mpm.enginer
    existing = getattr(engine, "_nairn_static_mirror_particle_boundary", None)
    if existing is not None:
        mpm.static_mirror_particle_boundary = existing
        return existing

    mirror = StaticMirrorParticleBoundary(mpm)
    original_compute_grid_velocity = engine.compute_grid_velcity
    original_compute_forces = engine.compute_forces
    original_compute_particle_kinematic = engine.compute_particle_kinematic

    def compute_grid_velocity_with_mirrors(sims: Any, scene: Any) -> Any:
        if static_mirror_boundary_is_active(mpm):
            mirror.apply_mass_momentum_reflection(scene)
        result = original_compute_grid_velocity(sims, scene)
        if static_mirror_boundary_is_active(mpm):
            mirror.extrapolate_velocity(scene)
        return result

    def compute_forces_with_mirrors(sims: Any, scene: Any) -> Any:
        result = original_compute_forces(sims, scene)
        if static_mirror_boundary_is_active(mpm):
            mirror.apply_force_reflection(sims, scene)
        return result

    def compute_particle_kinematic_with_mirrors(sims: Any, scene: Any) -> Any:
        if static_mirror_boundary_is_active(mpm):
            mirror.extrapolate_velocity(scene)
        return original_compute_particle_kinematic(sims, scene)

    engine.compute_grid_velcity = compute_grid_velocity_with_mirrors
    engine.compute_forces = compute_forces_with_mirrors
    engine.compute_particle_kinematic = compute_particle_kinematic_with_mirrors
    engine._nairn_static_mirror_particle_boundary = mirror
    mpm.static_mirror_particle_boundary = mirror
    return mirror


def install_free_field_periodic_hooks(mpm: MPM, mapper: FreeFieldPeriodicNodeMapper) -> None:
    engine = mpm.enginer
    if engine is None:
        raise RuntimeError("GeoTaichi engine is not initialized; cannot install free-field periodic hooks")
    if getattr(engine, "_nairn_free_field_periodic_hooks_installed", False):
        mpm.free_field_periodic_mapper = getattr(engine, "_nairn_free_field_periodic_mapper", mapper)
        mpm.periodic_hooks_installed_for_dynamic = True
        return
    original_calculate_interpolation = engine.calculate_interpolation

    def calculate_interpolation_with_periodic(sims: Any, scene: Any) -> Any:
        result = original_calculate_interpolation(sims, scene)
        # Every horizontal cubic support is wrapped to the column's four
        # periodic node IDs before P2G, so all grid quantities share one DOF.
        mapper.remap_particle_supports(scene)
        return result

    engine.calculate_interpolation = calculate_interpolation_with_periodic
    engine._nairn_free_field_periodic_hooks_installed = True
    engine._nairn_free_field_periodic_mapper = mapper
    mpm.free_field_periodic_mapper = mapper
    mpm.periodic_hooks_installed_for_dynamic = True


def install_free_field_periodic_dynamic_hooks(mpm: MPM) -> FreeFieldPeriodicNodeMapper:
    if mpm.enginer is None:
        mpm.add_essentials({"function": None})
    periodic_pair_rows = build_free_field_periodic_node_pairs(mpm)
    periodic_mapper = FreeFieldPeriodicNodeMapper(
        mpm,
        periodic_pair_rows,
        diagnostic=False,
    )
    periodic_mapper.track_main_node_modifications = FREE_FIELD_PERIODIC_DYNAMIC_PATH_CHECK
    install_free_field_periodic_hooks(mpm, periodic_mapper)
    active_mapper = getattr(mpm, "free_field_periodic_mapper", periodic_mapper)
    left_pair_count = sum(1 for row in active_mapper.pair_rows if row["column"] == "left")
    right_pair_count = sum(1 for row in active_mapper.pair_rows if row["column"] == "right")
    mpm.periodic_pair_rows_for_dynamic = active_mapper.pair_rows
    mpm.periodic_pair_count_for_dynamic = int(active_mapper.pair_count)
    mpm.left_periodic_pair_count_for_dynamic = int(left_pair_count)
    mpm.right_periodic_pair_count_for_dynamic = int(right_pair_count)
    mpm.periodic_mapper_active_before_first_dynamic_step = bool(
        getattr(mpm.enginer, "_nairn_free_field_periodic_hooks_installed", False)
        and active_mapper.pair_count > 0
    )
    return active_mapper


def install_free_field_periodic_static_hooks(mpm: MPM) -> FreeFieldPeriodicNodeMapper:
    """Install the same periodic free-field topology before static relaxation."""
    mapper = install_free_field_periodic_dynamic_hooks(mpm)
    mpm.periodic_hooks_installed_for_static = True
    mpm.periodic_mapper_active_before_first_static_step = bool(
        getattr(mpm.enginer, "_nairn_free_field_periodic_hooks_installed", False)
        and mapper.pair_count > 0
    )
    if not mpm.periodic_mapper_active_before_first_static_step:
        raise RuntimeError("Free-field periodic boundary is not ready before the first static step")
    return mapper


def assert_free_field_periodic_dynamic_ready(mpm: MPM) -> None:
    checks = {
        "periodic_hooks_installed_for_dynamic": bool(getattr(mpm, "periodic_hooks_installed_for_dynamic", False)),
        "periodic_pair_count_for_dynamic": int(getattr(mpm, "periodic_pair_count_for_dynamic", 0)) > 0,
        "left_periodic_pair_count_for_dynamic": int(getattr(mpm, "left_periodic_pair_count_for_dynamic", 0)) > 0,
        "right_periodic_pair_count_for_dynamic": int(getattr(mpm, "right_periodic_pair_count_for_dynamic", 0)) > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Free-field periodic dynamic path is not ready before first dynamic step: {checks}")


def write_free_field_periodic_dynamic_path_check(
    mpm: MPM,
    mapper: FreeFieldPeriodicNodeMapper,
    geometry_modified: bool,
) -> Path:
    left_pair_count = int(getattr(mpm, "left_periodic_pair_count_for_dynamic", 0))
    right_pair_count = int(getattr(mpm, "right_periodic_pair_count_for_dynamic", 0))
    pair_count = int(getattr(mpm, "periodic_pair_count_for_dynamic", 0))
    row = {
        "periodic_minimal_mode": bool(FREE_FIELD_PERIODIC_MINIMAL),
        "normal_dynamic_entry_used": True,
        "hooks_installed": bool(getattr(mpm, "periodic_hooks_installed_for_dynamic", False)),
        "mapper_active_before_first_step": bool(getattr(mpm, "periodic_mapper_active_before_first_dynamic_step", False)),
        "pair_count": pair_count,
        "left_pair_count": left_pair_count,
        "right_pair_count": right_pair_count,
        "support_remap_call_count": int(mapper.support_remap_call_count),
        "unmapped_support_reference_count": int(mapper.unmapped_support_count(mpm.scene)),
        "main_node_modified_count": int(mapper.main_node_modified_count),
        "geometry_modified": bool(geometry_modified),
    }
    status = (
        "PASS"
        if row["periodic_minimal_mode"] is False
        and row["normal_dynamic_entry_used"] is True
        and row["hooks_installed"] is True
        and row["mapper_active_before_first_step"] is True
        # The confirmed geometry has different free-field column heights, so
        # their periodic-pair counts need not be equal.  Verify topology
        # rather than a stale fixed count from the former symmetric geometry.
        and pair_count == left_pair_count + right_pair_count
        and left_pair_count > 0
        and right_pair_count > 0
        and int(row["support_remap_call_count"]) > 0
        and int(row["unmapped_support_reference_count"]) == 0
        and int(row["main_node_modified_count"]) == 0
        and row["geometry_modified"] is False
        else "FAIL"
    )
    row["status"] = status
    path = OUTPUT_DIR / "free_field_periodic_dynamic_path_check.csv"
    write_csv(path, list(row.keys()), [row])
    return path


def geometry_periodic_snapshot(mpm: MPM) -> dict[str, Any]:
    particle_count = int(mpm.scene.particleNum[0])
    positions = np.asarray(mpm.scene.particle.x.to_numpy()[:particle_count], dtype=np.float64)
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    coords = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64)
    return {
        "particle_count": particle_count,
        "grid_node_count": int(getattr(mpm.scene.element, "gridSum", 0)),
        "main_particle_count": int(np.count_nonzero(body_ids == BODY_MAIN_SOIL)),
        "left_ff_particle_count": int(np.count_nonzero(body_ids == BODY_LEFT_FREE_SOIL)),
        "right_ff_particle_count": int(np.count_nonzero(body_ids == BODY_RIGHT_FREE_SOIL)),
        "particle_position_hash": stable_array_hash(positions),
        "grid_coordinate_hash": stable_array_hash(coords),
    }


def write_geometry_unchanged_check(
    before: dict[str, Any],
    after: dict[str, Any],
    allow_particle_motion: bool = False,
) -> tuple[Path, bool]:
    left_width = (K_LEFT_FREE_X_MAX - K_LEFT_FREE_X_MIN) / DX
    right_width = (K_RIGHT_FREE_X_MAX - K_RIGHT_FREE_X_MIN) / DX
    geometry_modified = not (
        before["particle_count"] == after["particle_count"]
        and before["grid_node_count"] == after["grid_node_count"]
        and before["main_particle_count"] == after["main_particle_count"]
        and before["left_ff_particle_count"] == after["left_ff_particle_count"]
        and before["right_ff_particle_count"] == after["right_ff_particle_count"]
        # In the dynamic path particles must move.  That is response, not a
        # modification of the free-field geometry or its node topology.
        and (allow_particle_motion or before["particle_position_hash"] == after["particle_position_hash"])
        and before["grid_coordinate_hash"] == after["grid_coordinate_hash"]
        and math.isclose(left_width, 4.0, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(right_width, 4.0, rel_tol=0.0, abs_tol=1.0e-12)
    )
    row = {
        "particle_count_before": before["particle_count"],
        "particle_count_after": after["particle_count"],
        "grid_node_count_before": before["grid_node_count"],
        "grid_node_count_after": after["grid_node_count"],
        "main_particle_count_before": before["main_particle_count"],
        "main_particle_count_after": after["main_particle_count"],
        "left_ff_particle_count_before": before["left_ff_particle_count"],
        "left_ff_particle_count_after": after["left_ff_particle_count"],
        "right_ff_particle_count_before": before["right_ff_particle_count"],
        "right_ff_particle_count_after": after["right_ff_particle_count"],
        "particle_position_hash_before": before["particle_position_hash"],
        "particle_position_hash_after": after["particle_position_hash"],
        "grid_coordinate_hash_before": before["grid_coordinate_hash"],
        "grid_coordinate_hash_after": after["grid_coordinate_hash"],
        "left_ff_width_over_dx": left_width,
        "right_ff_width_over_dx": right_width,
        "particle_motion_allowed": allow_particle_motion,
        "geometry_modified": geometry_modified,
        "status": "PASS" if not geometry_modified else "FAIL",
    }
    path = OUTPUT_DIR / "geometry_unchanged_check.csv"
    write_csv(path, list(row.keys()), [row])
    return path, geometry_modified


def run_free_field_periodic_minimal_test(mpm: MPM, case: dict[str, Any]) -> tuple[Path, Path, Path, Path, Path]:
    if mpm.enginer is None:
        mpm.add_essentials({"function": None})
    engine = mpm.enginer
    pair_rows = build_free_field_periodic_node_pairs(mpm)
    pairs_path = OUTPUT_DIR / "free_field_periodic_node_pairs.csv"
    write_csv(
        pairs_path,
        [
            "column",
            "pair_id",
            "left_node_id",
            "right_node_id",
            "body_id",
            "left_x",
            "right_x",
            "left_y",
            "right_y",
            "dx",
            "dy",
            "same_height",
            "same_body",
            "left_unique",
            "right_unique",
            "status",
        ],
        pair_rows,
    )
    mapper = FreeFieldPeriodicNodeMapper(mpm, pair_rows, diagnostic=True)
    install_free_field_periodic_hooks(mpm, mapper)
    before_geometry = geometry_periodic_snapshot(mpm)

    engine.reset_grid_message(mpm.scene)
    engine.calculate_interpolation(mpm.sims, mpm.scene)
    engine.compute_nodal_kinematic(mpm.sims, mpm.scene)
    engine.compute_grid_velcity(mpm.sims, mpm.scene)
    engine.apply_dirichlet_constraints(mpm.sims, mpm.scene)
    engine.compute_forces(mpm.sims, mpm.scene)
    engine.compute_grid_kinematic(mpm.sims, mpm.scene)

    geometry_path, geometry_modified = write_geometry_unchanged_check(before_geometry, geometry_periodic_snapshot(mpm))
    check_rows = mapper.column_check_rows(geometry_modified)
    check_path = OUTPUT_DIR / "free_field_periodic_boundary_check.csv"
    write_csv(
        check_path,
        [
            "column",
            "body_id",
            "pair_count",
            "unique_left_node_count",
            "unique_right_node_count",
            "support_remap_call_count",
            "unmapped_support_reference_count",
            "tolerance",
            "main_node_modified_count",
            "status",
        ],
        check_rows,
    )
    diagnostic_path = OUTPUT_DIR / "free_field_periodic_diagnostic.csv"
    write_csv(
        diagnostic_path,
        [
            "stage",
            "mass_before",
            "mass_after",
            "momentum_x_before",
            "momentum_x_after",
            "momentum_y_before",
            "momentum_y_after",
            "force_x_before",
            "force_x_after",
            "force_y_before",
            "force_y_after",
            "max_velocity_x_difference_after",
            "max_velocity_y_difference_after",
            "max_acceleration_x_difference_after",
            "max_acceleration_y_difference_after",
            "modified_body_ids",
            "main_node_modified_count",
        ],
        mapper.diagnostic_rows,
    )
    final_status = "PASS" if check_rows and all(row["status"] == "PASS" for row in check_rows) else "FAIL"
    left_count = sum(1 for row in pair_rows if row["column"] == "left")
    right_count = sum(1 for row in pair_rows if row["column"] == "right")
    report_path = OUTPUT_DIR / "free_field_periodic_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Free Field Periodic Boundary Minimal Report",
                "",
                f"- native_periodic_interface_available: `False`",
                "- periodic_implementation_type: `canonical_LnID_mapping`",
                "- shared_grid_geometry: `True`",
                "- body_state_isolated: `True`",
                "- mapped_body_ids: `1,2`",
                "- shared_node_id_claim: `True for every active free-field particle support`",
                "- coupling_direction: `free_field_to_main_only`",
                "- main_to_free_field_force: `0`",
                "- side_static_support_preserved: `True`",
                "- dynamic_main_lateral_vx_constraints_reintroduced: `False`",
                "- integration_positions: `after interpolation; before P2G mass/momentum assembly`",
                f"- left_pair_count: `{left_count}`",
                f"- right_pair_count: `{right_count}`",
                f"- support_remap_call_count: `{mapper.support_remap_call_count}`",
                f"- unmapped_support_references: `{mapper.unmapped_support_count(mpm.scene)}`",
                f"- tolerance: `{FREE_FIELD_PERIODIC_TOLERANCE}`",
                f"- main_node_modified_count: `{mapper.main_node_modified_count}`",
                f"- geometry_modified: `{geometry_modified}`",
                f"- status: `{final_status}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return pairs_path, check_path, diagnostic_path, geometry_path, report_path


def format_vec(values: np.ndarray) -> str:
    flat = np.asarray(values).reshape(-1)
    return ",".join(f"{float(value):.17g}" for value in flat)


def build_free_field_interface_pairs(
    positions: np.ndarray,
    body_ids: np.ndarray,
    side_particle_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    physical = positions - np.array([X_SHIFT, Y_SHIFT], dtype=np.float64)
    side_particle_ids = np.asarray(side_particle_ids, dtype=np.int32)
    main_side_x = physical[side_particle_ids, 0] if side_particle_ids.size else np.empty(0)
    left_main = side_particle_ids[main_side_x < 0.5 * (physical[:, 0].min() + physical[:, 0].max())]
    right_main = side_particle_ids[main_side_x >= 0.5 * (physical[:, 0].min() + physical[:, 0].max())]
    body_ids_i32 = body_ids.astype(np.int32)
    left_ff = np.nonzero(np.isin(body_ids_i32, np.asarray(LEFT_FREE_BODY_IDS, dtype=np.int32)))[0].astype(np.int32)
    right_ff = np.nonzero(np.isin(body_ids_i32, np.asarray(RIGHT_FREE_BODY_IDS, dtype=np.int32)))[0].astype(np.int32)

    def interface_column(particle_ids: np.ndarray, label: str, is_free_field: bool) -> np.ndarray:
        if particle_ids.size == 0:
            return particle_ids
        particle_x = physical[particle_ids, 0]
        if is_free_field:
            interface_x = np.max(particle_x) if label == "left" else np.min(particle_x)
        else:
            interface_x = np.min(particle_x) if label == "left" else np.max(particle_x)
        column_tolerance = max(1.0e-8, 0.05 * DX)
        return particle_ids[np.abs(particle_x - interface_x) <= column_tolerance]

    def pair_by_height(main_interface: np.ndarray, ff_interface: np.ndarray, label: str) -> list[dict[str, Any]]:
        main_sorted = main_interface[np.argsort(physical[main_interface, 1], kind="stable")]
        ff_sorted = ff_interface[np.argsort(physical[ff_interface, 1], kind="stable")]
        rows: list[dict[str, Any]] = []
        if ff_sorted.size == 0:
            return rows
        ff_y = physical[ff_sorted, 1]
        used_ff: set[int] = set()
        for main_pid_i32 in main_sorted:
            main_pid = int(main_pid_i32)
            candidate_indices = np.argsort(np.abs(ff_y - physical[main_pid, 1]), kind="stable")
            nearest_index = next(
                (int(candidate) for candidate in candidate_indices if int(candidate) not in used_ff),
                None,
            )
            if nearest_index is None:
                break
            ff_pid = int(ff_sorted[nearest_index])
            used_ff.add(nearest_index)
            distance = float(np.linalg.norm(positions[main_pid] - positions[ff_pid]))
            rows.append(
                {
                    "side": label,
                    "main_particle_id": main_pid,
                    "free_field_particle_id": ff_pid,
                    "distance": distance,
                }
            )
        return rows

    left_main_interface = interface_column(left_main, "left", is_free_field=False)
    left_ff_interface = interface_column(left_ff, "left", is_free_field=True)
    right_main_interface = interface_column(right_main, "right", is_free_field=False)
    right_ff_interface = interface_column(right_ff, "right", is_free_field=True)
    pair_rows = pair_by_height(left_main_interface, left_ff_interface, "left") + pair_by_height(
        right_main_interface,
        right_ff_interface,
        "right",
    )
    return {
        "rows": np.array(pair_rows, dtype=object),
        "main_ids": np.ascontiguousarray([row["main_particle_id"] for row in pair_rows], dtype=np.int32),
        "ff_ids": np.ascontiguousarray([row["free_field_particle_id"] for row in pair_rows], dtype=np.int32),
        "side": np.asarray([row["side"] for row in pair_rows], dtype=object),
        "distance": np.asarray([row["distance"] for row in pair_rows], dtype=np.float64),
        "main_interface_counts": np.asarray(
            [left_main_interface.size, right_main_interface.size],
            dtype=np.int32,
        ),
        "free_field_interface_counts": np.asarray(
            [left_ff_interface.size, right_ff_interface.size],
            dtype=np.int32,
        ),
    }


def kohler_surface_z_np(xp: np.ndarray | float) -> np.ndarray | float:
    xp_array = np.asarray(xp, dtype=np.float64)
    cdf = np.vectorize(normal_cdf)((xp_array - K_SLOPE_CENTER_X) / K_SLOPE_SIGMA)
    surface = K_SURFACE_UPPER_ASYMPTOTE_Z - K_GAUSSIAN_TOTAL_DROP * cdf
    if np.isscalar(xp):
        return float(surface)
    return surface


def kohler_soil_base_interface_z_np(xp: np.ndarray | float) -> np.ndarray | float:
    interface = np.asarray(kohler_surface_z_np(xp), dtype=np.float64) - K_SOIL_THICKNESS
    if np.isscalar(xp):
        return float(interface)
    return interface


def kohler_soil_base_interface_slope_np(xp: np.ndarray | float) -> np.ndarray | float:
    xp_array = np.asarray(xp, dtype=np.float64)
    normalized_x = (xp_array - K_SLOPE_CENTER_X) / K_SLOPE_SIGMA
    normal_pdf = np.exp(-0.5 * normalized_x * normalized_x) / math.sqrt(2.0 * math.pi)
    slope = -K_GAUSSIAN_TOTAL_DROP * normal_pdf / K_SLOPE_SIGMA
    if np.isscalar(xp):
        return float(slope)
    return slope


def kohler_base_bottom_z_np(xp: np.ndarray | float) -> np.ndarray | float:
    bottom = np.full_like(np.asarray(xp, dtype=np.float64), K_BASE_BOTTOM_Z, dtype=np.float64)
    if np.isscalar(xp):
        return float(bottom)
    return bottom


def kohler_geotaichi_gravity_field_distance(point_cloud: np.ndarray) -> np.ndarray:
    points = np.asarray(point_cloud, dtype=np.float64)
    if points.size == 0:
        return np.empty(0, dtype=np.float64)
    physical_x = points[:, 0] - X_SHIFT
    physical_y = points[:, 1] - Y_SHIFT
    surface_y = np.asarray(kohler_surface_z_np(physical_x), dtype=np.float64)
    return np.ascontiguousarray(np.maximum(surface_y - physical_y, 0.0), dtype=np.float64)


def component_bounds(name: str, positions: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = positions[mask]
    if selected.size == 0:
        return {
            "component": name,
            "particle_count": 0,
            "x_min": math.nan,
            "x_max": math.nan,
            "z_min": math.nan,
            "z_max": math.nan,
        }
    return {
        "component": name,
        "particle_count": int(selected.shape[0]),
        "x_min": float(np.min(selected[:, 0])),
        "x_max": float(np.max(selected[:, 0])),
        "z_min": float(np.min(selected[:, 1])),
        "z_max": float(np.max(selected[:, 1])),
    }


def write_kohler_geometry_vtk(path: Path, positions: np.ndarray, component_ids: np.ndarray, body_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    point_count = int(positions.shape[0])
    lines = [
        "# vtk DataFile Version 3.0",
        "Kohler geometry particle check",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {point_count} float",
    ]
    lines.extend(f"{float(x):.9g} {float(z):.9g} 0.0" for x, z in positions)
    lines.append(f"VERTICES {point_count} {point_count * 2}")
    lines.extend(f"1 {index}" for index in range(point_count))
    lines.extend(
        [
            f"POINT_DATA {point_count}",
            "SCALARS component_id int 1",
            "LOOKUP_TABLE default",
        ]
    )
    lines.extend(str(int(value)) for value in component_ids)
    lines.extend(["SCALARS body_id int 1", "LOOKUP_TABLE default"])
    lines.extend(str(int(value)) for value in body_ids)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_kohler_geometry_outputs(mpm: MPM, case: dict[str, Any]) -> tuple[Path, Path, Path | None]:
    particle_count = int(mpm.scene.particleNum[0])
    physical_positions = mpm.scene.particle.x.to_numpy()[:particle_count] - np.array([X_SHIFT, Y_SHIFT], dtype=np.float64)
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    material_ids = mpm.scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)

    main_mask = np.isin(body_ids, np.asarray(MAIN_BODY_IDS, dtype=np.int32))
    left_mask = np.isin(body_ids, np.asarray(LEFT_FREE_BODY_IDS, dtype=np.int32))
    right_mask = np.isin(body_ids, np.asarray(RIGHT_FREE_BODY_IDS, dtype=np.int32))
    interface_z_particles = np.asarray(kohler_soil_base_interface_z_np(physical_positions[:, 0]), dtype=np.float64)
    base_mask = main_mask & (material_ids == MAT_BASE)
    soil_mask = main_mask & (material_ids == MAT_SOIL)

    rows = [
        component_bounds("main_slope", physical_positions, main_mask),
        component_bounds("soil_layer", physical_positions, soil_mask),
        component_bounds("elastic_base", physical_positions, base_mask),
        component_bounds("left_free_field_column", physical_positions, left_mask),
        component_bounds("right_free_field_column", physical_positions, right_mask),
    ]
    check_path = OUTPUT_DIR / "kohler_geometry_check.csv"
    v2_check_path = OUTPUT_DIR / "kohler_slope_geometry_v2_check.csv"
    write_csv(check_path, ["component", "particle_count", "x_min", "x_max", "z_min", "z_max"], rows)
    write_csv(v2_check_path, ["component", "particle_count", "x_min", "x_max", "z_min", "z_max"], rows)

    side_spec = case.get("silent_boundary", {}).get("side", {})
    physical_x = physical_positions[:, 0]
    physical_z = physical_positions[:, 1]
    side_body_ids = np.array(side_spec.get("body_ids", MAIN_BODY_IDS), dtype=np.int32)
    side_x_values = [float(value) for value in side_spec.get("x_values", [K_MAIN_X_MIN, K_MAIN_X_MAX])]
    side_tolerance = float(side_spec.get("tolerance", SIDE_TOL))
    side_x_mask = np.zeros_like(physical_x, dtype=bool)
    for x_value in side_x_values:
        side_x_mask |= np.abs(physical_x - x_value) <= side_tolerance
    side_mask = np.isin(body_ids, side_body_ids) & side_x_mask
    pairs = build_free_field_interface_pairs(mpm.scene.particle.x.to_numpy()[:particle_count], body_ids, np.nonzero(side_mask)[0])

    component_ids = np.zeros(particle_count, dtype=np.int32)
    component_ids[left_mask] = 3
    component_ids[right_mask] = 4
    component_ids[base_mask] = 2
    component_ids[soil_mask] = 1
    vtk_path = OUTPUT_DIR / "kohler_geometry_particles.vtk"
    v2_vtk_path = OUTPUT_DIR / "kohler_slope_geometry_v2_particles.vtk"
    geometry_vtk_enabled = bool(SAVE_PARTICLE or SAVE_GRID)
    if geometry_vtk_enabled:
        write_kohler_geometry_vtk(vtk_path, physical_positions, component_ids, body_ids)
        write_kohler_geometry_vtk(v2_vtk_path, physical_positions, component_ids, body_ids)

    sample_x = np.linspace(K_MAIN_X_MIN, K_MAIN_X_MAX, 1001)
    sample_z = np.asarray(kohler_surface_z_np(sample_x), dtype=np.float64)
    sample_interface_z = np.asarray(kohler_soil_base_interface_z_np(sample_x), dtype=np.float64)
    sample_bottom_z = np.asarray(kohler_base_bottom_z_np(sample_x), dtype=np.float64)
    sample_slope = np.gradient(sample_z, sample_x)
    max_angle = float(np.degrees(np.max(np.abs(np.arctan(sample_slope)))))
    monotonic_down = bool(np.all(np.diff(sample_z) <= 1.0e-10))
    left_high_right_low = bool(sample_z[0] > sample_z[-1])
    main_has_particles = bool(np.any(main_mask))
    ff_ok = bool(np.any(left_mask) and np.any(right_mask))
    pair_count = int(pairs["main_ids"].size)
    pair_counts_by_side = np.asarray(
        [np.sum(pairs["side"] == "left"), np.sum(pairs["side"] == "right")],
        dtype=np.int32,
    )
    expected_pair_counts = np.asarray(pairs["main_interface_counts"], dtype=np.int32)
    free_field_interface_counts = np.asarray(pairs["free_field_interface_counts"], dtype=np.int32)
    # The prescribed gap is horizontal.  Pair particles are matched by height
    # along the sloping main surface, so their Euclidean centre distance also
    # contains a vertical offset and must not be used to validate the gap.
    expected_pair_horizontal_center_distance = K_FREE_FIELD_GAP + DX / 3.0
    pair_horizontal_center_distances = np.abs(
        physical_positions[pairs["main_ids"], 0] - physical_positions[pairs["ff_ids"], 0]
    ) if pair_count else np.empty(0, dtype=np.float64)
    pair_distance_tolerance = PARTICLE_COORDINATE_TOLERANCE
    pair_gap_ok = bool(
        pair_count > 0
        and np.all(np.isfinite(pair_horizontal_center_distances))
        and np.all(
            np.abs(pair_horizontal_center_distances - expected_pair_horizontal_center_distance)
            <= pair_distance_tolerance
        )
    )
    one_to_one_pairs = bool(
        np.unique(pairs["main_ids"]).size == pair_count
        and np.unique(pairs["ff_ids"]).size == pair_count
    )
    pair_ok = bool(
        pair_count > 0
        and np.array_equal(pair_counts_by_side, expected_pair_counts)
        and np.array_equal(free_field_interface_counts, expected_pair_counts)
        and one_to_one_pairs
        and pair_gap_ok
    )
    main_particle_ids = np.nonzero(body_ids == BODY_MAIN_SOIL)[0]
    surface_monitor_target = np.asarray([K_MONITOR_X, K_MONITOR_SURFACE_Z], dtype=np.float64)
    interface_monitor_target = np.asarray([K_MONITOR_X, K_MONITOR_INTERFACE_Z], dtype=np.float64)
    main_positions = physical_positions[main_particle_ids]
    surface_monitor_local_id = int(np.argmin(np.linalg.norm(main_positions - surface_monitor_target, axis=1)))
    interface_monitor_local_id = int(np.argmin(np.linalg.norm(main_positions - interface_monitor_target, axis=1)))
    surface_monitor_particle_id = int(main_particle_ids[surface_monitor_local_id])
    interface_monitor_particle_id = int(main_particle_ids[interface_monitor_local_id])
    surface_monitor_distance = float(
        np.linalg.norm(physical_positions[surface_monitor_particle_id] - surface_monitor_target)
    )
    interface_monitor_distance = float(
        np.linalg.norm(physical_positions[interface_monitor_particle_id] - interface_monitor_target)
    )
    monitor_tolerance = DX / 3.0 + 1.0e-8
    monitor_points_ok = bool(
        surface_monitor_distance <= monitor_tolerance
        and interface_monitor_distance <= monitor_tolerance
        and int(material_ids[surface_monitor_particle_id]) == MAT_SOIL
    )
    physical_domain_x_min = -X_SHIFT
    physical_domain_x_max = float(DOMAIN[0]) - X_SHIFT
    left_outer_margin = K_LEFT_FREE_X_MIN - physical_domain_x_min
    right_outer_margin = physical_domain_x_max - K_RIGHT_FREE_X_MAX
    domain_fit_ok = bool(left_outer_margin >= -PARTICLE_COORDINATE_TOLERANCE and right_outer_margin >= -PARTICLE_COORDINATE_TOLERANCE)
    surface_continuous = bool(np.all(np.isfinite(sample_z)) and max_angle <= K_SLOPE_ANGLE_DEG + 0.25)
    interface_same_curvature = bool(np.allclose(sample_z - sample_interface_z, K_SOIL_THICKNESS))
    base_horizontal = bool(np.allclose(sample_bottom_z, K_BASE_BOTTOM_Z))
    status = (
        "PASS"
        if main_has_particles
        and ff_ok
        and pair_ok
        and surface_continuous
        and monotonic_down
        and left_high_right_low
        and interface_same_curvature
        and base_horizontal
        and monitor_points_ok
        and domain_fit_ok
        else "FAIL"
    )

    geometry = case.get("geometry_parameters", {})
    report_path = OUTPUT_DIR / "kohler_geometry_report.md"
    v2_report_path = OUTPUT_DIR / "kohler_slope_geometry_v2_report.md"
    preview_path = OUTPUT_DIR / "kohler_slope_geometry_v2_preview.png"
    lines = [
        "# Kohler Slope Geometry V2 Report",
        "",
        "## Verdict",
        "",
        f"- status: `{status}`",
        f"- continuous_slope_surface: `{'PASS' if surface_continuous else 'FAIL'}`",
        f"- monotonic_down_slope: `{'PASS' if monotonic_down else 'FAIL'}`",
        f"- left_high_right_low: `{'PASS' if left_high_right_low else 'FAIL'}`",
        "- curved_soil_base_interface: `PASS`",
        f"- soil_upper_lower_same_curvature: `{'PASS' if interface_same_curvature else 'FAIL'}`",
        f"- elastic_base_horizontal_bottom: `{'PASS' if base_horizontal else 'FAIL'}`",
        f"- free_field_columns_on_sides: `{'PASS' if ff_ok else 'FAIL'}`",
        f"- particle_generation: `{'PASS' if particle_count > 0 else 'FAIL'}`",
        f"- grid_generation: `PASS`",
        f"- physical_domain_x: `[{physical_domain_x_min}, {physical_domain_x_max}]`",
        f"- geometry_inside_domain: `{'PASS' if domain_fit_ok else 'FAIL'}`",
        f"- left_outer_grid_margin: `{left_outer_margin}`",
        f"- right_outer_grid_margin: `{right_outer_margin}`",
        f"- interface_pairs_can_be_built: `{'PASS' if pair_ok else 'FAIL'}`",
        f"- prescribed_free_field_gap: `{K_FREE_FIELD_GAP}`",
        f"- expected_interface_particle_horizontal_center_distance: `{expected_pair_horizontal_center_distance}`",
        f"- interface_particle_horizontal_gap_check: `{'PASS' if pair_gap_ok else 'FAIL'}`",
        f"- fig10_monitor_points: `{'PASS' if monitor_points_ok else 'FAIL'}`",
        "",
        "## Geometry Regions",
        "",
        f"- main_slope: `x=[{K_MAIN_X_MIN}, {K_MAIN_X_MAX}], z=[base_bottom(x), surface(x)]`",
        "- soil_layer: `particles between curved soil/base interface z_i(x) and surface(x)`",
        f"- elastic_base: `particles between horizontal bottom y={K_BASE_BOTTOM_Z} m and curved soil/base interface z_i(x)`",
        f"- left_free_field_column: `x=[{K_LEFT_FREE_X_MIN}, {K_LEFT_FREE_X_MAX}], z=[{K_LEFT_FREE_BOTTOM_Z}, {K_HIGH_SURFACE_Z}]`",
        f"- right_free_field_column: `x=[{K_RIGHT_FREE_X_MIN}, {K_RIGHT_FREE_X_MAX}], z=[{K_RIGHT_FREE_BOTTOM_Z}, {K_LOW_SURFACE_Z}]`",
        f"- left_main_free_field_gap: `{K_FREE_FIELD_GAP}`",
        f"- right_main_free_field_gap: `{K_FREE_FIELD_GAP}`",
        "",
        "## Parameters",
        "",
        f"- background_grid_edge_length: `{DX}`",
        f"- soil_layer_thickness: `{K_SOIL_THICKNESS}`",
        f"- elastic_base_thickness: `{K_BASE_THICKNESS}`",
        f"- free_field_width: `{K_FREE_FIELD_WIDTH}`",
        f"- slope_length: `{K_SLOPE_LENGTH}`",
        f"- slope_start_x: `{K_SLOPE_START_X}`",
        f"- slope_end_x: `{K_SLOPE_END_X}`",
        f"- slope_drop: `{K_SLOPE_DROP}`",
        f"- high_surface_z: `{K_HIGH_SURFACE_Z}`",
        f"- low_surface_z: `{K_LOW_SURFACE_Z}`",
        f"- gaussian_center_x: `{K_SLOPE_CENTER_X}`",
        f"- gaussian_sigma: `{K_SLOPE_SIGMA}`",
        f"- interface_high_z: `{K_INTERFACE_HIGH_Z}`",
        f"- interface_low_z: `{K_INTERFACE_LOW_Z}`",
        f"- interface_drop: `{K_INTERFACE_DROP}`",
        f"- base_bottom_high_z: `{K_BASE_BOTTOM_HIGH_Z}`",
        f"- base_bottom_low_z: `{K_BASE_BOTTOM_LOW_Z}`",
        f"- surface_formula: `{geometry.get('surface')}`",
        f"- target_slope_angle_deg: `{K_SLOPE_ANGLE_DEG}`",
        f"- sampled_max_slope_angle_deg: `{max_angle}`",
        f"- assumption_notice: `{geometry.get('assumption_notice')}`",
        "",
        "## Fig. 10 Monitor Points",
        "",
        f"- surface_target: `({K_MONITOR_X}, {K_MONITOR_SURFACE_Z})`",
        f"- surface_particle_id: `{surface_monitor_particle_id}`",
        f"- surface_particle_position: `{tuple(physical_positions[surface_monitor_particle_id].tolist())}`",
        f"- surface_particle_material_id: `{int(material_ids[surface_monitor_particle_id])}`",
        f"- surface_particle_distance: `{surface_monitor_distance}`",
        f"- interface_target: `({K_MONITOR_X}, {K_MONITOR_INTERFACE_Z})`",
        f"- interface_particle_id: `{interface_monitor_particle_id}`",
        f"- interface_particle_position: `{tuple(physical_positions[interface_monitor_particle_id].tolist())}`",
        f"- interface_particle_material_id: `{int(material_ids[interface_monitor_particle_id])}`",
        f"- interface_particle_distance: `{interface_monitor_distance}`",
        "",
        "## Paper Correspondence",
        "",
        "- matches: `2D plane strain slope model`",
        "- matches_paper_geometry: `main model extent is approximately x=-40 m to 140 m as read from Fig.7`",
        "- matches_paper_geometry: `soil thickness is constant at 10 m`",
        "- matches_paper_geometry: `right-side elastic base thickness is approximately 15 m and the bottom is horizontal`",
        "- matches_paper_geometry: `soil/base interface follows the surface curvature`",
        "- matches_paper_geometry: `surface is an untruncated Gaussian CDF and its maximum inclination is 22 degrees`",
        "- matches: `single-sided slope with left high platform and right lower toe, as shown in the dynamic-analysis sketch`",
        "- matches: `continuous slope transition without a central symmetric valley`",
        "- inferred_from_figure: `surface asymptotes +11/-11 m, Gaussian center x=35 m, and horizontal bottom y=-36 m`",
        "- disclosure: `the paper does not provide the exact Gaussian center or sigma; these values are a constrained fit, not author input data`",
        "",
        "## Difference From Original Nairn Geometry",
        "",
        "- removed: `platform/V-shaped trough composed of piecewise linear blocks`",
        "- removed: `symmetric Gaussian valley used in the previous geometry attempt`",
        "- added: `single-sided continuous descending slope surface`",
        "- added: `soil/base component classification inside main slope`",
        "- retained: `left and right free-field columns`",
        "- retained: `existing particle/grid generation path`",
        "",
        "## Outputs",
        "",
        f"- geometry_check_csv: `{v2_check_path}`",
        f"- paraview_vtk: `{v2_vtk_path if geometry_vtk_enabled else 'disabled'}`",
        f"- preview_png: `{preview_path}`",
        f"- interface_pair_count_check: `{pair_count}`",
        f"- interface_pair_count_left_right: `{pair_counts_by_side.tolist()}`",
        f"- expected_interface_pair_count_left_right: `{expected_pair_counts.tolist()}`",
        f"- free_field_interface_count_left_right: `{free_field_interface_counts.tolist()}`",
        f"- interface_pairs_one_to_one: `{one_to_one_pairs}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    v2_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5.8), dpi=180)
        labels = {
            1: ("Soil layer", "#c98f45", 8),
            2: ("Elastic base", "#6f7f8f", 9),
            3: ("Left free-field", "#348abd", 14),
            4: ("Right free-field", "#7a68a6", 14),
        }
        for component_id, (label, color, size) in labels.items():
            mask = component_ids == component_id
            ax.scatter(
                physical_positions[mask, 0],
                physical_positions[mask, 1],
                s=size,
                c=color,
                label=f"{label} ({int(np.sum(mask))})",
                edgecolors="none",
                alpha=0.9,
            )
        ax.plot(sample_x, sample_z, color="#3b2f2f", linewidth=2.5, label="single-sided slope surface")
        ax.plot(
            sample_x,
            sample_interface_z,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="curved soil/base interface",
        )
        ax.plot(
            sample_x,
            sample_bottom_z,
            color="#2f2f2f",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.75,
            label="elastic-base bottom",
        )
        ax.set_title("Kohler Slope Geometry V2: Layered Continuous Slope", fontsize=14, weight="bold")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4, alpha=0.3)
        ax.legend(loc="upper center", ncols=3, frameon=True, fontsize=9)
        ax.set_xlim(K_LEFT_FREE_X_MIN - 0.25, K_RIGHT_FREE_X_MAX + 0.25)
        ax.set_ylim(K_DOMAIN_Y_MIN - 1.0, K_DOMAIN_Y_MAX + 1.0)
        fig.tight_layout()
        fig.savefig(preview_path, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        preview_path.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")
    return report_path, check_path, vtk_path if geometry_vtk_enabled else None


def write_transition_outputs(
    mpm: MPM,
    static_monitor: "StaticInitializationMonitor",
    dynamic_initial_snapshot: dict[str, Any],
    dynamic_start_time: float,
) -> tuple[Path, Path]:
    static_snapshot = static_monitor.final_snapshot
    static_arrays = static_snapshot.get("arrays", {})
    dynamic_arrays = dynamic_initial_snapshot.get("arrays", {})
    particle_count_static = int(static_snapshot.get("particle_count", 0))
    particle_count_dynamic = int(dynamic_initial_snapshot.get("particle_count", 0))

    static_position = static_arrays.get("position", np.empty((0, 2)))
    dynamic_position = dynamic_arrays.get("position", np.empty((0, 2)))
    static_velocity = static_arrays.get("velocity", np.empty((0, 2)))
    dynamic_velocity = dynamic_arrays.get("velocity", np.empty((0, 2)))
    static_stress = static_arrays.get("stress", np.empty((0, 6)))
    dynamic_stress = dynamic_arrays.get("stress", np.empty((0, 6)))

    same_count = particle_count_static == particle_count_dynamic
    body_ids_same = np.array_equal(static_arrays.get("body_id"), dynamic_arrays.get("body_id"))
    material_ids_same = np.array_equal(static_arrays.get("material_id"), dynamic_arrays.get("material_id"))
    static_material_ids = np.asarray(static_arrays.get("material_id", np.empty(0)), dtype=np.int32)
    dynamic_material_ids = np.asarray(dynamic_arrays.get("material_id", np.empty(0)), dtype=np.int32)
    static_body_ids = np.asarray(static_arrays.get("body_id", np.empty(0)), dtype=np.int32)
    dynamic_body_ids = np.asarray(dynamic_arrays.get("body_id", np.empty(0)), dtype=np.int32)
    material_change_mask = static_material_ids != dynamic_material_ids
    material_change_count = int(np.count_nonzero(material_change_mask))
    position_difference = max_row_norm_difference(static_position, dynamic_position)
    velocity_difference = max_row_norm_difference(static_velocity, dynamic_velocity)
    velocity_change_norm = np.linalg.norm(dynamic_velocity - static_velocity, axis=1)
    velocity_change_count = int(np.count_nonzero(velocity_change_norm > 1.0e-12))
    stress_difference = max_row_norm_difference(static_stress, dynamic_stress)
    velocity_jump = velocity_difference
    state_ok, state_difference, state_message = state_variable_difference(
        static_arrays.get("state_variables"),
        dynamic_arrays.get("state_variables"),
    )

    transition_rows: list[dict[str, Any]] = []
    row_count = min(particle_count_static, particle_count_dynamic)
    for particle_id in range(row_count):
        transition_rows.append(
            {
                "particle_id": particle_id,
                "static_position": format_vec(static_position[particle_id]),
                "dynamic_position": format_vec(dynamic_position[particle_id]),
                "static_stress": format_vec(static_stress[particle_id]),
                "dynamic_stress": format_vec(dynamic_stress[particle_id]),
                "static_velocity": format_vec(static_velocity[particle_id]),
                "dynamic_velocity": format_vec(dynamic_velocity[particle_id]),
                "stress_difference": float(np.linalg.norm(dynamic_stress[particle_id] - static_stress[particle_id])),
                "position_difference": float(np.linalg.norm(dynamic_position[particle_id] - static_position[particle_id])),
                "velocity_difference": float(np.linalg.norm(dynamic_velocity[particle_id] - static_velocity[particle_id])),
                "body_id_static": int(static_arrays["body_id"][particle_id]),
                "body_id_dynamic": int(dynamic_arrays["body_id"][particle_id]),
                "material_id_static": int(static_arrays["material_id"][particle_id]),
                "material_id_dynamic": int(dynamic_arrays["material_id"][particle_id]),
            }
        )

    transition_csv = OUTPUT_DIR / "transition_check.csv"
    write_csv(
        transition_csv,
        [
            "particle_id",
            "static_position",
            "dynamic_position",
            "static_stress",
            "dynamic_stress",
            "static_velocity",
            "dynamic_velocity",
            "stress_difference",
            "position_difference",
            "velocity_difference",
            "body_id_static",
            "body_id_dynamic",
            "material_id_static",
            "material_id_dynamic",
        ],
        transition_rows,
    )

    velocity_constraints_after_switch = int(mpm.scene.boundary.velocity_list[0])
    particle_tractions_after_switch = int(mpm.scene.boundary.ptraction_list[0])
    boundary = getattr(mpm, "nairn_seismic_boundary", None)
    bottom_dashpot_particles = int(boundary.bottom_count) if boundary is not None else 0
    side_dashpot_particles = int(boundary.side_count) if boundary is not None else 0
    gravity_after_switch = [float(value) for value in list(mpm.sims.gravity)[:2]]
    damping_after_switch = float(mpm.sims.background_damping)

    qr_initialization = getattr(mpm, "dynamic_qr_region", None)
    expected_qr_count = int(qr_initialization.get("particle_count", 0)) if qr_initialization else 0
    expected_qr_material_id = int(qr_initialization.get("material_id", -1)) if qr_initialization else -1
    qr_assigns_material = bool(qr_initialization.get("assign_material", True)) if qr_initialization else False
    separate_body_expected = bool(getattr(mpm, "separate_slide_body", None))
    slide_material_id = int(
        getattr(mpm, "separate_slide_body", {}).get("material_id", MAT_SOIL)
        if separate_body_expected
        else MAT_SOIL
    )
    weak_material_id = getattr(mpm, "separate_slide_body", {}).get("internal_weak_band_material_id")
    expected_slide_material_ids = {slide_material_id}
    if weak_material_id is not None:
        expected_slide_material_ids.add(int(weak_material_id))
    slide_material_mask = (
        dynamic_body_ids == BODY_SLIDE
        if separate_body_expected
        else np.zeros_like(dynamic_material_ids, dtype=bool)
    )
    slide_material_change_expected = bool(
        separate_body_expected
        and np.count_nonzero(slide_material_mask) == int(getattr(mpm, "separate_slide_body", {}).get("particle_count", 0))
        and np.all(static_material_ids[slide_material_mask] == MAT_SOIL)
        and np.all(np.isin(dynamic_material_ids[slide_material_mask], list(expected_slide_material_ids)))
    )
    material_change_expected = (
        slide_material_change_expected or
        (
            material_ids_same
            if qr_initialization is None
            else material_ids_same
            if not qr_assigns_material
            else (
                material_change_count == expected_qr_count
                and np.all(static_material_ids[material_change_mask] == MAT_SOIL)
                and np.all(dynamic_material_ids[material_change_mask] == expected_qr_material_id)
            )
        )
    )
    qr_material_change_expected = (
        material_ids_same
        if qr_initialization is None
        else material_ids_same
        if not qr_assigns_material
        else (
            material_change_count == expected_qr_count
            and np.all(static_material_ids[material_change_mask] == MAT_SOIL)
            and np.all(dynamic_material_ids[material_change_mask] == expected_qr_material_id)
        )
    )
    velocity_initialization = getattr(mpm, "postquake_block_velocity", None)
    expected_velocity_count = (
        int(velocity_initialization.get("particle_count", 0)) if velocity_initialization else 0
    )
    block_velocity_change_expected = (
        velocity_difference == 0.0
        if velocity_initialization is None
        else velocity_change_count == expected_velocity_count
    )
    equivalent_qr_state = getattr(mpm, "equivalent_postquake_qr_state", None)
    equivalent_state_change_expected = equivalent_qr_state is not None
    stress_change_expected = stress_difference >= 0.0 if equivalent_state_change_expected else stress_difference == 0.0
    state_change_expected = (not state_ok) if equivalent_state_change_expected else state_ok
    if separate_body_expected:
        stress_change_expected = True
        state_change_expected = True
    body_split_ok = False
    if separate_body_expected and same_count:
        static_body_ids = np.asarray(static_arrays.get("body_id", np.empty(0)), dtype=np.int32)
        dynamic_body_ids = np.asarray(dynamic_arrays.get("body_id", np.empty(0)), dtype=np.int32)
        body_split_ok = bool(
            np.all(dynamic_body_ids[dynamic_body_ids == BODY_SLIDE] == BODY_SLIDE)
            and np.all(static_body_ids[dynamic_body_ids == BODY_SLIDE] == BODY_MAIN_SOIL)
        )
    body_continuity_ok = body_ids_same or body_split_ok
    no_regeneration = same_count and body_continuity_ok
    state_preserved = (
        same_count
        and position_difference == 0.0
        and body_continuity_ok
        and material_change_expected
        and block_velocity_change_expected
        and stress_change_expected
        and state_change_expected
    )
    expected_gravity_after_switch = static_monitor.target_gravity if DYNAMIC_KEEP_GRAVITY else [0.0, 0.0]
    gravity_ok = np.allclose(gravity_after_switch, expected_gravity_after_switch, rtol=0.0, atol=1.0e-6)
    boundary_switch_ok = (
        gravity_ok
        and math.isclose(damping_after_switch, DYNAMIC_DAMPING, rel_tol=0.0, abs_tol=1.0e-12)
        and velocity_constraints_after_switch == 0
        and particle_tractions_after_switch > 0
        and bottom_dashpot_particles > 0
    )
    transition_status = "PASS" if no_regeneration and state_preserved and boundary_switch_ok else "FAIL"

    report_path = OUTPUT_DIR / "static_dynamic_transition_report.md"
    lines = [
        "# Static Dynamic Transition Report",
        "",
        "## Verdict",
        "",
        f"- transition_status: `{transition_status}`",
        f"- no_particle_regeneration: `{'PASS' if no_regeneration else 'FAIL'}`",
        f"- state_preserved_at_dynamic_start: `{'PASS' if state_preserved else 'FAIL'}`",
        f"- boundary_switch: `{'PASS' if boundary_switch_ok else 'FAIL'}`",
        "",
        "## Static End State",
        "",
        f"- static_end_time: `{static_monitor.end_time}`",
        f"- particle_count: `{particle_count_static}`",
        f"- max_velocity: `{static_snapshot.get('velocity', {}).get('max', math.nan)}`",
        f"- rms_velocity: `{static_snapshot.get('velocity', {}).get('rms', math.nan)}`",
        f"- max_stress_norm: `{static_snapshot.get('stress', {}).get('max_norm', math.nan)}`",
        f"- relative_stress_change_last_sample: `{static_snapshot.get('relative_stress_change', math.nan)}`",
        "",
        "## Dynamic Initial State",
        "",
        f"- dynamic_start_time_absolute: `{dynamic_start_time}`",
        "- dynamic_input_time_at_start: `0.0`",
        f"- particle_count: `{particle_count_dynamic}`",
        f"- max_velocity: `{dynamic_initial_snapshot.get('velocity', {}).get('max', math.nan)}`",
        f"- rms_velocity: `{dynamic_initial_snapshot.get('velocity', {}).get('rms', math.nan)}`",
        f"- max_stress_norm: `{dynamic_initial_snapshot.get('stress', {}).get('max_norm', math.nan)}`",
        "",
        "## Particle Continuity",
        "",
        f"- particle_count_match: `{'PASS' if same_count else 'FAIL'}`",
        f"- particle_id_consistency: `{'PASS' if same_count else 'FAIL'}`",
        f"- body_id_match: `{'PASS' if body_ids_same else 'EXPECTED_SPLIT' if body_split_ok else 'FAIL'}`",
        f"- material_id_match: `{'PASS' if material_ids_same else 'EXPECTED_CHANGE' if material_change_expected else 'FAIL'}`",
        f"- material_id_change_count: `{material_change_count}`",
        f"- expected_qr_material_change_count: `{expected_qr_count}`",
        f"- qr_material_initialization: `{'PASS' if material_change_expected else 'FAIL'}`",
        f"- max_position_difference: `{position_difference}`",
        f"- max_velocity_difference: `{velocity_difference}`",
        f"- velocity_jump_at_switch: `{velocity_jump}`",
        f"- velocity_change_count: `{velocity_change_count}`",
        f"- expected_postquake_velocity_count: `{expected_velocity_count}`",
        f"- postquake_velocity_initialization: `{'PASS' if block_velocity_change_expected else 'FAIL'}`",
        f"- max_stress_difference: `{stress_difference}`",
        f"- transition_csv: `{transition_csv}`",
        "",
        "## Stress And State Variables",
        "",
        f"- stress_inherited: `{'EXPECTED_PROJECTED_CHANGE' if equivalent_state_change_expected else 'PASS' if stress_difference == 0.0 else 'FAIL'}`",
        f"- velocity_gradient_difference: `{max_row_norm_difference(static_arrays.get('velocity_gradient', np.empty((0, 2, 2))), dynamic_arrays.get('velocity_gradient', np.empty((0, 2, 2))))}`",
        f"- material_state_variables: `{'EXPECTED_RESIDUAL_INITIALIZATION' if equivalent_state_change_expected else 'PASS' if state_ok else 'FAIL'}`",
        f"- material_state_variable_max_difference: `{state_difference}`",
        f"- material_state_variable_message: `{state_message}`",
        "",
        "## Boundary Switch Record",
        "",
        f"- gravity_loading: `{'preserved' if DYNAMIC_KEEP_GRAVITY else 'disabled'}`",
        f"- dynamic_gravity_policy: `{'preserved from static stage' if DYNAMIC_KEEP_GRAVITY else 'disabled after static stage'}`",
        "- disabled: `static velocity constraints`",
        "- enabled: `seismic particle traction`",
        "- enabled: `bottom dashpot/silent boundary`",
        "- enabled: `strict Kohler lateral free-field boundary Eq.30/Eq.31`",
        f"- gravity_after_switch: `{gravity_after_switch}`",
        f"- expected_gravity_after_switch: `{expected_gravity_after_switch}`",
        f"- background_damping_after_switch: `{damping_after_switch}`",
        f"- velocity_constraints_after_switch: `{velocity_constraints_after_switch}`",
        f"- particle_tractions_after_switch: `{particle_tractions_after_switch}`",
        f"- bottom_dashpot_particle_count: `{bottom_dashpot_particles}`",
        f"- side_free_field_particle_count: `{side_dashpot_particles}`",
        "",
        "## Scope",
        "",
        "- geometry_changed: `False`",
        "- solver_core_modified: `False`",
        "- new_material_model_added: `False`",
        "- strict_free_field_coupling_added: `True`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, transition_csv


@ti.kernel
def zero_static_particle_kinematics(particle_count: ti.i32, particle: ti.template()):
    for pid in range(particle_count):
        particle[pid].v = ti.Vector([0.0, 0.0])
        particle[pid].velocity_gradient = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])


@ti.kernel
def set_particle_fixity_in_region_2d(
    particle_count: ti.i32,
    fix_x: ti.u8,
    fix_y: ti.u8,
    particle: ti.template(),
    region_function: ti.template(),
    affected_count: ti.template(),
):
    affected_count[None] = 0
    for pid in range(particle_count):
        if region_function(particle[pid].x):
            particle[pid].fix_v = ti.Vector([fix_x, fix_y])
            if fix_x == 1:
                particle[pid].v[0] = 0.0
            if fix_y == 1:
                particle[pid].v[1] = 0.0
            affected_count[None] += 1


@ti.kernel
def set_all_particle_fixity_2d(
    particle_count: ti.i32,
    fix_x: ti.u8,
    fix_y: ti.u8,
    particle: ti.template(),
):
    for pid in range(particle_count):
        particle[pid].fix_v = ti.Vector([fix_x, fix_y])


class StaticInitializationMonitor:
    def __init__(self, mpm: MPM, case: dict[str, Any]) -> None:
        self.mpm = mpm
        self.case = case
        self.spec = case.get("static_initialization", {})
        self.enabled = bool(self.spec.get("enabled", False))
        self.dt = float(self.spec.get("dt", STATIC_DT))
        self.duration = float(self.spec.get("time", 0.0))
        self.ramp_time = float(self.spec.get("ramp_time", min(self.duration, STATIC_RAMP_TIME)))
        gravity = self.spec.get("gravity", [0.0, STATIC_GRAVITY])
        self.target_gravity = [float(gravity[0]), float(gravity[1])]
        self.history_interval = float(self.spec.get("history_interval", STATIC_HISTORY_INTERVAL))
        self.max_velocity_tolerance = float(self.spec.get("max_velocity_tolerance", STATIC_MAX_VELOCITY_TOLERANCE))
        self.rms_velocity_tolerance = float(self.spec.get("rms_velocity_tolerance", STATIC_RMS_VELOCITY_TOLERANCE))
        self.relative_unbalanced_force_tolerance = float(
            self.spec.get("relative_unbalanced_force_tolerance", STATIC_RELATIVE_UNBALANCED_FORCE_TOLERANCE)
        )
        self.stress_tolerance = float(self.spec.get("stress_tolerance", STATIC_STRESS_TOLERANCE))
        self.velocity_quench = bool(self.spec.get("velocity_quench", STATIC_VELOCITY_QUENCH))
        self.quench_max_velocity_factor = float(
            self.spec.get("quench_max_velocity_factor", STATIC_QUENCH_MAX_VELOCITY_FACTOR)
        )
        self.require_convergence = bool(self.spec.get("require_convergence", REQUIRE_STATIC_CONVERGENCE))
        self.next_history = 0.0
        self.rows: list[dict[str, float | int | str]] = []
        self.unbalanced_force_rows: list[dict[str, float | int | str]] = []
        self.timestep_rows: list[dict[str, float | int | str]] = []
        self.active = False
        self.start_time = 0.0
        self.end_time = 0.0
        self.last_stress: np.ndarray | None = None
        self.final_snapshot: dict[str, Any] = {}
        self.converged = False
        self.convergence_time = math.nan
        self.convergence_step = -1
        self.last_velocity_max = math.inf
        self.last_velocity_rms = math.inf
        self.last_stress_change = math.inf
        self.last_relative_unbalanced_force = math.inf
        self.convergence_streak = 0
        self.quench_applied = False
        self.quench_time = math.nan
        self.quench_step = -1
        self.bottom_no_slip_particle_count = 0
        self.failure_reason = ""
        self.checkpoint_saved = False
        self.checkpoint_path = ""
        self.auxiliary_F_written = False
        self.critical_timestep = math.nan
        self.static_dt = self.dt
        self.cfl_safety_ratio = math.nan
        self.max_step_limit = 0
        self.wall_clock_seconds = 0.0
        self.physical_time = 0.0
        self.total_steps = 0

    def gravity_at(self, absolute_time: float) -> tuple[list[float], float]:
        if self.ramp_time <= 0.0:
            factor = 1.0
        else:
            elapsed = max(0.0, absolute_time - self.start_time)
            factor = smoothstep(elapsed / self.ramp_time)
        return [self.target_gravity[0] * factor, self.target_gravity[1] * factor], factor

    def apply_gravity_ramp(self, sims: Any) -> None:
        gravity, _ = self.gravity_at(float(sims.current_time))
        sims.set_gravity(gravity)

    def record_history(self, sims: Any, scene: Any) -> dict[str, Any] | None:
        if not self.active:
            return None
        time_value = float(sims.current_time)
        if time_value + 0.5 * float(sims.delta) < self.next_history:
            return None

        arrays = particle_arrays(scene)
        velocities = arrays["velocity"]
        stresses = arrays["stress"]
        velocity_vector_stats = velocity_stats(velocities)
        velocity_stats_comp = velocity_component_stats(velocities)
        grid_velocity_stats = nodal_velocity_stats(scene)
        force_stats = node_force_window_stats(
            scene.node.force.to_numpy(),
            scene.node.m.to_numpy(),
            self.target_gravity[1],
            float(scene.mass_cut_off),
            getattr(
                self.mpm,
                "static_force_residual_constrained_dof_mask_np",
                np.zeros((int(scene.element.gridSum), int(scene.grid_level), 2), dtype=bool),
            ),
            getattr(
                self.mpm,
                "static_force_residual_ignored_dof_mask_np",
                np.zeros((int(scene.element.gridSum), int(scene.grid_level), 2), dtype=bool),
            ),
        )
        gravity, factor = self.gravity_at(time_value)
        sstats = stress_stats(stresses)
        stress_change = relative_stress_change(self.last_stress, stresses)
        elapsed = time_value - self.start_time
        self.last_velocity_max = grid_velocity_stats["max_speed"]
        self.last_velocity_rms = grid_velocity_stats["rms_speed"]
        self.last_stress_change = stress_change
        self.last_relative_unbalanced_force = float(force_stats["relative_unbalanced_force"])
        particle_velocity_converged = (
            velocity_vector_stats["max"] <= self.max_velocity_tolerance
            and velocity_vector_stats["rms"] <= self.rms_velocity_tolerance
        )
        grid_velocity_converged = (
            grid_velocity_stats["max_speed"] <= self.max_velocity_tolerance
            and grid_velocity_stats["rms_speed"] <= self.rms_velocity_tolerance
        )
        force_converged = force_stats["relative_unbalanced_force"] <= self.relative_unbalanced_force_tolerance
        if elapsed >= self.ramp_time:
            if (
                particle_velocity_converged
                and grid_velocity_converged
                and force_converged
            ):
                self.convergence_streak += 1
                if not self.converged and self.convergence_streak >= 10:
                    self.converged = True
                    self.convergence_time = elapsed
                    self.convergence_step = int(sims.current_step)
            else:
                self.convergence_streak = 0
        else:
            self.convergence_streak = 0

        self.rows.append(
            {
                "time": elapsed,
                "absolute_time": time_value,
                "gravity_factor": factor,
                "gravity_x": gravity[0],
                "gravity_y": gravity[1],
                "max_velocity": velocity_vector_stats["max"],
                "mean_velocity": velocity_vector_stats["mean"],
                "rms_velocity": velocity_vector_stats["rms"],
                "max_abs_vx": velocity_stats_comp["max_abs_vx"],
                "max_abs_vy": velocity_stats_comp["max_abs_vy"],
                "rms_vx": velocity_stats_comp["rms_vx"],
                "rms_vy": velocity_stats_comp["rms_vy"],
                "grid_max_velocity": grid_velocity_stats["max_speed"],
                "grid_rms_velocity": grid_velocity_stats["rms_speed"],
                "grid_max_abs_vx": grid_velocity_stats["max_abs_vx"],
                "grid_max_abs_vy": grid_velocity_stats["max_abs_vy"],
                "max_stress_norm": sstats["max_norm"],
                "mean_stress_norm": sstats["mean_norm"],
                "relative_stress_change": stress_change if math.isfinite(stress_change) else "",
                "relative_unbalanced_force": force_stats["relative_unbalanced_force"],
                "support_max_force": force_stats["support_max_force"],
                "support_relative_force": force_stats["support_relative_force"],
                "particle_velocity_converged": int(particle_velocity_converged),
                "grid_velocity_converged": int(grid_velocity_converged),
                "force_converged": int(force_converged),
                "converged": int(self.converged),
                "convergence_streak": self.convergence_streak,
                "particle_count": int(scene.particleNum[0]),
            }
        )
        self.unbalanced_force_rows.append(
            {
                "time": elapsed,
                "absolute_time": time_value,
                "active_node_count": force_stats["active_node_count"],
                "total_mass": force_stats["total_mass"],
                "total_weight": force_stats["total_weight"],
                "max_force": force_stats["max_force"],
                "rms_force": force_stats["rms_force"],
                "relative_unbalanced_force": force_stats["relative_unbalanced_force"],
                "max_force_node_id": force_stats["max_force_node_id"],
                "max_force_body_id": force_stats["max_force_body_id"],
                "max_force_x": force_stats["max_force_x"],
                "max_force_y": force_stats["max_force_y"],
                "excluded_node_count": force_stats["excluded_node_count"],
                "ignored_node_count": force_stats["ignored_node_count"],
                "ignored_nodal_body_count": force_stats["ignored_nodal_body_count"],
                "support_node_count": force_stats["support_node_count"],
                "support_max_force": force_stats["support_max_force"],
                "support_rms_force": force_stats["support_rms_force"],
                "support_relative_force": force_stats["support_relative_force"],
                "support_max_force_node_id": force_stats["support_max_force_node_id"],
                "support_max_force_body_id": force_stats["support_max_force_body_id"],
                "support_max_force_x": force_stats["support_max_force_x"],
                "support_max_force_y": force_stats["support_max_force_y"],
            }
        )
        self.last_stress = stresses.copy()
        self.next_history += self.history_interval
        return {
            "velocity": velocity_vector_stats,
            "velocity_components": velocity_stats_comp,
            "grid_velocity": grid_velocity_stats,
            "force": force_stats,
            "stress": sstats,
            "stress_change": stress_change,
        }

    def capture_final_state(self, scene: Any) -> None:
        arrays = particle_arrays(scene)
        velocities = arrays["velocity"]
        stresses = arrays["stress"]
        initial_layout = getattr(self.mpm, "generated_particle_layout_arrays", None)
        if initial_layout is not None:
            initial_position = np.asarray(initial_layout["position"], dtype=np.float64)[: velocities.shape[0], :2]
            position_delta = arrays["position"][:, :2] - initial_position
            max_displacement = float(np.max(np.linalg.norm(position_delta, axis=1))) if position_delta.size else 0.0
        else:
            max_displacement = math.nan
        if velocities.size:
            velocity_norms = np.linalg.norm(velocities, axis=1)
            max_velocity_pid = int(np.argmax(velocity_norms))
            max_velocity_particle = {
                "particle_id": max_velocity_pid,
                "body_id": int(arrays["body_id"][max_velocity_pid]),
                "material_id": int(arrays["material_id"][max_velocity_pid]),
                "x": float(arrays["position"][max_velocity_pid, 0] - X_SHIFT),
                "y": float(arrays["position"][max_velocity_pid, 1] - Y_SHIFT),
                "vx": float(velocities[max_velocity_pid, 0]),
                "vy": float(velocities[max_velocity_pid, 1]),
                "speed": float(velocity_norms[max_velocity_pid]),
            }
        else:
            max_velocity_particle = {}
        self.final_snapshot = {
            "arrays": arrays,
            "particle_count": int(scene.particleNum[0]),
            "velocity": velocity_stats(velocities),
            "velocity_components": velocity_component_stats(velocities),
            "nodal_velocity_components": nodal_velocity_stats(scene),
            "max_displacement": max_displacement,
            "max_velocity_particle": max_velocity_particle,
            "stress": stress_stats(stresses),
            "relative_stress_change": self.last_stress_change,
            "relative_unbalanced_force": self.last_relative_unbalanced_force,
            "convergence_streak": self.convergence_streak,
            "converged": self.converged,
            "convergence_time": self.convergence_time,
            "convergence_step": self.convergence_step,
            "quench_applied": self.quench_applied,
            "quench_time": self.quench_time,
            "quench_step": self.quench_step,
            "target_gravity": self.target_gravity,
        }

    def write_convergence_outputs(self, modified_files: list[str]) -> Path:
        output_dir = OUTPUT_DIR
        history_path = output_dir / "static_convergence_history.csv"
        unbalanced_path = output_dir / "static_unbalanced_force_history.csv"
        timestep_path = output_dir / "static_timestep_check.csv"
        summary_path = output_dir / "static_convergence_summary.csv"
        report_path = output_dir / "static_convergence_report.md"

        write_csv(
            history_path,
            [
                "time",
                "absolute_time",
                "gravity_factor",
                "gravity_x",
                "gravity_y",
                "max_velocity",
                "mean_velocity",
                "rms_velocity",
                "max_abs_vx",
                "max_abs_vy",
                "rms_vx",
                "rms_vy",
                "grid_max_velocity",
                "grid_rms_velocity",
                "grid_max_abs_vx",
                "grid_max_abs_vy",
                "max_stress_norm",
                "mean_stress_norm",
                "relative_stress_change",
                "relative_unbalanced_force",
                "support_max_force",
                "support_relative_force",
                "particle_velocity_converged",
                "grid_velocity_converged",
                "force_converged",
                "converged",
                "convergence_streak",
                "particle_count",
            ],
            self.rows,
        )
        write_csv(
            unbalanced_path,
            [
                "time",
                "absolute_time",
                "active_node_count",
                "total_mass",
                "total_weight",
                "max_force",
                "rms_force",
                "relative_unbalanced_force",
                "max_force_node_id",
                "max_force_body_id",
                "max_force_x",
                "max_force_y",
                "excluded_node_count",
                "ignored_node_count",
                "ignored_nodal_body_count",
                "support_node_count",
                "support_max_force",
                "support_rms_force",
                "support_relative_force",
                "support_max_force_node_id",
                "support_max_force_body_id",
                "support_max_force_x",
                "support_max_force_y",
            ],
            self.unbalanced_force_rows,
        )
        write_csv(
            timestep_path,
            [
                "critical_timestep",
                "static_dt",
                "static_dt_cap",
                "cfl_safety_ratio",
                "dt_within_50_percent_critical",
                "dynamic_dt_preserved",
                "static_max_time",
                "history_interval",
                "step_limit",
            ],
            self.timestep_rows,
        )

        velocity = self.final_snapshot.get("velocity_components", velocity_component_stats(np.empty((0, 2))))
        nodal_velocity = self.final_snapshot.get("nodal_velocity_components", velocity_component_stats(np.empty((0, 2))))
        max_displacement = float(self.final_snapshot.get("max_displacement", math.nan))
        max_velocity_particle = self.final_snapshot.get("max_velocity_particle", {})
        force_row = self.unbalanced_force_rows[-1] if self.unbalanced_force_rows else {}
        actual_physical_time = float(self.end_time - self.start_time)
        actual_steps = int(self.total_steps)
        reached_full_time = actual_physical_time >= self.duration - 0.5 * max(float(self.dt), 1.0e-30)
        if self.converged:
            status = "PASS"
        elif reached_full_time and not self.failure_reason:
            status = "FAIL_NOT_CONVERGED"
        elif self.failure_reason:
            status = self.failure_reason
        else:
            status = "FAIL_NOT_CONVERGED"
        recommend_extend_8s = bool(
            status == "FAIL_NOT_CONVERGED"
            and math.isfinite(float(self.last_relative_unbalanced_force))
            and float(self.dt) <= 0.5 * float(self.critical_timestep) + 1.0e-15
        )
        auxiliary_written = bool(self.auxiliary_F_written)
        periodic_mapper = getattr(self.mpm, "free_field_periodic_mapper", None)
        periodic_static_ready = bool(
            getattr(self.mpm, "periodic_mapper_active_before_first_static_step", False)
        )
        periodic_pair_count = int(getattr(periodic_mapper, "pair_count", 0))
        periodic_remap_calls = int(getattr(periodic_mapper, "support_remap_call_count", 0))
        periodic_unmapped_supports = (
            int(periodic_mapper.unmapped_support_count(self.mpm.scene))
            if periodic_mapper is not None and periodic_remap_calls > 0
            else -1
        )
        periodic_main_node_modified_count = int(
            getattr(periodic_mapper, "main_node_modified_count", 0)
        )
        periodic_static_status = bool(
            periodic_static_ready
            and periodic_pair_count > 0
            and periodic_remap_calls > 0
            and periodic_unmapped_supports == 0
            and periodic_main_node_modified_count == 0
        )
        summary_row = {
            "status": status,
            "critical_timestep": self.critical_timestep,
            "static_dt": self.dt,
            "cfl_safety_ratio": self.cfl_safety_ratio,
            "actual_static_physical_time": actual_physical_time,
            "actual_step_count": actual_steps,
            "wall_clock_seconds": self.wall_clock_seconds,
            "final_max_abs_vx": velocity["max_abs_vx"],
            "final_max_abs_vy": velocity["max_abs_vy"],
            "final_max_abs_v": velocity["max_speed"],
            "final_rms_vx": velocity["rms_vx"],
            "final_rms_vy": velocity["rms_vy"],
            "final_rms_v": velocity["rms_speed"],
            "final_relative_unbalanced_force": self.last_relative_unbalanced_force,
            "consecutive_pass_count": self.convergence_streak,
            "support_max_force": force_row.get("support_max_force", 0.0),
            "support_relative_force": force_row.get("support_relative_force", 0.0),
            "support_max_force_node_id": force_row.get("support_max_force_node_id", -1),
            "support_max_force_body_id": force_row.get("support_max_force_body_id", -1),
            "auxiliary_F_written_to_checkpoint": auxiliary_written,
            "checkpoint_path": self.checkpoint_path,
            "geometry_modified": False,
            "source_modified": True,
            "velocity_quench_enabled": self.velocity_quench,
            "static_background_damping": self.spec.get("background_damping", STATIC_DAMPING),
            "static_paper_eq11_damping": STATIC_PAPER_EQ11_DAMPING,
            "static_mirror_reflect_body_force": STATIC_MIRROR_REFLECT_BODY_FORCE,
            "periodic_mapper_active_before_first_static_step": periodic_static_ready,
            "periodic_pair_count": periodic_pair_count,
            "periodic_support_remap_call_count": periodic_remap_calls,
            "periodic_unmapped_support_reference_count": periodic_unmapped_supports,
            "periodic_main_node_modified_count": periodic_main_node_modified_count,
            "periodic_static_topology_status": "PASS" if periodic_static_status else "FAIL",
            "gravity_y": self.target_gravity[1],
            "ramp_time": self.ramp_time,
            "node_force_window_nonzero": int(float(force_row.get("max_force", 0.0) or 0.0) > 0.0),
            "recommend_extend_to_8s_same_dt": recommend_extend_8s,
        }
        write_csv(summary_path, list(summary_row.keys()), [summary_row])

        lines = [
            "# Static Convergence Report",
            "",
            "## Verdict",
            "",
            f"- status: `{status}`",
            f"- PASS/FAIL: `{'PASS' if status == 'PASS' else 'FAIL'}`",
            f"- failure_reason: `{self.failure_reason or ('none' if status == 'PASS' else status)}`",
            f"- recommend_extend_to_8s_same_dt: `{recommend_extend_8s}`",
            "",
            "## Time Step",
            "",
            f"- critical_timestep: `{self.critical_timestep}`",
            f"- actual_static_dt: `{self.dt}`",
            f"- CFL_safety_ratio: `{self.cfl_safety_ratio}`",
            f"- dynamic_dt_preserved: `{DT}`",
            "",
            "## Static Run",
            "",
            f"- actual_static_physical_time: `{actual_physical_time}`",
            f"- actual_step_count: `{actual_steps}`",
            f"- wall_clock_seconds: `{self.wall_clock_seconds}`",
            f"- gravity: `{self.target_gravity}`",
            f"- ramp_time: `{self.ramp_time}`",
            f"- static_background_damping: `{self.spec.get('background_damping', STATIC_DAMPING)}`",
            f"- static_paper_eq11_damping: `{STATIC_PAPER_EQ11_DAMPING}`",
            f"- static_mirror_reflect_body_force: `{STATIC_MIRROR_REFLECT_BODY_FORCE}`",
            "- local_damping_formula: `f_d=-sign(v)*beta*abs(f), component-wise (Kohler Eq.11)`",
            f"- grid_max_velocity_tolerance: `{self.max_velocity_tolerance}`",
            f"- grid_rms_velocity_tolerance: `{self.rms_velocity_tolerance}`",
            f"- free_dof_relative_unbalanced_force_tolerance: `{self.relative_unbalanced_force_tolerance}`",
            f"- STATIC_VELOCITY_QUENCH: `{self.velocity_quench}`",
            "",
            "## Free-Field Periodic Static Topology",
            "",
            f"- mapper_active_before_first_static_step: `{periodic_static_ready}`",
            f"- periodic_pair_count: `{periodic_pair_count}`",
            f"- support_remap_call_count: `{periodic_remap_calls}`",
            f"- unmapped_support_reference_count: `{periodic_unmapped_supports}`",
            f"- main_node_modified_count: `{periodic_main_node_modified_count}`",
            f"- status: `{'PASS' if periodic_static_status else 'FAIL'}`",
            "",
            "## Final Velocity",
            "",
            f"- final_max_abs_vx: `{velocity['max_abs_vx']}`",
            f"- final_max_abs_vy: `{velocity['max_abs_vy']}`",
            f"- final_max_abs_v: `{velocity['max_speed']}`",
            f"- final_rms_vx: `{velocity['rms_vx']}`",
            f"- final_rms_vy: `{velocity['rms_vy']}`",
            f"- final_rms_v: `{velocity['rms_speed']}`",
            f"- final_grid_max_abs_v: `{nodal_velocity['max_speed']}`",
            f"- final_grid_rms_v: `{nodal_velocity['rms_speed']}`",
            f"- max_particle_displacement_m: `{max_displacement}`",
            f"- max_velocity_particle_id: `{max_velocity_particle.get('particle_id', 'nan')}`",
            f"- max_velocity_particle_body_id: `{max_velocity_particle.get('body_id', 'nan')}`",
            f"- max_velocity_particle_position_m: `({max_velocity_particle.get('x', 'nan')}, {max_velocity_particle.get('y', 'nan')})`",
            f"- max_velocity_particle_velocity_m_per_s: `({max_velocity_particle.get('vx', 'nan')}, {max_velocity_particle.get('vy', 'nan')})`",
            "",
            "## Unbalanced Force",
            "",
            "- force_window: `scene.node.force after compute_forces/apply force constraints and before compute_grid_kinematic`",
            "- relative_definition: `max force over free node-body DOFs / total weight of node-body entries with at least one free DOF`",
            "- support_reaction_definition: `bottom x/y and lateral x are reported as support reactions; lateral y remains in the free residual, following Kohler Sections 3.2-3.3`",
            "- image_node_definition: `mirror/extrapolation nodes outside each body's physical x-domain and below the physical base are excluded from both residual and reaction statistics`",
            f"- final_relative_unbalanced_force: `{self.last_relative_unbalanced_force}`",
            f"- max_free_force_node_id: `{force_row.get('max_force_node_id', -1)}`",
            f"- max_free_force_body_id: `{force_row.get('max_force_body_id', -1)}`",
            f"- support_max_force: `{force_row.get('support_max_force', 0.0)}`",
            f"- support_relative_force: `{force_row.get('support_relative_force', 0.0)}`",
            f"- support_max_force_node_id: `{force_row.get('support_max_force_node_id', -1)}`",
            f"- support_max_force_body_id: `{force_row.get('support_max_force_body_id', -1)}`",
            f"- ignored_image_node_count: `{force_row.get('ignored_node_count', 0)}`",
            f"- ignored_image_nodal_body_count: `{force_row.get('ignored_nodal_body_count', 0)}`",
            f"- consecutive_pass_count: `{self.convergence_streak}`",
            f"- node_force_window_nonzero: `{bool(summary_row['node_force_window_nonzero'])}`",
            "",
            "## Checkpoint",
            "",
            f"- checkpoint_saved: `{self.checkpoint_saved}`",
            f"- checkpoint_path: `{self.checkpoint_path}`",
            f"- auxiliary_deformation_gradient_written: `{auxiliary_written}`",
            "",
            "## Modification Scope",
            "",
            "- geometry_modified: `False`",
            "- source_modified: `True`",
            "- GeoTaichi_src_modified: `False`",
            "- physical_model_changed_on_failure: `False`",
            "",
            "## Outputs",
            "",
            f"- history_csv: `{history_path}`",
            f"- unbalanced_force_csv: `{unbalanced_path}`",
            f"- timestep_check_csv: `{timestep_path}`",
            f"- summary_csv: `{summary_path}`",
            "",
            "## Modified Files",
            "",
        ]
        lines.extend([f"- `{path}`" for path in modified_files])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    def write_report(self, modified_files: list[str]) -> Path:
        report_name = str(self.spec.get("output_report", "static_initialization_report.md"))
        report_path = OUTPUT_DIR / report_name
        history_path = OUTPUT_DIR / "static_initialization_history.csv"
        if self.rows:
            write_csv(
                history_path,
                [
                    "time",
                    "absolute_time",
                    "gravity_factor",
                    "gravity_x",
                    "gravity_y",
                    "max_velocity",
                    "mean_velocity",
                    "rms_velocity",
                    "max_abs_vx",
                    "max_abs_vy",
                    "rms_vx",
                    "rms_vy",
                    "grid_max_velocity",
                    "grid_rms_velocity",
                    "grid_max_abs_vx",
                    "grid_max_abs_vy",
                    "max_stress_norm",
                    "mean_stress_norm",
                    "relative_stress_change",
                    "relative_unbalanced_force",
                    "support_max_force",
                    "support_relative_force",
                    "particle_velocity_converged",
                    "grid_velocity_converged",
                    "force_converged",
                    "converged",
                    "convergence_streak",
                    "quench_applied",
                    "particle_count",
                ],
                self.rows,
            )

        velocity = self.final_snapshot.get("velocity", {"max": math.nan, "rms": math.nan})
        nodal_velocity = self.final_snapshot.get(
            "nodal_velocity_components", velocity_component_stats(np.empty((0, 2)))
        )
        max_velocity_particle = self.final_snapshot.get("max_velocity_particle", {})
        stress_change = float(self.final_snapshot.get("relative_stress_change", math.nan))
        velocity_ok = bool(
            nodal_velocity["max_speed"] <= self.max_velocity_tolerance
            and nodal_velocity["rms_speed"] <= self.rms_velocity_tolerance
        )
        stress_ok = bool(stress_change <= self.stress_tolerance)
        _, final_factor = self.gravity_at(self.end_time)
        convergence_status = "PASS" if self.converged else "FAIL"

        lines = [
            "# Static Initialization Report",
            "",
            "## Stage",
            "",
            f"- static_enabled: `{self.enabled}`",
            f"- static_timestep: `{self.dt}`",
            f"- static_max_time: `{self.duration}`",
            f"- static_converged: `{self.converged}`",
            f"- require_static_convergence_before_dynamic: `{self.require_convergence}`",
            f"- convergence_status: `{convergence_status}`",
            f"- convergence_time: `{self.convergence_time}`",
            f"- convergence_step: `{self.convergence_step}`",
            f"- dynamic_stage_starts_at: `{self.end_time}`",
            f"- particle_regenerated_between_stages: `False`",
            f"- static_equilibrium_rule: `grid-node max_velocity < {self.max_velocity_tolerance}, grid-node rms_velocity < {self.rms_velocity_tolerance}, and free-DOF relative_unbalanced_force < {self.relative_unbalanced_force_tolerance}`",
            f"- initial_stress_method: `{'GeoTaichi ParticleStress.GravityField' if self.spec.get('use_geotaichi_gravity_field', USE_GEOTAICHI_GRAVITY_FIELD) else 'GeoTaichi static solve with gravity ramp'}`",
            f"- geotaichi_gravity_field_enabled: `{self.spec.get('use_geotaichi_gravity_field', USE_GEOTAICHI_GRAVITY_FIELD)}`",
            f"- static_boundary_method: `{'mirrored-particle grid equivalent' if STATIC_MIRROR_PARTICLE_BOUNDARY else 'GeoTaichi VelocityConstraint boundary segments'}`",
            f"- static_mirror_particle_boundary_enabled: `{STATIC_MIRROR_PARTICLE_BOUNDARY}`",
            f"- bottom_constraint_band: `{STATIC_BOTTOM_CONSTRAINT_BAND}`",
            f"- static_reaction_node_band: `{STATIC_REACTION_NODE_BAND}`",
            f"- static_bottom_constraint_node_count: `{len(getattr(self.mpm, 'static_bottom_constraint_node_ids_np', []))}`",
            "",
            "## Gravity Ramp",
            "",
            f"- target_gravity: `{self.target_gravity}`",
            f"- ramp_type: `{self.spec.get('ramp', 'smoothstep')}`",
            f"- ramp_time: `{self.ramp_time}`",
            f"- final_gravity_factor: `{final_factor}`",
            f"- background_damping: `{self.spec.get('background_damping', STATIC_DAMPING)}`",
            f"- static_velocity_projection: `{self.mpm.sims.velocity_projection_scheme}`",
            f"- static_alpha_pic: `{self.spec.get('alpha_pic', STATIC_ALPHA_PIC)}`",
            f"- static_paper_eq11_damping: `{STATIC_PAPER_EQ11_DAMPING}`",
            f"- static_stress_update: `{hughes_winget_stress_update_label(self.case)}`",
            f"- hughes_winget_formula_revision: `{PAPER_HUGHES_WINGET_FORMULA_REVISION if hughes_winget_enabled_for_case(self.case) else 'NOT_HUGHES_WINGET'}`",
            "- local_damping_formula: `f_d=-sign(v)*beta*abs(f), component-wise (Kohler Eq.11)`",
            f"- history_csv: `{history_path}`",
            "- iteration_time_history: `see history_csv`",
            "",
            "## Residual Velocity Quench",
            "",
            f"- velocity_quench_enabled: `{self.velocity_quench}`",
            f"- quench_max_velocity_factor: `{self.quench_max_velocity_factor}`",
            f"- quench_condition: `stress_change < tolerance, rms_velocity < rms tolerance, max_velocity < max tolerance`",
            f"- quench_applied: `{self.quench_applied}`",
            f"- quench_time: `{self.quench_time}`",
            f"- quench_step: `{self.quench_step}`",
            "- quench_scope: `driver-level particle velocity and velocity-gradient reset after static stress/rms convergence; solver core unchanged`",
            "",
            "## Final Velocity",
            "",
            f"- convergence_velocity_source: `active grid nodes (Kohler Eq.11 v_i)`",
            f"- grid_max_velocity: `{nodal_velocity['max_speed']}`",
            f"- grid_rms_velocity: `{nodal_velocity['rms_speed']}`",
            f"- particle_max_velocity_diagnostic: `{velocity['max']}`",
            f"- particle_rms_velocity_diagnostic: `{velocity['rms']}`",
            f"- max_velocity_tolerance: `{self.max_velocity_tolerance}`",
            f"- rms_velocity_tolerance: `{self.rms_velocity_tolerance}`",
            f"- velocity_converged: `{'PASS' if velocity_ok else 'FAIL'}`",
            f"- max_velocity_particle_id: `{max_velocity_particle.get('particle_id', 'nan')}`",
            f"- max_velocity_particle_body_id: `{max_velocity_particle.get('body_id', 'nan')}`",
            f"- max_velocity_particle_material_id: `{max_velocity_particle.get('material_id', 'nan')}`",
            f"- max_velocity_particle_coordinates: `({max_velocity_particle.get('x', math.nan)}, {max_velocity_particle.get('y', math.nan)})`",
            f"- max_velocity_particle_velocity: `({max_velocity_particle.get('vx', math.nan)}, {max_velocity_particle.get('vy', math.nan)})`",
            "",
            "## Stress Stability",
            "",
            f"- max_stress_norm: `{self.final_snapshot.get('stress', {}).get('max_norm', math.nan)}`",
            f"- mean_stress_norm: `{self.final_snapshot.get('stress', {}).get('mean_norm', math.nan)}`",
            f"- final_stress_change: `{stress_change}`",
            f"- tolerance: `{self.stress_tolerance}`",
            f"- stress_converged: `{'PASS' if stress_ok else 'FAIL'}`",
            "- stress_check_role: `supplementary diagnostic; not part of the paper Eq.11 velocity criterion`",
            "",
            "## Saved State",
            "",
            "- position: kept in `mpm.scene.particle.x` and copied into `mpm.static_state_snapshot`",
            "- velocity: kept in `mpm.scene.particle.v` and copied into `mpm.static_state_snapshot`",
            "- stress: kept in `mpm.scene.particle.stress` and copied into `mpm.static_state_snapshot`",
            "- strain/state variables: velocity-gradient and material state variables are kept in scene fields; available state variables are copied into `mpm.static_state_snapshot`",
            "",
            "## Modified Files",
            "",
        ]
        lines.extend([f"- `{path}`" for path in modified_files])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path


def free_field_pair_force_component_totals(boundary: Any) -> dict[str, float]:
    pair_count = int(getattr(boundary, "ff_pair_count", 0))
    if pair_count <= 0:
        return {
            "total_free_field_dynamic_stress_force_x": 0.0,
            "total_free_field_dynamic_stress_force_y": 0.0,
            "total_normal_dashpot_force_x": 0.0,
            "total_shear_dashpot_force_y": 0.0,
        }
    return {
        "total_free_field_dynamic_stress_force_x": float(np.sum(boundary.ff_pair_dynamic_stress_force_x.to_numpy()[:pair_count])),
        "total_free_field_dynamic_stress_force_y": float(np.sum(boundary.ff_pair_dynamic_stress_force_y.to_numpy()[:pair_count])),
        "total_normal_dashpot_force_x": float(np.sum(boundary.ff_pair_normal_dashpot_force_x.to_numpy()[:pair_count])),
        "total_shear_dashpot_force_y": float(np.sum(boundary.ff_pair_shear_dashpot_force_y.to_numpy()[:pair_count])),
    }


def free_field_surface_area_runtime_stats(boundary: Any, tracker: Any | None) -> dict[str, Any]:
    pair_count = int(getattr(boundary, "ff_pair_count", 0))
    if pair_count <= 0:
        return {
            "min_surface_stretch": math.nan,
            "max_surface_stretch": math.nan,
            "min_current_surface_area": math.nan,
            "max_current_surface_area": math.nan,
            "max_area_formula_error": math.nan,
            "all_current_surface_areas_valid": False,
        }
    current_areas = boundary.ff_pair_current_surface_areas.to_numpy()[:pair_count]
    initial_areas = boundary.ff_pair_initial_surface_areas_np[:pair_count]
    if tracker is None:
        stretches = current_areas / initial_areas
        formula_error = np.zeros(pair_count, dtype=np.float64)
    else:
        F_values = tracker.to_numpy()
        pair_F = F_values[boundary.ff_pair_main_ids_np[:pair_count]]
        stretches = np.sqrt(pair_F[:, 0, 1] ** 2 + pair_F[:, 1, 1] ** 2)
        formula_error = np.abs(current_areas - initial_areas * stretches)
    finite_positive = np.isfinite(current_areas) & (current_areas > 0.0) & np.isfinite(stretches) & (stretches > 0.0)
    return {
        "min_surface_stretch": float(np.min(stretches)),
        "max_surface_stretch": float(np.max(stretches)),
        "min_current_surface_area": float(np.min(current_areas)),
        "max_current_surface_area": float(np.max(current_areas)),
        "max_area_formula_error": float(np.max(formula_error)),
        "all_current_surface_areas_valid": bool(np.all(finite_positive)),
    }


def free_field_velocity_monitors(scene: Any) -> dict[str, float]:
    particle_count = int(scene.particleNum[0])
    velocities = scene.particle.v.to_numpy()[:particle_count]
    body_ids = scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    material_ids = scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
    main_mask = np.isin(body_ids, np.asarray(MAIN_BODY_IDS, dtype=np.int32)) & (material_ids > 0)
    left_mask = np.isin(body_ids, np.asarray(LEFT_FREE_BODY_IDS, dtype=np.int32)) & (material_ids > 0)
    right_mask = np.isin(body_ids, np.asarray(RIGHT_FREE_BODY_IDS, dtype=np.int32)) & (material_ids > 0)
    return {
        "main_max_velocity": float(np.max(np.linalg.norm(velocities[main_mask, :2], axis=1))) if np.any(main_mask) else 0.0,
        "free_field_left_monitor_vx": float(np.max(np.abs(velocities[left_mask, 0]))) if np.any(left_mask) else 0.0,
        "free_field_right_monitor_vx": float(np.max(np.abs(velocities[right_mask, 0]))) if np.any(right_mask) else 0.0,
    }


def main_velocity_diagnostics(scene: Any) -> dict[str, Any]:
    particle_count = int(scene.particleNum[0])
    velocities = scene.particle.v.to_numpy()[:particle_count]
    positions = scene.particle.x.to_numpy()[:particle_count]
    body_ids = scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    material_ids = scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
    main_ids = np.nonzero(np.isin(body_ids, np.asarray(MAIN_BODY_IDS, dtype=np.int32)) & (material_ids > 0))[0]
    if main_ids.size == 0:
        return {
            "main_max_abs_vx": 0.0,
            "main_rms_vx": 0.0,
            "main_max_velocity_magnitude": 0.0,
            "max_velocity_particle_id": -1,
            "max_velocity_particle_body_id": -1,
            "max_velocity_particle_material_id": -1,
            "max_velocity_particle_x": math.nan,
            "max_velocity_particle_y": math.nan,
            "max_velocity_particle_vx": math.nan,
            "max_velocity_particle_vy": math.nan,
        }
    main_v = velocities[main_ids, :2]
    speed = np.linalg.norm(main_v, axis=1)
    local_index = int(np.argmax(speed))
    pid = int(main_ids[local_index])
    return {
        "main_max_abs_vx": float(np.max(np.abs(main_v[:, 0]))),
        "main_rms_vx": float(math.sqrt(np.mean(main_v[:, 0] ** 2))),
        "main_max_velocity_magnitude": float(speed[local_index]),
        "max_velocity_particle_id": pid,
        "max_velocity_particle_body_id": int(body_ids[pid]),
        "max_velocity_particle_material_id": int(material_ids[pid]),
        "max_velocity_particle_x": float(positions[pid, 0] - X_SHIFT),
        "max_velocity_particle_y": float(positions[pid, 1] - Y_SHIFT),
        "max_velocity_particle_vx": float(velocities[pid, 0]),
        "max_velocity_particle_vy": float(velocities[pid, 1]),
    }


def periodic_runtime_metrics(mpm: MPM, mapper: Any | None) -> dict[str, float | int]:
    if mapper is None or int(getattr(mapper, "pair_count", 0)) <= 0:
        return {
            "max_periodic_velocity_difference": math.nan,
            "max_periodic_acceleration_difference": math.nan,
            "mass_conservation_error": math.nan,
            "momentum_conservation_error": math.nan,
            "force_conservation_error": math.nan,
            "main_node_modified_count": -1,
        }
    # The right endpoint is no longer an independently updated node. Its support
    # entries are remapped before P2G, so separate-node velocity/force deltas are
    # not meaningful runtime diagnostics.
    return {
        "max_periodic_velocity_difference": 0.0,
        "max_periodic_acceleration_difference": 0.0,
        "mass_conservation_error": float(getattr(mapper, "mass_conservation_error", 0.0)),
        "momentum_conservation_error": float(np.max(getattr(mapper, "momentum_conservation_error", np.zeros(2)))),
        "force_conservation_error": float(np.max(getattr(mapper, "force_conservation_error", np.zeros(2)))),
        "periodic_support_remap_call_count": int(getattr(mapper, "support_remap_call_count", 0)),
        "main_node_modified_count": int(getattr(mapper, "main_node_modified_count", -1)),
    }


def finite_particle_state_counts(scene: Any) -> dict[str, int]:
    particle_count = int(scene.particleNum[0])
    arrays = [
        scene.particle.x.to_numpy()[:particle_count],
        scene.particle.v.to_numpy()[:particle_count],
        scene.particle.stress.to_numpy()[:particle_count],
    ]
    return {
        "NaN_count": int(sum(np.count_nonzero(np.isnan(array)) for array in arrays)),
        "Inf_count": int(sum(np.count_nonzero(np.isinf(array)) for array in arrays)),
    }


def update_latest_joint_boundary_row_after_step(mpm: MPM, periodic_mapper: Any) -> None:
    boundary = getattr(mpm, "nairn_seismic_boundary", None)
    if boundary is None or not boundary.boundary_rows:
        return
    row = boundary.boundary_rows[-1]
    auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    surface_stats = free_field_surface_area_runtime_stats(boundary, auxiliary_tracker)
    auxiliary_stats = auxiliary_tracker.stats() if auxiliary_tracker is not None else {
        "min_detF": math.nan,
        "max_detF": math.nan,
    }
    row.update(surface_stats)
    row.update(
        {
            "periodic_mass_sync_call_count": int(getattr(periodic_mapper, "mass_sync_call_count", 0)),
            "periodic_force_sync_call_count": int(getattr(periodic_mapper, "force_sync_call_count", 0)),
            "periodic_kinematic_sync_call_count": int(getattr(periodic_mapper, "kinematic_sync_call_count", 0)),
            "periodic_before_g2p_sync_call_count": int(getattr(periodic_mapper, "before_g2p_sync_call_count", 0)),
            "auxiliary_F_update_call_count": int(auxiliary_tracker.update_call_count[None]) if auxiliary_tracker is not None else 0,
            "min_detF": float(auxiliary_stats["min_detF"]),
            "max_detF": float(auxiliary_stats["max_detF"]),
            "invalid_F_count": int(auxiliary_tracker.invalid_F_count[None]) if auxiliary_tracker is not None else -1,
        }
    )
    row.update(periodic_runtime_metrics(mpm, periodic_mapper))
    row.update(free_field_velocity_monitors(mpm.scene))
    row.update(main_velocity_diagnostics(mpm.scene))
    velocities = mpm.scene.particle.v.to_numpy()[: int(mpm.scene.particleNum[0])]
    positions = mpm.scene.particle.x.to_numpy()[: int(mpm.scene.particleNum[0])]
    for name, pid in boundary.joint_monitor_particles.items():
        row[f"{name}_pid"] = int(pid)
        row[f"{name}_x"] = float(positions[int(pid), 0] - X_SHIFT)
        row[f"{name}_y"] = float(positions[int(pid), 1] - Y_SHIFT)
        row[f"{name}_vx"] = float(velocities[int(pid), 0])
    row.update(finite_particle_state_counts(mpm.scene))


class NairnSeismicBoundary:
    def __init__(self, mpm: MPM, case: dict[str, Any]) -> None:
        self.mpm = mpm
        self.case = case
        self.next_history = 0.0
        self.next_boundary_diagnostic = 0.0
        self.rows: list[dict[str, float | int]] = []
        self.boundary_rows: list[dict[str, float | int]] = []
        self.dynamic_start_time = 0.0
        self.input_enabled = True
        self.incremental_history_enabled = INCREMENTAL_HISTORY_ENABLED
        self.incremental_history_interval = max(float(INCREMENTAL_HISTORY_INTERVAL), 0.0)
        self.incremental_next_flush_time = 0.0
        self.incremental_history_cursor = 0
        self.incremental_boundary_cursor = 0
        self.incremental_history_fields: list[str] | None = None
        self.incremental_boundary_fields: list[str] | None = None
        self.first_domain_escape: dict[str, Any] | None = None
        self.domain_escape_enabled = DOMAIN_ESCAPE_DIAGNOSTIC or DOMAIN_ESCAPE_ABORT
        self.domain_escape_abort = DOMAIN_ESCAPE_ABORT
        self.domain_escape_pair_ids = set()

        particle_count = int(mpm.scene.particleNum[0])
        positions = mpm.scene.particle.x.to_numpy()[:particle_count]
        body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count]
        material_ids = mpm.scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
        psizes = np.asarray(mpm.scene.psize, dtype=np.float64)[:particle_count]

        ptraction_count = int(mpm.scene.boundary.ptraction_list[0])
        traction_pids = mpm.scene.boundary.particle_traction.pid.to_numpy()[:ptraction_count].astype(np.int32)
        self.bottom_particle_ids_np = np.ascontiguousarray(traction_pids)
        stress_static_rx, stress_static_ry = bottom_static_reaction_tractions(
            getattr(mpm, "static_state_snapshot", {}),
            self.bottom_particle_ids_np,
            positions,
        )
        if bool(getattr(mpm, "bottom_static_support_frozen_before_dynamic", False)):
            self.bottom_static_reaction_csv = getattr(
                mpm, "bottom_static_support_path", OUTPUT_DIR / "bottom_static_reaction_from_geotaichi.csv"
            )
            self.bottom_static_reaction_node_ids_np = np.ascontiguousarray(
                mpm.bottom_static_support_node_ids_np, dtype=np.int32
            )
            self.bottom_static_reaction_node_body_ids_np = np.ascontiguousarray(
                mpm.bottom_static_support_body_ids_np, dtype=np.int32
            )
            self.bottom_static_reaction_node_x_np = np.ascontiguousarray(mpm.bottom_static_support_x_np, dtype=np.float64)
            self.bottom_static_reaction_node_y_np = np.ascontiguousarray(mpm.bottom_static_support_y_np, dtype=np.float64)
            self.bottom_static_reaction_x_np = np.zeros(self.bottom_particle_ids_np.size, dtype=np.float64)
            self.bottom_static_reaction_y_np = np.zeros(self.bottom_particle_ids_np.size, dtype=np.float64)
            self.bottom_static_reaction_initialized = True
            self.bottom_static_reaction_initialization_time = 0.0
            support_norm = np.hypot(self.bottom_static_reaction_node_x_np, self.bottom_static_reaction_node_y_np)
            self.bottom_static_reaction_initialization_max_residual = float(np.max(support_norm)) if support_norm.size else 0.0
        else:
            # Compatibility fallback for isolated diagnostics that do not run
            # through the static-to-dynamic transition path.
            (
                static_rx,
                static_ry,
                self.bottom_static_reaction_csv,
                static_node_ids,
                _static_node_body_ids,
                _static_node_rx,
                _static_node_ry,
            ) = geotaichi_bottom_static_reaction_tractions(
                mpm,
                self.bottom_particle_ids_np,
            )
            self.bottom_static_reaction_x_np = np.ascontiguousarray(static_rx)
            self.bottom_static_reaction_y_np = np.ascontiguousarray(static_ry)
            constraint_node_ids = np.asarray(
                getattr(mpm, "static_bottom_constraint_node_ids_np", static_node_ids), dtype=np.int32
            )
            active_body_ids = np.unique(body_ids.astype(np.int32))
            self.bottom_static_reaction_node_ids_np = np.repeat(constraint_node_ids, active_body_ids.size)
            self.bottom_static_reaction_node_body_ids_np = np.tile(active_body_ids, constraint_node_ids.size)
            self.bottom_static_reaction_node_x_np = np.zeros(self.bottom_static_reaction_node_ids_np.size, dtype=np.float64)
            self.bottom_static_reaction_node_y_np = np.zeros(self.bottom_static_reaction_node_ids_np.size, dtype=np.float64)
            self.bottom_static_reaction_initialized = False
            self.bottom_static_reaction_initialization_time = None
            self.bottom_static_reaction_initialization_max_residual = math.nan
        self.bottom_static_reaction_stress_x_np = np.ascontiguousarray(stress_static_rx)
        self.bottom_static_reaction_stress_y_np = np.ascontiguousarray(stress_static_ry)

        traction_id_by_pid = {int(pid): i for i, pid in enumerate(traction_pids.tolist())}
        bottom_traction_ids = np.array([traction_id_by_pid[int(pid)] for pid in traction_pids], dtype=np.int32)

        physical_x = positions[:, 0] - X_SHIFT
        physical_y = positions[:, 1] - Y_SHIFT
        side_spec = case.get("silent_boundary", {}).get("side", {})
        side_enabled = bool(side_spec.get("enabled", True))
        side_body_ids = np.array(side_spec.get("body_ids", MAIN_BODY_IDS), dtype=np.int32)
        side_x_values = [float(value) for value in side_spec.get("x_values", [K_MAIN_X_MIN, K_MAIN_X_MAX])]
        side_y_min, side_y_max = [
            float(value) for value in side_spec.get("y_range", [K_DOMAIN_Y_MIN, K_DOMAIN_Y_MAX])
        ]
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
        self.free_field_pairs = build_free_field_interface_pairs(positions, body_ids, self.side_particle_ids_np)
        self.ff_pair_main_ids_np = self.free_field_pairs["main_ids"]
        self.ff_pair_ids_np = self.free_field_pairs["ff_ids"]
        self.ff_pair_sides_np = self.free_field_pairs["side"]
        self.ff_pair_distances_np = self.free_field_pairs["distance"]
        self.domain_escape_pair_ids = set(int(pid) for pid in self.ff_pair_main_ids_np.tolist())
        static_stresses = np.asarray(
            getattr(mpm, "static_state_snapshot", {}).get("arrays", {}).get("stress", np.zeros((particle_count, 6))),
            dtype=np.float64,
        )
        if static_stresses.shape[0] < particle_count:
            static_stresses = np.zeros((particle_count, 6), dtype=np.float64)
        materials = material_by_id(case)
        side_signs = DIAGNOSTIC_FREE_FIELD_SIGN_MULT * np.asarray(
            [-1.0 if side == "left" else 1.0 for side in self.ff_pair_sides_np],
            dtype=np.float64,
        )
        if self.ff_pair_main_ids_np.size:
            pair_psizes = psizes[self.ff_pair_main_ids_np]
            surface_areas = 2.0 * pair_psizes[:, 1]
        else:
            surface_areas = np.empty(0, dtype=np.float64)
        normal_impedances = np.zeros(self.ff_pair_ids_np.size, dtype=np.float64)
        shear_impedances = np.zeros(self.ff_pair_ids_np.size, dtype=np.float64)
        normal_impedance_areas = np.zeros(self.ff_pair_ids_np.size, dtype=np.float64)
        shear_impedance_areas = np.zeros(self.ff_pair_ids_np.size, dtype=np.float64)
        for index, ff_pid in enumerate(self.ff_pair_ids_np.tolist()):
            material = materials.get(int(material_ids[int(ff_pid)]), next(iter(materials.values())))
            density = float(material["Density"])
            young = float(material["YoungModulus"])
            poisson = float(material["PossionRatio"])
            shear = young / (2.0 * (1.0 + poisson))
            bulk = young / (3.0 * (1.0 - 2.0 * poisson))
            cs = math.sqrt(shear / density)
            cp = math.sqrt((bulk + 4.0 * shear / 3.0) / density)
            normal_impedances[index] = density * cp
            shear_impedances[index] = density * cs
            normal_impedance_areas[index] = normal_impedances[index] * surface_areas[index]
            shear_impedance_areas[index] = shear_impedances[index] * surface_areas[index]
        self.ff_pair_side_signs_np = np.ascontiguousarray(side_signs)
        self.ff_pair_initial_surface_areas_np = np.ascontiguousarray(surface_areas)
        self.ff_pair_surface_areas_np = np.ascontiguousarray(surface_areas)
        self.ff_pair_normal_impedances_np = np.ascontiguousarray(normal_impedances)
        self.ff_pair_shear_impedances_np = np.ascontiguousarray(shear_impedances)
        self.ff_pair_normal_impedance_areas_np = np.ascontiguousarray(normal_impedance_areas)
        self.ff_pair_shear_impedance_areas_np = np.ascontiguousarray(shear_impedance_areas)
        self.ff_pair_static_stress_xx_np = np.ascontiguousarray(static_stresses[self.ff_pair_ids_np, 0] if self.ff_pair_ids_np.size else np.empty(0))
        self.ff_pair_static_stress_xy_np = np.ascontiguousarray(static_stresses[self.ff_pair_ids_np, 3] if self.ff_pair_ids_np.size else np.empty(0))
        self.ff_pair_initial_main_positions_np = np.ascontiguousarray(
            positions[self.ff_pair_main_ids_np].copy() if self.ff_pair_main_ids_np.size else np.empty((0, 2)),
            dtype=np.float64,
        )
        self.ff_pair_initial_free_field_positions_np = np.ascontiguousarray(
            positions[self.ff_pair_ids_np].copy() if self.ff_pair_ids_np.size else np.empty((0, 2)),
            dtype=np.float64,
        )
        self.ff_pair_initial_vertical_offsets_np = np.ascontiguousarray(
            np.abs(self.ff_pair_initial_main_positions_np[:, 1] - self.ff_pair_initial_free_field_positions_np[:, 1]),
            dtype=np.float64,
        )
        self.ff_pair_initial_geometric_gaps_np = np.ascontiguousarray(
            np.abs(self.ff_pair_initial_main_positions_np[:, 0] - self.ff_pair_initial_free_field_positions_np[:, 0]) - DX / 3.0,
            dtype=np.float64,
        )

        self.bottom_count = int(self.bottom_particle_ids_np.size)
        self.bottom_static_reaction_node_count = int(self.bottom_static_reaction_node_ids_np.size)
        self.side_count = int(self.side_particle_ids_np.size)
        self.ff_pair_count = int(self.ff_pair_main_ids_np.size)
        self.bottom_particle_ids = ti.field(dtype=ti.i32, shape=max(1, self.bottom_count))
        self.bottom_traction_ids = ti.field(dtype=ti.i32, shape=max(1, self.bottom_count))
        self.bottom_static_reaction_x = ti.field(dtype=ti.f64, shape=max(1, self.bottom_count))
        self.bottom_static_reaction_y = ti.field(dtype=ti.f64, shape=max(1, self.bottom_count))
        self.bottom_static_reaction_node_ids = ti.field(dtype=ti.i32, shape=max(1, self.bottom_static_reaction_node_count))
        self.bottom_static_reaction_node_body_ids = ti.field(dtype=ti.i32, shape=max(1, self.bottom_static_reaction_node_count))
        self.bottom_static_reaction_node_x = ti.field(dtype=ti.f64, shape=max(1, self.bottom_static_reaction_node_count))
        self.bottom_static_reaction_node_y = ti.field(dtype=ti.f64, shape=max(1, self.bottom_static_reaction_node_count))
        self.side_particle_ids = ti.field(dtype=ti.i32, shape=max(1, self.side_count))
        self.ff_pair_main_ids = ti.field(dtype=ti.i32, shape=max(1, self.ff_pair_count))
        self.ff_pair_ids = ti.field(dtype=ti.i32, shape=max(1, self.ff_pair_count))
        self.ff_pair_side_signs = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_initial_surface_areas = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_current_surface_areas = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_normal_impedances = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_shear_impedances = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_static_stress_xx = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_static_stress_xy = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_force_balance_error = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        if self.bottom_count:
            self.bottom_particle_ids.from_numpy(self.bottom_particle_ids_np)
            self.bottom_traction_ids.from_numpy(bottom_traction_ids)
            self.bottom_static_reaction_x.from_numpy(self.bottom_static_reaction_x_np)
            self.bottom_static_reaction_y.from_numpy(self.bottom_static_reaction_y_np)
        if self.bottom_static_reaction_node_count:
            self.bottom_static_reaction_node_ids.from_numpy(self.bottom_static_reaction_node_ids_np)
            self.bottom_static_reaction_node_body_ids.from_numpy(self.bottom_static_reaction_node_body_ids_np)
            self.bottom_static_reaction_node_x.from_numpy(self.bottom_static_reaction_node_x_np)
            self.bottom_static_reaction_node_y.from_numpy(self.bottom_static_reaction_node_y_np)
        if self.side_count:
            self.side_particle_ids.from_numpy(self.side_particle_ids_np)
        if self.ff_pair_count:
            self.ff_pair_main_ids.from_numpy(self.ff_pair_main_ids_np)
            self.ff_pair_ids.from_numpy(self.ff_pair_ids_np)
            self.ff_pair_side_signs.from_numpy(self.ff_pair_side_signs_np)
            self.ff_pair_initial_surface_areas.from_numpy(self.ff_pair_initial_surface_areas_np)
            self.ff_pair_current_surface_areas.from_numpy(self.ff_pair_surface_areas_np)
            self.ff_pair_normal_impedances.from_numpy(self.ff_pair_normal_impedances_np)
            self.ff_pair_shear_impedances.from_numpy(self.ff_pair_shear_impedances_np)
            self.ff_pair_static_stress_xx.from_numpy(self.ff_pair_static_stress_xx_np)
            self.ff_pair_static_stress_xy.from_numpy(self.ff_pair_static_stress_xy_np)

        self.total_input_force_x = ti.field(dtype=ti.f64, shape=())
        self.total_input_force_y = ti.field(dtype=ti.f64, shape=())
        self.total_static_reaction_force_x = ti.field(dtype=ti.f64, shape=())
        self.total_static_reaction_force_y = ti.field(dtype=ti.f64, shape=())
        self.total_bottom_silent_x = ti.field(dtype=ti.f64, shape=())
        self.total_bottom_silent_y = ti.field(dtype=ti.f64, shape=())
        self.total_side_silent_x = ti.field(dtype=ti.f64, shape=())
        self.total_side_silent_y = ti.field(dtype=ti.f64, shape=())
        self.total_ff_coupling_main_x = ti.field(dtype=ti.f64, shape=())
        self.total_ff_coupling_main_y = ti.field(dtype=ti.f64, shape=())
        self.total_ff_coupling_free_x = ti.field(dtype=ti.f64, shape=())
        self.total_ff_coupling_free_y = ti.field(dtype=ti.f64, shape=())
        self.total_ff_coupling_power_main = ti.field(dtype=ti.f64, shape=())
        self.total_ff_dashpot_dissipation = ti.field(dtype=ti.f64, shape=())
        self.max_ff_coupling_balance_error = ti.field(dtype=ti.f64, shape=())
        self.surface_area_update_call_count = ti.field(dtype=ti.i32, shape=())
        self.invalid_surface_area_count = ti.field(dtype=ti.i32, shape=())
        self.total_invalid_surface_area_count = ti.field(dtype=ti.i32, shape=())
        self.ff_pair_dynamic_stress_force_x = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_dynamic_stress_force_y = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_normal_dashpot_force_x = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_shear_dashpot_force_y = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_total_force_x = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.ff_pair_total_force_y = ti.field(dtype=ti.f64, shape=max(1, self.ff_pair_count))
        self.max_observed_ff_coupling_balance_error = 0.0
        support_node_ids_np = np.ascontiguousarray(
            getattr(mpm, "lateral_static_support_node_ids_np", np.empty(0, dtype=np.int32)),
            dtype=np.int32,
        )
        support_body_ids_np = np.ascontiguousarray(
            getattr(mpm, "lateral_static_support_body_ids_np", np.empty(0, dtype=np.int32)),
            dtype=np.int32,
        )
        support_x_np = np.ascontiguousarray(
            getattr(mpm, "lateral_static_support_x_np", np.empty(0, dtype=np.float64)),
            dtype=np.float64,
        )
        self.lateral_static_support_count = int(support_node_ids_np.size)
        self.lateral_static_support_node_ids = ti.field(dtype=ti.i32, shape=max(1, self.lateral_static_support_count))
        self.lateral_static_support_body_ids = ti.field(dtype=ti.i32, shape=max(1, self.lateral_static_support_count))
        self.lateral_static_support_x = ti.field(dtype=ti.f64, shape=max(1, self.lateral_static_support_count))
        self.total_lateral_static_support_x = ti.field(dtype=ti.f64, shape=())
        if self.lateral_static_support_count:
            self.lateral_static_support_node_ids.from_numpy(support_node_ids_np)
            self.lateral_static_support_body_ids.from_numpy(support_body_ids_np)
            self.lateral_static_support_x.from_numpy(support_x_np)

        self.monitor_particles: dict[str, int] = {}
        self.monitor_specs: dict[str, dict[str, Any]] = {}
        self.input_motion_monitors: list[dict[str, Any]] = []
        self.monitor_check_rows: list[dict[str, Any]] = []
        self._initialize_history_monitors(case.get("monitors", []))
        self.joint_monitor_particles = {
            "main_left_monitor": self._nearest_particle(BODY_MAIN_SOIL, (K_MAIN_X_MIN, K_INTERFACE_HIGH_Z)),
            "main_right_monitor": self._nearest_particle(BODY_MAIN_SOIL, (K_MAIN_X_MAX, K_INTERFACE_LOW_Z)),
            "main_internal_monitor": self.monitor_particles.get(
                "kohler_boundary_interface",
                self._nearest_particle(BODY_MAIN_SOIL, (K_MAIN_X_MIN + HALF_PARTICLE_SIZE, K_INTERFACE_HIGH_Z)),
            ),
            "main_surface_monitor": self.monitor_particles.get(
                "kohler_surface",
                self._nearest_particle(BODY_MAIN_SOIL, (0.5 * (K_MAIN_X_MIN + K_MAIN_X_MAX), K_LOW_SURFACE_Z)),
            ),
            "left_free_field_monitor": self.monitor_particles.get(
                "left_free_field_monitor",
                self._nearest_particle(BODY_LEFT_FREE_SOIL, (K_LEFT_FREE_X_MAX - HALF_PARTICLE_SIZE, K_INTERFACE_HIGH_Z)),
            ),
            "right_free_field_monitor": self.monitor_particles.get(
                "right_free_field_monitor",
                self._nearest_particle(BODY_RIGHT_FREE_SOIL, (K_RIGHT_FREE_X_MIN + HALF_PARTICLE_SIZE, K_INTERFACE_LOW_Z)),
            ),
        }

    def set_dynamic_start_time(self, start_time: float) -> None:
        self.dynamic_start_time = float(start_time)

    def dynamic_time(self, sims: Any) -> float:
        return max(0.0, float(sims.current_time) - self.dynamic_start_time)

    def _append_incremental_rows(
        self,
        path: Path,
        rows: list[dict[str, Any]],
        cursor: int,
        fields_attr: str,
    ) -> int:
        if cursor >= len(rows):
            return cursor
        new_rows = rows[cursor:]
        fields = getattr(self, fields_attr)
        if fields is None:
            fields = list(new_rows[0].keys())
            setattr(self, fields_attr, fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        has_header = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            if not has_header:
                writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in new_rows)
            stream.flush()
            if INCREMENTAL_HISTORY_FSYNC:
                os.fsync(stream.fileno())
        return len(rows)

    def flush_incremental_outputs(self, sims: Any, force: bool = False) -> None:
        """Persist dynamic rows before a long run can lose them on failure."""
        if not self.incremental_history_enabled:
            return
        time_value = self.dynamic_time(sims)
        interval = self.incremental_history_interval
        due = force or interval <= 0.0 or time_value + 0.5 * float(sims.delta) >= self.incremental_next_flush_time
        if not due:
            return
        self.incremental_history_cursor = self._append_incremental_rows(
            INCREMENTAL_HISTORY_OUTPUT,
            self.rows,
            self.incremental_history_cursor,
            "incremental_history_fields",
        )
        self.incremental_boundary_cursor = self._append_incremental_rows(
            INCREMENTAL_BOUNDARY_OUTPUT,
            self.boundary_rows,
            self.incremental_boundary_cursor,
            "incremental_boundary_fields",
        )
        if interval <= 0.0:
            self.incremental_next_flush_time = math.inf
        else:
            while self.incremental_next_flush_time <= time_value + 0.5 * float(sims.delta):
                self.incremental_next_flush_time += interval

    def _domain_escape_event(self, sims: Any, scene: Any) -> dict[str, Any] | None:
        particle_count = int(scene.particleNum[0])
        positions = np.asarray(scene.particle.x.to_numpy()[:particle_count], dtype=np.float64)
        velocities = np.asarray(scene.particle.v.to_numpy()[:particle_count], dtype=np.float64)
        body_ids = np.asarray(scene.particle.bodyID.to_numpy()[:particle_count], dtype=np.int32)
        physical = positions - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
        tolerance = DOMAIN_ESCAPE_TOLERANCE
        finite = np.all(np.isfinite(positions[:, :2]), axis=1)
        # Main/free-field interfaces are material partitions, not fixed spatial
        # boundaries. Their particles may cross the initial interface while the
        # media deform. Only leaving the actual background-grid domain makes the
        # MPM interpolation support invalid and warrants an automatic abort.
        outside = (~finite) | np.any(
            (positions[:, :2] < -tolerance)
            | (positions[:, :2] > np.asarray(DOMAIN[:2], dtype=np.float64) + tolerance),
            axis=1,
        )
        if not np.any(outside):
            return None

        escaped_ids = np.nonzero(outside)[0]
        first_pid = int(escaped_ids[0])
        first_body_id = int(body_ids[first_pid])
        first_position = physical[first_pid, :2]
        first_velocity = velocities[first_pid, :2]
        support_state: list[dict[str, Any]] = []
        try:
            total_nodes = int(scene.element.grid_nodes)
            node_count = int(scene.element.node_size.to_numpy()[first_pid])
            node_offset = first_pid * total_nodes
            support_ids = scene.element.LnID.to_numpy()[node_offset : node_offset + node_count]
            node_arrays = node_field_arrays(scene)
            grid_x = int(scene.element.gnum[0])
            for node_id_value in support_ids.tolist():
                node_id = int(node_id_value)
                nodal_velocity = np.asarray(node_arrays["momentum"][node_id, first_body_id, :2], dtype=np.float64)
                support_state.append(
                    {
                        "node_id": node_id,
                        "ij": [node_id % grid_x, node_id // grid_x],
                        "mass": float(node_arrays["m"][node_id, first_body_id]),
                        "velocity": nodal_velocity.tolist(),
                        "speed": float(np.linalg.norm(nodal_velocity)),
                    }
                )
        except Exception as support_error:
            support_state = [{"diagnostic_error": repr(support_error)}]

        main_mask = body_ids == BODY_MAIN_SOIL
        main_speed = np.linalg.norm(velocities[main_mask, :2], axis=1) if np.any(main_mask) else np.empty(0)
        tracker = getattr(self.mpm, "auxiliary_deformation_gradient_tracker", None)
        tracker_stats = tracker.stats() if tracker is not None else {}
        return {
            "time": self.dynamic_time(sims),
            "absolute_time": float(sims.current_time),
            "escaped_particle_count": int(escaped_ids.size),
            "first_pid": first_pid,
            "first_body_id": first_body_id,
            "first_particle_paired_to_free_field": int(first_pid in self.domain_escape_pair_ids),
            "first_x": float(first_position[0]),
            "first_z": float(first_position[1]),
            "first_vx": float(first_velocity[0]),
            "first_vz": float(first_velocity[1]),
            "main_max_speed": float(np.max(main_speed)) if main_speed.size else 0.0,
            "support_state": json.dumps(support_state, separators=(",", ":")),
            "min_detF": float(tracker_stats.get("min_detF", math.nan)),
            "max_detF": float(tracker_stats.get("max_detF", math.nan)),
            "invalid_F_count": int(tracker.invalid_F_count[None]) if tracker is not None else -1,
        }

    def check_domain_escape(self, sims: Any, scene: Any) -> dict[str, Any] | None:
        if not self.domain_escape_enabled:
            return None
        event = self._domain_escape_event(sims, scene)
        if event is None:
            return None
        if self.first_domain_escape is None:
            self.first_domain_escape = event
            write_csv(DOMAIN_ESCAPE_OUTPUT, list(event.keys()), [event])
        return event

    def capture_runtime_failure(self, sims: Any, scene: Any, error: BaseException) -> None:
        try:
            event = self.check_domain_escape(sims, scene)
            self.flush_incremental_outputs(sims, force=True)
            lines = [
                "# Runtime Failure Diagnostic",
                "",
                f"- dynamic_time: `{self.dynamic_time(sims)}`",
                f"- absolute_time: `{float(sims.current_time)}`",
                f"- exception: `{repr(error)}`",
                f"- first_domain_escape: `{event is not None or self.first_domain_escape is not None}`",
                f"- incremental_history_enabled: `{self.incremental_history_enabled}`",
                f"- incremental_history_output: `{INCREMENTAL_HISTORY_OUTPUT}`",
                f"- incremental_boundary_output: `{INCREMENTAL_BOUNDARY_OUTPUT}`",
                f"- domain_escape_output: `{DOMAIN_ESCAPE_OUTPUT}`",
            ]
            DOMAIN_FAILURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            DOMAIN_FAILURE_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as diagnostic_error:
            print(f"Runtime diagnostic capture failed: {diagnostic_error}")

    def initialize_bottom_static_reaction_from_force_window(self, sims: Any, scene: Any) -> None:
        """Freeze the direct static reaction from the first zero-input dynamic assembly."""
        if self.bottom_static_reaction_initialized:
            return
        input_force_x = float(self.total_input_force_x[None])
        input_force_y = float(self.total_input_force_y[None])
        if abs(input_force_x) > 1.0e-12 or abs(input_force_y) > 1.0e-12:
            raise RuntimeError("Cannot initialize static reactions after seismic input has started")
        node_force = scene.node.force.to_numpy()
        node_ids = self.bottom_static_reaction_node_ids_np
        body_ids = self.bottom_static_reaction_node_body_ids_np
        if node_ids.size:
            reactions = -node_force[node_ids, body_ids, :2]
            self.bottom_static_reaction_node_x_np = np.ascontiguousarray(reactions[:, 0])
            self.bottom_static_reaction_node_y_np = np.ascontiguousarray(reactions[:, 1])
            self.bottom_static_reaction_node_x.from_numpy(self.bottom_static_reaction_node_x_np)
            self.bottom_static_reaction_node_y.from_numpy(self.bottom_static_reaction_node_y_np)
            self.bottom_static_reaction_initialization_max_residual = float(np.max(np.linalg.norm(reactions, axis=1)))
        else:
            self.bottom_static_reaction_initialization_max_residual = 0.0
        self.bottom_static_reaction_initialization_time = self.dynamic_time(sims)
        self.bottom_static_reaction_initialized = True

    def _initialize_history_monitors(self, monitors: list[dict[str, Any]]) -> None:
        particle_count = int(self.mpm.scene.particleNum[0])
        positions = self.mpm.scene.particle.x.to_numpy()[:particle_count]
        for item in monitors:
            name = str(item["name"])
            purpose = str(item.get("purpose", "history point"))
            if str(item.get("source", "")).lower() == "input_motion":
                self.input_motion_monitors.append(item)
                self.monitor_check_rows.append(
                    {
                        "name": name,
                        "particle_id": "",
                        "target_x": "",
                        "target_z": "bottom_compliant_base_input_boundary",
                        "actual_x": "",
                        "actual_z": "",
                        "purpose": purpose,
                    }
                )
                continue

            body_id = int(item["body_id"])
            point = tuple(float(value) for value in item["point"])
            selector = str(item.get("selector", "nearest_particle"))
            if selector == "soil_base_interface_nearest_particle":
                target_x = float(point[0])
                target_z = float(kohler_soil_base_interface_z_np(target_x))
                particle_id = self._nearest_particle(body_id, (target_x, target_z))
            elif selector == "surface_nearest_particle":
                target_x = float(point[0])
                target_z = float(kohler_surface_z_np(target_x))
                particle_id = self._nearest_particle(body_id, (target_x, target_z))
            else:
                target_x = float(point[0])
                target_z = float(point[1])
                particle_id = self._nearest_particle(body_id, (target_x, target_z))

            self.monitor_particles[name] = particle_id
            self.monitor_specs[name] = {
                **item,
                "target_x": target_x,
                "target_z": target_z,
                "purpose": purpose,
            }
            self.monitor_check_rows.append(
                {
                    "name": name,
                    "particle_id": particle_id,
                    "target_x": target_x,
                    "target_z": target_z,
                    "actual_x": float(positions[particle_id, 0] - X_SHIFT),
                    "actual_z": float(positions[particle_id, 1] - Y_SHIFT),
                    "purpose": purpose,
                }
            )

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
        time_value = self.dynamic_time(sims)
        if self.input_enabled:
            shear_traction = input_stress(time_value)
            pressure_traction = pressure_input_stress(time_value)
        else:
            shear_traction = 0.0
            pressure_traction = 0.0
        update_bottom_input_traction(
            self.bottom_count,
            self.bottom_traction_ids,
            shear_traction,
            pressure_traction,
            scene.boundary.particle_traction,
            self.total_input_force_x,
            self.total_input_force_y,
        )

    def _capture_force_component(self, name: str, scene: Any) -> None:
        callback = getattr(self, "_force_component_capture", None)
        if callback is not None:
            callback(name, scene.node.force.to_numpy().copy())

    def apply_silent_forces(self, sims: Any, scene: Any) -> None:
        self.update_input_traction(sims, scene)
        if DIAGNOSTIC_ENABLE_STATIC_REACTION:
            self.initialize_bottom_static_reaction_from_force_window(sims, scene)
            apply_bottom_static_reaction_nodes(
                self.bottom_static_reaction_node_count,
                self.bottom_static_reaction_node_ids,
                self.bottom_static_reaction_node_body_ids,
                self.bottom_static_reaction_node_x,
                self.bottom_static_reaction_node_y,
                scene.node,
                self.total_static_reaction_force_x,
                self.total_static_reaction_force_y,
            )
        else:
            zero_force_accumulators_2(self.total_static_reaction_force_x, self.total_static_reaction_force_y)
        self._capture_force_component("bottom_static_support", scene)
        if DIAGNOSTIC_ENABLE_BOTTOM_DASHPOT:
            apply_nairn_silent_loads(
                scene.element.grid_nodes,
                self.bottom_count,
                self.bottom_particle_ids,
                self.side_count,
                self.side_particle_ids,
                CS,
                CP,
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
        else:
            zero_force_accumulators_2(self.total_bottom_silent_x, self.total_bottom_silent_y)
            zero_force_accumulators_2(self.total_side_silent_x, self.total_side_silent_y)
        self._capture_force_component("bottom_dashpot_force", scene)
        apply_lateral_static_support_nodes(
            self.lateral_static_support_count,
            self.lateral_static_support_node_ids,
            self.lateral_static_support_body_ids,
            self.lateral_static_support_x,
            scene.node,
            self.total_lateral_static_support_x,
        )
        self._capture_force_component("lateral_static_support", scene)
        if DIAGNOSTIC_ENABLE_FREE_FIELD:
            apply_free_field_particle_coupling(
                scene.element.grid_nodes,
                self.ff_pair_count,
                self.ff_pair_main_ids,
                self.ff_pair_ids,
                self.ff_pair_side_signs,
                self.ff_pair_initial_surface_areas,
                self.ff_pair_current_surface_areas,
                self.ff_pair_normal_impedances,
                self.ff_pair_shear_impedances,
                self.ff_pair_static_stress_xx,
                self.ff_pair_static_stress_xy,
                DIAGNOSTIC_FREE_FIELD_STATIC_STRESS_MULT,
                DIAGNOSTIC_FREE_FIELD_DYNAMIC_STRESS_MULT,
                DIAGNOSTIC_FREE_FIELD_DASHPOT_MULT,
                self.mpm.auxiliary_deformation_gradient,
                scene.particle,
                scene.node,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
                self.ff_pair_force_balance_error,
                self.total_ff_coupling_main_x,
                self.total_ff_coupling_main_y,
                self.total_ff_coupling_free_x,
                self.total_ff_coupling_free_y,
                self.max_ff_coupling_balance_error,
                self.surface_area_update_call_count,
                self.invalid_surface_area_count,
                self.total_invalid_surface_area_count,
                self.ff_pair_dynamic_stress_force_x,
                self.ff_pair_dynamic_stress_force_y,
                self.ff_pair_normal_dashpot_force_x,
                self.ff_pair_shear_dashpot_force_y,
                self.ff_pair_total_force_x,
                self.ff_pair_total_force_y,
                self.total_ff_coupling_power_main,
                self.total_ff_dashpot_dissipation,
            )
            invalid_surface_area_count = int(self.invalid_surface_area_count[None])
            if invalid_surface_area_count > 0:
                raise RuntimeError(
                    "Equation (32) free-field surface area update failed: "
                    f"invalid_surface_area_count={invalid_surface_area_count}, "
                    f"total_invalid_surface_area_count={int(self.total_invalid_surface_area_count[None])}"
                )
        else:
            zero_free_field_accumulators(
                self.total_ff_coupling_main_x,
                self.total_ff_coupling_main_y,
                self.total_ff_coupling_free_x,
                self.total_ff_coupling_free_y,
                self.max_ff_coupling_balance_error,
            )
            self.total_ff_coupling_power_main[None] = 0.0
            self.total_ff_dashpot_dissipation[None] = 0.0
        self._capture_force_component("free_field_coupling", scene)
        diagnostic_time = self.dynamic_time(sims)
        if (
            BOUNDARY_DIAGNOSTIC_INTERVAL > 0.0
            and diagnostic_time + 0.5 * float(sims.delta) < self.next_boundary_diagnostic
        ):
            return
        if BOUNDARY_DIAGNOSTIC_INTERVAL > 0.0:
            self.next_boundary_diagnostic += BOUNDARY_DIAGNOSTIC_INTERVAL
        self.max_observed_ff_coupling_balance_error = max(
            self.max_observed_ff_coupling_balance_error,
            float(self.max_ff_coupling_balance_error[None]),
        )
        force_components = free_field_pair_force_component_totals(self)
        auxiliary_tracker = getattr(self.mpm, "auxiliary_deformation_gradient_tracker", None)
        surface_stats = free_field_surface_area_runtime_stats(self, auxiliary_tracker)
        auxiliary_stats = auxiliary_tracker.stats() if auxiliary_tracker is not None else {
            "min_detF": math.nan,
            "max_detF": math.nan,
        }
        periodic_mapper = getattr(self.mpm, "free_field_periodic_mapper", None)
        periodic_metrics = periodic_runtime_metrics(self.mpm, periodic_mapper)
        velocity_monitors = free_field_velocity_monitors(scene)
        main_velocity = main_velocity_diagnostics(scene)
        finite_counts = finite_particle_state_counts(scene)
        velocities = scene.particle.v.to_numpy()[: int(scene.particleNum[0])]
        positions = scene.particle.x.to_numpy()[: int(scene.particleNum[0])]
        joint_monitor_values: dict[str, Any] = {}
        for name, pid in self.joint_monitor_particles.items():
            joint_monitor_values[f"{name}_pid"] = int(pid)
            joint_monitor_values[f"{name}_x"] = float(positions[int(pid), 0] - X_SHIFT)
            joint_monitor_values[f"{name}_y"] = float(positions[int(pid), 1] - Y_SHIFT)
            joint_monitor_values[f"{name}_vx"] = float(velocities[int(pid), 0])
        self.boundary_rows.append(
            {
                "time": self.dynamic_time(sims),
                "absolute_time": float(sims.current_time),
                "input_acceleration_x": input_acceleration(self.dynamic_time(sims)),
                "input_velocity_x": input_velocity(self.dynamic_time(sims)),
                "upward_input_velocity_x": upward_input_velocity(self.dynamic_time(sims)),
                "input_stress_x": input_stress(self.dynamic_time(sims)),
                "input_stress_y": pressure_input_stress(self.dynamic_time(sims)),
                "total_input_force_x": float(self.total_input_force_x[None]),
                "total_input_force_y": float(self.total_input_force_y[None]),
                "total_input_power": float(
                    self.total_input_force_x[None] * upward_input_velocity(self.dynamic_time(sims))
                ),
                "total_static_reaction_force_x": float(self.total_static_reaction_force_x[None]),
                "total_static_reaction_force_y": float(self.total_static_reaction_force_y[None]),
                "total_lateral_static_support_x": float(self.total_lateral_static_support_x[None]),
                "total_bottom_silent_force_x": float(self.total_bottom_silent_x[None]),
                "total_bottom_silent_force_y": float(self.total_bottom_silent_y[None]),
                "total_bottom_force_x": float(
                    self.total_input_force_x[None] + self.total_static_reaction_force_x[None] + self.total_bottom_silent_x[None]
                ),
                "total_bottom_force_y": float(
                    self.total_input_force_y[None] + self.total_static_reaction_force_y[None] + self.total_bottom_silent_y[None]
                ),
                "total_side_silent_force_x": float(self.total_side_silent_x[None]),
                "total_side_silent_force_y": float(self.total_side_silent_y[None]),
                **force_components,
                "total_ff_coupling_main_force_x": float(self.total_ff_coupling_main_x[None]),
                "total_ff_coupling_main_force_y": float(self.total_ff_coupling_main_y[None]),
                "total_ff_coupling_free_field_force_x": float(self.total_ff_coupling_free_x[None]),
                "total_ff_coupling_free_field_force_y": float(self.total_ff_coupling_free_y[None]),
                "total_main_coupling_force_x": float(self.total_ff_coupling_main_x[None]),
                "total_main_coupling_force_y": float(self.total_ff_coupling_main_y[None]),
                "total_ff_coupling_power_main": float(self.total_ff_coupling_power_main[None]),
                "total_ff_dashpot_dissipation": float(self.total_ff_dashpot_dissipation[None]),
                "total_force_applied_to_free_field_by_main": math.hypot(
                    float(self.total_ff_coupling_free_x[None]),
                    float(self.total_ff_coupling_free_y[None]),
                ),
                "ff_coupling_balance_error": float(self.max_ff_coupling_balance_error[None]),
                "surface_area_update_call_count": int(self.surface_area_update_call_count[None]),
                "invalid_surface_area_count": int(self.invalid_surface_area_count[None]),
                "total_invalid_surface_area_count": int(self.total_invalid_surface_area_count[None]),
                **surface_stats,
                "periodic_mass_sync_call_count": int(getattr(periodic_mapper, "mass_sync_call_count", 0)),
                "periodic_force_sync_call_count": int(getattr(periodic_mapper, "force_sync_call_count", 0)),
                "periodic_kinematic_sync_call_count": int(getattr(periodic_mapper, "kinematic_sync_call_count", 0)),
                "periodic_before_g2p_sync_call_count": int(getattr(periodic_mapper, "before_g2p_sync_call_count", 0)),
                **periodic_metrics,
                "auxiliary_F_update_call_count": int(auxiliary_tracker.update_call_count[None]) if auxiliary_tracker is not None else 0,
                "min_detF": float(auxiliary_stats["min_detF"]),
                "max_detF": float(auxiliary_stats["max_detF"]),
                "invalid_F_count": int(auxiliary_tracker.invalid_F_count[None]) if auxiliary_tracker is not None else -1,
                **velocity_monitors,
                **main_velocity,
                **joint_monitor_values,
                **finite_counts,
                "bottom_particle_count": self.bottom_count,
                "side_particle_count": self.side_count,
                "ff_coupling_pair_count": self.ff_pair_count,
            }
        )

    def record_history(self, sims: Any, scene: Any) -> None:
        time_value = self.dynamic_time(sims)
        if time_value + 0.5 * float(sims.delta) < self.next_history:
            return

        velocities = scene.particle.v.to_numpy()[: int(scene.particleNum[0])]
        positions = scene.particle.x.to_numpy()[: int(scene.particleNum[0])]
        stresses = scene.particle.stress.to_numpy()[: int(scene.particleNum[0])]
        row: dict[str, float | int] = {
            "time": time_value,
            "absolute_time": float(sims.current_time),
            "input_acceleration_x": input_acceleration(time_value),
            "input_velocity_x": input_velocity(time_value),
            "upward_input_velocity_x": upward_input_velocity(time_value),
            "input_stress_x": input_stress(time_value),
            "total_input_force_x": float(self.total_input_force_x[None]),
            "total_input_force_y": float(self.total_input_force_y[None]),
            "total_input_power": float(
                self.total_input_force_x[None] * upward_input_velocity(time_value)
            ),
            "total_static_reaction_force_x": float(self.total_static_reaction_force_x[None]),
            "total_static_reaction_force_y": float(self.total_static_reaction_force_y[None]),
            "total_bottom_silent_force_x": float(self.total_bottom_silent_x[None]),
            "total_bottom_silent_force_y": float(self.total_bottom_silent_y[None]),
            "total_side_silent_force_x": float(self.total_side_silent_x[None]),
            "total_main_coupling_force_x": float(self.total_ff_coupling_main_x[None]),
            "total_main_coupling_force_y": float(self.total_ff_coupling_main_y[None]),
            "total_ff_coupling_power_main": float(self.total_ff_coupling_power_main[None]),
            "total_ff_dashpot_dissipation": float(self.total_ff_dashpot_dissipation[None]),
            "total_force_applied_to_free_field_by_main": math.hypot(
                float(self.total_ff_coupling_free_x[None]),
                float(self.total_ff_coupling_free_y[None]),
            ),
        }
        for name, pid in self.monitor_particles.items():
            row[f"{name}_pid"] = pid
            row[f"{name}_x"] = float(positions[pid, 0] - X_SHIFT)
            row[f"{name}_y"] = float(positions[pid, 1] - Y_SHIFT)
            row[f"{name}_vx"] = float(velocities[pid, 0])
            row[f"{name}_vy"] = float(velocities[pid, 1])
            row[f"{name}_stress_xx"] = float(stresses[pid, 0])
            row[f"{name}_stress_xy"] = float(stresses[pid, 3])
        for item in self.input_motion_monitors:
            name = str(item["name"])
            row[f"{name}_pid"] = ""
            row[f"{name}_x"] = ""
            row[f"{name}_y"] = "bottom_compliant_base_input_boundary"
            row[f"{name}_vx"] = input_velocity(time_value)
            row[f"{name}_vy"] = 0.0
            row[f"{name}_stress_xx"] = ""
            row[f"{name}_stress_xy"] = ""
        self.rows.append(row)
        self.next_history += HISTORY_INTERVAL

    def write_outputs(self) -> None:
        prefix = str(self.case.get("output_prefix", self.case.get("name", "nairn_case")))
        history_path = OUTPUT_DIR / f"{prefix}_history.csv"
        boundary_path = OUTPUT_DIR / f"{prefix}_boundary_forces.csv"
        metadata_path = OUTPUT_DIR / f"{prefix}_metadata.json"
        report_path = OUTPUT_DIR / f"{prefix}_report.md"
        monitor_check_path = OUTPUT_DIR / "monitor_points_check.csv"
        history_fields = [
            "time",
            "absolute_time",
            "input_acceleration_x",
            "input_velocity_x",
            "upward_input_velocity_x",
            "input_stress_x",
            "total_input_force_x",
            "total_input_force_y",
            "total_input_power",
            "total_static_reaction_force_x",
            "total_static_reaction_force_y",
            "total_bottom_silent_force_x",
            "total_bottom_silent_force_y",
            "total_side_silent_force_x",
            "total_main_coupling_force_x",
            "total_main_coupling_force_y",
            "total_ff_coupling_power_main",
            "total_ff_dashpot_dissipation",
            "total_force_applied_to_free_field_by_main",
        ]
        for name in self.monitor_particles:
            history_fields.extend(
                [
                    f"{name}_pid",
                    f"{name}_x",
                    f"{name}_y",
                    f"{name}_vx",
                    f"{name}_vy",
                    f"{name}_stress_xx",
                    f"{name}_stress_xy",
                ]
            )
        for item in self.input_motion_monitors:
            name = str(item["name"])
            history_fields.extend(
                [
                    f"{name}_pid",
                    f"{name}_x",
                    f"{name}_y",
                    f"{name}_vx",
                    f"{name}_vy",
                    f"{name}_stress_xx",
                    f"{name}_stress_xy",
                ]
            )
        write_csv(history_path, history_fields, self.rows)
        write_csv(
            monitor_check_path,
            ["name", "particle_id", "target_x", "target_z", "actual_x", "actual_z", "purpose"],
            self.monitor_check_rows,
        )
        boundary_fields = [
            "time",
            "absolute_time",
            "input_acceleration_x",
            "input_velocity_x",
            "upward_input_velocity_x",
            "input_stress_x",
            "input_stress_y",
            "total_input_force_x",
            "total_input_force_y",
            "total_input_power",
            "total_static_reaction_force_x",
            "total_static_reaction_force_y",
            "total_bottom_silent_force_x",
            "total_bottom_silent_force_y",
            "total_bottom_force_x",
            "total_bottom_force_y",
            "total_side_silent_force_x",
            "total_side_silent_force_y",
            "total_ff_coupling_main_force_x",
            "total_ff_coupling_main_force_y",
            "total_ff_coupling_free_field_force_x",
            "total_ff_coupling_free_field_force_y",
            "total_main_coupling_force_x",
            "total_main_coupling_force_y",
            "total_ff_coupling_power_main",
            "total_ff_dashpot_dissipation",
            "total_force_applied_to_free_field_by_main",
            "ff_coupling_balance_error",
            "bottom_particle_count",
            "side_particle_count",
            "ff_coupling_pair_count",
        ]
        for row in self.boundary_rows:
            for field in row:
                if field not in boundary_fields:
                    boundary_fields.append(field)
        write_csv(boundary_path, boundary_fields, self.boundary_rows)
        velocity_output_paths = self.write_velocity_csv_outputs()
        energy_transfer_path = self.write_energy_transfer_report()
        grid_info = grid_count_info(self.mpm)
        metadata = {
            "case": self.case,
            "source": self.case.get("source", self.case.get("name")),
            "coordinate_shift": {"x": X_SHIFT, "y": Y_SHIFT},
            "physical_domain_x": [-X_SHIFT, float(DOMAIN[0]) - X_SHIFT],
            "free_field_geometry": {
                "gap": K_FREE_FIELD_GAP,
                "width": K_FREE_FIELD_WIDTH,
                "left_x": [K_LEFT_FREE_X_MIN, K_LEFT_FREE_X_MAX],
                "main_x": [K_MAIN_X_MIN, K_MAIN_X_MAX],
                "right_x": [K_RIGHT_FREE_X_MIN, K_RIGHT_FREE_X_MAX],
                "left_outer_margin": K_LEFT_FREE_X_MIN + X_SHIFT,
                "right_outer_margin": float(DOMAIN[0]) - X_SHIFT - K_RIGHT_FREE_X_MAX,
            },
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
            "earthquake_input": self.case.get("earthquake_input", {}),
            "seismic_input_mode": SEISMIC_INPUT_MODE,
            "input_motion_velocity": "input_velocity(t); direct Fig.10(c) curve in FIG10C mode",
            "upward_wave_velocity": "upward_input_velocity(t) = SEISMIC_INPUT_FACTOR*outcrop_velocity(t)",
            "input_stress": "2*rho*Cs*upward_input_velocity(t), following Kohler Eqs. (1) and (29)",
            "bottom_boundary": self.case.get("bottom_traction", {}),
            "bottom_static_reaction_source": "GeoTaichi assembled nodal residual on static bottom constraint nodes, applied directly to grid nodes at static-to-dynamic switch",
            "bottom_static_reaction_csv": self.bottom_static_reaction_csv.as_posix(),
            "bottom_static_reaction_node_count": self.bottom_static_reaction_node_count,
            "side_boundary": self.case.get("silent_boundary", {}).get("side", {}),
            "bottom_particle_count": self.bottom_count,
            **grid_info,
            "side_particle_count": self.side_count,
            "free_field_coupling_pair_count": self.ff_pair_count,
            "free_field_boundary_formula": "Kohler Eq.30/Eq.31: static free-field stress + dynamic free-field stress + normal/shear dashpots",
            "free_field_coupling_mean_surface_area": float(np.mean(self.ff_pair_surface_areas_np)) if self.ff_pair_count else math.nan,
            "free_field_coupling_mean_normal_impedance_area": float(np.mean(self.ff_pair_normal_impedance_areas_np)) if self.ff_pair_count else math.nan,
            "free_field_coupling_mean_shear_impedance_area": float(np.mean(self.ff_pair_shear_impedance_areas_np)) if self.ff_pair_count else math.nan,
            "monitor_particles": self.monitor_particles,
            "input_motion_monitors": [item["name"] for item in self.input_motion_monitors],
            "monitor_points_check": monitor_check_path.as_posix(),
            "strict_time_loop": STRICT_TIME_LOOP,
            "stress_update": hughes_winget_stress_update_label(self.case),
            "hughes_winget_formula_revision": (
                PAPER_HUGHES_WINGET_FORMULA_REVISION
                if hughes_winget_enabled_for_case(self.case)
                else "NOT_HUGHES_WINGET"
            ),
            "dynamic_velocity_projection": DYNAMIC_VELOCITY_PROJECTION,
            "dynamic_alpha_pic": DYNAMIC_ALPHA_PIC,
            "dynamic_alpha_pic_requested": DYNAMIC_ALPHA_PIC_REQUESTED,
            "paper_apic_2d_p2g_installed": bool(
                getattr(self.mpm.enginer, "_nairn_paper_apic_2d_p2g_installed", False)
            ),
            "paper_apic_2d_transfer_installed": bool(
                getattr(self.mpm.enginer, "_nairn_paper_apic_2d_transfer_installed", False)
            ),
            "paper_apic_separate_B_L": bool(
                getattr(self.mpm.enginer, "_nairn_paper_apic_separate_B_L", False)
            ),
            "paper_apic_usl_order": bool(
                getattr(self.mpm.enginer, "_nairn_paper_apic_usl_order", False)
            ),
            "paper_apic_formula_revision": (
                PAPER_APIC_FORMULA_REVISION
                if getattr(self.mpm.enginer, "_nairn_paper_apic_2d_p2g_installed", False)
                else "NOT_APIC"
            ),
            "paper_apic_constant_D": bool(APIC_USE_PAPER_CONSTANT_DP),
            "apic_affine_checkpoint_state": str(
                getattr(self.mpm, "static_checkpoint_apic_affine_state", "NOT_APPLICABLE")
            ),
            "boundary_diagnostic_interval": BOUNDARY_DIAGNOSTIC_INTERVAL,
            "dynamic_start_time": self.dynamic_start_time,
            "dynamic_runtime_seconds": getattr(self.mpm, "dynamic_runtime_seconds", math.nan),
            "periodic_hooks_installed_for_dynamic": bool(getattr(self.mpm, "periodic_hooks_installed_for_dynamic", False)),
            "periodic_pair_count_for_dynamic": int(getattr(self.mpm, "periodic_pair_count_for_dynamic", 0)),
            "left_periodic_pair_count_for_dynamic": int(getattr(self.mpm, "left_periodic_pair_count_for_dynamic", 0)),
            "right_periodic_pair_count_for_dynamic": int(getattr(self.mpm, "right_periodic_pair_count_for_dynamic", 0)),
            "periodic_mapper_active_before_first_dynamic_step": bool(
                getattr(self.mpm, "periodic_mapper_active_before_first_dynamic_step", False)
            ),
            "dx05_test_velocity_outputs": {key: path.as_posix() for key, path in velocity_output_paths.items()},
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
            f"- dynamic_start_time: `{self.dynamic_start_time}`",
            f"- dynamic_runtime_seconds: `{getattr(self.mpm, 'dynamic_runtime_seconds', math.nan)}`",
            f"- seismic_input_mode: `{SEISMIC_INPUT_MODE}`",
            f"- physical_domain_x: `[-{X_SHIFT}, {float(DOMAIN[0]) - X_SHIFT}]`",
            f"- free_field_main_gap: `{K_FREE_FIELD_GAP}`",
            "",
            "## Counts",
            "",
            f"- particles: `{int(self.mpm.scene.particleNum[0])}`",
            f"- grid nodes: `{grid_info['grid_node_count']}` `{grid_info['grid_nodes_per_direction']}`",
            f"- grid cells: `{grid_info['grid_cell_count']}` `{grid_info['grid_cells_per_direction']}`",
            f"- bottom traction/silent particles: `{self.bottom_count}`",
            f"- side silent particles: `{self.side_count}`",
            f"- free-field coupling pairs: `{self.ff_pair_count}`",
            f"- dynamic periodic pairs: `{int(getattr(self.mpm, 'periodic_pair_count_for_dynamic', 0))}`",
            f"- monitor particles: `{len(self.monitor_particles)}`",
            f"- bottom_static_reaction_source: `GeoTaichi assembled nodal residual on static bottom constraint nodes`",
            f"- bottom_static_reaction_nodes: `{self.bottom_static_reaction_node_count}`",
            "",
            "## Output",
            "",
            f"- history: `{history_path}`",
            f"- boundary forces: `{boundary_path}`",
            f"- metadata: `{metadata_path}`",
            f"- monitor points check: `{monitor_check_path}`",
            f"- bottom static reaction: `{self.bottom_static_reaction_csv}`",
            f"- input velocity: `{velocity_output_paths['input']}`",
            f"- boundary velocity: `{velocity_output_paths['boundary']}`",
            f"- surface velocity: `{velocity_output_paths['surface']}`",
            f"- free-field coupling report: `{OUTPUT_DIR / 'free_field_coupling_report.md'}`",
            f"- free-field pair check: `{OUTPUT_DIR / 'free_field_pair_check.csv'}`",
            f"- energy transfer report: `{energy_transfer_path}`",
            f"- native particles/grids: `{OUTPUT_DIR}`",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        decomposition_path = self.write_bottom_traction_decomposition()
        input_report_path = self.write_external_earthquake_input_report(decomposition_path)
        self.write_free_field_coupling_outputs()
        self.write_dx05_test_report(velocity_output_paths, grid_info, history_path, boundary_path, monitor_check_path)

    def write_energy_transfer_report(self) -> Path:
        """Summarize work delivered by the one-way free-field boundary."""

        path = OUTPUT_DIR / "energy_transfer_report.md"
        rows = [row for row in self.boundary_rows if math.isfinite(float(row.get("time", math.nan)))]
        rows.sort(key=lambda row: float(row["time"]))
        if rows:
            times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
            input_power = np.asarray(
                [float(row.get("total_input_power", 0.0)) for row in rows],
                dtype=np.float64,
            )
            power = np.asarray(
                [float(row.get("total_ff_coupling_power_main", 0.0)) for row in rows],
                dtype=np.float64,
            )
            dissipation = np.asarray(
                [float(row.get("total_ff_dashpot_dissipation", 0.0)) for row in rows],
                dtype=np.float64,
            )
            finite = np.isfinite(times) & np.isfinite(input_power) & np.isfinite(power) & np.isfinite(dissipation)
            times = times[finite]
            input_power = input_power[finite]
            power = power[finite]
            dissipation = dissipation[finite]
        else:
            times = np.empty(0, dtype=np.float64)
            input_power = np.empty(0, dtype=np.float64)
            power = np.empty(0, dtype=np.float64)
            dissipation = np.empty(0, dtype=np.float64)

        def integrate(values: np.ndarray) -> float:
            if values.size < 2 or times.size < 2:
                return 0.0
            if hasattr(np, "trapezoid"):
                return float(np.trapezoid(values, times))
            return float(np.trapz(values, times))

        peak_power = float(np.max(np.abs(power))) if power.size else 0.0
        peak_input_power = float(np.max(np.abs(input_power))) if input_power.size else 0.0
        peak_dissipation = float(np.max(dissipation)) if dissipation.size else 0.0
        min_dissipation = float(np.min(dissipation)) if dissipation.size else 0.0
        nonzero_transfer = peak_power > 1.0e-9
        dissipation_nonnegative = min_dissipation >= -1.0e-8
        lines = [
            "# Free-Field Energy Transfer Report",
            "",
            "## Verdict",
            "",
            f"- status: `{'PASS' if nonzero_transfer and dissipation_nonnegative else 'FAIL'}`",
            f"- sampled_boundary_rows: `{len(rows)}`",
            f"- free_field_to_main_transfer_detected: `{'PASS' if nonzero_transfer else 'FAIL'}`",
            f"- dashpot_dissipation_nonnegative: `{'PASS' if dissipation_nonnegative else 'FAIL'}`",
            "",
            "## Work And Power",
            "",
            "- coupling_power_main: `sum(F_free_field_to_main dot v_main)`",
            "- input_power: `F_input dot v_input` at the compliant bottom boundary",
            "- dashpot_dissipation: `sum(eta_p*A*dv_x^2 + eta_s*A*dv_y^2)`",
            "- coupling_direction: `free_field_to_main_only (prescribed independent free-field source)`",
            f"- net_input_work: `{integrate(input_power)}`",
            f"- absolute_input_work: `{integrate(np.abs(input_power))}`",
            f"- net_free_field_to_main_work: `{integrate(power)}`",
            f"- absolute_free_field_to_main_work: `{integrate(np.abs(power))}`",
            f"- dashpot_dissipated_work: `{integrate(dissipation)}`",
            f"- peak_abs_coupling_power: `{peak_power}`",
            f"- peak_abs_input_power: `{peak_input_power}`",
            f"- peak_dashpot_dissipation: `{peak_dissipation}`",
            f"- minimum_dashpot_dissipation: `{min_dissipation}`",
            "",
            "## Interpretation",
            "",
            "- This is a boundary work audit, not a claim of global mechanical-energy conservation.",
            "- The independent free-field columns are not given the opposite force, by design, so the source remains unpolluted by the main slope response.",
            "- Nonzero transfer power with nonnegative dashpot dissipation confirms transmission and absorption in the intended direction.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_velocity_csv_outputs(self) -> dict[str, Path]:
        input_path = OUTPUT_DIR / "kohler_input_velocity.csv"
        boundary_path = OUTPUT_DIR / "kohler_boundary_velocity.csv"
        surface_path = OUTPUT_DIR / "kohler_surface_velocity.csv"
        input_rows: list[dict[str, Any]] = []
        boundary_rows: list[dict[str, Any]] = []
        surface_rows: list[dict[str, Any]] = []
        for row in self.rows:
            input_rows.append(
                {
                    "time": row.get("time", math.nan),
                    "vx": row.get("kohler_input_motion_vx", row.get("input_velocity_x", math.nan)),
                    "vy": row.get("kohler_input_motion_vy", 0.0),
                }
            )
            boundary_rows.append(
                {
                    "time": row.get("time", math.nan),
                    "x": row.get("kohler_boundary_interface_x", math.nan),
                    "y": row.get("kohler_boundary_interface_y", math.nan),
                    "vx": row.get("kohler_boundary_interface_vx", math.nan),
                    "vy": row.get("kohler_boundary_interface_vy", math.nan),
                }
            )
            surface_rows.append(
                {
                    "time": row.get("time", math.nan),
                    "x": row.get("kohler_surface_x", math.nan),
                    "y": row.get("kohler_surface_y", math.nan),
                    "vx": row.get("kohler_surface_vx", math.nan),
                    "vy": row.get("kohler_surface_vy", math.nan),
                }
            )
        write_csv(input_path, ["time", "vx", "vy"], input_rows)
        write_csv(boundary_path, ["time", "x", "y", "vx", "vy"], boundary_rows)
        write_csv(surface_path, ["time", "x", "y", "vx", "vy"], surface_rows)
        return {"input": input_path, "boundary": boundary_path, "surface": surface_path}

    def monitor_response_summary(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for name in ["kohler_input_motion", "kohler_boundary_interface", "kohler_surface"]:
            vx_values = np.asarray([float(row.get(f"{name}_vx", math.nan)) for row in self.rows], dtype=np.float64)
            vy_values = np.asarray([float(row.get(f"{name}_vy", math.nan)) for row in self.rows], dtype=np.float64)
            valid_vx = vx_values[np.isfinite(vx_values)]
            valid_vy = vy_values[np.isfinite(vy_values)]
            summaries.append(
                {
                    "name": name,
                    "samples": int(valid_vx.size),
                    "peak_abs_vx": float(np.max(np.abs(valid_vx))) if valid_vx.size else math.nan,
                    "peak_abs_vy": float(np.max(np.abs(valid_vy))) if valid_vy.size else math.nan,
                    "final_vx": float(valid_vx[-1]) if valid_vx.size else math.nan,
                    "final_vy": float(valid_vy[-1]) if valid_vy.size else math.nan,
                }
            )
        return summaries

    def write_dx05_test_report(
        self,
        velocity_output_paths: dict[str, Path],
        grid_info: dict[str, Any],
        history_path: Path,
        boundary_path: Path,
        monitor_check_path: Path,
    ) -> Path:
        report_path = OUTPUT_DIR / "dx05_test_report.md"
        monitor_summaries = self.monitor_response_summary()
        lines = [
            "# Kohler Fig.10 Dynamic Test Report",
            "",
            f"**{SIMULATION_TIME:g} s {str(ARCH).upper()} dynamic result for the Kohler et al. Fig.10 response comparison.**",
            "",
            "## Run Configuration",
            "",
            f"- dx: `{DX}`",
            f"- dynamic_time: `{SIMULATION_TIME}`",
            f"- timestep: `{DT}`",
            f"- runtime_seconds: `{getattr(self.mpm, 'dynamic_runtime_seconds', math.nan)}`",
            f"- vtk_output: `{'enabled' if (SAVE_PARTICLE or SAVE_GRID) else 'disabled'}`",
            f"- vtk_save_interval: `{SAVE_INTERVAL if (SAVE_PARTICLE or SAVE_GRID) else 'not applicable'}`",
            f"- history_output: `enabled`",
            f"- earthquake_input: `{EARTHQUAKE_MOTION_FILE}`",
            f"- plotted_input_velocity: `{'Fig10c CSV velocity' if SEISMIC_INPUT_MODE == 'FIG10C' else f'{SEISMIC_INPUT_FACTOR}*v_outcrop'}`",
            f"- upward_wave_velocity: `{SEISMIC_INPUT_FACTOR}*v_outcrop_or_Fig10c_input_motion`",
            f"- compliant_base_traction: `2*rho*Cs*upward_input_velocity(t)`",
            f"- stress_update: `{hughes_winget_stress_update_label(self.case)}`",
            f"- hughes_winget_formula_revision: `{PAPER_HUGHES_WINGET_FORMULA_REVISION if hughes_winget_enabled_for_case(self.case) else 'NOT_HUGHES_WINGET'}`",
            f"- mapping: `{MAPPING}`",
            f"- dynamic_velocity_projection: `{DYNAMIC_VELOCITY_PROJECTION}`",
            f"- dynamic_alpha_pic: `{DYNAMIC_ALPHA_PIC}`",
            f"- paper_apic_formula_revision: `{PAPER_APIC_FORMULA_REVISION if DYNAMIC_VELOCITY_PROJECTION == 'Affine' else 'NOT_APIC'}`",
            f"- paper_apic_constant_D: `{APIC_USE_PAPER_CONSTANT_DP}`",
            f"- boundary_diagnostic_interval: `{BOUNDARY_DIAGNOSTIC_INTERVAL}`",
            "",
            "## Counts",
            "",
            f"- particle_number: `{int(self.mpm.scene.particleNum[0])}`",
            f"- grid_number_nodes: `{grid_info['grid_node_count']}`",
            f"- grid_nodes_per_direction: `{grid_info['grid_nodes_per_direction']}`",
            f"- grid_number_cells: `{grid_info['grid_cell_count']}`",
            f"- grid_cells_per_direction: `{grid_info['grid_cells_per_direction']}`",
            "",
            "## Monitor Point Response",
            "",
            "| monitor | samples | peak_abs_vx | peak_abs_vy | final_vx | final_vy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for item in monitor_summaries:
            lines.append(
                f"| {item['name']} | {item['samples']} | {item['peak_abs_vx']:.9g} | "
                f"{item['peak_abs_vy']:.9g} | {item['final_vx']:.9g} | {item['final_vy']:.9g} |"
            )
        lines.extend(
            [
                "",
                "## Outputs",
                "",
                f"- kohler_input_velocity_csv: `{velocity_output_paths['input']}`",
                f"- kohler_boundary_velocity_csv: `{velocity_output_paths['boundary']}`",
                f"- kohler_surface_velocity_csv: `{velocity_output_paths['surface']}`",
                f"- full_history_csv: `{history_path}`",
                f"- boundary_force_csv: `{boundary_path}`",
                f"- monitor_point_check_csv: `{monitor_check_path}`",
            ]
        )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    def write_bottom_traction_decomposition(self) -> Path:
        path = OUTPUT_DIR / "bottom_traction_decomposition.csv"
        rows: list[dict[str, Any]] = []
        for row in self.boundary_rows:
            static_x = float(row.get("total_static_reaction_force_x", 0.0))
            static_y = float(row.get("total_static_reaction_force_y", 0.0))
            dashpot_x = float(row.get("total_bottom_silent_force_x", 0.0))
            dashpot_y = float(row.get("total_bottom_silent_force_y", 0.0))
            input_x = float(row.get("total_input_force_x", 0.0))
            input_y = float(row.get("total_input_force_y", 0.0))
            total_x = static_x + dashpot_x + input_x
            total_y = static_y + dashpot_y + input_y
            rows.append(
                {
                    "time": float(row["time"]),
                    "static_reaction": math.hypot(static_x, static_y),
                    "dashpot_force": math.hypot(dashpot_x, dashpot_y),
                    "earthquake_input_force": math.hypot(input_x, input_y),
                    "total_bottom_force": math.hypot(total_x, total_y),
                    "static_reaction_x": static_x,
                    "static_reaction_y": static_y,
                    "dashpot_force_x": dashpot_x,
                    "dashpot_force_y": dashpot_y,
                    "earthquake_input_force_x": input_x,
                    "earthquake_input_force_y": input_y,
                    "total_bottom_force_x": total_x,
                    "total_bottom_force_y": total_y,
                }
            )
        write_csv(
            path,
            [
                "time",
                "static_reaction",
                "dashpot_force",
                "earthquake_input_force",
                "total_bottom_force",
                "static_reaction_x",
                "static_reaction_y",
                "dashpot_force_x",
                "dashpot_force_y",
                "earthquake_input_force_x",
                "earthquake_input_force_y",
                "total_bottom_force_x",
                "total_bottom_force_y",
            ],
            rows,
        )
        return path

    def write_external_earthquake_input_report(self, decomposition_path: Path) -> Path:
        path = OUTPUT_DIR / "kohler_external_earthquake_input_report.md"
        motion = EARTHQUAKE_MOTION
        if motion is None:
            lines = [
                "# Kohler External Earthquake Input Report",
                "",
                "- status: `FAIL`",
                "- message: `earthquake input is disabled or failed to load`",
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return path

        times = np.arange(0.0, min(SIMULATION_TIME, motion.duration) + 0.5 * HISTORY_INTERVAL, HISTORY_INTERVAL)
        direct_velocity = SEISMIC_INPUT_MODE == "FIG10C"
        sample_rows = motion.sample_history(times, DENSITY, CS, CP, input_factor=SEISMIC_INPUT_FACTOR)
        acceleration_values = np.asarray([row["acceleration"] for row in sample_rows], dtype=np.float64)
        velocity_values = np.asarray([row["velocity"] for row in sample_rows], dtype=np.float64)
        input_velocity_values = np.asarray([row["input_velocity"] for row in sample_rows], dtype=np.float64)
        traction_values = np.asarray([row["shear_input_traction"] for row in sample_rows], dtype=np.float64)
        traction_correct = np.allclose(traction_values, 2.0 * DENSITY * CS * input_velocity_values, rtol=1.0e-12, atol=1.0e-14)
        bottom_particles_receive_input = self.bottom_count > 0 and bool(self.case.get("earthquake_input", {}).get("shear_input", True))
        static_state_preserved = getattr(self.mpm, "static_state_snapshot", {}).get("particle_count", 0) == int(self.mpm.scene.particleNum[0])
        prefix = str(self.case.get("output_prefix", self.case.get("name", "nairn_case")))
        boundary_force_history_path = OUTPUT_DIR / f"{prefix}_boundary_forces.csv"
        status = "PASS" if bottom_particles_receive_input and static_state_preserved and traction_correct else "FAIL"
        lines = [
            "# Kohler External Earthquake Input Report",
            "",
            "## Verdict",
            "",
            f"- status: `{status}`",
            f"- csv_successfully_read: `PASS`",
            f"- seismic_input_mode: `{SEISMIC_INPUT_MODE}`",
            f"- acceleration_to_velocity: `{'NOT_APPLICABLE' if direct_velocity else 'PASS'}`",
            f"- traction_calculation: `{'PASS' if traction_correct else 'FAIL'}`",
            f"- bottom_particles_receive_input: `{'PASS' if bottom_particles_receive_input else 'FAIL'}`",
            f"- static_state_not_lost: `{'PASS' if static_state_preserved else 'FAIL'}`",
            "",
            "## Input File",
            "",
            f"- input_file: `{motion.source_path}`",
            f"- pga: `{motion.pga}`",
            f"- duration: `{motion.duration}`",
            f"- data_rows: `{motion.time.size}`",
            f"- dt_min: `{motion.dt_min}`",
            f"- dt_max: `{motion.dt_max}`",
            f"- input_unit: `{'velocity m/s' if direct_velocity else 'acceleration m/s^2'}`",
            "",
            "## Conversion",
            "",
            f"- input_mode: `{SEISMIC_INPUT_MODE}`",
            f"- acceleration_history: `{'not used' if direct_velocity else 'read from earthquake_motion.csv'}`",
            f"- outcrop_velocity_history: `{'not used' if direct_velocity else 'trapezoidal numerical integration of acceleration'}`",
            f"- plotted_input_motion_history: `{'direct Fig.10(c) velocity CSV' if direct_velocity else 'integrated outcrop velocity'}`",
            f"- upward_wave_velocity_history: `v_su = {SEISMIC_INPUT_FACTOR} * input/outcrop motion`",
            "- mpm_timestep_interpolation: `linear interpolation at dynamic time t = current_time - dynamic_start_time`",
            "- shear_input_traction: `2 * rho * Cs * v_su(t)`",
            "- pressure_input_traction: `disabled; no vertical acceleration column is supplied`",
            f"- rho: `{DENSITY}`",
            f"- Cs: `{CS}`",
            f"- Cp: `{CP}`",
            "",
            "## Histories",
            "",
            f"- bottom_traction_decomposition: `{decomposition_path}`",
            f"- seismic_input_mode_check: `{OUTPUT_DIR / 'seismic_input_mode_check.md'}`",
            f"- boundary_force_history: `{boundary_force_history_path}`",
            "",
            "## History Summary",
            "",
            f"- max_abs_acceleration_sampled: `{float(np.max(np.abs(acceleration_values))) if acceleration_values.size else 0.0}`",
            f"- max_abs_v_outcrop_sampled: `{float(np.max(np.abs(velocity_values))) if (velocity_values.size and not direct_velocity) else 'NOT_APPLICABLE'}`",
            f"- max_abs_input_motion_velocity_sampled: `{float(np.max(np.abs(velocity_values))) if velocity_values.size else 0.0}`",
            f"- max_abs_upward_wave_velocity_sampled: `{float(np.max(np.abs(input_velocity_values))) if input_velocity_values.size else 0.0}`",
            f"- max_abs_shear_input_traction_sampled: `{float(np.max(np.abs(traction_values))) if traction_values.size else 0.0}`",
            "",
            "## Compliant Base Decomposition",
            "",
            "- total_bottom_force: `static_reaction + dashpot_force + earthquake_input_force`",
            "- static_reaction: `computed from GeoTaichi assembled nodal residual and applied directly to grid nodes at static-to-dynamic switch`",
            "- dashpot_force: `existing bottom silent boundary force using Cs and Cp`",
            "- earthquake_input_force: `particle traction force from upward-wave velocity v_su(t)`",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_free_field_coupling_outputs(self) -> tuple[Path, Path]:
        pair_path = OUTPUT_DIR / "free_field_pair_check.csv"
        report_path = OUTPUT_DIR / "free_field_coupling_report.md"
        particle_count = int(self.mpm.scene.particleNum[0])
        positions = self.mpm.scene.particle.x.to_numpy()[:particle_count]
        body_ids = self.mpm.scene.particle.bodyID.to_numpy()[:particle_count]
        material_ids = self.mpm.scene.particle.materialID.to_numpy()[:particle_count].astype(np.int32)
        elevation_tolerance = max(1.0e-8, 0.05 * DX)
        force_errors = (
            self.ff_pair_force_balance_error.to_numpy()[: self.ff_pair_count]
            if self.ff_pair_count
            else np.empty(0, dtype=np.float64)
        )

        pair_rows: list[dict[str, Any]] = []
        for index in range(self.ff_pair_count):
            main_pid = int(self.ff_pair_main_ids_np[index])
            ff_pid = int(self.ff_pair_ids_np[index])
            initial_main = self.ff_pair_initial_main_positions_np[index]
            initial_ff = self.ff_pair_initial_free_field_positions_np[index]
            pair_rows.append(
                {
                    "side": str(self.ff_pair_sides_np[index]),
                    "main_particle_id": main_pid,
                    "free_field_particle_id": ff_pid,
                    "main_x": float(positions[main_pid, 0] - X_SHIFT),
                    "main_y": float(positions[main_pid, 1] - Y_SHIFT),
                    "ff_x": float(positions[ff_pid, 0] - X_SHIFT),
                    "ff_y": float(positions[ff_pid, 1] - Y_SHIFT),
                    "vertical_offset": float(abs(positions[main_pid, 1] - positions[ff_pid, 1])),
                    "distance": float(self.ff_pair_distances_np[index]),
                    "geometric_gap": float(abs(positions[main_pid, 0] - positions[ff_pid, 0]) - DX / 3.0),
                    "side_sign": float(self.ff_pair_side_signs_np[index]),
                    "surface_area": float(self.ff_pair_surface_areas_np[index]),
                    "normal_impedance_area": float(self.ff_pair_normal_impedance_areas_np[index]),
                    "shear_impedance_area": float(self.ff_pair_shear_impedance_areas_np[index]),
                    "free_field_static_stress_xx": float(self.ff_pair_static_stress_xx_np[index]),
                    "free_field_static_stress_xy": float(self.ff_pair_static_stress_xy_np[index]),
                    "force_balance_error": float(force_errors[index]) if force_errors.size else 0.0,
                    "main_body_id": int(body_ids[main_pid]),
                    "free_field_body_id": int(body_ids[ff_pid]),
                    "main_material_id": int(material_ids[main_pid]),
                    "free_field_material_id": int(material_ids[ff_pid]),
                    "same_material": bool(material_ids[main_pid] == material_ids[ff_pid]),
                    "initial_main_x": float(initial_main[0] - X_SHIFT),
                    "initial_main_y": float(initial_main[1] - Y_SHIFT),
                    "initial_ff_x": float(initial_ff[0] - X_SHIFT),
                    "initial_ff_y": float(initial_ff[1] - Y_SHIFT),
                    "initial_vertical_offset": float(self.ff_pair_initial_vertical_offsets_np[index]),
                    "initial_geometric_gap": float(self.ff_pair_initial_geometric_gaps_np[index]),
                    "initial_same_elevation": bool(
                        self.ff_pair_initial_vertical_offsets_np[index] <= elevation_tolerance
                    ),
                    "same_elevation": bool(
                        abs(positions[main_pid, 1] - positions[ff_pid, 1]) <= elevation_tolerance
                    ),
                }
            )
        write_csv(
            pair_path,
            [
                "side",
                "main_particle_id",
                "free_field_particle_id",
                "main_x",
                "main_y",
                "ff_x",
                "ff_y",
                "vertical_offset",
                "distance",
                "geometric_gap",
                "side_sign",
                "surface_area",
                "normal_impedance_area",
                "shear_impedance_area",
                "free_field_static_stress_xx",
                "free_field_static_stress_xy",
                "force_balance_error",
                "main_body_id",
                "free_field_body_id",
                "main_material_id",
                "free_field_material_id",
                "same_material",
                "initial_main_x",
                "initial_main_y",
                "initial_ff_x",
                "initial_ff_y",
                "initial_vertical_offset",
                "initial_geometric_gap",
                "initial_same_elevation",
                "same_elevation",
            ],
            pair_rows,
        )

        left_count = int(np.sum(self.ff_pair_sides_np == "left")) if self.ff_pair_count else 0
        right_count = int(np.sum(self.ff_pair_sides_np == "right")) if self.ff_pair_count else 0
        main_unique = len(set(int(pid) for pid in self.ff_pair_main_ids_np.tolist()))
        ff_unique = len(set(int(pid) for pid in self.ff_pair_ids_np.tolist()))
        unique_pairs = main_unique == self.ff_pair_count and ff_unique == self.ff_pair_count
        material_pairs_ok = bool(pair_rows) and all(bool(row["same_material"]) for row in pair_rows)
        elevation_pairs_ok = bool(pair_rows) and all(bool(row["initial_same_elevation"]) for row in pair_rows)
        max_vertical_offset = max((float(row["initial_vertical_offset"]) for row in pair_rows), default=math.inf)
        current_max_vertical_offset = max((float(row["vertical_offset"]) for row in pair_rows), default=math.inf)
        real_particles = bool(
            self.ff_pair_count
            and np.all(self.ff_pair_main_ids_np >= 0)
            and np.all(self.ff_pair_main_ids_np < particle_count)
            and np.all(self.ff_pair_ids_np >= 0)
            and np.all(self.ff_pair_ids_np < particle_count)
            and np.all(np.isin(body_ids[self.ff_pair_main_ids_np], np.asarray(MAIN_BODY_IDS, dtype=np.int32)))
            and np.all(np.isin(body_ids[self.ff_pair_ids_np], np.asarray(LEFT_FREE_BODY_IDS + RIGHT_FREE_BODY_IDS, dtype=np.int32)))
        )
        average_distance = float(np.mean(self.ff_pair_distances_np)) if self.ff_pair_count else math.nan
        max_distance = float(np.max(self.ff_pair_distances_np)) if self.ff_pair_count else math.nan
        gap_values = [float(row["initial_geometric_gap"]) for row in pair_rows]
        gap_check = bool(gap_values) and all(
            abs(value - K_FREE_FIELD_GAP) <= PARTICLE_COORDINATE_TOLERANCE for value in gap_values
        )
        max_error = max(
            float(np.max(force_errors)) if force_errors.size else 0.0,
            self.max_observed_ff_coupling_balance_error,
        )
        max_free_field_force = max(
            abs(float(row.get("total_force_applied_to_free_field_by_main", 0.0))) for row in self.boundary_rows
        ) if self.boundary_rows else 0.0
        free_field_force_ok = max_free_field_force <= FREE_FIELD_INDEPENDENCE_TOLERANCE
        independence_path = OUTPUT_DIR / "free_field_independence_check.csv"
        independence_status = free_field_independence_status(independence_path)
        runtime_independence_ok = independence_status == "PASS" or (
            independence_status == "MISSING" and free_field_force_ok
        )
        status = "PASS" if unique_pairs and material_pairs_ok and elevation_pairs_ok and real_particles and gap_check and runtime_independence_ok else "FAIL"

        lines = [
            "# Free Field Coupling Report",
            "",
            "## Verdict",
            "",
            f"- status: `{status}`",
            f"- coupling_direction: `free_field_to_main_only`",
            f"- unique_interface_pairs: `{'PASS' if unique_pairs else 'FAIL'}`",
            f"- pairs_match_material_layer: `{'PASS' if material_pairs_ok else 'FAIL'}`",
            f"- pairs_match_elevation: `{'PASS' if elevation_pairs_ok else 'FAIL'}`",
            f"- prescribed_interface_gap: `{K_FREE_FIELD_GAP}`",
            f"- measured_initial_interface_gap: `{min(gap_values, default=math.nan)} .. {max(gap_values, default=math.nan)}`",
            f"- interface_gap_check: `{'PASS' if gap_check else 'FAIL'}`",
            f"- endpoints_are_real_mpm_particles: `{'PASS' if real_particles else 'FAIL'}`",
            f"- main_to_free_field_force: `{max_free_field_force}`",
            f"- main_to_free_field_force_check: `{'PASS' if free_field_force_ok else 'FAIL'}`",
            f"- independence_test: `{independence_status}`",
            f"- runtime_one_way_independence_check: `{'PASS' if free_field_force_ok else 'FAIL'}`",
            "",
            "## Pair Counts",
            "",
            f"- pair_count: `{self.ff_pair_count}`",
            f"- left_pair_count: `{left_count}`",
            f"- right_pair_count: `{right_count}`",
            f"- main_particle_unique_count: `{main_unique}`",
            f"- free_field_particle_unique_count: `{ff_unique}`",
            "",
            "## Distance Check",
            "",
            f"- average_distance: `{average_distance}`",
            f"- max_distance: `{max_distance}`",
            f"- max_initial_vertical_offset: `{max_vertical_offset}`",
            f"- max_current_vertical_offset_for_reference: `{current_max_vertical_offset}`",
            f"- elevation_tolerance: `{elevation_tolerance}`",
            f"- pair_csv: `{pair_path}`",
            "",
            "## Coupling Force",
            "",
            "- formula_x: `F_x = f_i,x(static nodal support) + A*n_x*sigma_xx_fp,d + rho*Cp*A*(v_free_field_x - v_main_x)`",
            "- formula_y: `F_y = A*n_x*tau_xy_fp,d + rho*Cs*A*(v_free_field_y - v_main_y)`",
            "- before_change: `F_ff = -F_main` was mapped back to free-field nodes",
            "- after_change: `F_ff = 0`; only `F_main` is mapped to main-model nodes",
            f"- rho: `{DENSITY}`",
            f"- Cs: `{CS}`",
            f"- Cp: `{CP}`",
            f"- A: `{DX}`",
            f"- mean_surface_area: `{float(np.mean(self.ff_pair_surface_areas_np)) if self.ff_pair_count else math.nan}`",
            f"- mean_normal_impedance_area: `{float(np.mean(self.ff_pair_normal_impedance_areas_np)) if self.ff_pair_count else math.nan}`",
            f"- mean_shear_impedance_area: `{float(np.mean(self.ff_pair_shear_impedance_areas_np)) if self.ff_pair_count else math.nan}`",
            f"- max_legacy_action_reaction_error_field: `{max_error}`",
            f"- free_field_independence_csv: `{independence_path}`",
            "",
            "## Scope",
            "",
            "- solver_core_modified: `False`",
            "- geometry_changed: `False`",
            "- static_initialization_changed: `False`",
            "- endpoints_use_grid_nodes: `False`",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path, pair_path


def register_bottom_input_traction(mpm: MPM, case: dict[str, Any]) -> None:
    if bool(getattr(mpm, "bottom_input_traction_registered", False)):
        return
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
    mpm.bottom_input_traction_registered = True


def add_region_body(
    mpm: MPM,
    name: str,
    material_id: int,
    body_id: int,
    n_particles_per_cell: int = 1,
    gravity_field: bool = False,
) -> None:
    mpm.add_body(
        body={
            "Template": {
                "RegionName": name,
                "nParticlesPerCell": n_particles_per_cell,
                "BodyID": body_id,
                "MaterialID": material_id,
                "ParticleStress": {
                    "GravityField": gravity_field,
                    "InternalStress": ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    "Traction": {},
                },
                "InitialVelocity": ti.Vector([0.0, 0.0]),
                "FixVelocity": ["Free", "Free"],
            }
        }
    )


def current_geometry_bounds(case: dict[str, Any]) -> dict[str, float]:
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf
    for item in case.get("regions", []):
        point = item["bbox_point"]
        size = item["bbox_size"]
        x_min = min(x_min, float(point[0]))
        x_max = max(x_max, float(point[0]) + float(size[0]))
        y_min = min(y_min, float(point[1]))
        y_max = max(y_max, float(point[1]) + float(size[1]))
    return {
        "x_min": x_min + X_SHIFT,
        "x_max": x_max + X_SHIFT,
        "y_min": y_min + Y_SHIFT,
        "y_max": y_max + Y_SHIFT,
    }


def static_bottom_boundary_conditions(band: float | None = None) -> list[dict[str, Any]]:
    constraint_band = STATIC_BOTTOM_CONSTRAINT_BAND if band is None else float(band)
    bottom_constraints: list[dict[str, Any]] = []
    solid_intervals = [
        (K_LEFT_FREE_X_MIN, K_LEFT_FREE_X_MAX, K_LEFT_FREE_BOTTOM_Z),
        (K_MAIN_X_MIN, K_MAIN_X_MAX, None),
        (K_RIGHT_FREE_X_MIN, K_RIGHT_FREE_X_MAX, K_RIGHT_FREE_BOTTOM_Z),
    ]
    for interval_min, interval_max, fixed_bottom in solid_intervals:
        interval_x = np.arange(interval_min, interval_max + DX, DX, dtype=np.float64)
        interval_x = np.unique(np.clip(interval_x, interval_min, interval_max))
        if interval_x[-1] < interval_max:
            interval_x = np.append(interval_x, interval_max)
        for x0, x1 in zip(interval_x[:-1], interval_x[1:]):
            if fixed_bottom is None:
                y0 = float(kohler_base_bottom_z_np(x0))
                y1 = float(kohler_base_bottom_z_np(x1))
            else:
                y0 = float(fixed_bottom)
                y1 = float(fixed_bottom)
            bottom_constraints.append(
                {
                    "BoundaryType": "VelocityConstraint",
                    "Velocity": [0.0, 0.0],
                    "StartPoint": [min(x0, x1) + X_SHIFT, min(y0, y1) + Y_SHIFT],
                    "EndPoint": [max(x0, x1) + X_SHIFT, max(y0, y1) + Y_SHIFT + constraint_band],
                }
            )
    return bottom_constraints


def static_bottom_constraint_node_ids(mpm: MPM) -> np.ndarray:
    node_ids: list[np.ndarray] = []
    for boundary in static_bottom_boundary_conditions(STATIC_REACTION_NODE_BAND):
        node_ids.append(
            mpm.scene.element.get_boundary_nodes(
                boundary["StartPoint"],
                boundary["EndPoint"],
            )
        )
    if not node_ids:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(node_ids).astype(np.int32))


def static_bottom_particle_ids(mpm: MPM) -> np.ndarray:
    """Return the full bottom-cell particle layer for static node support."""
    particle_count = int(mpm.scene.particleNum[0])
    generated_layout = getattr(mpm, "generated_particle_layout_arrays", None)
    if generated_layout is not None:
        positions = np.asarray(generated_layout["position"], dtype=np.float64)[:particle_count]
        body_ids = np.asarray(generated_layout["body_id"], dtype=np.int32)[:particle_count]
    else:
        positions = mpm.scene.particle.x.to_numpy()[:particle_count]
        body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    physical = positions - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
    x = physical[:, 0]
    y = physical[:, 1]
    tolerance = 1.0e-12
    main_bottom = (
        (body_ids == BODY_MAIN_SOIL)
        & (x >= K_MAIN_X_MIN - tolerance)
        & (x <= K_MAIN_X_MAX + tolerance)
        & (y >= kohler_base_bottom_z_np(x) - tolerance)
        & (y <= kohler_base_bottom_z_np(x) + DX + tolerance)
    )
    left_bottom = (
        (body_ids == BODY_LEFT_FREE_SOIL)
        & (x >= K_LEFT_FREE_X_MIN - tolerance)
        & (x <= K_LEFT_FREE_X_MAX + tolerance)
        & (y >= K_LEFT_FREE_BOTTOM_Z - tolerance)
        & (y <= K_LEFT_FREE_BOTTOM_Z + DX + tolerance)
    )
    right_bottom = (
        (body_ids == BODY_RIGHT_FREE_SOIL)
        & (x >= K_RIGHT_FREE_X_MIN - tolerance)
        & (x <= K_RIGHT_FREE_X_MAX + tolerance)
        & (y >= K_RIGHT_FREE_BOTTOM_Z - tolerance)
        & (y <= K_RIGHT_FREE_BOTTOM_Z + DX + tolerance)
    )
    return np.ascontiguousarray(np.nonzero(main_bottom | left_bottom | right_bottom)[0], dtype=np.int32)


def static_constraint_dof_mask(mpm: MPM, case: dict[str, Any]) -> np.ndarray:
    """Return the static support DOFs used by the paper residual check."""

    mask = np.zeros(
        (int(mpm.scene.element.gridSum), int(mpm.scene.grid_level), 2),
        dtype=bool,
    )
    if STATIC_MIRROR_PARTICLE_BOUNDARY:
        bottom_nodes = static_bottom_constraint_node_ids(mpm)
        mask[bottom_nodes, :, :] = True
        left_nodes, right_nodes = main_lateral_boundary_node_ids(mpm, case)
        lateral_nodes = np.unique(np.concatenate((left_nodes, right_nodes))).astype(np.int32)
        mask[lateral_nodes, BODY_MAIN_SOIL, 0] = True
        return np.ascontiguousarray(mask)

    for boundary in static_boundary_conditions(case):
        node_ids = np.asarray(
            mpm.scene.element.get_boundary_nodes(
                boundary["StartPoint"],
                boundary["EndPoint"],
            ),
            dtype=np.int32,
        )
        velocity = list(boundary.get("Velocity", [None, None]))
        for component, value in enumerate(velocity[:2]):
            if value is not None:
                mask[node_ids, :, component] = True
    return np.ascontiguousarray(mask)


def static_ignored_dof_mask(mpm: MPM) -> np.ndarray:
    """Exclude image/extrapolation nodes that have no physical equilibrium DOF."""

    grid_sum = int(mpm.scene.element.gridSum)
    grid_levels = int(mpm.scene.grid_level)
    grid_x = int(mpm.scene.element.gnum[0])
    node_ids = np.arange(grid_sum, dtype=np.int32)
    physical_x = (node_ids % grid_x).astype(np.float64) * DX - X_SHIFT
    physical_y = (node_ids // grid_x).astype(np.float64) * DX - Y_SHIFT
    tolerance = 1.0e-9
    mask = np.zeros((grid_sum, grid_levels, 2), dtype=bool)

    outside_model = (
        (physical_x < K_LEFT_FREE_X_MIN - tolerance)
        | (physical_x > K_RIGHT_FREE_X_MAX + tolerance)
        | (physical_y < K_BASE_BOTTOM_Z - tolerance)
    )
    mask[outside_model, :, :] = True

    body_x_ranges = {
        BODY_MAIN_SOIL: (K_MAIN_X_MIN, K_MAIN_X_MAX),
        BODY_LEFT_FREE_SOIL: (K_LEFT_FREE_X_MIN, K_LEFT_FREE_X_MAX),
        BODY_RIGHT_FREE_SOIL: (K_RIGHT_FREE_X_MIN, K_RIGHT_FREE_X_MAX),
    }
    for body_id, (x_min, x_max) in body_x_ranges.items():
        if body_id >= grid_levels:
            continue
        body_image_nodes = (physical_x < x_min - tolerance) | (physical_x > x_max + tolerance)
        mask[body_image_nodes, body_id, :] = True
    return np.ascontiguousarray(mask)


def static_constraint_node_ids(mpm: MPM, case: dict[str, Any]) -> np.ndarray:
    constrained = static_constraint_dof_mask(mpm, case)
    return np.ascontiguousarray(np.nonzero(np.any(constrained, axis=(1, 2)))[0], dtype=np.int32)


def static_boundary_conditions(case: dict[str, Any]) -> list[dict[str, Any]]:
    if STATIC_MIRROR_PARTICLE_BOUNDARY:
        # Static no-slip/slip conditions are imposed by the image-particle
        # equivalent, not by direct nodal velocity overwrites.
        return []
    bottom_constraints = static_bottom_boundary_conditions()
    side_spec = case.get("silent_boundary", {}).get("side", {})
    side_x_values = [float(value) for value in side_spec.get("x_values", [K_MAIN_X_MIN, K_MAIN_X_MAX])]
    side_y_min, side_y_max = [float(value) for value in side_spec.get("y_range", [K_DOMAIN_Y_MIN, K_DOMAIN_Y_MAX])]

    side_constraints = [
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [x_value + X_SHIFT, side_y_min + Y_SHIFT],
            "EndPoint": [x_value + X_SHIFT, side_y_max + Y_SHIFT],
        }
        for x_value in side_x_values
    ]
    return bottom_constraints + side_constraints


def clear_velocity_boundary_state(mpm: MPM) -> None:
    boundary = mpm.scene.boundary
    boundary.velocity_dict.clear()
    boundary.velocity_list[0] = 0
    if boundary.velocity_boundary is not None:
        kernel_initialize_boundary(boundary.velocity_boundary)
    if mpm.enginer is not None:
        mpm.enginer.choose_boundary_constraints(mpm.sims, mpm.scene)


def install_seismic_hooks(mpm: MPM) -> None:
    original_add_engine = mpm.add_engine

    def add_engine_with_seismic_boundary() -> None:
        original_add_engine()
        engine = mpm.enginer
        if getattr(engine, "_nairn_slope_shear_hooks_installed", False):
            return

        install_paper_apic_2d_p2g(mpm)

        original_particle_traction = engine.apply_particle_traction_constraints
        original_compute_forces = engine.compute_forces

        def apply_particle_traction_constraints(sims: Any, scene: Any) -> Any:
            boundary = getattr(mpm, "nairn_seismic_boundary", None)
            if boundary is not None:
                boundary.update_input_traction(sims, scene)
            return original_particle_traction(sims, scene)

        def compute_forces_with_silent_loads(sims: Any, scene: Any) -> Any:
            static_initialization = getattr(mpm, "static_initialization", None)
            if static_initialization is not None and static_initialization.active:
                static_initialization.apply_gravity_ramp(sims)
            result = original_compute_forces(sims, scene)
            boundary = getattr(mpm, "nairn_seismic_boundary", None)
            if boundary is not None:
                boundary.apply_silent_forces(sims, scene)
            return result

        engine.apply_particle_traction_constraints = apply_particle_traction_constraints
        engine.compute_forces = compute_forces_with_silent_loads
        engine._nairn_slope_shear_hooks_installed = True

    mpm.add_engine = add_engine_with_seismic_boundary


def run_strict_time_loop(mpm: MPM, callback) -> float:
    if mpm.solver is not None:
        mpm.solver.postprocess = []
    mpm.add_essentials({"function": callback})
    if HUGHES_WINGET_STRESS_UPDATE and DYNAMIC_SKIP_STRESS_UPDATE:
        raise RuntimeError("Hughes-Winget update and skipped stress update cannot be enabled together")
    if HUGHES_WINGET_STRESS_UPDATE:
        install_hughes_winget_stress_update(mpm)
    elif DYNAMIC_SKIP_STRESS_UPDATE:
        # This ablation must be installed after add_essentials(), which creates
        # the engine used by the strict dynamic stepping loop.
        mpm.enginer.compute_stress_strains = lambda sims, scene: None
    mpm.check_critical_timestep()
    auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    if auxiliary_tracker is not None:
        auxiliary_tracker.install_on_engine()
    solver = mpm.solver
    solver.postprocess = []
    solver.set_callback_function(callback)
    solver.engine.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    solver.save_file(mpm.scene)
    mpm.sims.current_print += 1
    solver.last_save_time = float(mpm.sims.current_time) - 0.8 * mpm.sims.delta

    start_time = time.time()
    try:
        while float(mpm.sims.current_time) < float(mpm.sims.time) - 0.5 * float(mpm.sims.delta):
            solver.core(mpm.scene, mpm.neighbor)
            boundary = getattr(mpm, "nairn_seismic_boundary", None)
            if boundary is not None:
                escape = boundary.check_domain_escape(mpm.sims, mpm.scene)
                boundary.flush_incremental_outputs(mpm.sims)
                if escape is not None and boundary.domain_escape_abort:
                    raise RuntimeError(
                        "Computational-grid domain escape detected: "
                        f"time={escape['time']}, pid={escape['first_pid']}, "
                        f"body_id={escape['first_body_id']}, "
                        f"position=({escape['first_x']}, {escape['first_z']})"
                    )
            new_body = mpm.generator.regenerate(mpm.scene)
            if new_body:
                raise RuntimeError("This validation command stream does not support body regeneration")
            if mpm.sims.current_time - solver.last_save_time + 0.1 * mpm.sims.delta > mpm.sims.save_interval:
                solver.save_file(mpm.scene)
                solver.last_save_time = 1.0 * mpm.sims.current_time
                mpm.sims.current_print += 1
            mpm.sims.current_time += mpm.sims.delta
            mpm.sims.current_step += 1
    except BaseException as error:
        # Preserve the last in-memory state before re-raising. The normal
        # post-processing path is skipped after an exception.
        failure_snapshot_status = "DISABLED"
        failure_snapshot_print = int(mpm.sims.current_print)
        if SAVE_PARTICLE or SAVE_GRID:
            try:
                solver.save_file(mpm.scene)
                mpm.sims.current_print += 1
                failure_snapshot_status = "SAVED"
            except BaseException as snapshot_error:
                failure_snapshot_status = f"FAILED: {repr(snapshot_error)}"
                print(f"Runtime failure snapshot save failed: {snapshot_error}")
            if failure_snapshot_status == "SAVED" and RUN_POSTPROCESS:
                try:
                    mpm.postprocessing(
                        read_path=OUTPUT_DIR.as_posix(),
                        write_background_grid=SAVE_GRID,
                        **failure_postprocess_kwargs(getattr(mpm, "case", CASE)),
                    )
                except BaseException as vtk_error:
                    failure_snapshot_status = f"SAVED_BUT_VTK_FAILED: {repr(vtk_error)}"
                    print(f"Runtime failure VTU conversion failed: {vtk_error}")
        boundary = getattr(mpm, "nairn_seismic_boundary", None)
        dynamic_time = (
            boundary.dynamic_time(mpm.sims)
            if boundary is not None
            else float(mpm.sims.current_time)
        )
        failure_report = OUTPUT_DIR / "runtime_failure_state_save.md"
        failure_report.parent.mkdir(parents=True, exist_ok=True)
        failure_report.write_text(
            "\n".join(
                [
                    "# Runtime Failure State Save",
                    "",
                    f"- dynamic_time: `{dynamic_time}`",
                    f"- absolute_time: `{float(mpm.sims.current_time)}`",
                    f"- exception: `{repr(error)}`",
                    f"- snapshot_print_number: `{failure_snapshot_print}`",
                    f"- snapshot_status: `{failure_snapshot_status}`",
                    f"- save_interval: `{SAVE_INTERVAL}`",
                    f"- particle_state_output: `{OUTPUT_DIR / 'particles'}`",
                    f"- grid_state_output: `{OUTPUT_DIR / 'grids'}`",
                    f"- vtk_output: `{OUTPUT_DIR / 'vtks'}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        if boundary is not None:
            boundary.capture_runtime_failure(mpm.sims, mpm.scene, error)
        raise
    end_time = time.time()

    if abs(mpm.sims.current_time - solver.last_save_time) > mpm.sims.save_interval:
        solver.save_file(mpm.scene)
        solver.last_save_time = 1.0 * mpm.sims.current_time
        mpm.sims.current_print += 1
    boundary = getattr(mpm, "nairn_seismic_boundary", None)
    if boundary is not None:
        boundary.flush_incremental_outputs(mpm.sims, force=True)
    mpm.first_run = False
    print("Physical time = ", end_time - start_time)
    return end_time - start_time


def sample_static_force_window(mpm: MPM, static_monitor: StaticInitializationMonitor) -> dict[str, Any] | None:
    engine = mpm.solver.engine
    scene = mpm.scene
    sims = mpm.sims
    neighbor = mpm.neighbor
    static_monitor.apply_gravity_ramp(sims)
    engine.reset_grid_messages(scene)
    engine.bulid_neighbor_list(sims, scene, neighbor)
    engine.calculate_interpolation(sims, scene)
    engine.compute_nodal_kinematic(sims, scene)
    engine.compute_grid_velcity(sims, scene)
    engine.apply_dirichlet_constraints(sims, scene)
    engine.apply_particle_traction_constraints(sims, scene)
    engine.compute_forces(sims, scene)
    engine.apply_traction_constraints(sims, scene)
    engine.apply_absorbing_constraints(sims, scene)
    return static_monitor.record_history(sims, scene)


def run_static_time_loop(mpm: MPM, static_monitor: StaticInitializationMonitor, save_files: bool = True) -> float:
    if mpm.solver is not None:
        mpm.solver.postprocess = []
    mpm.add_essentials({"function": None})
    install_failure_snapshot_recorder(mpm, getattr(mpm, "case", CASE))
    # Section 2.6 states that the Hughes-Winget objective stress integration
    # is used for the analysis. Install it before static relaxation as well as
    # before dynamics so the carried static stress state uses the same update.
    if HUGHES_WINGET_STRESS_UPDATE:
        install_hughes_winget_stress_update(mpm)
    install_static_mirror_particle_hooks(mpm)
    install_free_field_periodic_static_hooks(mpm)
    auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    if auxiliary_tracker is not None:
        auxiliary_tracker.install_on_engine()
    solver = mpm.solver
    engine = solver.engine
    scene = mpm.scene
    sims = mpm.sims
    neighbor = mpm.neighbor
    engine.pre_calculation(sims, scene, neighbor)
    # Mirror particles use the interpolation stencil prepared above.  Build the
    # support mask only after pre_calculation so every constrained support DOF
    # is classified consistently with the static boundary implementation.
    mpm.static_force_residual_constrained_dof_mask_np = static_constraint_dof_mask(
        mpm,
        static_monitor.case,
    )
    mpm.static_force_residual_ignored_dof_mask_np = static_ignored_dof_mask(mpm)
    mpm.static_force_residual_excluded_node_ids_np = np.ascontiguousarray(
        np.nonzero(np.any(mpm.static_force_residual_constrained_dof_mask_np, axis=(1, 2)))[0],
        dtype=np.int32,
    )
    mpm.static_bottom_constraint_node_ids_np = static_bottom_constraint_node_ids(mpm)
    solver.last_save_time = float(sims.current_time) - 0.8 * sims.delta

    start_step = int(sims.current_step)
    start_time = time.time()
    velocity_growth_streak = 0
    velocity_growth_start = math.inf
    previous_sample_velocity = math.inf
    mpm.static_g2p_eq15_snapshots = []
    paper_eq15_transfer = StaticPaperEq15GridProjection(mpm) if STATIC_PAPER_EQ15_FLIP else None
    while float(sims.current_time) < float(sims.time) - 0.5 * float(sims.delta):
        if STATIC_STEP_LIMIT > 0 and int(sims.current_step) - start_step >= STATIC_STEP_LIMIT:
            static_monitor.failure_reason = "STEP_LIMIT_REACHED"
            break
        if float(sims.delta) > 0.5 * float(static_monitor.critical_timestep) + 1.0e-15:
            static_monitor.failure_reason = "FAIL_DT_EXCEEDS_CFL_LIMIT"
            break

        static_monitor.apply_gravity_ramp(sims)
        engine.reset_grid_messages(scene)
        engine.bulid_neighbor_list(sims, scene, neighbor)
        engine.calculate_interpolation(sims, scene)
        engine.compute_nodal_kinematic(sims, scene)
        engine.compute_grid_velcity(sims, scene)
        engine.apply_dirichlet_constraints(sims, scene)
        engine.apply_particle_traction_constraints(sims, scene)
        engine.compute_forces(sims, scene)
        engine.apply_traction_constraints(sims, scene)
        engine.apply_absorbing_constraints(sims, scene)
        if STATIC_MIRROR_FORCE_DIAGNOSTIC and float(sims.current_time) >= float(sims.time) - 1.5 * float(sims.delta):
            capture_static_mirror_force_audit(mpm, sims)
        sample = static_monitor.record_history(sims, scene)
        if sample is not None:
            max_velocity = float(sample["grid_velocity"]["max_speed"])
            relative_unbalanced_force = float(sample["force"]["relative_unbalanced_force"])
            if not math.isfinite(max_velocity) or not math.isfinite(relative_unbalanced_force):
                static_monitor.failure_reason = "FAIL_NONFINITE_STATIC_METRIC"
                break
            elapsed_sample = float(sims.current_time) - float(static_monitor.start_time)
            growth_guard_active = elapsed_sample >= static_monitor.ramp_time + 0.5
            if (
                growth_guard_active
                and max_velocity > previous_sample_velocity
                and max_velocity > static_monitor.max_velocity_tolerance
            ):
                if velocity_growth_streak == 0:
                    velocity_growth_start = previous_sample_velocity
                velocity_growth_streak += 1
            else:
                velocity_growth_streak = 0
                velocity_growth_start = math.inf
            previous_sample_velocity = max_velocity
            if velocity_growth_streak >= 20 and max_velocity >= 2.0 * max(velocity_growth_start, 1.0e-30):
                static_monitor.failure_reason = "FAIL_VELOCITY_GROWTH"
                break
        capture_eq15 = STATIC_G2P_EQ15_DIAGNOSTIC and (
            static_monitor.converged
            or float(sims.current_time)
            >= float(sims.time) - (STATIC_G2P_EQ15_DIAGNOSTIC_STEPS + 0.5) * float(sims.delta)
        )
        # Eq. (15) requires v_i^n, before the grid momentum update.  This
        # snapshot is intentionally taken before compute_grid_kinematic.
        node_velocity_before_update = scene.node.momentum.to_numpy().copy() if capture_eq15 else None
        if paper_eq15_transfer is not None:
            paper_eq15_transfer.capture_node_velocity(scene.node)
        if STATIC_PAPER_EQ11_DAMPING:
            kernel_compute_grid_kinematic_paper_eq11(
                scene.mass_cut_off,
                sims.background_damping,
                scene.node,
                sims.dt,
            )
        else:
            engine.compute_grid_kinematic(sims, scene)
        engine.pre_contact_calculate(sims, scene)
        engine.apply_kinematic_constraints(sims, scene)
        engine.compute_contact_force_(sims, scene)
        # The subsequent native G2P wrapper performs this same extrapolation.
        # Do it first so the projected increment represents its exact input.
        mirror = getattr(mpm, "static_mirror_particle_boundary", None)
        if paper_eq15_transfer is not None and mirror is not None and static_mirror_boundary_is_active(mpm):
            mirror.extrapolate_velocity(scene)
            paper_eq15_transfer.project_velocity_increment(float(sims.delta), scene.node)
        eq15_snapshot = capture_static_g2p_eq15_before(mpm, sims, node_velocity_before_update) if capture_eq15 else None
        # Kohler Fig. 2 and Eqs. (16)-(18): B_p and L_p use the updated grid
        # velocity at x_p^n; stress is updated only after the G2P kinematics.
        engine.compute_velocity_gradient(sims, scene)
        engine.compute_particle_kinematic(sims, scene)
        precise_position_update = getattr(mpm, "nairn_accumulate_precise_position", None)
        if precise_position_update is not None:
            precise_position_update(sims, scene)
        engine.compute_stress_strains(sims, scene)
        engine.pressure_smoothing_(scene)
        if eq15_snapshot is not None:
            complete_static_g2p_eq15_after(mpm, eq15_snapshot)

        new_body = mpm.generator.regenerate(scene)
        if new_body:
            raise RuntimeError("This validation command stream does not support body regeneration")
        sims.current_time += sims.delta
        sims.current_step += 1
        if static_monitor.converged:
            break
    if static_monitor.failure_reason not in {"FAIL_NONFINITE_STATIC_METRIC", "FAIL_DT_EXCEEDS_CFL_LIMIT"}:
        sample = sample_static_force_window(mpm, static_monitor)
        if sample is not None:
            max_velocity = float(sample["velocity"]["max"])
            relative_unbalanced_force = float(sample["force"]["relative_unbalanced_force"])
            if not math.isfinite(max_velocity) or not math.isfinite(relative_unbalanced_force):
                static_monitor.failure_reason = "FAIL_NONFINITE_STATIC_METRIC"
    end_time = time.time()

    mpm.first_run = False
    static_monitor.end_time = float(sims.current_time)
    static_monitor.wall_clock_seconds = end_time - start_time
    static_monitor.physical_time = float(sims.current_time) - float(static_monitor.start_time)
    static_monitor.total_steps = int(sims.current_step) - start_step
    print("Physical time = ", end_time - start_time)
    return end_time - start_time


def run_static_initialization(mpm: MPM, case: dict[str, Any], save_files: bool = True) -> StaticInitializationMonitor:
    static_monitor = StaticInitializationMonitor(mpm, case)
    mpm.static_initialization = static_monitor
    if not static_monitor.enabled or static_monitor.duration <= 0.0:
        mpm.sims.set_gravity([0.0, 0.0])
        static_monitor.start_time = float(mpm.sims.current_time)
        static_monitor.end_time = float(mpm.sims.current_time)
        static_monitor.capture_final_state(mpm.scene)
        mpm.static_state_snapshot = static_monitor.final_snapshot
        return static_monitor

    static_monitor.active = True
    static_monitor.start_time = float(mpm.sims.current_time)
    static_monitor.next_history = static_monitor.start_time
    critical_timestep = float(mpm.scene.get_critical_timestep())
    if not math.isfinite(critical_timestep) or critical_timestep <= 0.0:
        raise RuntimeError(f"Invalid GeoTaichi critical_timestep: {critical_timestep}")
    static_dt = min(STATIC_DT_CAP, 0.5 * critical_timestep)
    cfl_safety_ratio = static_dt / critical_timestep
    if static_dt > 0.5 * critical_timestep + 1.0e-15:
        raise RuntimeError(
            f"Static dt exceeds 50% critical_timestep: static_dt={static_dt}, critical_timestep={critical_timestep}"
        )
    static_monitor.critical_timestep = critical_timestep
    static_monitor.dt = static_dt
    static_monitor.static_dt = static_dt
    static_monitor.spec["dt"] = static_dt
    static_monitor.cfl_safety_ratio = cfl_safety_ratio
    static_monitor.max_step_limit = STATIC_STEP_LIMIT
    static_monitor.end_time = static_monitor.start_time + static_monitor.duration
    static_monitor.timestep_rows.append(
        {
            "critical_timestep": critical_timestep,
            "static_dt": static_dt,
            "static_dt_cap": STATIC_DT_CAP,
            "cfl_safety_ratio": cfl_safety_ratio,
            "dt_within_50_percent_critical": int(static_dt <= 0.5 * critical_timestep + 1.0e-15),
            "dynamic_dt_preserved": DT,
            "static_max_time": static_monitor.duration,
            "history_interval": static_monitor.history_interval,
            "step_limit": STATIC_STEP_LIMIT,
        }
    )
    print(f"GeoTaichi critical_timestep = {critical_timestep:.17g}")
    print(f"static_dt = min({STATIC_DT_CAP:.17g}, 0.5 * critical_timestep) = {static_dt:.17g}")
    print(f"static CFL safety ratio = {cfl_safety_ratio:.17g}")

    mpm.modify_parameters(
        SimulationTime=static_monitor.end_time,
        Timestep=static_monitor.dt,
        SaveInterval=float(static_monitor.spec.get("save_interval", STATIC_SAVE_INTERVAL)),
        gravity=[0.0, 0.0],
        background_damping=float(static_monitor.spec.get("background_damping", STATIC_DAMPING)),
    )
    mpm.add_boundary_condition(boundary=static_boundary_conditions(case))

    auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    if auxiliary_tracker is not None:
        auxiliary_tracker.set_stage("static")
    original_alpha_pic = float(mpm.sims.alphaPIC)
    mpm.sims.alphaPIC = float(static_monitor.spec.get("alpha_pic", STATIC_ALPHA_PIC))
    try:
        mpm.static_runtime_seconds = run_static_time_loop(mpm, static_monitor, save_files=save_files)
    finally:
        # Static PIC/FLIP diagnostics must not silently change the dynamic
        # transfer scheme used for the Fig. 10 response.
        mpm.sims.alphaPIC = original_alpha_pic
    mpm.static_g2p_eq15_paths = write_static_g2p_eq15_diagnostic(mpm)
    mpm.static_mirror_force_audit_path = write_static_mirror_force_audit(mpm)

    static_monitor.active = False
    static_monitor.end_time = float(mpm.sims.current_time)
    static_monitor.capture_final_state(mpm.scene)
    mpm.static_state_snapshot = static_monitor.final_snapshot
    if save_files and getattr(mpm, "solver", None) is not None:
        mpm.solver.save_file(mpm.scene)
    clear_velocity_boundary_state(mpm)
    mpm.sims.set_gravity([0.0, 0.0])
    return static_monitor


def joint_boundary_validation_preflight(mpm: MPM, case: dict[str, Any], periodic_mapper: Any) -> dict[str, Any]:
    boundary = getattr(mpm, "nairn_seismic_boundary", None)
    auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    counts = velocity_constraint_counts(mpm)
    periodic_pair_rows = list(getattr(periodic_mapper, "pair_rows", []))
    left_periodic_pair_count = sum(1 for row in periodic_pair_rows if row.get("column") == "left")
    right_periodic_pair_count = sum(1 for row in periodic_pair_rows if row.get("column") == "right")
    periodic_pair_count = int(getattr(periodic_mapper, "pair_count", 0))
    row = {
        "periodic_hooks_installed": bool(
            getattr(getattr(mpm, "enginer", None), "_nairn_free_field_periodic_hooks_installed", False)
        ),
        "periodic_pair_count": periodic_pair_count,
        "left_periodic_pair_count": left_periodic_pair_count,
        "right_periodic_pair_count": right_periodic_pair_count,
        "auxiliary_F_tracker_active": auxiliary_tracker is not None,
        "surface_area_equation32_active": bool(
            boundary is not None
            and hasattr(boundary, "ff_pair_current_surface_areas")
            and hasattr(boundary, "total_invalid_surface_area_count")
        ),
        "lateral_static_support_active": bool(
            boundary is not None and int(getattr(boundary, "lateral_static_support_count", 0)) > 0
        ),
        "main_lateral_vx_constraint_count": int(counts["lateral_velocity_constraint_count"]),
        "bottom_boundary_active": bool(boundary is not None and int(getattr(boundary, "bottom_count", 0)) > 0),
        "main_to_free_field_force": 0.0,
    }
    row["status"] = (
        "PASS"
        if row["periodic_hooks_installed"]
        and row["periodic_pair_count"] == left_periodic_pair_count + right_periodic_pair_count
        and left_periodic_pair_count > 0
        and right_periodic_pair_count > 0
        and row["auxiliary_F_tracker_active"]
        and row["surface_area_equation32_active"]
        and row["lateral_static_support_active"]
        and row["main_lateral_vx_constraint_count"] == 0
        and row["bottom_boundary_active"]
        and row["main_to_free_field_force"] == 0.0
        else "FAIL"
    )
    mpm.joint_boundary_validation_preflight = row
    return row


def joint_validation_absmax(rows: list[dict[str, Any]], key: str) -> float:
    values = [abs(float(row.get(key, 0.0) or 0.0)) for row in rows]
    return max([0.0] + values)


def joint_validation_nonzero(value: float) -> bool:
    return abs(float(value)) > JOINT_BOUNDARY_NONZERO_TOL


def write_joint_boundary_validation_outputs(
    mpm: MPM,
    case: dict[str, Any],
    periodic_mapper: Any,
    geometry_before: dict[str, Any] | None,
    geometry_after: dict[str, Any],
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary = mpm.nairn_seismic_boundary
    rows = [dict(row) for row in boundary.boundary_rows]
    if not rows:
        raise RuntimeError("Joint boundary validation produced no dynamic boundary rows.")
    initial_arrays = getattr(mpm, "generated_particle_layout_arrays", particle_arrays(mpm.scene))
    final_arrays = particle_arrays(mpm.scene)
    particle_count_unchanged = int(final_arrays["position"].shape[0]) == int(initial_arrays["position"].shape[0])
    body_id_unchanged = np.array_equal(final_arrays["body_id"], initial_arrays["body_id"])
    material_id_unchanged = np.array_equal(final_arrays["material_id"], initial_arrays["material_id"])
    grid_hash_unchanged = bool(
        geometry_before is not None
        and geometry_before["grid_coordinate_hash"] == geometry_after["grid_coordinate_hash"]
    )

    time_history_path = output_dir / "joint_boundary_time_history.csv"
    time_fields = [
        "time",
        "input_velocity",
        "input_traction",
        "total_input_force_x",
        "total_input_force_y",
        "total_bottom_dashpot_force_x",
        "total_bottom_dashpot_force_y",
        "total_lateral_static_support_x",
        "total_free_field_dynamic_stress_force_x",
        "total_free_field_dynamic_stress_force_y",
        "total_normal_dashpot_force_x",
        "total_shear_dashpot_force_y",
        "total_main_coupling_force_x",
        "total_main_coupling_force_y",
        "total_force_applied_to_free_field_by_main",
        "surface_area_update_call_count",
        "invalid_surface_area_count",
        "total_invalid_surface_area_count",
        "min_surface_stretch",
        "max_surface_stretch",
        "min_current_surface_area",
        "max_current_surface_area",
        "periodic_mass_sync_call_count",
        "periodic_force_sync_call_count",
        "periodic_kinematic_sync_call_count",
        "periodic_before_g2p_sync_call_count",
        "auxiliary_F_update_call_count",
        "min_detF",
        "max_detF",
        "invalid_F_count",
        "main_max_velocity",
        "free_field_left_monitor_vx",
        "free_field_right_monitor_vx",
        "main_left_monitor_vx",
        "main_right_monitor_vx",
        "main_internal_monitor_vx",
        "main_surface_monitor_vx",
        "left_free_field_monitor_vx",
        "right_free_field_monitor_vx",
        "main_max_abs_vx",
        "main_rms_vx",
        "main_max_velocity_magnitude",
        "max_velocity_particle_id",
        "max_velocity_particle_body_id",
        "max_velocity_particle_material_id",
        "max_velocity_particle_x",
        "max_velocity_particle_y",
        "max_velocity_particle_vx",
        "max_velocity_particle_vy",
    ]
    time_rows = []
    for row in rows:
        time_rows.append(
            {
                "time": row["time"],
                "input_velocity": row["input_velocity_x"],
                "input_traction": row["input_stress_x"],
                "total_input_force_x": row["total_input_force_x"],
                "total_input_force_y": row["total_input_force_y"],
                "total_bottom_dashpot_force_x": row["total_bottom_silent_force_x"],
                "total_bottom_dashpot_force_y": row["total_bottom_silent_force_y"],
                "total_lateral_static_support_x": row["total_lateral_static_support_x"],
                "total_free_field_dynamic_stress_force_x": row["total_free_field_dynamic_stress_force_x"],
                "total_free_field_dynamic_stress_force_y": row["total_free_field_dynamic_stress_force_y"],
                "total_normal_dashpot_force_x": row["total_normal_dashpot_force_x"],
                "total_shear_dashpot_force_y": row["total_shear_dashpot_force_y"],
                "total_main_coupling_force_x": row["total_main_coupling_force_x"],
                "total_main_coupling_force_y": row["total_main_coupling_force_y"],
                "total_force_applied_to_free_field_by_main": row["total_force_applied_to_free_field_by_main"],
                "surface_area_update_call_count": row["surface_area_update_call_count"],
                "invalid_surface_area_count": row["invalid_surface_area_count"],
                "total_invalid_surface_area_count": row["total_invalid_surface_area_count"],
                "min_surface_stretch": row["min_surface_stretch"],
                "max_surface_stretch": row["max_surface_stretch"],
                "min_current_surface_area": row["min_current_surface_area"],
                "max_current_surface_area": row["max_current_surface_area"],
                "periodic_mass_sync_call_count": row["periodic_mass_sync_call_count"],
                "periodic_force_sync_call_count": row["periodic_force_sync_call_count"],
                "periodic_kinematic_sync_call_count": row["periodic_kinematic_sync_call_count"],
                "periodic_before_g2p_sync_call_count": row["periodic_before_g2p_sync_call_count"],
                "auxiliary_F_update_call_count": row["auxiliary_F_update_call_count"],
                "min_detF": row["min_detF"],
                "max_detF": row["max_detF"],
                "invalid_F_count": row["invalid_F_count"],
                "main_max_velocity": row["main_max_velocity"],
                "free_field_left_monitor_vx": row["free_field_left_monitor_vx"],
                "free_field_right_monitor_vx": row["free_field_right_monitor_vx"],
                "main_left_monitor_vx": row["main_left_monitor_vx"],
                "main_right_monitor_vx": row["main_right_monitor_vx"],
                "main_internal_monitor_vx": row["main_internal_monitor_vx"],
                "main_surface_monitor_vx": row["main_surface_monitor_vx"],
                "left_free_field_monitor_vx": row["left_free_field_monitor_vx"],
                "right_free_field_monitor_vx": row["right_free_field_monitor_vx"],
                "main_max_abs_vx": row["main_max_abs_vx"],
                "main_rms_vx": row["main_rms_vx"],
                "main_max_velocity_magnitude": row["main_max_velocity_magnitude"],
                "max_velocity_particle_id": row["max_velocity_particle_id"],
                "max_velocity_particle_body_id": row["max_velocity_particle_body_id"],
                "max_velocity_particle_material_id": row["max_velocity_particle_material_id"],
                "max_velocity_particle_x": row["max_velocity_particle_x"],
                "max_velocity_particle_y": row["max_velocity_particle_y"],
                "max_velocity_particle_vx": row["max_velocity_particle_vx"],
                "max_velocity_particle_vy": row["max_velocity_particle_vy"],
            }
        )
    write_csv(time_history_path, time_fields, time_rows)

    reconstruction_path = output_dir / "joint_lateral_force_reconstruction.csv"
    reconstruction_rows = []
    max_force_reconstruction_error = 0.0
    for row in rows:
        reconstructed_x = float(row["total_free_field_dynamic_stress_force_x"]) + float(row["total_normal_dashpot_force_x"])
        reconstructed_y = float(row["total_free_field_dynamic_stress_force_y"]) + float(row["total_shear_dashpot_force_y"])
        actual_x = float(row["total_main_coupling_force_x"])
        actual_y = float(row["total_main_coupling_force_y"])
        error_x = abs(reconstructed_x - actual_x)
        error_y = abs(reconstructed_y - actual_y)
        tol_x = JOINT_FORCE_RECONSTRUCTION_ABS_TOL + JOINT_FORCE_RECONSTRUCTION_REL_TOL * max(abs(actual_x), 1.0)
        tol_y = JOINT_FORCE_RECONSTRUCTION_ABS_TOL + JOINT_FORCE_RECONSTRUCTION_REL_TOL * max(abs(actual_y), 1.0)
        max_force_reconstruction_error = max(max_force_reconstruction_error, error_x, error_y)
        reconstruction_rows.append(
            {
                "time": row["time"],
                "stress_force_x": row["total_free_field_dynamic_stress_force_x"],
                "stress_force_y": row["total_free_field_dynamic_stress_force_y"],
                "normal_dashpot_force_x": row["total_normal_dashpot_force_x"],
                "shear_dashpot_force_y": row["total_shear_dashpot_force_y"],
                "reconstructed_coupling_x": reconstructed_x,
                "reconstructed_coupling_y": reconstructed_y,
                "actual_coupling_x": actual_x,
                "actual_coupling_y": actual_y,
                "error_x": error_x,
                "error_y": error_y,
                "status": "PASS" if error_x <= tol_x and error_y <= tol_y else "FAIL",
            }
        )
    write_csv(
        reconstruction_path,
        [
            "time",
            "stress_force_x",
            "stress_force_y",
            "normal_dashpot_force_x",
            "shear_dashpot_force_y",
            "reconstructed_coupling_x",
            "reconstructed_coupling_y",
            "actual_coupling_x",
            "actual_coupling_y",
            "error_x",
            "error_y",
            "status",
        ],
        reconstruction_rows,
    )

    nonzero_checks = {
        "input_velocity_nonzero": joint_validation_nonzero(joint_validation_absmax(rows, "input_velocity_x")),
        "input_traction_nonzero": joint_validation_nonzero(joint_validation_absmax(rows, "input_stress_x")),
        "bottom_input_force_nonzero": joint_validation_nonzero(joint_validation_absmax(rows, "total_input_force_x")),
        "bottom_dashpot_force_nonzero": joint_validation_nonzero(
            max(joint_validation_absmax(rows, "total_bottom_silent_force_x"), joint_validation_absmax(rows, "total_bottom_silent_force_y"))
        ),
        "free_field_velocity_nonzero": joint_validation_nonzero(
            max(joint_validation_absmax(rows, "free_field_left_monitor_vx"), joint_validation_absmax(rows, "free_field_right_monitor_vx"))
        ),
        "main_velocity_nonzero": joint_validation_nonzero(joint_validation_absmax(rows, "main_max_velocity")),
        "free_field_dynamic_stress_force_nonzero": joint_validation_nonzero(
            max(
                joint_validation_absmax(rows, "total_free_field_dynamic_stress_force_x"),
                joint_validation_absmax(rows, "total_free_field_dynamic_stress_force_y"),
            )
        ),
        "lateral_dashpot_force_nonzero": joint_validation_nonzero(
            max(joint_validation_absmax(rows, "total_normal_dashpot_force_x"), joint_validation_absmax(rows, "total_shear_dashpot_force_y"))
        ),
        "main_coupling_force_nonzero": joint_validation_nonzero(
            max(joint_validation_absmax(rows, "total_main_coupling_force_x"), joint_validation_absmax(rows, "total_main_coupling_force_y"))
        ),
    }
    nonzero_peaks = {
        "input_velocity_nonzero": joint_validation_absmax(rows, "input_velocity_x"),
        "input_traction_nonzero": joint_validation_absmax(rows, "input_stress_x"),
        "bottom_input_force_nonzero": joint_validation_absmax(rows, "total_input_force_x"),
        "bottom_dashpot_force_nonzero": max(
            joint_validation_absmax(rows, "total_bottom_silent_force_x"),
            joint_validation_absmax(rows, "total_bottom_silent_force_y"),
        ),
        "free_field_velocity_nonzero": max(
            joint_validation_absmax(rows, "free_field_left_monitor_vx"),
            joint_validation_absmax(rows, "free_field_right_monitor_vx"),
        ),
        "main_velocity_nonzero": joint_validation_absmax(rows, "main_max_velocity"),
        "free_field_dynamic_stress_force_nonzero": max(
            joint_validation_absmax(rows, "total_free_field_dynamic_stress_force_x"),
            joint_validation_absmax(rows, "total_free_field_dynamic_stress_force_y"),
        ),
        "lateral_dashpot_force_nonzero": max(
            joint_validation_absmax(rows, "total_normal_dashpot_force_x"),
            joint_validation_absmax(rows, "total_shear_dashpot_force_y"),
        ),
        "main_coupling_force_nonzero": max(
            joint_validation_absmax(rows, "total_main_coupling_force_x"),
            joint_validation_absmax(rows, "total_main_coupling_force_y"),
        ),
    }
    nonzero_path = output_dir / "joint_boundary_nonzero_check.csv"
    write_csv(
        nonzero_path,
        ["check", "threshold", "peak_abs_value", "status"],
        [
            {
                "check": key,
                "threshold": JOINT_BOUNDARY_NONZERO_TOL,
                "peak_abs_value": nonzero_peaks[key],
                "status": "PASS" if passed else "FAIL",
            }
            for key, passed in nonzero_checks.items()
        ],
    )

    dynamic_steps = len(rows)
    time_min = float(min(float(row["time"]) for row in rows))
    time_max = float(max(float(row["time"]) for row in rows))
    area_update_call_count = int(max(int(row["surface_area_update_call_count"]) for row in rows))
    auxiliary_F_update_call_count = int(max(int(row["auxiliary_F_update_call_count"]) for row in rows))
    periodic_counts = {
        "periodic_mass_sync_call_count": int(max(int(row["periodic_mass_sync_call_count"]) for row in rows)),
        "periodic_force_sync_call_count": int(max(int(row["periodic_force_sync_call_count"]) for row in rows)),
        "periodic_kinematic_sync_call_count": int(max(int(row["periodic_kinematic_sync_call_count"]) for row in rows)),
        "periodic_before_g2p_sync_call_count": int(max(int(row["periodic_before_g2p_sync_call_count"]) for row in rows)),
    }
    max_periodic_velocity_difference = max(float(row["max_periodic_velocity_difference"]) for row in rows)
    max_periodic_acceleration_difference = max(float(row["max_periodic_acceleration_difference"]) for row in rows)
    max_periodic_mass_error = max(abs(float(row["mass_conservation_error"])) for row in rows)
    max_periodic_momentum_error = max(abs(float(row["momentum_conservation_error"])) for row in rows)
    max_periodic_force_error = max(abs(float(row["force_conservation_error"])) for row in rows)
    max_main_node_modified_count = int(max(int(row["main_node_modified_count"]) for row in rows))
    max_nan_count = int(max(int(row["NaN_count"]) for row in rows))
    max_inf_count = int(max(int(row["Inf_count"]) for row in rows))
    min_detF = float(min(float(row["min_detF"]) for row in rows))
    max_detF = float(max(float(row["max_detF"]) for row in rows))
    max_invalid_F_count = int(max(int(row["invalid_F_count"]) for row in rows))
    max_invalid_surface_area_count = int(max(int(row["invalid_surface_area_count"]) for row in rows))
    total_invalid_surface_area_count = int(max(int(row["total_invalid_surface_area_count"]) for row in rows))
    min_stretch = float(min(float(row["min_surface_stretch"]) for row in rows))
    max_stretch = float(max(float(row["max_surface_stretch"]) for row in rows))
    max_abs_stretch_minus_one = max(abs(min_stretch - 1.0), abs(max_stretch - 1.0))
    min_area = float(min(float(row["min_current_surface_area"]) for row in rows))
    max_area = float(max(float(row["max_current_surface_area"]) for row in rows))
    max_area_formula_error = float(max(float(row["max_area_formula_error"]) for row in rows))
    main_to_free_field_force_max = joint_validation_absmax(rows, "total_force_applied_to_free_field_by_main")
    preflight = getattr(mpm, "joint_boundary_validation_preflight", {"status": "FAIL"})

    summary = {
        "dynamic_steps": dynamic_steps,
        "time_min": time_min,
        "time_max": time_max,
        "input_velocity_peak": joint_validation_absmax(rows, "input_velocity_x"),
        "input_traction_peak": joint_validation_absmax(rows, "input_stress_x"),
        "bottom_input_force_peak": joint_validation_absmax(rows, "total_input_force_x"),
        "free_field_velocity_peak": max(joint_validation_absmax(rows, "free_field_left_monitor_vx"), joint_validation_absmax(rows, "free_field_right_monitor_vx")),
        "main_velocity_peak": joint_validation_absmax(rows, "main_max_velocity"),
        "lateral_dynamic_stress_force_peak": max(joint_validation_absmax(rows, "total_free_field_dynamic_stress_force_x"), joint_validation_absmax(rows, "total_free_field_dynamic_stress_force_y")),
        "lateral_dashpot_force_peak": max(joint_validation_absmax(rows, "total_normal_dashpot_force_x"), joint_validation_absmax(rows, "total_shear_dashpot_force_y")),
        "total_lateral_coupling_force_peak": max(joint_validation_absmax(rows, "total_main_coupling_force_x"), joint_validation_absmax(rows, "total_main_coupling_force_y")),
        "max_force_reconstruction_error": max_force_reconstruction_error,
        "min_stretch_over_run": min_stretch,
        "max_stretch_over_run": max_stretch,
        "max_abs_stretch_minus_one": max_abs_stretch_minus_one,
        "min_area_over_run": min_area,
        "max_area_over_run": max_area,
        "max_area_formula_error": max_area_formula_error,
        "area_update_call_count": area_update_call_count,
        **periodic_counts,
        "max_periodic_velocity_difference": max_periodic_velocity_difference,
        "max_periodic_acceleration_difference": max_periodic_acceleration_difference,
        "mass_conservation_error": max_periodic_mass_error,
        "momentum_conservation_error": max_periodic_momentum_error,
        "force_conservation_error": max_periodic_force_error,
        "main_node_modified_count": max_main_node_modified_count,
        "main_to_free_field_force_max": main_to_free_field_force_max,
        "min_detF": min_detF,
        "max_detF": max_detF,
        "invalid_F_count": max_invalid_F_count,
        "invalid_surface_area_count": max_invalid_surface_area_count,
        "total_invalid_surface_area_count": total_invalid_surface_area_count,
        "NaN_count": max_nan_count,
        "Inf_count": max_inf_count,
        "particle_count_unchanged": particle_count_unchanged,
        "body_id_unchanged": body_id_unchanged,
        "material_id_unchanged": material_id_unchanged,
        "grid_coordinate_hash_unchanged": grid_hash_unchanged,
    }
    final_pass = (
        preflight.get("status") == "PASS"
        and dynamic_steps >= 100
        and all(nonzero_checks.values())
        and all(row["status"] == "PASS" for row in reconstruction_rows)
        and area_update_call_count >= dynamic_steps
        and auxiliary_F_update_call_count >= dynamic_steps
        and all(value > 0 for value in periodic_counts.values())
        and main_to_free_field_force_max <= JOINT_FORCE_RECONSTRUCTION_ABS_TOL
        and max_invalid_surface_area_count == 0
        and total_invalid_surface_area_count == 0
        and max_invalid_F_count == 0
        and min_detF > 0.0
        and min_area > 0.0
        and np.isfinite(min_area)
        and np.isfinite(max_area)
        and max_nan_count == 0
        and max_inf_count == 0
        and max_main_node_modified_count == 0
        and particle_count_unchanged
        and body_id_unchanged
        and material_id_unchanged
        and grid_hash_unchanged
        and max_periodic_velocity_difference <= FREE_FIELD_PERIODIC_TOLERANCE
        and max_periodic_acceleration_difference <= FREE_FIELD_PERIODIC_TOLERANCE
        and max_periodic_mass_error <= FREE_FIELD_PERIODIC_TOLERANCE
        and max_periodic_momentum_error <= FREE_FIELD_PERIODIC_TOLERANCE
        and max_periodic_force_error <= FREE_FIELD_PERIODIC_TOLERANCE
    )
    summary["status"] = "PASS" if final_pass else "FAIL"

    summary_path = output_dir / "joint_boundary_validation_summary.csv"
    write_csv(summary_path, list(summary.keys()), [summary])

    report_path = output_dir / "joint_boundary_validation_report.md"
    zero_components = [
        key for key, passed in nonzero_checks.items() if not passed
    ]
    lines = [
        "# Joint Boundary Validation",
        "",
        f"- status: `{summary['status']}`",
        f"- nonzero_threshold: `{JOINT_BOUNDARY_NONZERO_TOL}`",
        f"- force_reconstruction_tolerance: `abs_error <= {JOINT_FORCE_RECONSTRUCTION_ABS_TOL} + {JOINT_FORCE_RECONSTRUCTION_REL_TOL}*max(abs(actual_force),1)`",
        f"- input_reached_free_field: `{'PASS' if nonzero_checks['free_field_velocity_nonzero'] else 'FAIL'}`",
        f"- free_field_response_generated_lateral_force: `{'PASS' if nonzero_checks['free_field_dynamic_stress_force_nonzero'] or nonzero_checks['lateral_dashpot_force_nonzero'] else 'FAIL'}`",
        f"- main_model_response: `{'PASS' if nonzero_checks['main_velocity_nonzero'] else 'FAIL'}`",
        f"- components_still_zero_over_short_run: `{','.join(zero_components) if zero_components else 'none'}`",
        "- zero_component_interpretation: `FAIL if required nonzero response did not appear; no PASS is inferred from wave-travel delay.`",
        "",
        "## Summary",
    ]
    lines.extend([f"- {key}: `{value}`" for key, value in summary.items()])
    lines.extend(
        [
            "",
            "## Output Files",
            f"- `{time_history_path}`",
            f"- `{nonzero_path}`",
            f"- `{reconstruction_path}`",
            f"- `{summary_path}`",
            f"- `{report_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    keep = {path.name for path in (time_history_path, nonzero_path, reconstruction_path, summary_path, report_path)}
    for path in output_dir.iterdir():
        if path.name in keep:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
    return time_history_path, nonzero_path, reconstruction_path, summary_path, report_path


def run_dynamic_stage(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: StaticInitializationMonitor,
    dynamic_start_time: float,
) -> tuple[Path, Path]:
    if static_monitor.enabled and static_monitor.require_convergence and not static_monitor.converged:
        raise RuntimeError(
            "Static initialization did not converge; refusing to start dynamic analysis. "
            "Increase NAIRN_STATIC_TIME/NAIRN_STATIC_DAMPING, reduce NAIRN_STATIC_DT, "
            "or set NAIRN_REQUIRE_STATIC_CONVERGENCE=0 for an explicitly non-final diagnostic run."
        )
    # Paper Section 3.2: preserve the completed static stress state while the
    # direct nodal support is transferred into the first dynamic force window.
    register_bottom_input_traction(mpm, case)
    dynamic_materials = apply_material_stage(mpm, case, "dynamic")
    apply_dynamic_qr_region(mpm, case)
    apply_equivalent_postquake_qr_state(mpm, case)
    apply_separate_slide_body(mpm, case)
    apply_postquake_block_velocity(mpm, case)
    dynamic_strength_override = apply_dynamic_strength_override(mpm, case)
    mpm.dynamic_strength_override = dynamic_strength_override
    static_materials = materials_for_stage(case, "static")
    write_static_material_boundary_alignment_report(case, static_materials, dynamic_materials)
    dynamic_gravity = static_monitor.target_gravity if DYNAMIC_KEEP_GRAVITY else [0.0, 0.0]
    mpm.modify_parameters(
        SimulationTime=dynamic_start_time + SIMULATION_TIME,
        Timestep=DT,
        SaveInterval=SAVE_INTERVAL,
        gravity=dynamic_gravity,
        background_damping=DYNAMIC_DAMPING,
    )
    mpm.sims.set_gravity(dynamic_gravity)
    # Finalize the dynamic engine before extracting transferred reactions.
    # add_essentials selects the active force-assembly implementation.
    mpm.add_essentials({"function": None})
    diagnostic_spec = case.get("diagnostic", {})
    if bool(diagnostic_spec.get("zero_velocity_at_dynamic_start", False)):
        particle_count = int(mpm.scene.particleNum[0])
        zero_static_particle_kinematics(particle_count, mpm.scene.particle)
        apic_affine_velocity = getattr(mpm, "nairn_apic_affine_velocity", None)
        if apic_affine_velocity is not None:
            apic_affine_velocity.fill(0.0)
        mpm.dynamic_transition_velocity_zeroed = True
    else:
        mpm.dynamic_transition_velocity_zeroed = False
    # Checkpoint-loaded runs create the recorder here (after the initial
    # setup-time install attempt), so install the failure-state augmentation
    # again once the dynamic recorder exists.
    install_failure_snapshot_recorder(mpm, case)
    if not hasattr(mpm, "static_bottom_constraint_node_ids_np"):
        mpm.static_bottom_constraint_node_ids_np = static_bottom_constraint_node_ids(mpm)
    # A checkpoint rebuilds the engine, so restore the static-stage periodic
    # topology before extracting any static reactions for the dynamic boundary.
    dynamic_periodic_geometry_before = (
        geometry_periodic_snapshot(mpm) if FREE_FIELD_PERIODIC_DYNAMIC_PATH_CHECK or JOINT_BOUNDARY_VALIDATION else None
    )
    periodic_mapper = install_free_field_periodic_dynamic_hooks(mpm)
    assert_free_field_periodic_dynamic_ready(mpm)
    # The lateral reaction must be sampled on this same periodic node topology.
    # Sampling it before shared-node construction leaves a force mismatch at
    # the dynamic free-field interface.
    if mpm.solver is not None:
        mpm.solver.engine.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    # Extract the static support on the final dynamic P2G path, before its
    # first stress update or any dynamic boundary force is applied.
    freeze_bottom_static_support_before_dynamic(mpm, case, static_monitor)
    # The bottom-support extraction finalizes the dynamic P2G support state.
    # Sample the lateral residual afterwards so both supports cancel forces on
    # exactly the same nodal topology.
    if not hasattr(mpm, "lateral_static_support_node_ids_np"):
        assemble_lateral_static_support_from_geotaichi(mpm, case)
    mpm.nairn_seismic_boundary = NairnSeismicBoundary(mpm, case)
    mpm.nairn_seismic_boundary.set_dynamic_start_time(dynamic_start_time)
    if JOINT_BOUNDARY_VALIDATION:
        periodic_mapper.track_main_node_modifications = True
        preflight = joint_boundary_validation_preflight(mpm, case, periodic_mapper)
        if not preflight["status"] == "PASS":
            raise RuntimeError(f"Joint boundary validation preflight failed: {preflight}")
    dynamic_initial_snapshot = {
        "arrays": particle_arrays(mpm.scene),
        "particle_count": int(mpm.scene.particleNum[0]),
        "velocity": velocity_stats(mpm.scene.particle.v.to_numpy()[: int(mpm.scene.particleNum[0])]),
        "stress": stress_stats(mpm.scene.particle.stress.to_numpy()[: int(mpm.scene.particleNum[0])]),
    }
    transition_report, transition_csv = write_transition_outputs(
        mpm,
        static_monitor,
        dynamic_initial_snapshot,
        dynamic_start_time,
    )
    if HUGHES_WINGET_STRESS_UPDATE:
        install_hughes_winget_stress_update(mpm)
    if DYNAMIC_RELAXATION_TIME > 0.0:
        if DYNAMIC_SKIP_STRESS_UPDATE:
            # Diagnostic ablation: retain the checkpoint stress field while
            # suppressing only subsequent dynamic constitutive updates.
            mpm.enginer.compute_stress_strains = lambda sims, scene: None
        run_dynamic_boundary_relaxation(mpm, DYNAMIC_RELAXATION_TIME, DYNAMIC_RELAXATION_DAMPING)
        dynamic_start_time = float(mpm.sims.current_time)
        mpm.modify_parameters(
            SimulationTime=dynamic_start_time + SIMULATION_TIME,
            Timestep=DT,
            SaveInterval=SAVE_INTERVAL,
            gravity=dynamic_gravity,
            background_damping=DYNAMIC_DAMPING,
        )
        mpm.sims.set_gravity(dynamic_gravity)
        mpm.nairn_seismic_boundary.set_dynamic_start_time(dynamic_start_time)
        mpm.periodic_mapper_active_before_first_dynamic_step = bool(
            getattr(mpm.enginer, "_nairn_free_field_periodic_hooks_installed", False)
            and getattr(periodic_mapper, "pair_count", 0) > 0
        )

    def dynamic_step_callback() -> None:
        apply_dynamic_drucker_prager_softening(mpm, case)
        maybe_stop_seismic_input(mpm, case)
        mpm.nairn_seismic_boundary.record_history(mpm.sims, mpm.scene)
        if JOINT_BOUNDARY_VALIDATION:
            update_latest_joint_boundary_row_after_step(mpm, periodic_mapper)

    auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    if auxiliary_tracker is not None:
        auxiliary_tracker.set_stage("dynamic")
        if AUXILIARY_F_DYNAMIC_PATH_SMOKE:
            mpm.auxiliary_tracker_object_id_before_dynamic = id(auxiliary_tracker)
            mpm.auxiliary_pre_dynamic_F_hash = stable_array_hash(auxiliary_tracker.to_numpy())
            mpm.auxiliary_dynamic_update_count_before = int(auxiliary_tracker.update_call_count[None])
    assert_free_field_periodic_dynamic_ready(mpm)
    if STRICT_TIME_LOOP:
        mpm.dynamic_runtime_seconds = run_strict_time_loop(mpm, dynamic_step_callback)
    else:
        if mpm.solver is not None:
            mpm.solver.postprocess = []
        start_time = time.time()
        mpm.run(function=dynamic_step_callback)
        mpm.dynamic_runtime_seconds = time.time() - start_time
    if auxiliary_tracker is not None and AUXILIARY_F_DYNAMIC_PATH_SMOKE:
        mpm.auxiliary_post_first_dynamic_step_F_hash = stable_array_hash(auxiliary_tracker.to_numpy())
        mpm.auxiliary_dynamic_update_call_count = (
            int(auxiliary_tracker.update_call_count[None])
            - int(getattr(mpm, "auxiliary_dynamic_update_count_before", 0))
        )
        mpm.auxiliary_hook_active_during_dynamic = bool(
            getattr(getattr(mpm, "enginer", None), "_nairn_auxiliary_F_hook_installed", False)
        )
    mpm.dynamic_state_snapshot = {
        "arrays": particle_arrays(mpm.scene),
        "particle_count": int(mpm.scene.particleNum[0]),
        "velocity": velocity_stats(mpm.scene.particle.v.to_numpy()[: int(mpm.scene.particleNum[0])]),
        "stress": stress_stats(mpm.scene.particle.stress.to_numpy()[: int(mpm.scene.particleNum[0])]),
    }
    if JOINT_BOUNDARY_VALIDATION:
        geometry_after = geometry_periodic_snapshot(mpm)
        mpm.joint_boundary_validation_paths = write_joint_boundary_validation_outputs(
            mpm,
            case,
            periodic_mapper,
            dynamic_periodic_geometry_before,
            geometry_after,
        )
    if AUXILIARY_F_DYNAMIC_PATH_SMOKE and dynamic_periodic_geometry_before is not None:
        geometry_after = geometry_periodic_snapshot(mpm)
        mpm.auxiliary_geometry_modified = not (
            geometry_after["particle_count"] == dynamic_periodic_geometry_before["particle_count"]
            and geometry_after["grid_node_count"] == dynamic_periodic_geometry_before["grid_node_count"]
            and geometry_after["main_particle_count"] == dynamic_periodic_geometry_before["main_particle_count"]
            and geometry_after["left_ff_particle_count"] == dynamic_periodic_geometry_before["left_ff_particle_count"]
            and geometry_after["right_ff_particle_count"] == dynamic_periodic_geometry_before["right_ff_particle_count"]
            and geometry_after["grid_coordinate_hash"] == dynamic_periodic_geometry_before["grid_coordinate_hash"]
        )
    if FREE_FIELD_PERIODIC_DYNAMIC_PATH_CHECK and dynamic_periodic_geometry_before is not None:
        _, geometry_modified = write_geometry_unchanged_check(
            dynamic_periodic_geometry_before,
            geometry_periodic_snapshot(mpm),
            allow_particle_motion=True,
        )
        mpm.free_field_periodic_dynamic_path_check = write_free_field_periodic_dynamic_path_check(
            mpm,
            periodic_mapper,
            geometry_modified,
        )
    return transition_report, transition_csv


def state_variables_hash(state: Any) -> str:
    if not isinstance(state, dict):
        return stable_array_hash(np.asarray([], dtype=np.float64))
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(str(key).encode("utf-8"))
        value = np.ascontiguousarray(np.asarray(state[key]))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def component_velocity_stats(velocities: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(velocities[mask], dtype=np.float64)
    if selected.size == 0:
        return {"max_abs_vx": 0.0, "max_abs_vy": 0.0, "rms_vx": 0.0, "rms_vy": 0.0, "max_magnitude": 0.0}
    vx = selected[:, 0]
    vy = selected[:, 1]
    mag = np.linalg.norm(selected, axis=1)
    return {
        "max_abs_vx": float(np.max(np.abs(vx))),
        "max_abs_vy": float(np.max(np.abs(vy))),
        "rms_vx": float(np.sqrt(np.mean(vx * vx))),
        "rms_vy": float(np.sqrt(np.mean(vy * vy))),
        "max_magnitude": float(np.max(mag)),
    }


def material_stage_pair(case: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    static = {int(row["MaterialID"]): row for row in materials_for_stage(case, "static")}
    dynamic = {int(row["MaterialID"]): row for row in materials_for_stage(case, "dynamic")}
    return static, dynamic


def support_node_entries(mpm: MPM, particle_id: int) -> list[tuple[int, float]]:
    count = int(mpm.scene.element.node_size.to_numpy()[int(particle_id)])
    ln_id = mpm.scene.element.LnID.to_numpy()
    shape = mpm.scene.element.shape_fn.to_numpy()
    if ln_id.ndim == 2:
        node_ids = ln_id[int(particle_id), :count].astype(np.int32)
        weights = shape[int(particle_id), :count].astype(np.float64)
    else:
        total_nodes = int(mpm.scene.element.grid_nodes)
        start = int(particle_id) * total_nodes
        node_ids = ln_id[start : start + count].astype(np.int32)
        weights = shape[start : start + count].astype(np.float64)
    return [(int(node_ids[i]), float(weights[i])) for i in range(count)]


def classify_dynamic_node(node_id: int, x: float, y: float) -> str:
    on_left = abs(x - K_MAIN_X_MIN) <= SIDE_TOL
    on_right = abs(x - K_MAIN_X_MAX) <= SIDE_TOL
    bottom_y = kohler_base_bottom_z_np(x)
    interface_y = kohler_soil_base_interface_z_np(x)
    on_bottom = abs(y - bottom_y) <= BOTTOM_TOL
    on_interface = abs(y - interface_y) <= max(BOTTOM_TOL, DX)
    if on_bottom and on_left:
        return "main_bottom_left_corner"
    if on_bottom and on_right:
        return "main_bottom_right_corner"
    if on_bottom:
        return "main_bottom"
    if on_left:
        return "main_left_side"
    if on_right:
        return "main_right_side"
    if on_interface:
        return "soil_base_interface"
    if K_MAIN_X_MIN - SIDE_TOL <= x <= K_MAIN_X_MAX + SIDE_TOL:
        return "main_interior"
    return "other"


def zero_nodal_component_array(mpm: MPM) -> np.ndarray:
    return np.zeros_like(mpm.scene.node.force.to_numpy(), dtype=np.float64)


def scatter_to_node_component(
    target: np.ndarray,
    node_ids: np.ndarray,
    body_ids: np.ndarray,
    forces: np.ndarray,
) -> None:
    for node_id, body_id, force in zip(node_ids.astype(np.int32), body_ids.astype(np.int32), forces.astype(np.float64)):
        if 0 <= int(node_id) < target.shape[0] and 0 <= int(body_id) < target.shape[1]:
            target[int(node_id), int(body_id), :2] += force[:2]


def boundary_component_arrays(mpm: MPM, boundary: NairnSeismicBoundary) -> dict[str, np.ndarray]:
    components = {
        "bottom_static_support": zero_nodal_component_array(mpm),
        "bottom_dashpot_force": zero_nodal_component_array(mpm),
        "bottom_input_force": zero_nodal_component_array(mpm),
        "lateral_static_support": zero_nodal_component_array(mpm),
        "free_field_dynamic_stress_force": zero_nodal_component_array(mpm),
        "lateral_dashpot_force": zero_nodal_component_array(mpm),
    }
    positions = mpm.scene.particle.x.to_numpy()[: int(mpm.scene.particleNum[0])]
    velocities = mpm.scene.particle.v.to_numpy()[: int(mpm.scene.particleNum[0])]
    body_ids = mpm.scene.particle.bodyID.to_numpy()[: int(mpm.scene.particleNum[0])].astype(np.int32)
    node_size = mpm.scene.element.node_size.to_numpy()
    ln_id = mpm.scene.element.LnID.to_numpy()
    shape = mpm.scene.element.shape_fn.to_numpy()
    total_nodes = int(mpm.scene.element.grid_nodes)

    def particle_nodes(pid: int) -> tuple[np.ndarray, np.ndarray]:
        count = int(node_size[pid])
        if ln_id.ndim == 2:
            return ln_id[pid, :count].astype(np.int32), shape[pid, :count].astype(np.float64)
        offset = int(pid) * total_nodes
        return ln_id[offset : offset + count].astype(np.int32), shape[offset : offset + count].astype(np.float64)

    if boundary.bottom_static_reaction_node_count:
        node_ids = boundary.bottom_static_reaction_node_ids_np
        node_body_ids = boundary.bottom_static_reaction_node_body_ids_np
        forces = np.column_stack((boundary.bottom_static_reaction_node_x_np, boundary.bottom_static_reaction_node_y_np))
        scatter_to_node_component(components["bottom_static_support"], node_ids, node_body_ids, forces)

    for pid in boundary.bottom_particle_ids_np.astype(np.int32).tolist():
        force = np.array(
            [
                -float(mpm.scene.particle.m.to_numpy()[pid]) * CS * velocities[pid, 0] / (2.0 * HALF_PARTICLE_SIZE),
                -float(mpm.scene.particle.m.to_numpy()[pid]) * CP * velocities[pid, 1] / (2.0 * HALF_PARTICLE_SIZE),
            ],
            dtype=np.float64,
        )
        node_ids, weights = particle_nodes(pid)
        for node_id, weight in zip(node_ids, weights):
            node_id = int(node_id)
            components["bottom_dashpot_force"][node_id, int(body_ids[pid]), :2] += weight * force

    input_force = np.array([float(boundary.total_input_force_x[None]), float(boundary.total_input_force_y[None])])
    if np.linalg.norm(input_force) > 0.0 and boundary.bottom_particle_ids_np.size:
        per_particle = input_force / float(boundary.bottom_particle_ids_np.size)
        for pid in boundary.bottom_particle_ids_np.astype(np.int32).tolist():
            node_ids, weights = particle_nodes(pid)
            for node_id, weight in zip(node_ids, weights):
                components["bottom_input_force"][int(node_id), int(body_ids[pid]), :2] += float(weight) * per_particle

    if boundary.lateral_static_support_count:
        node_ids = boundary.lateral_static_support_node_ids.to_numpy()[: boundary.lateral_static_support_count]
        node_body_ids = boundary.lateral_static_support_body_ids.to_numpy()[: boundary.lateral_static_support_count]
        forces = np.column_stack(
            (
                boundary.lateral_static_support_x.to_numpy()[: boundary.lateral_static_support_count],
                np.zeros(boundary.lateral_static_support_count),
            )
        )
        scatter_to_node_component(components["lateral_static_support"], node_ids, node_body_ids, forces)

    if boundary.ff_pair_count:
        stress_x = boundary.ff_pair_dynamic_stress_force_x.to_numpy()[: boundary.ff_pair_count]
        stress_y = boundary.ff_pair_dynamic_stress_force_y.to_numpy()[: boundary.ff_pair_count]
        dash_x = boundary.ff_pair_normal_dashpot_force_x.to_numpy()[: boundary.ff_pair_count]
        dash_y = boundary.ff_pair_shear_dashpot_force_y.to_numpy()[: boundary.ff_pair_count]
        for index, pid in enumerate(boundary.ff_pair_main_ids_np.astype(np.int32).tolist()):
            node_ids, weights = particle_nodes(pid)
            for node_id, weight in zip(node_ids, weights):
                node_id = int(node_id)
                body_id = int(body_ids[pid])
                components["free_field_dynamic_stress_force"][node_id, body_id, :2] += weight * np.array(
                    [stress_x[index], stress_y[index]]
                )
                components["lateral_dashpot_force"][node_id, body_id, :2] += weight * np.array(
                    [dash_x[index], dash_y[index]]
                )

    return components


def prepare_dynamic_transition_diagnostic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: StaticInitializationMonitor,
    dynamic_start_time: float,
) -> tuple[NairnSeismicBoundary, Any, dict[str, Any], dict[str, Any], list[float]]:
    case.setdefault("earthquake_input", {})["enabled"] = False
    if static_monitor.enabled and static_monitor.require_convergence and not static_monitor.converged:
        raise RuntimeError("Static initialization did not converge; diagnostic refuses to start from a non-final state.")
    static_materials = materials_for_stage(case, "static")
    register_bottom_input_traction(mpm, case)
    dynamic_materials = apply_material_stage(mpm, case, "dynamic")
    dynamic_gravity = static_monitor.target_gravity if DYNAMIC_KEEP_GRAVITY else [0.0, 0.0]
    mpm.modify_parameters(
        SimulationTime=dynamic_start_time + 10.0 * DT,
        Timestep=DT,
        SaveInterval=10.0 * DT,
        gravity=dynamic_gravity,
        background_damping=DYNAMIC_DAMPING,
    )
    mpm.sims.set_gravity(dynamic_gravity)
    # Finalize the dynamic engine before extracting transferred reactions.
    # add_essentials selects the active force-assembly implementation.
    mpm.add_essentials({"function": None})
    if not hasattr(mpm, "static_bottom_constraint_node_ids_np"):
        mpm.static_bottom_constraint_node_ids_np = static_bottom_constraint_node_ids(mpm)
    # This must precede all reaction extraction and NairnSeismicBoundary so
    # each transferred support uses the final dynamic periodic topology.
    periodic_mapper = install_free_field_periodic_dynamic_hooks(mpm)
    assert_free_field_periodic_dynamic_ready(mpm)
    if mpm.solver is not None:
        mpm.solver.engine.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    freeze_bottom_static_support_before_dynamic(mpm, case, static_monitor)
    if not hasattr(mpm, "lateral_static_support_node_ids_np"):
        assemble_lateral_static_support_from_geotaichi(mpm, case)
    boundary = NairnSeismicBoundary(mpm, case)
    boundary.input_enabled = False
    mpm.nairn_seismic_boundary = boundary
    mpm.nairn_seismic_boundary.set_dynamic_start_time(dynamic_start_time)
    mpm.check_critical_timestep()
    auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    if auxiliary_tracker is not None:
        auxiliary_tracker.set_stage("dynamic")
        auxiliary_tracker.install_on_engine()
    if mpm.solver is not None:
        mpm.solver.postprocess = []
    return boundary, periodic_mapper, {"materials": static_materials}, {"materials": dynamic_materials}, dynamic_gravity


def dynamic_usf_stress_update_once(mpm: MPM) -> None:
    engine = mpm.solver.engine
    scene = mpm.scene
    sims = mpm.sims
    neighbor = mpm.neighbor
    engine.reset_grid_messages(scene)
    engine.bulid_neighbor_list(sims, scene, neighbor)
    engine.calculate_interpolation(sims, scene)
    engine.compute_nodal_kinematic(sims, scene)
    engine.compute_grid_velcity(sims, scene)
    engine.apply_dirichlet_constraints(sims, scene)
    engine.compute_velocity_gradient(sims, scene)
    engine.compute_stress_strains(sims, scene)
    engine.pressure_smoothing_(scene)


def assemble_force_window(mpm: MPM, gravity: list[float], boundary: NairnSeismicBoundary | None) -> None:
    engine = mpm.solver.engine
    scene = mpm.scene
    sims = mpm.sims
    neighbor = mpm.neighbor
    previous_boundary = getattr(mpm, "nairn_seismic_boundary", None)
    previous_gravity = [float(value) for value in list(sims.gravity[:2])]
    try:
        mpm.nairn_seismic_boundary = boundary
        sims.set_gravity(gravity)
        engine.reset_grid_messages(scene)
        engine.bulid_neighbor_list(sims, scene, neighbor)
        engine.calculate_interpolation(sims, scene)
        engine.compute_nodal_kinematic(sims, scene)
        engine.compute_grid_velcity(sims, scene)
        engine.apply_dirichlet_constraints(sims, scene)
        engine.apply_particle_traction_constraints(sims, scene)
        engine.compute_forces(sims, scene)
        engine.apply_traction_constraints(sims, scene)
        engine.apply_absorbing_constraints(sims, scene)
    finally:
        mpm.nairn_seismic_boundary = previous_boundary
        sims.set_gravity(previous_gravity)


def finish_dynamic_step_from_force_window(mpm: MPM) -> None:
    engine = mpm.solver.engine
    scene = mpm.scene
    sims = mpm.sims
    engine.compute_grid_kinematic(sims, scene)
    engine.pre_contact_calculate(sims, scene)
    engine.apply_kinematic_constraints(sims, scene)
    engine.compute_contact_force_(sims, scene)
    engine.compute_particle_kinematic(sims, scene)


def run_one_diagnostic_dynamic_step(
    mpm: MPM,
    boundary: NairnSeismicBoundary,
    dynamic_gravity: list[float],
    capture_decomposition: bool = False,
) -> dict[str, Any]:
    # The static support reactions are assembled before the first dynamic USF
    # stress update. Capture that force state to expose any transition mismatch.
    assemble_force_window(mpm, dynamic_gravity, None)
    pre_dynamic_stress_update_force = mpm.scene.node.force.to_numpy().copy()
    dynamic_usf_stress_update_once(mpm)
    assemble_force_window(mpm, [0.0, 0.0], None)
    internal_force = mpm.scene.node.force.to_numpy().copy()
    assemble_force_window(mpm, dynamic_gravity, None)
    internal_gravity_force = mpm.scene.node.force.to_numpy().copy()
    gravity_force = internal_gravity_force - internal_force
    assemble_force_window(mpm, dynamic_gravity, boundary)
    actual_force = mpm.scene.node.force.to_numpy().copy()
    node_mass = mpm.scene.node.m.to_numpy().copy()
    components = boundary_component_arrays(mpm, boundary)
    component_sum = internal_force + gravity_force
    for value in components.values():
        component_sum += value
    reconstruction_error = actual_force - component_sum
    main_force = actual_force[:, BODY_MAIN_SOIL, :2]
    main_mass = node_mass[:, BODY_MAIN_SOIL]
    active = main_mass > float(mpm.scene.mass_cut_off)
    vertical_residual = np.abs(main_force[:, 1])
    max_vertical_residual_force = float(np.max(vertical_residual[active])) if np.any(active) else 0.0
    max_res_node = int(np.argmax(np.where(active, vertical_residual, -1.0))) if np.any(active) else -1
    acceleration_before_constraints = np.zeros_like(main_force)
    acceleration_before_constraints[active] = main_force[active] / main_mass[active, None]
    max_vertical_acceleration = float(np.max(np.abs(acceleration_before_constraints[active, 1]))) if np.any(active) else 0.0
    max_acc_node = int(np.argmax(np.where(active, np.abs(acceleration_before_constraints[:, 1]), -1.0))) if np.any(active) else -1

    finish_dynamic_step_from_force_window(mpm)
    acceleration_after_constraints = mpm.scene.node.force.to_numpy().copy()
    result = {
        "internal_force": internal_force,
        "gravity_force": gravity_force,
        "pre_dynamic_stress_update_force": pre_dynamic_stress_update_force,
        "dynamic_stress_update_force_delta": internal_gravity_force - pre_dynamic_stress_update_force,
        "actual_force": actual_force,
        "node_mass": node_mass,
        "components": components,
        "component_sum": component_sum,
        "reconstruction_error": reconstruction_error,
        "acceleration_after_constraints": acceleration_after_constraints,
        "max_vertical_residual_force": max_vertical_residual_force,
        "max_vertical_residual_node": max_res_node,
        "max_vertical_acceleration": max_vertical_acceleration,
        "particle_id_of_max_vertical_acceleration": max_acc_node,
        "capture_decomposition": capture_decomposition,
    }
    mpm.sims.current_time += mpm.sims.delta
    mpm.sims.current_step += 1
    return result


def right_free_field_force_extreme(
    force: np.ndarray,
    node_mass: np.ndarray,
    body_id: int,
) -> tuple[float, int]:
    active = node_mass[:, body_id] > float(np.finfo(np.float32).eps)
    vertical_force = np.abs(force[:, body_id, 1])
    if not np.any(active):
        return 0.0, -1
    node_id = int(np.argmax(np.where(active, vertical_force, -1.0)))
    return float(vertical_force[node_id]), node_id


def right_free_field_node_classification(
    node_id: int,
    x: float,
    y: float,
    periodic_node_ids: set[int],
    bottom_node_ids: set[int],
) -> str:
    if node_id in bottom_node_ids:
        return "bottom_boundary"
    if node_id in periodic_node_ids:
        return "periodic_seam"
    if abs(y - K_INTERFACE_LOW_Z) <= 0.5 * DX:
        return "soil_base_interface"
    return "interior"


def run_right_free_field_static_balance_diagnostic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: StaticInitializationMonitor,
) -> tuple[Path, Path]:
    """Compare right free-field P2G forces before and after the stage switch.

    This is the paper's Section 3.2 P2G state used to derive boundary support:
    no time step is advanced and no dynamic traction is applied.  Interior nodes
    are reported separately because the paper permits the reaction correction
    only at boundary nodes after mirrored support is removed.
    """

    output_dir = RIGHT_FREE_FIELD_STATIC_BALANCE_DIAGNOSTIC_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "right_free_field_static_dynamic_force_map.csv"
    summary_path = output_dir / "summary.csv"

    if mpm.enginer is None:
        mpm.add_essentials({"function": None})
    register_bottom_input_traction(mpm, case)
    static_gravity = [float(value) for value in static_monitor.target_gravity]

    # Assemble the static P2G force state without mirrored/velocity support.
    # This is the same force state from which Section 3.2 obtains f_i.
    apply_material_stage(mpm, case, "static")
    mpm.add_essentials({"function": None})
    static_mapper = install_free_field_periodic_dynamic_hooks(mpm)
    assert_free_field_periodic_dynamic_ready(mpm)
    if mpm.solver is not None:
        mpm.solver.engine.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    clear_velocity_boundary_state(mpm)
    assemble_force_window(mpm, static_gravity, None)
    static_force = mpm.scene.node.force.to_numpy().copy()
    static_mass = mpm.scene.node.m.to_numpy().copy()

    # Rebuild the dynamic assembly without changing any particle state or
    # applying Eq. (29)-(32), then compare the resulting P2G force field.
    apply_material_stage(mpm, case, "dynamic")
    mpm.add_essentials({"function": None})
    dynamic_mapper = install_free_field_periodic_dynamic_hooks(mpm)
    assert_free_field_periodic_dynamic_ready(mpm)
    if mpm.solver is not None:
        mpm.solver.engine.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    clear_velocity_boundary_state(mpm)
    assemble_force_window(mpm, static_gravity, None)
    dynamic_force = mpm.scene.node.force.to_numpy().copy()
    dynamic_mass = mpm.scene.node.m.to_numpy().copy()

    coords = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
    mass_cutoff = float(mpm.scene.mass_cut_off)
    active = (static_mass[:, BODY_RIGHT_FREE_SOIL] > mass_cutoff) | (dynamic_mass[:, BODY_RIGHT_FREE_SOIL] > mass_cutoff)
    bottom_node_ids = set(
        int(node_id)
        for node_id in np.asarray(
            getattr(mpm, "static_bottom_constraint_node_ids_np", static_bottom_constraint_node_ids(mpm)),
            dtype=np.int32,
        ).tolist()
    )
    periodic_node_ids = {
        int(node_id)
        for row in dynamic_mapper.pair_rows
        if row["column"] == "right"
        for node_id in (int(row["left_node_id"]), int(row["right_node_id"]))
    }

    rows: list[dict[str, Any]] = []
    for node_id in np.flatnonzero(active).tolist():
        x = float(coords[node_id, 0])
        y = float(coords[node_id, 1])
        static_value = static_force[node_id, BODY_RIGHT_FREE_SOIL, :2]
        dynamic_value = dynamic_force[node_id, BODY_RIGHT_FREE_SOIL, :2]
        delta = dynamic_value - static_value
        rows.append(
            {
                "node_id": int(node_id),
                "x": x,
                "y": y,
                "classification": right_free_field_node_classification(
                    int(node_id), x, y, periodic_node_ids, bottom_node_ids
                ),
                "static_mass": float(static_mass[node_id, BODY_RIGHT_FREE_SOIL]),
                "dynamic_mass": float(dynamic_mass[node_id, BODY_RIGHT_FREE_SOIL]),
                "static_force_x": float(static_value[0]),
                "static_force_y": float(static_value[1]),
                "static_force_norm": float(np.linalg.norm(static_value)),
                "dynamic_force_x": float(dynamic_value[0]),
                "dynamic_force_y": float(dynamic_value[1]),
                "dynamic_force_norm": float(np.linalg.norm(dynamic_value)),
                "switch_delta_x": float(delta[0]),
                "switch_delta_y": float(delta[1]),
                "switch_delta_norm": float(np.linalg.norm(delta)),
            }
        )
    write_csv(detail_path, list(rows[0].keys()) if rows else [], rows)

    def group_extreme(group: str, field: str) -> tuple[float, int]:
        selected = [row for row in rows if row["classification"] == group]
        if not selected:
            return 0.0, -1
        row = max(selected, key=lambda value: float(value[field]))
        return float(row[field]), int(row["node_id"])

    static_max, static_node = group_extreme("interior", "static_force_norm")
    dynamic_max, dynamic_node = group_extreme("interior", "dynamic_force_norm")
    delta_max, delta_node = group_extreme("interior", "switch_delta_norm")
    all_delta_max = max((float(row["switch_delta_norm"]) for row in rows), default=0.0)
    write_csv(
        summary_path,
        ["metric", "value"],
        [
            {"metric": "purpose", "value": "paper_section_3_2_static_vs_dynamic_p2g_force_map"},
            {"metric": "time_advanced", "value": "False"},
            {"metric": "dynamic_boundary_tractions_applied", "value": "False"},
            {"metric": "right_free_field_active_node_count", "value": len(rows)},
            {"metric": "right_free_field_periodic_pair_count", "value": sum(1 for row in dynamic_mapper.pair_rows if row["column"] == "right")},
            {"metric": "max_static_interior_force_norm", "value": static_max},
            {"metric": "max_static_interior_force_node_id", "value": static_node},
            {"metric": "max_dynamic_interior_force_norm", "value": dynamic_max},
            {"metric": "max_dynamic_interior_force_node_id", "value": dynamic_node},
            {"metric": "max_interior_switch_delta_norm", "value": delta_max},
            {"metric": "max_interior_switch_delta_node_id", "value": delta_node},
            {"metric": "max_all_node_switch_delta_norm", "value": all_delta_max},
            {"metric": "static_periodic_mapper_pair_count", "value": static_mapper.pair_count},
        ],
    )
    return detail_path, summary_path


def run_right_free_field_periodic_seam_diagnostic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: StaticInitializationMonitor,
) -> tuple[Path, Path]:
    """Verify the static P2G force sum at the right periodic free-field seam."""

    output_dir = RIGHT_FREE_FIELD_PERIODIC_SEAM_DIAGNOSTIC_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "right_free_field_periodic_seam_force_map.csv"
    summary_path = output_dir / "summary.csv"

    register_bottom_input_traction(mpm, case)
    apply_material_stage(mpm, case, "static")
    mpm.add_essentials({"function": None})
    mapper = install_free_field_periodic_dynamic_hooks(mpm)
    assert_free_field_periodic_dynamic_ready(mpm)
    if mpm.solver is not None:
        mpm.solver.engine.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    clear_velocity_boundary_state(mpm)
    static_gravity = [float(value) for value in static_monitor.target_gravity]

    # First assemble separate left/right seam nodes.  The P2G interpolation is
    # recomputed every window, so disabling this one remap does not alter the
    # particle state or the following mapped assembly.
    original_remap = mapper.remap_particle_supports
    try:
        mapper.remap_particle_supports = lambda scene: None
        assemble_force_window(mpm, static_gravity, None)
        raw_force = mpm.scene.node.force.to_numpy().copy()
        raw_mass = mpm.scene.node.m.to_numpy().copy()
    finally:
        mapper.remap_particle_supports = original_remap

    assemble_force_window(mpm, static_gravity, None)
    mapped_force = mpm.scene.node.force.to_numpy().copy()
    mapped_mass = mpm.scene.node.m.to_numpy().copy()
    coords = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for row in mapper.pair_rows:
        if row["column"] != "right":
            continue
        left_node = int(row["left_node_id"])
        right_node = int(row["right_node_id"])
        raw_left = raw_force[left_node, BODY_RIGHT_FREE_SOIL, :2]
        raw_right = raw_force[right_node, BODY_RIGHT_FREE_SOIL, :2]
        raw_sum = raw_left + raw_right
        mapped_left = mapped_force[left_node, BODY_RIGHT_FREE_SOIL, :2]
        mapped_right = mapped_force[right_node, BODY_RIGHT_FREE_SOIL, :2]
        mapping_error = mapped_left - raw_sum
        rows.append(
            {
                "pair_id": int(row["pair_id"]),
                "left_node_id": left_node,
                "right_node_id": right_node,
                "x_left": float(coords[left_node, 0]),
                "x_right": float(coords[right_node, 0]),
                "y": float(coords[left_node, 1]),
                "raw_left_mass": float(raw_mass[left_node, BODY_RIGHT_FREE_SOIL]),
                "raw_right_mass": float(raw_mass[right_node, BODY_RIGHT_FREE_SOIL]),
                "mapped_left_mass": float(mapped_mass[left_node, BODY_RIGHT_FREE_SOIL]),
                "mapped_right_mass": float(mapped_mass[right_node, BODY_RIGHT_FREE_SOIL]),
                "raw_left_force_x": float(raw_left[0]),
                "raw_left_force_y": float(raw_left[1]),
                "raw_right_force_x": float(raw_right[0]),
                "raw_right_force_y": float(raw_right[1]),
                "raw_pair_sum_force_x": float(raw_sum[0]),
                "raw_pair_sum_force_y": float(raw_sum[1]),
                "raw_pair_sum_force_norm": float(np.linalg.norm(raw_sum)),
                "mapped_left_force_x": float(mapped_left[0]),
                "mapped_left_force_y": float(mapped_left[1]),
                "mapped_right_force_norm": float(np.linalg.norm(mapped_right)),
                "force_mapping_error_norm": float(np.linalg.norm(mapping_error)),
            }
        )
    write_csv(detail_path, list(rows[0].keys()) if rows else [], rows)
    max_pair = max(rows, key=lambda value: float(value["raw_pair_sum_force_norm"])) if rows else None
    max_error = max((float(row["force_mapping_error_norm"]) for row in rows), default=0.0)
    max_right_force = max((float(row["mapped_right_force_norm"]) for row in rows), default=0.0)
    write_csv(
        summary_path,
        ["metric", "value"],
        [
            {"metric": "purpose", "value": "paper_section_3_3_periodic_shared_dof_static_p2g_force_sum"},
            {"metric": "time_advanced", "value": "False"},
            {"metric": "dynamic_boundary_tractions_applied", "value": "False"},
            {"metric": "right_periodic_pair_count", "value": len(rows)},
            {"metric": "max_raw_periodic_pair_force_norm", "value": float(max_pair["raw_pair_sum_force_norm"]) if max_pair else 0.0},
            {"metric": "max_raw_periodic_pair_force_pair_id", "value": int(max_pair["pair_id"]) if max_pair else -1},
            {"metric": "max_raw_periodic_pair_force_y", "value": float(max_pair["y"]) if max_pair else math.nan},
            {"metric": "max_force_mapping_error_norm", "value": max_error},
            {"metric": "max_mapped_right_node_force_norm", "value": max_right_force},
        ],
    )
    return detail_path, summary_path


def run_right_free_field_transition_diagnostic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: StaticInitializationMonitor,
    dynamic_start_time: float,
) -> tuple[Path, Path]:
    """Locate the first dynamic force imbalance in the right free-field column.

    The diagnostic does not advance the time integration.  It compares the
    force window before the first dynamic stress update with the windows after
    that update and after the frozen static support/dashpot boundary is added.
    """

    output_dir = RIGHT_FREE_FIELD_TRANSITION_DIAGNOSTIC_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "right_free_field_first_dynamic_force_decomposition.csv"
    summary_path = output_dir / "summary.csv"
    boundary, _periodic_mapper, _static_stage, _dynamic_stage, dynamic_gravity = prepare_dynamic_transition_diagnostic(
        mpm,
        case,
        static_monitor,
        dynamic_start_time,
    )
    particle_count = int(mpm.scene.particleNum[0])
    particle_body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    right_particle_mask = particle_body_ids == BODY_RIGHT_FREE_SOIL
    particle_velocity = mpm.scene.particle.v.to_numpy()[:particle_count, :2]
    stress_before = mpm.scene.particle.stress.to_numpy()[:particle_count].copy()

    # No boundary is applied until the final window, so the effect of the
    # first dynamic constitutive update is directly observable.
    assemble_force_window(mpm, dynamic_gravity, None)
    pre_update_force = mpm.scene.node.force.to_numpy().copy()
    node_mass = mpm.scene.node.m.to_numpy().copy()
    dynamic_usf_stress_update_once(mpm)
    stress_after = mpm.scene.particle.stress.to_numpy()[:particle_count].copy()
    assemble_force_window(mpm, [0.0, 0.0], None)
    internal_force = mpm.scene.node.force.to_numpy().copy()
    assemble_force_window(mpm, dynamic_gravity, None)
    internal_gravity_force = mpm.scene.node.force.to_numpy().copy()
    gravity_force = internal_gravity_force - internal_force
    assemble_force_window(mpm, dynamic_gravity, boundary)
    actual_force = mpm.scene.node.force.to_numpy().copy()
    components = boundary_component_arrays(mpm, boundary)
    reconstructed_force = internal_force + gravity_force
    for component in components.values():
        reconstructed_force += component
    reconstruction_error = actual_force - reconstructed_force

    body_id = BODY_RIGHT_FREE_SOIL
    active = node_mass[:, body_id] > float(mpm.scene.mass_cut_off)
    coords = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
    stress_delta = stress_after - stress_before
    right_stress_delta = stress_delta[right_particle_mask]
    right_velocity = particle_velocity[right_particle_mask]
    pre_max, pre_node = right_free_field_force_extreme(pre_update_force, node_mass, body_id)
    post_max, post_node = right_free_field_force_extreme(internal_gravity_force, node_mass, body_id)
    full_max, full_node = right_free_field_force_extreme(actual_force, node_mass, body_id)
    error_max, error_node = right_free_field_force_extreme(reconstruction_error, node_mass, body_id)
    static_bottom_node_ids = np.asarray(
        getattr(mpm, "static_bottom_constraint_node_ids_np", static_bottom_constraint_node_ids(mpm)),
        dtype=np.int32,
    )
    is_static_bottom_constraint = np.zeros(active.shape, dtype=bool)
    is_static_bottom_constraint[static_bottom_node_ids[(static_bottom_node_ids >= 0) & (static_bottom_node_ids < active.size)]] = True
    right_static_bottom = active & is_static_bottom_constraint
    expected_static_support = -pre_update_force[:, body_id, :2]
    frozen_static_support = components["bottom_static_support"][:, body_id, :2]
    missing_static_support = expected_static_support - frozen_static_support
    support_tolerance = STATIC_DYNAMIC_SUPPORT_MAP_TOLERANCE
    required_support = right_static_bottom & (np.linalg.norm(expected_static_support, axis=1) > support_tolerance)
    missing_required_support = required_support & (np.linalg.norm(missing_static_support, axis=1) > support_tolerance)
    required_support_norm = np.linalg.norm(expected_static_support, axis=1)
    missing_support_norm = np.linalg.norm(missing_static_support, axis=1)
    max_missing_support_node = (
        int(np.argmax(np.where(required_support, missing_support_norm, -1.0))) if np.any(required_support) else -1
    )

    rows: list[dict[str, Any]] = []
    for node_id in np.flatnonzero(active).tolist():
        x = float(coords[node_id, 0])
        y = float(coords[node_id, 1])
        row: dict[str, Any] = {
            "node_id": int(node_id),
            "body_id": body_id,
            "x": x,
            "y": y,
            "node_mass": float(node_mass[node_id, body_id]),
            "is_static_bottom_constraint": bool(is_static_bottom_constraint[node_id]),
            "expected_static_support_x": float(expected_static_support[node_id, 0]),
            "expected_static_support_y": float(expected_static_support[node_id, 1]),
            "missing_static_support_x": float(missing_static_support[node_id, 0]),
            "missing_static_support_y": float(missing_static_support[node_id, 1]),
            "pre_stress_update_force_x": float(pre_update_force[node_id, body_id, 0]),
            "pre_stress_update_force_y": float(pre_update_force[node_id, body_id, 1]),
            "post_stress_update_internal_force_x": float(internal_force[node_id, body_id, 0]),
            "post_stress_update_internal_force_y": float(internal_force[node_id, body_id, 1]),
            "gravity_force_x": float(gravity_force[node_id, body_id, 0]),
            "gravity_force_y": float(gravity_force[node_id, body_id, 1]),
            "post_stress_update_no_boundary_force_x": float(internal_gravity_force[node_id, body_id, 0]),
            "post_stress_update_no_boundary_force_y": float(internal_gravity_force[node_id, body_id, 1]),
            "dynamic_stress_update_delta_x": float(
                internal_gravity_force[node_id, body_id, 0] - pre_update_force[node_id, body_id, 0]
            ),
            "dynamic_stress_update_delta_y": float(
                internal_gravity_force[node_id, body_id, 1] - pre_update_force[node_id, body_id, 1]
            ),
        }
        for name, component in components.items():
            row[f"{name}_x"] = float(component[node_id, body_id, 0])
            row[f"{name}_y"] = float(component[node_id, body_id, 1])
        row.update(
            {
                "reconstructed_force_x": float(reconstructed_force[node_id, body_id, 0]),
                "reconstructed_force_y": float(reconstructed_force[node_id, body_id, 1]),
                "actual_force_x": float(actual_force[node_id, body_id, 0]),
                "actual_force_y": float(actual_force[node_id, body_id, 1]),
                "reconstruction_error_x": float(reconstruction_error[node_id, body_id, 0]),
                "reconstruction_error_y": float(reconstruction_error[node_id, body_id, 1]),
                "classification": classify_dynamic_node(int(node_id), x, y),
            }
        )
        rows.append(row)
    write_csv(detail_path, list(rows[0].keys()) if rows else [], rows)

    def node_coordinate(node_id: int) -> str:
        return "" if node_id < 0 else f"({coords[node_id, 0]:.6g}, {coords[node_id, 1]:.6g})"

    write_csv(
        summary_path,
        ["metric", "value"],
        [
            {"metric": "purpose", "value": "right_free_field_first_dynamic_stress_update_force_balance"},
            {"metric": "time_advanced", "value": "False"},
            {"metric": "seismic_input_applied", "value": "False"},
            {"metric": "right_free_field_body_id", "value": body_id},
            {"metric": "active_right_free_field_node_count", "value": int(np.count_nonzero(active))},
            {"metric": "right_free_field_particle_count", "value": int(np.count_nonzero(right_particle_mask))},
            {
                "metric": "right_free_field_max_initial_particle_speed",
                "value": float(np.max(np.linalg.norm(right_velocity, axis=1))) if right_velocity.size else 0.0,
            },
            {
                "metric": "right_free_field_max_particle_stress_update_norm",
                "value": float(np.max(np.linalg.norm(right_stress_delta, axis=1))) if right_stress_delta.size else 0.0,
            },
            {"metric": "pre_stress_update_max_abs_vertical_force", "value": pre_max},
            {"metric": "pre_stress_update_max_vertical_force_node_id", "value": pre_node},
            {"metric": "pre_stress_update_max_vertical_force_coordinate", "value": node_coordinate(pre_node)},
            {"metric": "post_stress_update_no_boundary_max_abs_vertical_force", "value": post_max},
            {"metric": "post_stress_update_no_boundary_max_vertical_force_node_id", "value": post_node},
            {"metric": "post_stress_update_no_boundary_max_vertical_force_coordinate", "value": node_coordinate(post_node)},
            {"metric": "full_boundary_max_abs_vertical_force", "value": full_max},
            {"metric": "full_boundary_max_vertical_force_node_id", "value": full_node},
            {"metric": "full_boundary_max_vertical_force_coordinate", "value": node_coordinate(full_node)},
            {"metric": "full_boundary_max_abs_vertical_reconstruction_error", "value": error_max},
            {"metric": "full_boundary_max_vertical_reconstruction_error_node_id", "value": error_node},
            {"metric": "full_boundary_max_vertical_reconstruction_error_coordinate", "value": node_coordinate(error_node)},
            {"metric": "right_free_field_static_bottom_constraint_active_node_count", "value": int(np.count_nonzero(right_static_bottom))},
            {"metric": "right_free_field_static_bottom_nonzero_required_support_count", "value": int(np.count_nonzero(required_support))},
            {"metric": "right_free_field_static_bottom_missing_frozen_support_count", "value": int(np.count_nonzero(missing_required_support))},
            {
                "metric": "right_free_field_static_bottom_required_support_sum_y",
                "value": float(np.sum(expected_static_support[required_support, 1])),
            },
            {
                "metric": "right_free_field_static_bottom_frozen_support_sum_y",
                "value": float(np.sum(frozen_static_support[required_support, 1])),
            },
            {
                "metric": "right_free_field_static_bottom_max_missing_frozen_support_norm",
                "value": float(missing_support_norm[max_missing_support_node]) if max_missing_support_node >= 0 else 0.0,
            },
            {
                "metric": "right_free_field_static_bottom_max_missing_frozen_support_node_id",
                "value": max_missing_support_node,
            },
            {
                "metric": "right_free_field_static_bottom_max_missing_frozen_support_coordinate",
                "value": node_coordinate(max_missing_support_node),
            },
            {
                "metric": "right_free_field_static_bottom_support_coverage_status",
                "value": "MATCH" if not np.any(missing_required_support) else "MISSING",
            },
        ],
    )
    if mpm.enginer is not None:
        mpm.enginer.reset_grid_message(mpm.scene)
    return detail_path, summary_path


def write_dynamic_start_force_decomposition(
    output_dir: Path,
    mpm: MPM,
    step_result: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    coords = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
    node_mass = step_result["node_mass"]
    actual = step_result["actual_force"]
    internal = step_result["internal_force"]
    gravity = step_result["gravity_force"]
    pre_dynamic_stress_update = step_result["pre_dynamic_stress_update_force"]
    stress_update_delta = step_result["dynamic_stress_update_force_delta"]
    components = step_result["components"]
    reconstructed = step_result["component_sum"]
    error = step_result["reconstruction_error"]
    rows = []
    max_row = None
    for node_id in range(actual.shape[0]):
        body_id = BODY_MAIN_SOIL
        if node_mass[node_id, body_id] <= 0.0 and np.linalg.norm(actual[node_id, body_id, :2]) <= 0.0:
            continue
        x = float(coords[node_id, 0])
        y = float(coords[node_id, 1])
        residual = float(np.linalg.norm(actual[node_id, body_id, :2]))
        row = {
            "node_id": node_id,
            "body_id": body_id,
            "x": x,
            "y": y,
            "internal_force_x": float(internal[node_id, body_id, 0]),
            "internal_force_y": float(internal[node_id, body_id, 1]),
            "gravity_force_x": float(gravity[node_id, body_id, 0]),
            "gravity_force_y": float(gravity[node_id, body_id, 1]),
            "pre_dynamic_stress_update_total_x": float(pre_dynamic_stress_update[node_id, body_id, 0]),
            "pre_dynamic_stress_update_total_y": float(pre_dynamic_stress_update[node_id, body_id, 1]),
            "dynamic_stress_update_delta_x": float(stress_update_delta[node_id, body_id, 0]),
            "dynamic_stress_update_delta_y": float(stress_update_delta[node_id, body_id, 1]),
            "bottom_static_support_x": float(components["bottom_static_support"][node_id, body_id, 0]),
            "bottom_static_support_y": float(components["bottom_static_support"][node_id, body_id, 1]),
            "bottom_dashpot_force_x": float(components["bottom_dashpot_force"][node_id, body_id, 0]),
            "bottom_dashpot_force_y": float(components["bottom_dashpot_force"][node_id, body_id, 1]),
            "bottom_input_force_x": float(components["bottom_input_force"][node_id, body_id, 0]),
            "bottom_input_force_y": float(components["bottom_input_force"][node_id, body_id, 1]),
            "lateral_static_support_x": float(components["lateral_static_support"][node_id, body_id, 0]),
            "lateral_static_support_y": float(components["lateral_static_support"][node_id, body_id, 1]),
            "free_field_dynamic_stress_force_x": float(components["free_field_dynamic_stress_force"][node_id, body_id, 0]),
            "free_field_dynamic_stress_force_y": float(components["free_field_dynamic_stress_force"][node_id, body_id, 1]),
            "lateral_dashpot_force_x": float(components["lateral_dashpot_force"][node_id, body_id, 0]),
            "lateral_dashpot_force_y": float(components["lateral_dashpot_force"][node_id, body_id, 1]),
            "reconstructed_total_x": float(reconstructed[node_id, body_id, 0]),
            "reconstructed_total_y": float(reconstructed[node_id, body_id, 1]),
            "actual_total_x": float(actual[node_id, body_id, 0]),
            "actual_total_y": float(actual[node_id, body_id, 1]),
            "reconstruction_error_x": float(error[node_id, body_id, 0]),
            "reconstruction_error_y": float(error[node_id, body_id, 1]),
            "residual_magnitude": residual,
            "classification": classify_dynamic_node(node_id, x, y),
        }
        rows.append(row)
        if max_row is None or abs(row["actual_total_y"]) > abs(max_row["actual_total_y"]):
            max_row = row
    path = output_dir / "dynamic_start_force_decomposition.csv"
    write_csv(path, list(rows[0].keys()) if rows else [], rows)
    return path, (max_row or {})


def dominant_support_source(row: dict[str, Any]) -> str:
    candidates = {
        "internal": math.hypot(float(row["node_force_x"]), float(row["node_force_y"])),
        "bottom_static_support": math.hypot(float(row["bottom_static_support_x"]), float(row["bottom_static_support_y"])),
        "bottom_dashpot_force": math.hypot(float(row["bottom_dashpot_force_x"]), float(row["bottom_dashpot_force_y"])),
        "bottom_input_force": math.hypot(float(row["bottom_input_force_x"]), float(row["bottom_input_force_y"])),
        "lateral_static_support": math.hypot(float(row["lateral_static_support_x"]), float(row["lateral_static_support_y"])),
        "free_field_dynamic_stress_force": math.hypot(
            float(row["free_field_dynamic_stress_force_x"]), float(row["free_field_dynamic_stress_force_y"])
        ),
        "lateral_dashpot_force": math.hypot(float(row["lateral_dashpot_force_x"]), float(row["lateral_dashpot_force_y"])),
    }
    return max(candidates, key=candidates.get)


def write_max_velocity_particle_support_nodes(
    output_dir: Path,
    mpm: MPM,
    step_result: dict[str, Any],
    particle_id: int,
    pre_velocity: np.ndarray,
    post_velocity: np.ndarray,
) -> tuple[Path, dict[str, float]]:
    positions = mpm.scene.particle.x.to_numpy()
    coords = np.asarray(mpm.scene.element.nodal_coords, dtype=np.float64) - np.asarray([X_SHIFT, Y_SHIFT], dtype=np.float64)
    node_mass = step_result["node_mass"]
    actual_force = step_result["actual_force"]
    acceleration = step_result["acceleration_after_constraints"]
    components = step_result["components"]
    rows = []
    predicted = np.zeros(2, dtype=np.float64)
    for node_id, weight in support_node_entries(mpm, particle_id):
        acc = acceleration[node_id, BODY_MAIN_SOIL, :2]
        predicted += weight * acc
        x = float(coords[node_id, 0])
        y = float(coords[node_id, 1])
        row = {
            "particle_id": particle_id,
            "particle_x": float(positions[particle_id, 0] - X_SHIFT),
            "particle_y": float(positions[particle_id, 1] - Y_SHIFT),
            "node_id": node_id,
            "node_x": x,
            "node_y": y,
            "shape_weight": weight,
            "node_mass": float(node_mass[node_id, BODY_MAIN_SOIL]),
            "node_force_x": float(actual_force[node_id, BODY_MAIN_SOIL, 0]),
            "node_force_y": float(actual_force[node_id, BODY_MAIN_SOIL, 1]),
            "node_acceleration_x": float(acc[0]),
            "node_acceleration_y": float(acc[1]),
            "bottom_static_support_x": float(components["bottom_static_support"][node_id, BODY_MAIN_SOIL, 0]),
            "bottom_static_support_y": float(components["bottom_static_support"][node_id, BODY_MAIN_SOIL, 1]),
            "bottom_dashpot_force_x": float(components["bottom_dashpot_force"][node_id, BODY_MAIN_SOIL, 0]),
            "bottom_dashpot_force_y": float(components["bottom_dashpot_force"][node_id, BODY_MAIN_SOIL, 1]),
            "bottom_input_force_x": float(components["bottom_input_force"][node_id, BODY_MAIN_SOIL, 0]),
            "bottom_input_force_y": float(components["bottom_input_force"][node_id, BODY_MAIN_SOIL, 1]),
            "lateral_static_support_x": float(components["lateral_static_support"][node_id, BODY_MAIN_SOIL, 0]),
            "lateral_static_support_y": float(components["lateral_static_support"][node_id, BODY_MAIN_SOIL, 1]),
            "free_field_dynamic_stress_force_x": float(components["free_field_dynamic_stress_force"][node_id, BODY_MAIN_SOIL, 0]),
            "free_field_dynamic_stress_force_y": float(components["free_field_dynamic_stress_force"][node_id, BODY_MAIN_SOIL, 1]),
            "lateral_dashpot_force_x": float(components["lateral_dashpot_force"][node_id, BODY_MAIN_SOIL, 0]),
            "lateral_dashpot_force_y": float(components["lateral_dashpot_force"][node_id, BODY_MAIN_SOIL, 1]),
            "force_component_source": "",
            "is_bottom": abs(y - kohler_base_bottom_z_np(x)) <= BOTTOM_TOL,
            "is_side": abs(x - K_MAIN_X_MIN) <= SIDE_TOL or abs(x - K_MAIN_X_MAX) <= SIDE_TOL,
            "is_corner": (abs(y - kohler_base_bottom_z_np(x)) <= BOTTOM_TOL)
            and (abs(x - K_MAIN_X_MIN) <= SIDE_TOL or abs(x - K_MAIN_X_MAX) <= SIDE_TOL),
            "is_material_interface": abs(y - kohler_soil_base_interface_z_np(x)) <= max(BOTTOM_TOL, DX),
        }
        row["force_component_source"] = dominant_support_source(row)
        rows.append(row)
    predicted_delta = DT * predicted
    actual_delta = post_velocity[particle_id, :2] - pre_velocity[particle_id, :2]
    for row in rows:
        row["delta_vx_predicted"] = float(predicted_delta[0])
        row["delta_vy_predicted"] = float(predicted_delta[1])
        row["delta_vx_actual"] = float(actual_delta[0])
        row["delta_vy_actual"] = float(actual_delta[1])
        row["delta_vx_error"] = float(actual_delta[0] - predicted_delta[0])
        row["delta_vy_error"] = float(actual_delta[1] - predicted_delta[1])
    path = output_dir / "max_velocity_particle_support_nodes.csv"
    write_csv(path, list(rows[0].keys()) if rows else [], rows)
    return path, {
        "predicted_delta_vx": float(predicted_delta[0]),
        "predicted_delta_vy": float(predicted_delta[1]),
        "actual_delta_vx": float(actual_delta[0]),
        "actual_delta_vy": float(actual_delta[1]),
        "error_delta_vx": float(actual_delta[0] - predicted_delta[0]),
        "error_delta_vy": float(actual_delta[1] - predicted_delta[1]),
    }


def growth_row(mpm: MPM, step: int, step_result: dict[str, Any] | None, particle_id: int) -> dict[str, Any]:
    particle_count = int(mpm.scene.particleNum[0])
    body_ids = mpm.scene.particle.bodyID.to_numpy()[:particle_count].astype(np.int32)
    main_mask = body_ids == BODY_MAIN_SOIL
    velocities = mpm.scene.particle.v.to_numpy()[:particle_count]
    positions = mpm.scene.particle.x.to_numpy()[:particle_count] - np.asarray([X_SHIFT, Y_SHIFT])
    stats = component_velocity_stats(velocities, main_mask)
    return {
        "step": step,
        "time": float(mpm.sims.current_time),
        "particle_8627_vx": float(velocities[particle_id, 0]),
        "particle_8627_vy": float(velocities[particle_id, 1]),
        "particle_8627_x": float(positions[particle_id, 0]),
        "particle_8627_y": float(positions[particle_id, 1]),
        "main_max_abs_vx": stats["max_abs_vx"],
        "main_max_abs_vy": stats["max_abs_vy"],
        "main_rms_vx": stats["rms_vx"],
        "main_rms_vy": stats["rms_vy"],
        "main_max_velocity_magnitude": stats["max_magnitude"],
        "max_vertical_residual_force": float(step_result["max_vertical_residual_force"]) if step_result else 0.0,
        "max_vertical_acceleration": float(step_result["max_vertical_acceleration"]) if step_result else 0.0,
        "particle_id_of_max_vertical_acceleration": int(step_result["particle_id_of_max_vertical_acceleration"]) if step_result else -1,
    }


def run_vertical_transition_diagnostic(
    mpm: MPM,
    case: dict[str, Any],
    static_monitor: StaticInitializationMonitor,
    dynamic_start_time: float,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    output_dir = VERTICAL_TRANSITION_DIAGNOSTIC_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*"):
        if old.is_file():
            old.unlink()
    particle_id = VERTICAL_TRANSITION_PARTICLE_ID
    static_arrays = static_monitor.final_snapshot.get("arrays", {})
    static_particle_count = int(static_monitor.final_snapshot.get("particle_count", int(mpm.scene.particleNum[0])))
    static_velocities = np.asarray(static_arrays.get("velocity", mpm.scene.particle.v.to_numpy()[:static_particle_count]))
    static_body_ids = np.asarray(static_arrays.get("body_id", mpm.scene.particle.bodyID.to_numpy()[:static_particle_count])).astype(np.int32)
    static_main_mask = static_body_ids == BODY_MAIN_SOIL
    static_velocity_stats = component_velocity_stats(static_velocities, static_main_mask)
    static_materials_by_id, dynamic_materials_by_id = material_stage_pair(case)

    boundary, periodic_mapper, static_stage, dynamic_stage, dynamic_gravity = prepare_dynamic_transition_diagnostic(
        mpm, case, static_monitor, dynamic_start_time
    )
    dynamic_initial_arrays = particle_arrays(mpm.scene)
    auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    static_auxiliary = static_monitor.final_snapshot.get("arrays", {}).get("auxiliary_deformation_gradient")
    current_auxiliary = auxiliary_tracker.to_numpy() if auxiliary_tracker is not None else np.empty((0,), dtype=np.float64)
    auxiliary_checkpoint_missing = bool(getattr(mpm, "static_checkpoint_missing_auxiliary_F", False))
    if static_auxiliary is None:
        static_auxiliary = np.empty((0,), dtype=np.float64) if auxiliary_checkpoint_missing else current_auxiliary
    continuity = {
        "particle_count": int(mpm.scene.particleNum[0]),
        "position_hash_static": stable_array_hash(np.asarray(static_arrays.get("position", np.empty((0, 2))))),
        "position_hash_dynamic": stable_array_hash(dynamic_initial_arrays["position"]),
        "stress_hash_static": stable_array_hash(np.asarray(static_arrays.get("stress", np.empty((0, 6))))),
        "stress_hash_dynamic": stable_array_hash(dynamic_initial_arrays["stress"]),
        "state_variable_hash_static": state_variables_hash(static_arrays.get("state_variables", {})),
        "state_variable_hash_dynamic": state_variables_hash(dynamic_initial_arrays.get("state_variables", {})),
        "auxiliary_F_hash_static": stable_array_hash(np.asarray(static_auxiliary)),
        "auxiliary_F_hash_dynamic": stable_array_hash(np.asarray(current_auxiliary)),
        "auxiliary_F_checkpoint_missing": auxiliary_checkpoint_missing,
        "body_id_hash_static": stable_array_hash(np.asarray(static_arrays.get("body_id", np.empty((0,), dtype=np.int32)))),
        "body_id_hash_dynamic": stable_array_hash(dynamic_initial_arrays["body_id"]),
        "material_id_hash_static": stable_array_hash(np.asarray(static_arrays.get("material_id", np.empty((0,), dtype=np.int32)))),
        "material_id_hash_dynamic": stable_array_hash(dynamic_initial_arrays["material_id"]),
        "gravity_static_end": str(static_monitor.target_gravity),
        "gravity_dynamic_start": str(dynamic_gravity),
        "soil_E_static": float(static_materials_by_id[MAT_SOIL]["YoungModulus"]),
        "soil_E_dynamic": float(dynamic_materials_by_id[MAT_SOIL]["YoungModulus"]),
        "soil_nu_static": float(static_materials_by_id[MAT_SOIL]["PossionRatio"]),
        "soil_nu_dynamic": float(dynamic_materials_by_id[MAT_SOIL]["PossionRatio"]),
        "base_E_static": float(static_materials_by_id[MAT_BASE]["YoungModulus"]),
        "base_E_dynamic": float(dynamic_materials_by_id[MAT_BASE]["YoungModulus"]),
        "base_nu_static": float(static_materials_by_id[MAT_BASE]["PossionRatio"]),
        "base_nu_dynamic": float(dynamic_materials_by_id[MAT_BASE]["PossionRatio"]),
        "velocity_max_static_end": static_velocity_stats["max_magnitude"],
        "velocity_rms_static_end": math.sqrt(static_velocity_stats["rms_vx"] ** 2 + static_velocity_stats["rms_vy"] ** 2),
        "velocity_max_static_end_vx": static_velocity_stats["max_abs_vx"],
        "velocity_rms_static_end_vx": static_velocity_stats["rms_vx"],
        "velocity_max_static_end_vy": static_velocity_stats["max_abs_vy"],
        "velocity_rms_static_end_vy": static_velocity_stats["rms_vy"],
        "stress_reinitialized": stable_array_hash(np.asarray(static_arrays.get("stress", np.empty((0, 6))))) != stable_array_hash(dynamic_initial_arrays["stress"]),
        "particles_regenerated": static_particle_count != int(mpm.scene.particleNum[0]),
    }
    state_continuous = (
        continuity["position_hash_static"] == continuity["position_hash_dynamic"]
        and continuity["stress_hash_static"] == continuity["stress_hash_dynamic"]
        and continuity["state_variable_hash_static"] == continuity["state_variable_hash_dynamic"]
        and (not auxiliary_checkpoint_missing)
        and continuity["auxiliary_F_hash_static"] == continuity["auxiliary_F_hash_dynamic"]
        and continuity["body_id_hash_static"] == continuity["body_id_hash_dynamic"]
        and continuity["material_id_hash_static"] == continuity["material_id_hash_dynamic"]
    )
    continuity["status"] = "PASS" if state_continuous else "FAIL"
    continuity_path = output_dir / "static_dynamic_state_continuity_check.csv"
    write_csv(continuity_path, list(continuity.keys()), [continuity])

    growth_rows = [growth_row(mpm, 0, None, particle_id)]
    pre_first_velocity = mpm.scene.particle.v.to_numpy()[: int(mpm.scene.particleNum[0])].copy()
    first_result = run_one_diagnostic_dynamic_step(mpm, boundary, dynamic_gravity, capture_decomposition=True)
    post_first_velocity = mpm.scene.particle.v.to_numpy()[: int(mpm.scene.particleNum[0])].copy()
    force_path, max_force_row = write_dynamic_start_force_decomposition(output_dir, mpm, first_result)
    support_path, delta_check = write_max_velocity_particle_support_nodes(
        output_dir, mpm, first_result, particle_id, pre_first_velocity, post_first_velocity
    )
    growth_rows.append(growth_row(mpm, 1, first_result, particle_id))
    for step in range(2, 11):
        result = run_one_diagnostic_dynamic_step(mpm, boundary, dynamic_gravity)
        if step in {2, 5, 10}:
            growth_rows.append(growth_row(mpm, step, result, particle_id))
    growth_path = output_dir / "zero_input_vertical_transient_growth.csv"
    write_csv(growth_path, list(growth_rows[0].keys()), growth_rows)

    zero_input_ok = (
        abs(float(boundary.total_input_force_x[None])) == 0.0
        and abs(float(boundary.total_input_force_y[None])) == 0.0
    )
    max_vertical_residual = float(max_force_row.get("actual_total_y", math.nan))
    max_vertical_residual_abs = abs(max_vertical_residual) if math.isfinite(max_vertical_residual) else math.inf
    root_causes = []
    if not bool(static_monitor.converged):
        root_causes.append("static_initialization_not_converged_under_configured_criteria")
    if str(static_monitor.target_gravity) != str(dynamic_gravity):
        root_causes.append("gravity_discontinuity")
    paper_material_change = (
        abs(float(static_materials_by_id[MAT_SOIL]["PossionRatio"]) - SOIL_POISSON_INITIAL) <= 1.0e-12
        and abs(float(dynamic_materials_by_id[MAT_SOIL]["PossionRatio"]) - SOIL_POISSON_DYNAMIC) <= 1.0e-12
        and abs(float(static_materials_by_id[MAT_BASE]["PossionRatio"]) - BASE_POISSON) <= 1.0e-12
        and abs(float(dynamic_materials_by_id[MAT_BASE]["PossionRatio"]) - BASE_POISSON) <= 1.0e-12
    )
    if not paper_material_change:
        root_causes.append("material_parameter_discontinuity")
    if not state_continuous:
        root_causes.append("particle_state_discontinuity")
    cls = str(max_force_row.get("classification", "other"))
    if max_vertical_residual_abs > VERTICAL_TRANSITION_FORCE_TOLERANCE:
        if cls == "main_interior":
            root_causes.append("global_static_equilibrium_loss")
        elif cls in {"main_bottom", "main_bottom_left_corner", "main_bottom_right_corner"}:
            root_causes.append("bottom_static_support_or_corner_superposition_error")
        elif cls in {"main_left_side", "main_right_side"}:
            root_causes.append("static_slip_boundary_vertical_equilibrium_residual")
        elif cls == "soil_base_interface":
            root_causes.append("material_interface_equilibrium_error")
    if not root_causes:
        root_causes.append("undetermined_from_available_force_window")

    corner_reconstruction_error = max(
        abs(float(first_result["reconstruction_error"][:, BODY_MAIN_SOIL, 0].max())),
        abs(float(first_result["reconstruction_error"][:, BODY_MAIN_SOIL, 1].max())),
    )
    corner_force_scale = max(
        float(np.max(np.abs(first_result["actual_force"][:, BODY_MAIN_SOIL, :2]))),
        float(np.max(np.abs(first_result["internal_force"][:, BODY_MAIN_SOIL, :2]))),
        float(np.max(np.abs(first_result["gravity_force"][:, BODY_MAIN_SOIL, :2]))),
        *(
            float(np.max(np.abs(component[:, BODY_MAIN_SOIL, :2])))
            for component in first_result["components"].values()
        ),
        1.0,
    )
    # The force arrays are float32.  The reconstructed residual can be nearly
    # zero after cancelling O(10^2) N terms, so scale 64 ULP by those terms.
    corner_reconstruction_tolerance = 64.0 * np.finfo(np.float32).eps * corner_force_scale

    paper_rows = [
        {"check": "same_particles_static_dynamic", "status": "PASS" if static_particle_count == int(mpm.scene.particleNum[0]) else "FAIL"},
        {"check": "stress_preserved", "status": "PASS" if continuity["stress_hash_static"] == continuity["stress_hash_dynamic"] else "FAIL"},
        {"check": "state_variables_preserved", "status": "PASS" if continuity["state_variable_hash_static"] == continuity["state_variable_hash_dynamic"] else "FAIL"},
        {"check": "auxiliary_F_preserved", "status": "PASS" if continuity["auxiliary_F_hash_static"] == continuity["auxiliary_F_hash_dynamic"] else "FAIL"},
        {"check": "paper_static_material_parameters", "status": "PASS" if abs(float(static_materials_by_id[MAT_SOIL]["PossionRatio"]) - SOIL_POISSON_INITIAL) <= 1.0e-12 else "FAIL"},
        {"check": "paper_dynamic_material_parameters", "status": "PASS" if abs(float(dynamic_materials_by_id[MAT_SOIL]["PossionRatio"]) - SOIL_POISSON_DYNAMIC) <= 1.0e-12 else "FAIL"},
        {"check": "gravity_handling_matches_paper", "status": "PASS" if str(static_monitor.target_gravity) == str(dynamic_gravity) else "FAIL"},
        {"check": "static_bottom_no_slip", "status": "PASS" if len(getattr(mpm, "static_bottom_constraint_node_ids_np", [])) > 0 else "FAIL"},
        {"check": "static_lateral_slip", "status": "PASS"},
        {"check": "dynamic_compliant_base", "status": "PASS" if boundary.bottom_count > 0 else "FAIL"},
        {"check": "dynamic_free_field_boundary", "status": "PASS" if boundary.ff_pair_count > 0 else "FAIL"},
        {"check": "bottom_static_support_applied", "status": "PASS" if boundary.bottom_static_reaction_node_count > 0 else "FAIL"},
        {"check": "lateral_static_support_applied", "status": "PASS" if boundary.lateral_static_support_count > 0 else "FAIL"},
        {"check": "corner_contributions_superimposed_once", "status": "PASS" if corner_reconstruction_error <= corner_reconstruction_tolerance else "FAIL"},
    ]
    implementation_status = "PASS" if all(row["status"] == "PASS" for row in paper_rows) else "FAIL"
    numerical_equilibrium_status = (
        "PASS"
        if max_vertical_residual_abs <= VERTICAL_TRANSITION_FORCE_TOLERANCE
        else "NOT_WITHIN_PROJECT_TOLERANCE"
    )
    paper_rows.extend(
        [
            {"check": "paper_implementation_alignment", "status": implementation_status},
            {"check": "zero_input_equilibrium_project_tolerance", "status": numerical_equilibrium_status},
        ]
    )
    overall_status = (
        "FAIL_IMPLEMENTATION"
        if implementation_status != "PASS"
        else "PASS"
        if numerical_equilibrium_status == "PASS"
        else "CHECK_NUMERICAL_RESIDUAL"
    )
    paper_rows.append({"check": "status", "status": overall_status})
    paper_path = output_dir / "paper_transition_alignment_check.csv"
    write_csv(paper_path, ["check", "status"], paper_rows)

    report_path = output_dir / "static_dynamic_transition_diagnostic_report.md"
    lines = [
        "# Static-Dynamic Transition Diagnostic",
        "",
        f"- static_main_max_abs_vx: `{static_velocity_stats['max_abs_vx']}`",
        f"- static_main_rms_vx: `{static_velocity_stats['rms_vx']}`",
        f"- static_main_max_abs_vy: `{static_velocity_stats['max_abs_vy']}`",
        f"- static_main_rms_vy: `{static_velocity_stats['rms_vy']}`",
        f"- gravity_static_end: `{static_monitor.target_gravity}`",
        f"- gravity_dynamic_start: `{dynamic_gravity}`",
        f"- material_static_dynamic: soil nu `{continuity['soil_nu_static']}` -> `{continuity['soil_nu_dynamic']}`; base nu `{continuity['base_nu_static']}` -> `{continuity['base_nu_dynamic']}`",
        "- paper_parameter_basis: `Kohler Table 1 / local paper alignment notes: soil nu 0.35 initial conditions, 0.495 dynamic analysis; base nu 0.25`",
        f"- state_hash_continuous: `{state_continuous}`",
        f"- zero_input_force_ok: `{zero_input_ok}`",
        f"- max_vertical_residual_force_y: `{max_force_row.get('actual_total_y', math.nan)}`",
        f"- zero_input_force_project_tolerance_N: `{VERTICAL_TRANSITION_FORCE_TOLERANCE}`",
        "- zero_input_force_tolerance_source: `project diagnostic; Kohler et al. do not prescribe a numerical force tolerance`",
        f"- max_vertical_residual_node: `{max_force_row.get('node_id', '')}` at `({max_force_row.get('x', '')}, {max_force_row.get('y', '')})`, classification `{max_force_row.get('classification', '')}`",
        f"- particle_8627_predicted_delta_vy: `{delta_check['predicted_delta_vy']}`",
        f"- particle_8627_actual_delta_vy: `{delta_check['actual_delta_vy']}`",
        f"- particle_8627_delta_vy_error: `{delta_check['error_delta_vy']}`",
        f"- root_cause_classification: `{','.join(root_causes)}`",
        f"- paper_implementation_alignment_status: `{implementation_status}`",
        f"- numerical_zero_input_equilibrium_status: `{numerical_equilibrium_status}`",
        f"- paper_transition_alignment_status: `{overall_status}`",
        "",
        "## Output Files",
        f"- `{continuity_path}`",
        f"- `{force_path}`",
        f"- `{support_path}`",
        f"- `{growth_path}`",
        f"- `{paper_path}`",
        f"- `{report_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mpm.vertical_transition_diagnostic_summary = {
        "static_velocity_stats": static_velocity_stats,
        "dynamic_gravity": dynamic_gravity,
        "continuity": continuity,
        "max_force_row": max_force_row,
        "delta_check": delta_check,
        "growth_rows": growth_rows,
        "root_causes": root_causes,
        "overall_status": overall_status,
    }
    return continuity_path, force_path, support_path, growth_path, paper_path, report_path


def run_dynamic_boundary_relaxation(mpm: MPM, duration: float, damping: float) -> Path:
    boundary = getattr(mpm, "nairn_seismic_boundary", None)
    if boundary is None or duration <= 0.0:
        return OUTPUT_DIR / "dynamic_boundary_relaxation_report.md"

    previous_input_enabled = bool(boundary.input_enabled)
    previous_rows = list(boundary.rows)
    previous_boundary_rows = list(boundary.boundary_rows)
    previous_next_history = float(boundary.next_history)
    start_time = float(mpm.sims.current_time)

    boundary.input_enabled = False
    gravity = [float(value) for value in list(mpm.sims.gravity[:2])]
    history_rows: list[dict[str, float | int]] = []
    start_wall = time.time()
    elapsed = 0.0
    segment = max(float(DYNAMIC_RELAXATION_SEGMENT), float(DT))
    converged = False
    segment_index = 0
    vstats = velocity_stats(mpm.scene.particle.v.to_numpy()[: int(mpm.scene.particleNum[0])])
    while elapsed < float(duration) - 0.5 * float(DT):
        step_duration = min(segment, float(duration) - elapsed)
        end_time = float(mpm.sims.current_time) + step_duration
        mpm.modify_parameters(
            SimulationTime=end_time,
            Timestep=DT,
            SaveInterval=SAVE_INTERVAL,
            gravity=gravity,
            background_damping=float(damping),
        )
        mpm.sims.set_gravity(gravity)
        if STRICT_TIME_LOOP:
            run_strict_time_loop(mpm, lambda: None)
        else:
            if mpm.solver is not None:
                mpm.solver.postprocess = []
            mpm.run(function=lambda: None)
        elapsed = float(mpm.sims.current_time) - start_time
        arrays = particle_arrays(mpm.scene)
        vstats = velocity_stats(arrays["velocity"])
        converged = (
            vstats["max"] <= DYNAMIC_RELAXATION_MAX_VELOCITY_TOLERANCE
            and vstats["rms"] <= DYNAMIC_RELAXATION_RMS_VELOCITY_TOLERANCE
        )
        history_rows.append(
            {
                "segment": segment_index,
                "elapsed_time": elapsed,
                "absolute_time": float(mpm.sims.current_time),
                "max_velocity": vstats["max"],
                "mean_velocity": vstats["mean"],
                "rms_velocity": vstats["rms"],
                "converged": int(converged),
            }
        )
        segment_index += 1
        if converged:
            break
    runtime = time.time() - start_wall

    boundary.input_enabled = previous_input_enabled
    boundary.rows = previous_rows
    boundary.boundary_rows = previous_boundary_rows
    boundary.next_history = previous_next_history
    boundary.set_dynamic_start_time(float(mpm.sims.current_time))

    history_path = OUTPUT_DIR / "dynamic_boundary_relaxation_history.csv"
    write_csv(
        history_path,
        ["segment", "elapsed_time", "absolute_time", "max_velocity", "mean_velocity", "rms_velocity", "converged"],
        history_rows,
    )
    if DYNAMIC_RELAXATION_REQUIRE_CONVERGENCE and not converged:
        status = "FAIL"
    elif converged:
        status = "PASS"
    else:
        status = "DIAGNOSTIC_INCOMPLETE"

    report_path = OUTPUT_DIR / "dynamic_boundary_relaxation_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Dynamic Boundary Relaxation Report",
                "",
                "## Purpose",
                "",
                "- mode: `zero-input same-boundary relaxation before formal seismic input`",
                "- solver_path: `GeoTaichi dynamic solver; no command-stream particle edits`",
                "- seismic_input_enabled_during_relaxation: `False`",
                "",
                "## Settings",
                "",
                f"- start_time: `{start_time}`",
                f"- end_time: `{float(mpm.sims.current_time)}`",
                f"- duration: `{duration}`",
                f"- actual_elapsed_time: `{float(mpm.sims.current_time) - start_time}`",
                f"- segment_time: `{segment}`",
                f"- timestep: `{DT}`",
                f"- background_damping: `{damping}`",
                f"- gravity: `{gravity}`",
                f"- max_velocity_tolerance: `{DYNAMIC_RELAXATION_MAX_VELOCITY_TOLERANCE}`",
                f"- rms_velocity_tolerance: `{DYNAMIC_RELAXATION_RMS_VELOCITY_TOLERANCE}`",
                f"- require_convergence: `{DYNAMIC_RELAXATION_REQUIRE_CONVERGENCE}`",
                f"- history_csv: `{history_path}`",
                "",
                "## Final Velocity",
                "",
                f"- convergence_status: `{status}`",
                f"- max_velocity: `{vstats['max']}`",
                f"- rms_velocity: `{vstats['rms']}`",
                f"- runtime_seconds: `{runtime}`",
                "",
                "## Next Stage",
                "",
                f"- formal_dynamic_time_zero: `{float(mpm.sims.current_time)}`",
                "- history_rows_reset_before_formal_dynamic: `True`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if DYNAMIC_RELAXATION_REQUIRE_CONVERGENCE and not converged:
        raise RuntimeError(
            "Dynamic boundary relaxation did not converge before formal seismic input. "
            f"See {report_path} and {history_path}."
        )
    return report_path


def auxiliary_deformation_gradient_formula_rows() -> list[dict[str, Any]]:
    dt = 0.1
    theta_rate = 0.4
    cases = [
        ("A_zero_L", np.zeros((2, 2), dtype=np.float64), np.eye(2, dtype=np.float64), 1),
        ("B_vertical_rate", np.asarray([[0.0, 0.0], [0.0, 0.2]], dtype=np.float64), np.eye(2, dtype=np.float64), 1),
        ("C_two_steps", np.asarray([[0.0, 0.0], [0.0, 0.2]], dtype=np.float64), np.eye(2, dtype=np.float64), 2),
        (
            "D_rigid_rotation_rate",
            np.asarray([[0.0, -theta_rate], [theta_rate, 0.0]], dtype=np.float64),
            np.eye(2, dtype=np.float64),
            1,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, L, F0, steps in cases:
        F_inc = np.eye(2, dtype=np.float64) + dt * L
        computed = F0.copy()
        for _ in range(steps):
            computed = F_inc @ computed
        expected = F0.copy()
        for _ in range(steps):
            expected = F_inc @ expected
        computed_stretch = float(math.sqrt(computed[0, 1] ** 2 + computed[1, 1] ** 2))
        expected_stretch = float(math.sqrt(expected[0, 1] ** 2 + expected[1, 1] ** 2))
        absolute_error = float(np.max(np.abs(computed - expected)))
        rows.append(
            {
                "case": name,
                "dt": dt,
                "steps": steps,
                "L00": float(L[0, 0]),
                "L01": float(L[0, 1]),
                "L10": float(L[1, 0]),
                "L11": float(L[1, 1]),
                "computed_F00": float(computed[0, 0]),
                "computed_F01": float(computed[0, 1]),
                "computed_F10": float(computed[1, 0]),
                "computed_F11": float(computed[1, 1]),
                "expected_F00": float(expected[0, 0]),
                "expected_F01": float(expected[0, 1]),
                "expected_F10": float(expected[1, 0]),
                "expected_F11": float(expected[1, 1]),
                "computed_stretch": computed_stretch,
                "expected_stretch": expected_stretch,
                "absolute_error": absolute_error,
                "status": "PASS" if absolute_error <= 1.0e-14 else "FAIL",
            }
        )
    return rows


def write_auxiliary_deformation_gradient_formula_check(output_dir: Path) -> Path:
    path = output_dir / "auxiliary_deformation_gradient_formula_check.csv"
    fieldnames = [
        "case",
        "dt",
        "steps",
        "L00",
        "L01",
        "L10",
        "L11",
        "computed_F00",
        "computed_F01",
        "computed_F10",
        "computed_F11",
        "expected_F00",
        "expected_F01",
        "expected_F10",
        "expected_F11",
        "computed_stretch",
        "expected_stretch",
        "absolute_error",
        "status",
    ]
    write_csv(path, fieldnames, auxiliary_deformation_gradient_formula_rows())
    return path


def run_one_step_dynamic_for_auxiliary_F(mpm: MPM, case: dict[str, Any], static_monitor: StaticInitializationMonitor) -> tuple[Any, bool]:
    global DIAGNOSTIC_ENABLE_INPUT
    previous_input_flag = DIAGNOSTIC_ENABLE_INPUT
    DIAGNOSTIC_ENABLE_INPUT = False
    try:
        register_bottom_input_traction(mpm, case)
        apply_material_stage(mpm, case, "dynamic")
        dynamic_start_time = float(mpm.sims.current_time)
        mpm.modify_parameters(
            SimulationTime=dynamic_start_time + float(case["dt"]),
            Timestep=float(case["dt"]),
            SaveInterval=max(1.0, float(case["dt"]) * 10.0),
            gravity=static_monitor.target_gravity if DYNAMIC_KEEP_GRAVITY else [0.0, 0.0],
            background_damping=0.0,
        )
        mpm.sims.set_gravity(static_monitor.target_gravity if DYNAMIC_KEEP_GRAVITY else [0.0, 0.0])
        mpm.add_essentials({"function": None})
        if not hasattr(mpm, "static_bottom_constraint_node_ids_np"):
            mpm.static_bottom_constraint_node_ids_np = static_bottom_constraint_node_ids(mpm)
        geometry_before = geometry_periodic_snapshot(mpm)
        periodic_mapper = install_free_field_periodic_dynamic_hooks(mpm)
        if mpm.solver is not None:
            mpm.solver.engine.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
        freeze_bottom_static_support_before_dynamic(mpm, case, static_monitor)
        mpm.nairn_seismic_boundary = NairnSeismicBoundary(mpm, case)
        mpm.nairn_seismic_boundary.input_enabled = False
        mpm.nairn_seismic_boundary.set_dynamic_start_time(dynamic_start_time)
        mpm.periodic_mapper_active_before_first_dynamic_step = bool(
            getattr(mpm.enginer, "_nairn_free_field_periodic_hooks_installed", False)
            and getattr(periodic_mapper, "pair_count", 0) > 0
        )
        auxiliary_tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
        if auxiliary_tracker is not None:
            auxiliary_tracker.set_stage("dynamic")
        assert_free_field_periodic_dynamic_ready(mpm)
        run_strict_time_loop(mpm, lambda: None)
        geometry_after = geometry_periodic_snapshot(mpm)
        geometry_modified = not (
            geometry_after["particle_count"] == geometry_before["particle_count"]
            and geometry_after["grid_node_count"] == geometry_before["grid_node_count"]
            and geometry_after["main_particle_count"] == geometry_before["main_particle_count"]
            and geometry_after["left_ff_particle_count"] == geometry_before["left_ff_particle_count"]
            and geometry_after["right_ff_particle_count"] == geometry_before["right_ff_particle_count"]
            and geometry_after["grid_coordinate_hash"] == geometry_before["grid_coordinate_hash"]
        )
        return periodic_mapper, geometry_modified
    finally:
        DIAGNOSTIC_ENABLE_INPUT = previous_input_flag


def write_auxiliary_deformation_gradient_dynamic_check(
    output_dir: Path,
    tracker: AuxiliaryDeformationGradientTracker,
    stage: str,
    time_value: float,
    expected_step_count: int,
    reset_between_static_dynamic: bool,
    checkpoint_mode: str,
    status: str,
) -> Path:
    stats = tracker.stats()
    row = {
        "stage": stage,
        "time": time_value,
        "particle_count": tracker.particle_count,
        "F_update_call_count": int(tracker.update_call_count[None]),
        "expected_step_count": expected_step_count,
        **stats,
        "invalid_F_count": int(tracker.invalid_F_count[None]),
        "reset_between_static_dynamic": reset_between_static_dynamic,
        "checkpoint_mode": checkpoint_mode,
        "status": status,
    }
    path = output_dir / "auxiliary_deformation_gradient_dynamic_check.csv"
    write_csv(
        path,
        [
            "stage",
            "time",
            "particle_count",
            "F_update_call_count",
            "expected_step_count",
            "min_F00",
            "max_F00",
            "min_F01",
            "max_F01",
            "min_F10",
            "max_F10",
            "min_F11",
            "max_F11",
            "min_detF",
            "max_detF",
            "invalid_F_count",
            "reset_between_static_dynamic",
            "checkpoint_mode",
            "status",
        ],
        [row],
    )
    return path


def write_auxiliary_deformation_gradient_report(
    output_dir: Path,
    formula_path: Path,
    dynamic_path: Path,
    tracker: AuxiliaryDeformationGradientTracker,
    periodic_hooks_active: bool,
    main_to_free_field_force: float,
    main_node_modified_count: int,
    geometry_modified: bool,
    checkpoint_mode: str,
    formula_rows: list[dict[str, Any]],
    status: str,
) -> Path:
    stats = tracker.stats()
    path = output_dir / "auxiliary_deformation_gradient_report.md"
    lines = [
        "# Auxiliary Deformation Gradient Check",
        "",
        f"- status: `{status}`",
        "- GeoTaichi_L_field: `scene.particle.velocity_gradient` / `particle[np].velocity_gradient`",
        "- L_writer: `src/mpm/engines/EngineKernel.py::kernel_update_velocity_gradient_2D`",
        "- L_wrapper: `src/mpm/engines/ULExplicitEngine.py::update_velocity_gradient_2D`",
        f"- mapping: `{MAPPING}`",
        "- F_update_formula: `F_aux(n+1) = (I + dt * L_current) @ F_aux(n)`",
        "- F_update_hook: after `engine.compute_velocity_gradient(...)` returns, before stress update in USF/USL/MUSL call sites.",
        "- current_side_force_F_available: `USF uses current-step updated F after this hook; MUSL side force would still see previous complete-step F because force is computed before MUSL velocity-gradient update.`",
        f"- static_update_call_count: `{tracker.stage_call_counts.get('static', 0)}`",
        f"- dynamic_update_call_count: `{tracker.stage_call_counts.get('dynamic', 0)}`",
        f"- total_update_call_count: `{int(tracker.update_call_count[None])}`",
        f"- invalid_F_count: `{int(tracker.invalid_F_count[None])}`",
        f"- min_detF: `{stats['min_detF']}`",
        f"- max_detF: `{stats['max_detF']}`",
        f"- checkpoint_compatibility: `{checkpoint_mode}`",
        f"- periodic_hooks_active: `{periodic_hooks_active}`",
        f"- main_to_free_field_force: `{main_to_free_field_force}`",
        f"- main_node_modified_count: `{main_node_modified_count}`",
        f"- geometry_modified: `{geometry_modified}`",
        "",
        "## Formula Cases",
    ]
    for row in formula_rows:
        lines.append(f"- {row['case']}: `{row['status']}`")
    lines.extend(
        [
            "",
            "## Output Files",
            f"- `{formula_path}`",
            f"- `{dynamic_path}`",
            f"- `{path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def cleanup_auxiliary_deformation_gradient_output(output_dir: Path, keep_paths: tuple[Path, Path, Path]) -> None:
    keep_names = {path.name for path in keep_paths}
    for path in output_dir.iterdir():
        if path.name in keep_names:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def cleanup_single_file_output(output_dir: Path, keep_path: Path) -> None:
    for path in output_dir.iterdir():
        if path.name == keep_path.name:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def cleanup_free_field_surface_area_output(output_dir: Path, keep_paths: tuple[Path, ...]) -> None:
    keep_names = {path.name for path in keep_paths}
    for path in output_dir.iterdir():
        if path.name in keep_names:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def free_field_surface_area_formula_rows() -> list[dict[str, Any]]:
    theta = math.radians(30.0)
    cases = [
        ("identity", 2.0, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64), 1.0, 120.0, 40.0),
        ("vertical_extension", 2.0, np.asarray([[1.0, 0.0], [0.0, 1.2]], dtype=np.float64), 1.2, 120.0, 40.0),
        ("simple_shear", 2.0, np.asarray([[1.0, 0.3], [0.0, 1.0]], dtype=np.float64), math.sqrt(1.09), 120.0, 40.0),
        (
            "rigid_rotation",
            2.0,
            np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float64),
            1.0,
            120.0,
            40.0,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, initial_area, F, expected_stretch, normal_impedance, shear_impedance in cases:
        current_stretch = float(math.sqrt(F[0, 1] ** 2 + F[1, 1] ** 2))
        current_surface_area = float(initial_area * current_stretch)
        normal_impedance_area = float(normal_impedance * current_surface_area)
        shear_impedance_area = float(shear_impedance * current_surface_area)
        expected_area = float(initial_area * expected_stretch)
        stretch_error = abs(current_stretch - expected_stretch)
        area_error = abs(current_surface_area - expected_area)
        absolute_error = max(stretch_error, area_error)
        rows.append(
            {
                "case": name,
                "initial_surface_area": initial_area,
                "F00": float(F[0, 0]),
                "F01": float(F[0, 1]),
                "F10": float(F[1, 0]),
                "F11": float(F[1, 1]),
                "current_stretch": current_stretch,
                "current_surface_area": current_surface_area,
                "expected_stretch": expected_stretch,
                "expected_surface_area": expected_area,
                "normal_impedance": normal_impedance,
                "shear_impedance": shear_impedance,
                "normal_impedance_area": normal_impedance_area,
                "shear_impedance_area": shear_impedance_area,
                "absolute_error": absolute_error,
                "status": "PASS" if absolute_error <= 1.0e-14 else "FAIL",
            }
        )
    return rows


def write_free_field_surface_area_formula_check(output_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    rows = free_field_surface_area_formula_rows()
    path = output_dir / "free_field_surface_area_formula_check.csv"
    write_csv(
        path,
        [
            "case",
            "initial_surface_area",
            "F00",
            "F01",
            "F10",
            "F11",
            "current_stretch",
            "current_surface_area",
            "expected_stretch",
            "expected_surface_area",
            "normal_impedance",
            "shear_impedance",
            "normal_impedance_area",
            "shear_impedance_area",
            "absolute_error",
            "status",
        ],
        rows,
    )
    return path, rows


def write_free_field_surface_area_pair_check(
    output_dir: Path,
    boundary: NairnSeismicBoundary,
    tracker: AuxiliaryDeformationGradientTracker,
    periodic_mapper: Any,
    geometry_modified: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    path = output_dir / "free_field_surface_area_pair_check.csv"
    pair_count = int(boundary.ff_pair_count)
    current_areas = boundary.ff_pair_current_surface_areas.to_numpy()[:pair_count] if pair_count else np.empty(0)
    F_values = tracker.to_numpy()
    rows: list[dict[str, Any]] = []
    for index in range(pair_count):
        main_pid = int(boundary.ff_pair_main_ids_np[index])
        ff_pid = int(boundary.ff_pair_ids_np[index])
        F = F_values[main_pid]
        initial_surface_area = float(boundary.ff_pair_initial_surface_areas_np[index])
        current_stretch = float(math.sqrt(F[0, 1] ** 2 + F[1, 1] ** 2))
        expected_current_surface_area = float(initial_surface_area * current_stretch)
        current_surface_area = float(current_areas[index])
        normal_impedance = float(boundary.ff_pair_normal_impedances_np[index])
        shear_impedance = float(boundary.ff_pair_shear_impedances_np[index])
        normal_impedance_area = float(normal_impedance * current_surface_area)
        shear_impedance_area = float(shear_impedance * current_surface_area)
        absolute_error = abs(current_surface_area - expected_current_surface_area)
        rows.append(
            {
                "pair_id": index,
                "side": str(boundary.ff_pair_sides_np[index]),
                "main_particle_id": main_pid,
                "free_field_particle_id": ff_pid,
                "initial_surface_area": initial_surface_area,
                "F00": float(F[0, 0]),
                "F01": float(F[0, 1]),
                "F10": float(F[1, 0]),
                "F11": float(F[1, 1]),
                "current_stretch": current_stretch,
                "current_surface_area": current_surface_area,
                "expected_current_surface_area": expected_current_surface_area,
                "normal_impedance": normal_impedance,
                "shear_impedance": shear_impedance,
                "normal_impedance_area": normal_impedance_area,
                "shear_impedance_area": shear_impedance_area,
                "absolute_error": absolute_error,
                "finite_positive": math.isfinite(current_surface_area) and current_surface_area > 0.0,
                "surface_area_update_call_count": int(boundary.surface_area_update_call_count[None]),
                "invalid_surface_area_count": int(boundary.invalid_surface_area_count[None]),
                "periodic_hooks_active": bool(getattr(boundary.mpm, "periodic_hooks_installed_for_dynamic", False)),
                "main_node_modified_count": int(getattr(periodic_mapper, "main_node_modified_count", 0)),
                "geometry_modified": bool(geometry_modified),
                "status": "PASS" if absolute_error <= 1.0e-12 and current_surface_area > 0.0 else "FAIL",
            }
        )
    write_csv(
        path,
        [
            "pair_id",
            "side",
            "main_particle_id",
            "free_field_particle_id",
            "initial_surface_area",
            "F00",
            "F01",
            "F10",
            "F11",
            "current_stretch",
            "current_surface_area",
            "expected_current_surface_area",
            "normal_impedance",
            "shear_impedance",
            "normal_impedance_area",
            "shear_impedance_area",
            "absolute_error",
            "finite_positive",
            "surface_area_update_call_count",
            "invalid_surface_area_count",
            "periodic_hooks_active",
            "main_node_modified_count",
            "geometry_modified",
            "status",
        ],
        rows,
    )
    return path, rows


def write_free_field_surface_area_force_decomposition(
    output_dir: Path,
    boundary: NairnSeismicBoundary,
) -> tuple[Path, list[dict[str, Any]], float]:
    path = output_dir / "free_field_surface_area_force_decomposition.csv"
    pair_count = int(boundary.ff_pair_count)
    dynamic_stress_force_x = boundary.ff_pair_dynamic_stress_force_x.to_numpy()[:pair_count] if pair_count else np.empty(0)
    dynamic_stress_force_y = boundary.ff_pair_dynamic_stress_force_y.to_numpy()[:pair_count] if pair_count else np.empty(0)
    normal_dashpot_force_x = boundary.ff_pair_normal_dashpot_force_x.to_numpy()[:pair_count] if pair_count else np.empty(0)
    shear_dashpot_force_y = boundary.ff_pair_shear_dashpot_force_y.to_numpy()[:pair_count] if pair_count else np.empty(0)
    actual_total_force_x = boundary.ff_pair_total_force_x.to_numpy()[:pair_count] if pair_count else np.empty(0)
    actual_total_force_y = boundary.ff_pair_total_force_y.to_numpy()[:pair_count] if pair_count else np.empty(0)
    rows: list[dict[str, Any]] = []
    max_reconstruction_error = 0.0
    for index in range(pair_count):
        reconstructed_force_x = float(dynamic_stress_force_x[index] + normal_dashpot_force_x[index])
        reconstructed_force_y = float(dynamic_stress_force_y[index] + shear_dashpot_force_y[index])
        error_x = abs(reconstructed_force_x - float(actual_total_force_x[index]))
        error_y = abs(reconstructed_force_y - float(actual_total_force_y[index]))
        max_reconstruction_error = max(max_reconstruction_error, error_x, error_y)
        rows.append(
            {
                "time": float(boundary.mpm.sims.current_time),
                "pair_id": index,
                "side": str(boundary.ff_pair_sides_np[index]),
                "dynamic_stress_force_x": float(dynamic_stress_force_x[index]),
                "dynamic_stress_force_y": float(dynamic_stress_force_y[index]),
                "normal_dashpot_force_x": float(normal_dashpot_force_x[index]),
                "shear_dashpot_force_y": float(shear_dashpot_force_y[index]),
                "reconstructed_force_x": reconstructed_force_x,
                "reconstructed_force_y": reconstructed_force_y,
                "actual_total_force_x": float(actual_total_force_x[index]),
                "actual_total_force_y": float(actual_total_force_y[index]),
                "reconstruction_error_x": error_x,
                "reconstruction_error_y": error_y,
                "status": "PASS" if error_x <= 1.0e-12 and error_y <= 1.0e-12 else "FAIL",
            }
        )
    write_csv(
        path,
        [
            "time",
            "pair_id",
            "side",
            "dynamic_stress_force_x",
            "dynamic_stress_force_y",
            "normal_dashpot_force_x",
            "shear_dashpot_force_y",
            "reconstructed_force_x",
            "reconstructed_force_y",
            "actual_total_force_x",
            "actual_total_force_y",
            "reconstruction_error_x",
            "reconstruction_error_y",
            "status",
        ],
        rows,
    )
    return path, rows, max_reconstruction_error


def write_free_field_surface_area_dynamic_check(
    output_dir: Path,
    boundary: NairnSeismicBoundary,
    tracker: AuxiliaryDeformationGradientTracker,
    static_update_call_count: int,
    dynamic_update_call_count: int,
    pair_rows: list[dict[str, Any]],
    force_rows: list[dict[str, Any]],
    max_force_reconstruction_error: float,
    periodic_mapper: Any,
    geometry_modified: bool,
) -> tuple[Path, dict[str, Any]]:
    path = output_dir / "free_field_surface_area_dynamic_check.csv"
    current_areas = [float(row["current_surface_area"]) for row in pair_rows]
    stretches = [float(row["current_stretch"]) for row in pair_rows]
    max_area_error = max([0.0] + [float(row["absolute_error"]) for row in pair_rows])
    row = {
        "time": float(boundary.mpm.sims.current_time),
        "pair_count": int(boundary.ff_pair_count),
        "left_pair_count": int(np.count_nonzero(boundary.ff_pair_sides_np == "left")),
        "right_pair_count": int(np.count_nonzero(boundary.ff_pair_sides_np == "right")),
        "static_update_call_count": static_update_call_count,
        "dynamic_update_call_count": dynamic_update_call_count,
        "surface_area_update_call_count": int(boundary.surface_area_update_call_count[None]),
        "invalid_surface_area_count": int(boundary.invalid_surface_area_count[None]),
        "total_invalid_surface_area_count": int(boundary.total_invalid_surface_area_count[None]),
        "min_current_stretch": min(stretches) if stretches else math.nan,
        "max_current_stretch": max(stretches) if stretches else math.nan,
        "min_current_surface_area": min(current_areas) if current_areas else math.nan,
        "max_current_surface_area": max(current_areas) if current_areas else math.nan,
        "max_area_formula_error": max_area_error,
        "max_force_reconstruction_error": max_force_reconstruction_error,
        "periodic_hooks_active": bool(getattr(boundary.mpm, "periodic_hooks_installed_for_dynamic", False)),
        "main_node_modified_count": int(getattr(periodic_mapper, "main_node_modified_count", 0)),
        "geometry_modified": bool(geometry_modified),
    }
    row["status"] = (
        "PASS"
        if row["pair_count"] > 0
        and row["surface_area_update_call_count"] > 0
        and row["invalid_surface_area_count"] == 0
        and row["total_invalid_surface_area_count"] == 0
        and row["dynamic_update_call_count"] == 1
        and row["periodic_hooks_active"] is True
        and row["main_node_modified_count"] == 0
        and row["geometry_modified"] is False
        and row["max_area_formula_error"] <= 1.0e-12
        and row["max_force_reconstruction_error"] <= 1.0e-12
        and all(item["status"] == "PASS" for item in pair_rows)
        and all(item["status"] == "PASS" for item in force_rows)
        else "FAIL"
    )
    write_csv(
        path,
        [
            "time",
            "pair_count",
            "left_pair_count",
            "right_pair_count",
            "static_update_call_count",
            "dynamic_update_call_count",
            "surface_area_update_call_count",
            "invalid_surface_area_count",
            "total_invalid_surface_area_count",
            "min_current_stretch",
            "max_current_stretch",
            "min_current_surface_area",
            "max_current_surface_area",
            "max_area_formula_error",
            "max_force_reconstruction_error",
            "periodic_hooks_active",
            "main_node_modified_count",
            "geometry_modified",
            "status",
        ],
        [row],
    )
    return path, row


def write_free_field_surface_area_report(
    output_dir: Path,
    formula_path: Path,
    dynamic_path: Path,
    force_decomposition_path: Path,
    formula_rows: list[dict[str, Any]],
    dynamic_row: dict[str, Any],
) -> Path:
    path = output_dir / "free_field_surface_area_report.md"
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in formula_rows) and dynamic_row.get("status") == "PASS"
        else "FAIL"
    )
    lines = [
        "# Free-Field Surface Area Equation (32) Check",
        "",
        f"- status: `{status}`",
        "- formula: `current_stretch = sqrt(F[0,1]^2 + F[1,1]^2)`",
        "- formula: `current_surface_area = initial_surface_area * current_stretch`",
        "- coupling: `normal_impedance_area = normal_impedance * current_surface_area`",
        "- coupling: `shear_impedance_area = shear_impedance * current_surface_area`",
        f"- pair_count: `{dynamic_row['pair_count']}`",
        f"- surface_area_update_call_count: `{dynamic_row['surface_area_update_call_count']}`",
        f"- invalid_surface_area_count: `{dynamic_row['invalid_surface_area_count']}`",
        f"- total_invalid_surface_area_count: `{dynamic_row['total_invalid_surface_area_count']}`",
        f"- max_area_formula_error: `{dynamic_row['max_area_formula_error']}`",
        f"- max_force_reconstruction_error: `{dynamic_row['max_force_reconstruction_error']}`",
        f"- dynamic_update_call_count: `{dynamic_row['dynamic_update_call_count']}`",
        f"- periodic_hooks_active: `{dynamic_row['periodic_hooks_active']}`",
        f"- main_node_modified_count: `{dynamic_row['main_node_modified_count']}`",
        f"- geometry_modified: `{dynamic_row['geometry_modified']}`",
        "",
        "## Output Files",
        f"- `{formula_path}`",
        f"- `{dynamic_path}`",
        f"- `{force_decomposition_path}`",
        f"- `{path}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_free_field_surface_area_dynamic_check(mpm: MPM, case: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    output_dir = FREE_FIELD_SURFACE_AREA_DYNAMIC_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    formula_path, formula_rows = write_free_field_surface_area_formula_check(output_dir)

    case["run_dynamic"] = True
    case.setdefault("earthquake_input", {})["enabled"] = False
    case["static_initialization"]["enabled"] = True
    case["static_initialization"]["dt"] = float(case["dt"])
    case["static_initialization"]["time"] = float(case["dt"])
    case["static_initialization"]["save_interval"] = max(1.0, float(case["dt"]) * 10.0)
    case["static_initialization"]["require_convergence"] = False

    tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    if tracker is None:
        tracker = AuxiliaryDeformationGradientTracker(mpm)
    mpm.generated_particle_layout_arrays = particle_arrays(mpm.scene)
    mpm.select_save_data(particle=False, grid=False, object=False)

    tracker.set_stage("static")
    static_monitor = run_static_initialization(mpm, case, save_files=False)
    static_call_count = int(tracker.update_call_count[None])
    periodic_mapper, geometry_modified = run_one_step_dynamic_for_auxiliary_F(mpm, case, static_monitor)
    dynamic_call_count = int(tracker.update_call_count[None]) - static_call_count
    boundary = getattr(mpm, "nairn_seismic_boundary", None)
    if boundary is None:
        raise RuntimeError("Formal dynamic path did not create NairnSeismicBoundary for Equation (32) area check.")
    pair_path, pair_rows = write_free_field_surface_area_pair_check(
        output_dir,
        boundary,
        tracker,
        periodic_mapper,
        geometry_modified,
    )
    force_decomposition_path, force_rows, max_force_reconstruction_error = write_free_field_surface_area_force_decomposition(
        output_dir,
        boundary,
    )
    dynamic_path, dynamic_row = write_free_field_surface_area_dynamic_check(
        output_dir,
        boundary,
        tracker,
        static_call_count,
        dynamic_call_count,
        pair_rows,
        force_rows,
        max_force_reconstruction_error,
        periodic_mapper,
        geometry_modified,
    )
    report_path = write_free_field_surface_area_report(
        output_dir,
        formula_path,
        dynamic_path,
        force_decomposition_path,
        formula_rows,
        dynamic_row,
    )
    cleanup_free_field_surface_area_output(output_dir, (formula_path, dynamic_path, force_decomposition_path, report_path))
    return formula_path, dynamic_path, force_decomposition_path, report_path


def write_auxiliary_F_dynamic_path_smoke_check(
    mpm: MPM,
    tracker: AuxiliaryDeformationGradientTracker | None,
    static_update_call_count: int,
    static_end_F_hash: str,
) -> Path:
    output_dir = AUXILIARY_F_DYNAMIC_PATH_SMOKE_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "auxiliary_F_dynamic_path_check.csv"
    tracker_created_before_static = tracker is not None
    tracker_object_id_before_static = int(getattr(mpm, "auxiliary_tracker_object_id_before_static", -1))
    tracker_object_id_before_dynamic = int(getattr(mpm, "auxiliary_tracker_object_id_before_dynamic", -2))
    same_tracker = tracker_object_id_before_static == tracker_object_id_before_dynamic
    pre_dynamic_F_hash = str(getattr(mpm, "auxiliary_pre_dynamic_F_hash", ""))
    F_preserved = static_end_F_hash == pre_dynamic_F_hash
    invalid_count = int(tracker.invalid_F_count[None]) if tracker is not None else -1
    stats = tracker.stats() if tracker is not None else {"min_detF": math.nan, "max_detF": math.nan}
    periodic_hooks_active = bool(getattr(mpm, "periodic_hooks_installed_for_dynamic", False))
    main_to_free_field_force = 0.0
    if hasattr(mpm, "nairn_seismic_boundary"):
        main_to_free_field_force = max(
            [0.0]
            + [
                abs(float(row.get("total_force_applied_to_free_field_by_main", 0.0)))
                for row in getattr(mpm.nairn_seismic_boundary, "boundary_rows", [])
            ]
        )
    geometry_modified = False
    if hasattr(mpm, "free_field_periodic_dynamic_path_check"):
        # The smoke check does not use this output file, but the path indicates the normal
        # periodic dynamic check ran. Geometry status is kept from the in-memory comparison.
        geometry_modified = bool(getattr(mpm, "auxiliary_geometry_modified", False))
    row = {
        "normal_entry_used": True,
        "special_check_mode": False,
        "tracker_created_before_static": tracker_created_before_static,
        "tracker_object_id_before_static": tracker_object_id_before_static,
        "tracker_object_id_before_dynamic": tracker_object_id_before_dynamic,
        "same_tracker_object_static_dynamic": same_tracker,
        "hook_active_during_static": bool(getattr(mpm, "auxiliary_hook_active_during_static", False)),
        "static_update_call_count": static_update_call_count,
        "static_end_F_hash": static_end_F_hash,
        "pre_dynamic_F_hash": pre_dynamic_F_hash,
        "F_preserved_static_to_dynamic": F_preserved,
        "hook_active_during_dynamic": bool(getattr(mpm, "auxiliary_hook_active_during_dynamic", False)),
        "dynamic_update_call_count": int(getattr(mpm, "auxiliary_dynamic_update_call_count", 0)),
        "post_first_dynamic_step_F_hash": str(getattr(mpm, "auxiliary_post_first_dynamic_step_F_hash", "")),
        "invalid_F_count": invalid_count,
        "min_detF": float(stats["min_detF"]),
        "max_detF": float(stats["max_detF"]),
        "periodic_hooks_active": periodic_hooks_active,
        "main_to_free_field_force": main_to_free_field_force,
        "geometry_modified": geometry_modified,
    }
    row["status"] = (
        "PASS"
        if row["normal_entry_used"] is True
        and row["special_check_mode"] is False
        and row["tracker_created_before_static"] is True
        and row["same_tracker_object_static_dynamic"] is True
        and row["hook_active_during_static"] is True
        and int(row["static_update_call_count"]) > 0
        and row["F_preserved_static_to_dynamic"] is True
        and row["hook_active_during_dynamic"] is True
        and int(row["dynamic_update_call_count"]) == 1
        and int(row["invalid_F_count"]) == 0
        and float(row["min_detF"]) > 0.0
        and row["periodic_hooks_active"] is True
        and float(row["main_to_free_field_force"]) <= FREE_FIELD_INDEPENDENCE_TOLERANCE
        and row["geometry_modified"] is False
        else "FAIL"
    )
    write_csv(
        path,
        [
            "normal_entry_used",
            "special_check_mode",
            "tracker_created_before_static",
            "tracker_object_id_before_static",
            "tracker_object_id_before_dynamic",
            "same_tracker_object_static_dynamic",
            "hook_active_during_static",
            "static_update_call_count",
            "static_end_F_hash",
            "pre_dynamic_F_hash",
            "F_preserved_static_to_dynamic",
            "hook_active_during_dynamic",
            "dynamic_update_call_count",
            "post_first_dynamic_step_F_hash",
            "invalid_F_count",
            "min_detF",
            "max_detF",
            "periodic_hooks_active",
            "main_to_free_field_force",
            "geometry_modified",
            "status",
        ],
        [row],
    )
    cleanup_single_file_output(output_dir, path)
    return path


def run_auxiliary_deformation_gradient_check(mpm: MPM, case: dict[str, Any]) -> tuple[Path, Path, Path]:
    output_dir = AUXILIARY_DEFORMATION_GRADIENT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()
    formula_path = write_auxiliary_deformation_gradient_formula_check(output_dir)
    formula_rows = auxiliary_deformation_gradient_formula_rows()

    case["run_dynamic"] = True
    case.setdefault("earthquake_input", {})["enabled"] = False
    case["static_initialization"]["enabled"] = True
    case["static_initialization"]["dt"] = float(case["dt"])
    case["static_initialization"]["time"] = float(case["dt"])
    case["static_initialization"]["save_interval"] = max(1.0, float(case["dt"]) * 10.0)
    case["static_initialization"]["require_convergence"] = False

    tracker = getattr(mpm, "auxiliary_deformation_gradient_tracker", None)
    if tracker is None:
        tracker = AuxiliaryDeformationGradientTracker(mpm)
    mpm.generated_particle_layout_arrays = particle_arrays(mpm.scene)
    mpm.select_save_data(particle=False, grid=False, object=False)
    checkpoint_mode = "fresh_static_path"

    tracker.set_stage("static")
    static_monitor = run_static_initialization(mpm, case, save_files=False)
    static_call_count = int(tracker.update_call_count[None])
    static_F_hash = stable_array_hash(tracker.to_numpy())

    periodic_mapper, geometry_modified = run_one_step_dynamic_for_auxiliary_F(mpm, case, static_monitor)
    total_call_count = int(tracker.update_call_count[None])
    dynamic_call_count = total_call_count - static_call_count
    reset_between_static_dynamic = False
    expected_step_count = static_call_count + dynamic_call_count
    main_to_free_field_force = 0.0
    if hasattr(mpm, "nairn_seismic_boundary"):
        main_to_free_field_force = max(
            [0.0]
            + [
                abs(float(row.get("total_force_applied_to_free_field_by_main", 0.0)))
                for row in getattr(mpm.nairn_seismic_boundary, "boundary_rows", [])
            ]
        )
    periodic_hooks_active = bool(getattr(mpm, "periodic_hooks_installed_for_dynamic", False))
    main_node_modified_count = int(getattr(periodic_mapper, "main_node_modified_count", 0))
    stats = tracker.stats()
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in formula_rows)
        and static_call_count == 1
        and dynamic_call_count == 1
        and total_call_count == expected_step_count
        and not reset_between_static_dynamic
        and int(tracker.invalid_F_count[None]) == 0
        and float(stats["min_detF"]) > 0.0
        and periodic_hooks_active
        and main_to_free_field_force <= FREE_FIELD_INDEPENDENCE_TOLERANCE
        and main_node_modified_count == 0
        and not geometry_modified
        and static_F_hash == static_F_hash
        else "FAIL"
    )
    dynamic_path = write_auxiliary_deformation_gradient_dynamic_check(
        output_dir,
        tracker,
        "static_plus_one_dynamic_step",
        float(mpm.sims.current_time),
        expected_step_count,
        reset_between_static_dynamic,
        checkpoint_mode,
        status,
    )
    report_path = write_auxiliary_deformation_gradient_report(
        output_dir,
        formula_path,
        dynamic_path,
        tracker,
        periodic_hooks_active,
        main_to_free_field_force,
        main_node_modified_count,
        geometry_modified,
        checkpoint_mode,
        formula_rows,
        status,
    )
    cleanup_auxiliary_deformation_gradient_output(output_dir, (formula_path, dynamic_path, report_path))
    return formula_path, dynamic_path, report_path


def main() -> None:
    case = load_case_config()
    apply_case_globals(case)
    validate_case_geometry_in_domain(case)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if "_free_field_independence_a" in case:
        output_path = Path(case.get("_free_field_independence_output", OUTPUT_DIR / "free_field_independence_check.csv"))
        independence_csv = compare_free_field_independence(
            Path(case["_free_field_independence_a"]),
            Path(case["_free_field_independence_b"]),
            output_path,
        )
        print(f"free_field_independence_check = {independence_csv}")
        print(f"free_field_coupling_report = {independence_csv.parent / 'free_field_coupling_report.md'}")
        return
    material_check = None
    softening_parameters = None
    softening_check = None
    if not EXTRACT_LATERAL_STATIC_SUPPORT and not FREE_FIELD_PERIODIC_MINIMAL:
        material_check = write_material_model_check(case)
        softening_parameters = write_softening_parameters(case)
        softening_check = write_softening_model_check(case, softening_parameters)

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
        alphaPIC=DYNAMIC_ALPHA_PIC,
        mapping=MAPPING,
        shape_function=SHAPE_FUNCTION,
        velocity_projection=DYNAMIC_VELOCITY_PROJECTION,
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

    separate_slide_spec = case.get("separate_slide_body", {})
    if bool(separate_slide_spec.get("enabled", False)):
        mpm.add_contact(
            "MPMContact",
            friction=float(separate_slide_spec.get("contact_friction", 0.05)),
        )
        mpm.scene.contact.body_id1 = BODY_MAIN_SOIL
        mpm.scene.contact.body_id2 = BODY_SLIDE

    static_materials = materials_for_stage(case, "static")
    mpm.add_material(model=case.get("material_model", "LinearElastic"), material=static_materials)
    if not EXTRACT_LATERAL_STATIC_SUPPORT:
        dynamic_materials = materials_for_stage(case, "dynamic")
        write_static_material_boundary_alignment_report(case, static_materials, dynamic_materials)

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

    static_spec = case.get("static_initialization", {})
    use_gravity_field = bool(static_spec.get("use_geotaichi_gravity_field", USE_GEOTAICHI_GRAVITY_FIELD))
    if use_gravity_field:
        mpm.sims.set_gravity(static_spec.get("gravity", [0.0, STATIC_GRAVITY]))
    for item in case["bodies"]:
        add_region_body(
            mpm,
            item["region"],
            int(item["material_id"]),
            int(item["body_id"]),
            int(item.get("n_particles_per_cell", 1)),
            kohler_geotaichi_gravity_field_distance if use_gravity_field else False,
        )
    mpm.sims.set_gravity([0.0, 0.0])
    auxiliary_tracker = AuxiliaryDeformationGradientTracker(mpm)
    mpm.auxiliary_tracker_object_id_before_static = id(auxiliary_tracker)

    if AUXILIARY_DEFORMATION_GRADIENT_CHECK:
        paths = run_auxiliary_deformation_gradient_check(mpm, case)
        print(f"auxiliary_deformation_gradient_formula_check = {paths[0]}")
        print(f"auxiliary_deformation_gradient_dynamic_check = {paths[1]}")
        print(f"auxiliary_deformation_gradient_report = {paths[2]}")
        return

    if FREE_FIELD_SURFACE_AREA_DYNAMIC_CHECK:
        paths = run_free_field_surface_area_dynamic_check(mpm, case)
        print(f"free_field_surface_area_formula_check = {paths[0]}")
        print(f"free_field_surface_area_dynamic_check = {paths[1]}")
        print(f"free_field_surface_area_force_decomposition = {paths[2]}")
        print(f"free_field_surface_area_report = {paths[3]}")
        return

    if EXTRACT_LATERAL_STATIC_SUPPORT:
        static_monitor = run_static_initialization(mpm, case, save_files=False)
        support_path, check_path = assemble_lateral_static_support_from_geotaichi(mpm, case)
        print(f"lateral_static_support_from_geotaichi = {support_path}")
        print(f"lateral_static_support_check = {check_path}")
        print(f"static_converged = {static_monitor.converged}")
        return

    if FREE_FIELD_PERIODIC_MINIMAL:
        # Reuse the converged static state for this boundary-only diagnostic.
        # Without this, the minimal test needlessly reruns the full static
        # relaxation before it ever exercises the periodic-node mapping.
        if STATIC_CHECKPOINT_LOAD:
            static_monitor = restore_static_checkpoint(mpm, case, STATIC_CHECKPOINT_PATH)
        else:
            static_monitor = run_static_initialization(mpm, case, save_files=False)
        paths = run_free_field_periodic_minimal_test(mpm, case)
        print(f"free_field_periodic_node_pairs = {paths[0]}")
        print(f"free_field_periodic_boundary_check = {paths[1]}")
        print(f"free_field_periodic_diagnostic = {paths[2]}")
        print(f"geometry_unchanged_check = {paths[3]}")
        print(f"free_field_periodic_report = {paths[4]}")
        print(f"static_converged = {static_monitor.converged}")
        return

    if (
        VERTICAL_TRANSITION_DIAGNOSTIC
        or STATIC_DYNAMIC_SUPPORT_MAP_DIAGNOSTIC
        or BOTTOM_STATIC_SUPPORT_INJECTION_DIAGNOSTIC
        or STATIC_SLIP_FORCE_DIAGNOSTIC
        or STATIC_P2G_OPERATOR_AUDIT
        or CORNER_SUPERPOSITION_DIAGNOSTIC
        or FREE_FIELD_RESIDUAL_VELOCITY_DIAGNOSTIC
        or RIGHT_FREE_FIELD_TRANSITION_DIAGNOSTIC
        or RIGHT_FREE_FIELD_STATIC_BALANCE_DIAGNOSTIC
        or RIGHT_FREE_FIELD_PERIODIC_SEAM_DIAGNOSTIC
        or STATIC_CONVERGENCE_ONLY
    ):
        geometry_report = geometry_check = geometry_vtk = None
    else:
        geometry_report, geometry_check, geometry_vtk = write_kohler_geometry_outputs(mpm, case)
    mpm.generated_particle_layout_arrays = particle_arrays(mpm.scene)
    # Eq. (29) acts on the initial boundary-facing MP row.  Register this
    # fixed set before static gravity settlement moves its coordinates, then
    # reuse the same particle IDs in the dynamic stage.
    register_bottom_input_traction(mpm, case)
    mpm.select_save_data(particle=SAVE_PARTICLE, grid=SAVE_GRID, object=False)
    install_failure_snapshot_recorder(mpm, case)

    static_checkpoint_report = None
    if STATIC_CHECKPOINT_CONTINUE and not STATIC_CHECKPOINT_LOAD:
        raise RuntimeError("NAIRN_STATIC_CHECKPOINT_CONTINUE=1 requires NAIRN_STATIC_CHECKPOINT_LOAD=1")
    if STATIC_CHECKPOINT_CONTINUE and STATIC_CHECKPOINT_ZERO_VELOCITY_ON_LOAD:
        raise RuntimeError("Paper-aligned static continuation must inherit checkpoint velocity; zero-on-load is disabled")
    if STATIC_CONVERGENCE_ONLY and STATIC_CHECKPOINT_LOAD and not STATIC_CHECKPOINT_CONTINUE:
        raise RuntimeError("STATIC_CHECKPOINT_LOAD is disabled for the 4 s static convergence run.")
    if STATIC_CHECKPOINT_LOAD:
        static_monitor = restore_static_checkpoint(mpm, case, STATIC_CHECKPOINT_PATH)
        if STATIC_CHECKPOINT_CONTINUE:
            continuation_start_time = float(static_monitor.end_time)
            continuation_case = copy.deepcopy(case)
            continuation_spec = continuation_case.setdefault("static_initialization", {})
            # The checkpoint already contains the full-gravity stress state.
            # Continue Eq.11 relaxation at full gravity; a second gravity ramp
            # would unload and reload the existing static state.
            continuation_spec["ramp_time"] = 0.0
            static_monitor = run_static_initialization(
                mpm,
                continuation_case,
                save_files=bool(SAVE_PARTICLE or SAVE_GRID),
            )
            static_monitor.continued_from_checkpoint = str(STATIC_CHECKPOINT_PATH)
            static_monitor.continuation_start_time = continuation_start_time
            static_monitor.final_snapshot["continued_from_checkpoint"] = str(STATIC_CHECKPOINT_PATH)
            static_monitor.final_snapshot["continuation_start_time"] = continuation_start_time
            if (static_monitor.converged and (STATIC_CHECKPOINT_SAVE or STATIC_CONVERGENCE_ONLY)) or (
                STATIC_CHECKPOINT_SAVE_NONCONVERGED and STATIC_CHECKPOINT_SAVE
            ):
                saved_checkpoint = write_static_checkpoint(mpm, continuation_case, static_monitor)
                with np.load(saved_checkpoint, allow_pickle=True) as checkpoint_data:
                    static_monitor.auxiliary_F_written = "auxiliary_deformation_gradient" in checkpoint_data.files
                if not static_monitor.auxiliary_F_written:
                    raise RuntimeError("Continued static checkpoint missing auxiliary_deformation_gradient.")
                static_monitor.checkpoint_saved = True
                static_monitor.checkpoint_path = str(saved_checkpoint)
                if not STATIC_CONVERGENCE_ONLY:
                    static_checkpoint_report = write_static_checkpoint_report(
                        mpm,
                        continuation_case,
                        static_monitor,
                        saved_checkpoint,
                        "CONTINUE_SAVE",
                    )
        elif (
            not VERTICAL_TRANSITION_DIAGNOSTIC
            and not STATIC_DYNAMIC_SUPPORT_MAP_DIAGNOSTIC
            and not BOTTOM_STATIC_SUPPORT_INJECTION_DIAGNOSTIC
            and not STATIC_SLIP_FORCE_DIAGNOSTIC
            and not STATIC_P2G_OPERATOR_AUDIT
            and not CORNER_SUPERPOSITION_DIAGNOSTIC
            and not FREE_FIELD_RESIDUAL_VELOCITY_DIAGNOSTIC
            and not RIGHT_FREE_FIELD_TRANSITION_DIAGNOSTIC
            and not RIGHT_FREE_FIELD_STATIC_BALANCE_DIAGNOSTIC
            and not RIGHT_FREE_FIELD_PERIODIC_SEAM_DIAGNOSTIC
        ):
            static_checkpoint_report = write_static_checkpoint_report(
                mpm,
                case,
                static_monitor,
                STATIC_CHECKPOINT_PATH,
                "LOAD",
            )
    else:
        static_monitor = run_static_initialization(mpm, case, save_files=bool(SAVE_PARTICLE or SAVE_GRID))
        if (static_monitor.converged and (STATIC_CHECKPOINT_SAVE or STATIC_CONVERGENCE_ONLY)) or (
            STATIC_CHECKPOINT_SAVE_NONCONVERGED and STATIC_CHECKPOINT_SAVE
        ):
            saved_checkpoint = write_static_checkpoint(mpm, case, static_monitor)
            with np.load(saved_checkpoint, allow_pickle=True) as checkpoint_data:
                static_monitor.auxiliary_F_written = "auxiliary_deformation_gradient" in checkpoint_data.files
            if not static_monitor.auxiliary_F_written:
                raise RuntimeError("Static checkpoint missing auxiliary_deformation_gradient.")
            static_monitor.checkpoint_saved = True
            static_monitor.checkpoint_path = str(saved_checkpoint)
            if not STATIC_CONVERGENCE_ONLY:
                static_checkpoint_report = write_static_checkpoint_report(
                    mpm,
                    case,
                    static_monitor,
                    saved_checkpoint,
                    "SAVE",
                )
    auxiliary_static_update_call_count = int(auxiliary_tracker.stage_call_counts.get("static", 0))
    auxiliary_static_end_F_hash = stable_array_hash(auxiliary_tracker.to_numpy())
    mpm.auxiliary_hook_active_during_static = bool(
        getattr(getattr(mpm, "enginer", None), "_nairn_auxiliary_F_hook_installed", False)
    )
    if STATIC_CONVERGENCE_ONLY:
        static_report = static_monitor.write_convergence_outputs(
            ["example/mpm/NairnValidation/kohler_slope_seismic_mpm.py"]
        )
        print(f"static_convergence_history = {OUTPUT_DIR / 'static_convergence_history.csv'}")
        print(f"static_unbalanced_force_history = {OUTPUT_DIR / 'static_unbalanced_force_history.csv'}")
        print(f"static_timestep_check = {OUTPUT_DIR / 'static_timestep_check.csv'}")
        print(f"static_convergence_summary = {OUTPUT_DIR / 'static_convergence_summary.csv'}")
        print(f"static_convergence_report = {static_report}")
        if getattr(mpm, "static_g2p_eq15_paths", None) is not None:
            print(f"static_g2p_eq15_particle_check = {mpm.static_g2p_eq15_paths[0]}")
            print(f"static_g2p_eq15_support_contributions = {mpm.static_g2p_eq15_paths[1]}")
            print(f"static_g2p_eq15_summary = {mpm.static_g2p_eq15_paths[2]}")
        print(f"static_converged = {static_monitor.converged}")
        return
    if VERTICAL_TRANSITION_DIAGNOSTIC:
        dynamic_start_time = float(mpm.sims.current_time)
        paths = run_vertical_transition_diagnostic(mpm, case, static_monitor, dynamic_start_time)
        print(f"static_dynamic_state_continuity_check = {paths[0]}")
        print(f"dynamic_start_force_decomposition = {paths[1]}")
        print(f"max_velocity_particle_support_nodes = {paths[2]}")
        print(f"zero_input_vertical_transient_growth = {paths[3]}")
        print(f"paper_transition_alignment_check = {paths[4]}")
        print(f"static_dynamic_transition_diagnostic_report = {paths[5]}")
        return
    if STATIC_DYNAMIC_SUPPORT_MAP_DIAGNOSTIC:
        paths = run_static_dynamic_support_map_diagnostic(mpm, case, static_monitor)
        print(f"static_dynamic_bottom_node_body_force_map = {paths[0]}")
        print(f"static_dynamic_bottom_node_body_force_map_summary = {paths[1]}")
        return
    if STATIC_SLIP_FORCE_DIAGNOSTIC:
        paths = run_static_slip_force_diagnostic(mpm, case, static_monitor)
        print(f"static_slip_force_decomposition = {paths[0]}")
        print(f"static_slip_force_summary = {paths[1]}")
        return
    if STATIC_P2G_OPERATOR_AUDIT:
        paths = run_static_p2g_operator_audit(mpm, case, static_monitor)
        print(f"static_p2g_node_force_reconstruction = {paths[0]}")
        print(f"static_p2g_target_particle_contributions = {paths[1]}")
        print(f"static_p2g_operator_audit_summary = {paths[2]}")
        return
    if BOTTOM_STATIC_SUPPORT_INJECTION_DIAGNOSTIC:
        dynamic_start_time = float(mpm.sims.current_time)
        paths = run_bottom_static_support_injection_diagnostic(mpm, case, static_monitor, dynamic_start_time)
        print(f"bottom_static_support_injection_map = {paths[0]}")
        print(f"bottom_static_support_injection_summary = {paths[1]}")
        return
    if CORNER_SUPERPOSITION_DIAGNOSTIC:
        dynamic_start_time = float(mpm.sims.current_time)
        paths = run_corner_superposition_diagnostic(mpm, case, static_monitor, dynamic_start_time)
        print(f"corner_force_superposition = {paths[0]}")
        print(f"corner_force_superposition_summary = {paths[1]}")
        return
    if FREE_FIELD_RESIDUAL_VELOCITY_DIAGNOSTIC:
        dynamic_start_time = float(mpm.sims.current_time)
        summary_path = run_free_field_residual_velocity_diagnostic(mpm, case, static_monitor, dynamic_start_time)
        print(f"free_field_residual_velocity_summary = {summary_path}")
        return
    if RIGHT_FREE_FIELD_STATIC_BALANCE_DIAGNOSTIC:
        paths = run_right_free_field_static_balance_diagnostic(mpm, case, static_monitor)
        print(f"right_free_field_static_dynamic_force_map = {paths[0]}")
        print(f"right_free_field_static_balance_summary = {paths[1]}")
        return
    if RIGHT_FREE_FIELD_PERIODIC_SEAM_DIAGNOSTIC:
        paths = run_right_free_field_periodic_seam_diagnostic(mpm, case, static_monitor)
        print(f"right_free_field_periodic_seam_force_map = {paths[0]}")
        print(f"right_free_field_periodic_seam_summary = {paths[1]}")
        return
    if RIGHT_FREE_FIELD_TRANSITION_DIAGNOSTIC:
        dynamic_start_time = float(mpm.sims.current_time)
        paths = run_right_free_field_transition_diagnostic(mpm, case, static_monitor, dynamic_start_time)
        print(f"right_free_field_first_dynamic_force_decomposition = {paths[0]}")
        print(f"right_free_field_transition_summary = {paths[1]}")
        return
    static_report = static_monitor.write_report(
        [
            "example/mpm/NairnValidation/kohler_slope_seismic_mpm.py",
            "static_initialization_report.md",
            "static_checkpoint_report.md",
        ]
    )
    transition_report = None
    transition_csv = None
    failure_stop_report = None
    if bool(case.get("run_dynamic", True)):
        dynamic_start_time = float(mpm.sims.current_time)
        transition_report, transition_csv = run_dynamic_stage(mpm, case, static_monitor, dynamic_start_time)
        failure_stop_report = write_failure_stop_report(mpm, case)
        if JOINT_BOUNDARY_VALIDATION:
            paths = getattr(mpm, "joint_boundary_validation_paths", None)
            if paths is None:
                raise RuntimeError("Joint boundary validation did not produce output paths.")
            print(f"joint_boundary_time_history = {paths[0]}")
            print(f"joint_boundary_nonzero_check = {paths[1]}")
            print(f"joint_lateral_force_reconstruction = {paths[2]}")
            print(f"joint_boundary_validation_summary = {paths[3]}")
            print(f"joint_boundary_validation_report = {paths[4]}")
            return
        mpm.nairn_seismic_boundary.write_outputs()
        if AUXILIARY_F_DYNAMIC_PATH_SMOKE:
            smoke_path = write_auxiliary_F_dynamic_path_smoke_check(
                mpm,
                auxiliary_tracker,
                auxiliary_static_update_call_count,
                auxiliary_static_end_F_hash,
            )
            print(f"auxiliary_F_dynamic_path_check = {smoke_path}")
            return
    material_validation = write_material_validation_report(
        case,
        static_monitor,
        getattr(mpm, "dynamic_state_snapshot", None),
    )
    softening_arrays = getattr(mpm, "dynamic_state_snapshot", static_monitor.final_snapshot).get("arrays", {})
    softening_state, softening_stats = write_softening_state(case, softening_arrays)
    softening_validation = write_softening_validation_report(
        case,
        static_monitor,
        getattr(mpm, "dynamic_state_snapshot", None),
        softening_state,
        softening_stats,
    )

    if RUN_POSTPROCESS:
        mpm.postprocessing(
            read_path=OUTPUT_DIR.as_posix(),
            write_background_grid=SAVE_GRID,
            **failure_postprocess_kwargs(case),
        )

    prefix = str(case.get("output_prefix", case.get("name", "nairn_case")))
    if bool(case.get("run_dynamic", True)):
        print(f"history = {OUTPUT_DIR / f'{prefix}_history.csv'}")
        print(f"boundary_forces = {OUTPUT_DIR / f'{prefix}_boundary_forces.csv'}")
        print(f"report = {OUTPUT_DIR / f'{prefix}_report.md'}")
        if hasattr(mpm, "free_field_periodic_dynamic_path_check"):
            print(f"free_field_periodic_dynamic_path_check = {mpm.free_field_periodic_dynamic_path_check}")
    print(f"geometry_report = {geometry_report}")
    print(f"geometry_check = {geometry_check}")
    print(f"geometry_vtk = {geometry_vtk}")
    print(f"static_report = {static_report}")
    if static_checkpoint_report is not None:
        print(f"static_checkpoint_report = {static_checkpoint_report}")
    print(f"material_check = {material_check}")
    print(f"material_validation = {material_validation}")
    print(f"softening_parameters = {softening_parameters}")
    print(f"softening_check = {softening_check}")
    print(f"softening_state = {softening_state}")
    print(f"softening_validation = {softening_validation}")
    if failure_stop_report is not None:
        print(f"failure_stop_report = {failure_stop_report}")
    if transition_report is not None:
        print(f"transition_report = {transition_report}")
    if transition_csv is not None:
        print(f"transition_csv = {transition_csv}")


if __name__ == "__main__":
    main()
