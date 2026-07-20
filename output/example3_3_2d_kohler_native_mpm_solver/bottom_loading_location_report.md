# Bottom Loading Location Report

## Scope

- Checked file: `F:\dhh\GeoTaichi-dhh\scripts\run_geometry_corrected_dynamic_native_mpm.py`
- Request: locate `input_tx` and `t_dashpot` / dashpot traction application position.
- Code modification: none.

## 1. Load Application Function Location

Bottom input and dashpot are computed in:

- `scripts/run_geometry_corrected_dynamic_native_mpm.py:86` `update_corrected_particle_tractions()`
- bottom loop starts at `scripts/run_geometry_corrected_dynamic_native_mpm.py:118`
- `input_tx` is computed at `scripts/run_geometry_corrected_dynamic_native_mpm.py:122`
- `dashpot_tx` is computed at `scripts/run_geometry_corrected_dynamic_native_mpm.py:123`
- combined traction is written at `scripts/run_geometry_corrected_dynamic_native_mpm.py:127`

Current code path:

```python
input_tx = 2.0 * rho_cs * input_velocity
dashpot_tx = -rho_cs * particle[pid].v[0]
dashpot_tz = -rho_cp * particle[pid].v[1]
tx = input_tx + dashpot_tx
tz = dashpot_tz
particle_traction[tid].traction = ti.Vector([tx, tz])
```

The bottom particles receiving this are selected in:

- `scripts/run_geometry_corrected_dynamic_native_mpm.py:298` `_bottom_specs()`
- condition: `abs(pos[pid, 1] - BOTTOM_Z) <= 1.0e-10`
- each row stores `particle_id`, `traction_id`, `body_id`, and `area`

The particle traction constraints are registered in:

- `scripts/run_geometry_corrected_dynamic_native_mpm.py:231` `register_particle_tractions()`
- `scripts/run_geometry_corrected_dynamic_native_mpm.py:233` `scene.boundary.set_particle_traction(...)`

The custom update is called through the native trace hook from the legacy helper before native particle-traction application:

- `examples/example3_3_2d_kohler_native_mpm_solver.py:1388-1393`
- it calls `compliant.apply(sims, scene)`, which in this run is `CorrectedBoundary.apply()`.

## 2. Application Object

Answer: the load is attached to `particle_id` through a particle traction constraint, not directly to a `grid_node_id`.

Evidence:

- `CorrectedBoundary._bottom_specs()` collects bottom `particle_id` values.
- It resolves each bottom particle to a `traction_id` through `boundary.particle_traction.pid`.
- `update_corrected_particle_tractions()` writes `particle_traction[tid].traction`, where `tid` is associated with a particle id.
- No bottom `grid_node_id` is selected in this custom bottom input code.

Therefore:

- direct application object: `particle_id` / `particle_traction[traction_id]`
- not direct application object: `grid_node_id`

## 3. Complete Path

Current bottom loading path is:

```text
input_velocity(time)
  -> input_tx = 2*rho*Cs*input_velocity
  -> dashpot_tx = -rho*Cs*particle[bottom_particle_id].v_x
  -> tx = input_tx + dashpot_tx
  -> particle_traction[bottom_traction_id].traction = [tx, tz]
  -> native apply_particle_traction_constraints()
  -> constraints[nboundary]._compute_traction_force()
  -> shape_fn[ln] * traction_force
  -> node[nodeID, bodyID]._update_nodal_force(...)
  -> grid nodal force participates in grid kinematic update
```

Native mapping details:

- `src/mpm/engines/ULExplicitEngine.py:255` `usl_updating()` is the active update path because the run sets `mapping="USL"` in `scripts/run_geometry_corrected_dynamic_native_mpm.py:500`.
- `src/mpm/engines/ULExplicitEngine.py:259` calls `apply_particle_traction_constraints(sims, scene)`.
- `src/mpm/engines/ULExplicitEngine.py:260` then calls `compute_forces(sims, scene)`.
- `src/mpm/engines/Engine.py:436` routes `particle_traction_constraints()` to `apply_particle_traction_constraint(...)`.
- `src/mpm/boundaries/BoundaryCore.py:261` defines `apply_particle_traction_constraint(...)`.
- `src/mpm/boundaries/BoundaryCore.py:267` computes `traction = constraints[nboundary]._compute_traction_force()`.
- `src/mpm/boundaries/BoundaryCore.py:271` maps it to grid force with `node[nodeID, bodyID]._update_nodal_force(shape_fn[ln] * traction)`.
- `src/mpm/engines/ULExplicitEngine.py:153` / `src/mpm/engines/EngineKernel.py:552` then compute standard 2D force P2G from particle external/internal forces.

Important distinction:

- The custom bottom wave is not injected by directly writing `node[grid_node_id].force`.
- It is defined as a particle traction value, then the native particle-traction machinery distributes it to background grid nodes via MPM shape functions.

## 4. A or B Classification

The implementation is closest to:

A. particle force/traction stage

with this clarification:

- The formula is computed and stored at the particle traction constraint level.
- It becomes background grid nodal force only after native `apply_particle_traction_constraint()` maps it through shape functions.
- It is not a direct background `grid_node_id` force prescription in the custom boundary code.

## 5. Quiet Boundary Equivalence Judgment

Current implementation is not strictly equivalent to FLAC3D quiet boundary internals.

It is equivalent in boundary-condition idea at the reduced MPM level because:

- the input velocity is converted to impedance traction using `2*rho*Cs*v_input`;
- a dashpot term opposes current bottom particle velocity using `-rho*Cs*v_particle_x`;
- the resulting traction is applied to the dynamic solve through the native MPM force path.

It is not strict FLAC3D quiet boundary equivalence because:

- FLAC3D quiet boundaries are gridpoint/zone-face based in FLAC3D's own discretization;
- this implementation attaches tractions to material particles and then maps them to background grid nodes by MPM shape functions;
- the dashpot velocity source is `particle[pid].v`, not a FLAC3D gridpoint velocity;
- no persistent FLAC3D `grid_node_id` or zone-face quiet-boundary object exists here.

Final judgment:

```text
FLAC3D quiet-boundary idea: PARTIAL/PASS
Strict FLAC3D quiet-boundary implementation equivalence: FAIL / not exact
```

