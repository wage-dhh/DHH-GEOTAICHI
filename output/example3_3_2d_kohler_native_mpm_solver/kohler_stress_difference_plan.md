# Kohler Stress Difference Modification Plan

## Scope

- Target file: `scripts/run_geometry_corrected_dynamic_native_mpm.py`
- Only allowed source edit: `update_corrected_particle_tractions()` lateral interface section.
- Do not modify GeoTaichi solver, P2G, G2P, stress update, material model, particle generator, grid generator, `geometry_only_corrected`, bottom input velocity, bottom traction conversion, timestep, or monitor points.

## 1. Current Side Formula Location

Current side free-field traction is implemented in the Taichi kernel:

- function: `update_corrected_particle_tractions()`
- current lateral loop: `for i in range(pair_count):`
- current code pattern:

```python
vrel = particle[ff_pid].v - particle[main_pid].v
sigma_ff = particle[ff_pid].stress
tx = n * sigma_ff[0] + rho_cp * vrel[0]
tz = n * sigma_ff[3] + rho_cs * vrel[1]
traction = ti.Vector([tx, tz])
particle_traction[mtid].traction = traction
particle_traction[ftid].traction = -traction
```

Current formula is therefore:

```text
sigma_ff*n + rho*C*(v_ff-v_main)
```

## 2. Current Available Variables

Available directly inside the interface-pair loop:

- `main_pid`: main model particle id
- `ff_pid`: free-field particle id
- `n`: side normal sign, left = -1 and right = +1
- `particle[main_pid].v`: `v_main`
- `particle[ff_pid].v`: `v_ff`
- `particle[main_pid].stress`: `sigma_main` / `tau_main` source
- `particle[ff_pid].stress`: `sigma_ff` / `tau_ff` source
- `rho_cp`: x-direction impedance coefficient currently used by this script
- `rho_cs`: z-direction impedance coefficient currently used by this script

Required new variables can be obtained without changing solver or material code:

```python
sigma_main = particle[main_pid].stress
sigma_ff = particle[ff_pid].stress
stress_diff = sigma_ff - sigma_main
```

## 3. Stress Tensor Index Confirmation

GeoTaichi particle stress is a 6-component vector. Source checks in `src/mpm/structs/Particle.py` and `src/utils/MaterialKernel.py` show the canonical storage:

- `stress[0]`: `sigma_xx`
- `stress[1]`: `sigma_yy`
- `stress[2]`: `sigma_zz`
- `stress[3]`: `sigma_xy` in canonical 3D notation
- `stress[4]`: `sigma_yz`
- `stress[5]`: `sigma_xz`

Important implementation note for this existing 2D x-z example:

- `scripts/run_geometry_corrected_dynamic_native_mpm.py` currently uses `stress[3]` as the interface shear component in the z traction calculation.
- The requested formula says `sigma_xz`; to avoid changing stress-update conventions or solver internals, this modification keeps the same existing script convention and changes only from `ff_stress[3]` to `(ff_stress[3] - main_stress[3])`.
- No remapping to `stress[5]` will be made in this task because that would be a broader convention change outside the requested single-function formula change.

## Planned Code Change

Inside `update_corrected_particle_tractions()` lateral loop, replace:

```python
sigma_ff = particle[ff_pid].stress
tx = n * sigma_ff[0] + rho_cp * vrel[0]
tz = n * sigma_ff[3] + rho_cs * vrel[1]
```

with:

```python
sigma_ff = particle[ff_pid].stress
sigma_main = particle[main_pid].stress
stress_diff = sigma_ff - sigma_main
tx = n * stress_diff[0] + rho_cp * vrel[0]
tz = n * stress_diff[3] + rho_cs * vrel[1]
```

Keep unchanged:

```python
particle_traction[mtid].traction = traction
particle_traction[ftid].traction = -traction
```

## Validation Plan

After modifying the single function:

1. Set output directory for this modified run to `kohler_stress_difference_run/` so the previous `new_geometry_corrected_dynamic_run/` remains available for before/after comparison.
2. Keep `SIMULATION_TIME = 0.015`, `DT = 1e-5`, `INPUT_PERIOD = 0.01`, input velocity function, bottom traction conversion, and monitor targets unchanged.
3. Run `scripts/run_geometry_corrected_dynamic_native_mpm.py`.
4. Generate required files in `kohler_stress_difference_run/`:
   - `side_boundary_formula_after.md`
   - `side_force_balance_after.csv`
   - `figure3_9_kohler_stress_difference.png`
   - `kohler_stress_difference_validation_report.md`
5. Compare before/after metrics:
   - main peak error
   - free-field peak error
   - NRMSE
   - correlation
   - phase difference
6. Check:
   - action-reaction residual remains zero
   - side force magnitude does not show abnormal growth relative to before
   - reflection indicator improves or worsens based on existing reflection diagnostics.
