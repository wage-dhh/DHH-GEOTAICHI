# Side Boundary Formula After Modification

- source_checked: `F:\dhh\GeoTaichi-dhh\scripts\run_geometry_corrected_dynamic_native_mpm.py`
- only source function modified: `update_corrected_particle_tractions()` lateral interface loop.

## Before

`sigma_ff*n+rho*C*(v_ff-v_main)`

## After

`(sigma_ff-sigma_main)*n+rho*C*(v_ff-v_main)`

Component form now used by the kernel:

- `vrel = particle[ff_pid].v - particle[main_pid].v`
- `sigma_ff = particle[ff_pid].stress`
- `sigma_main = particle[main_pid].stress`
- `stress_diff = sigma_ff - sigma_main`
- `tx = n * stress_diff[0] + rho_cp * vrel[0]`
- `tz = n * stress_diff[3] + rho_cs * vrel[1]`
- `main_particle_traction = traction`
- `ff_particle_traction = -traction`

Stress-index note: this run preserves the existing script convention that uses `stress[3]` for the interface shear component in this 2D x-z implementation; no solver stress storage remapping was changed.
