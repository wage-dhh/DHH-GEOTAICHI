# Native Compliant Base Report

- Native solver integration: PASS
- Compliant-base force execution: PASS
- Compliant-base physical response: FAIL
- native solver used: YES
- boundary insertion file: `examples/example3_3_2d_kohler_native_mpm_solver.py`
- boundary insertion function: `NativeCompliantBase.apply()` / `update_kohler_particle_traction()`
- insertion stage: `compute_grid_velcity(...) -> update_kohler_particle_traction(...) -> apply_particle_traction_constraints(...) -> compute_forces(...)`
- direct prescribed input velocity: NO
- horizontal input traction uses `2*rho*Cs*v_su`: YES
- dashpot active: YES
- bottom node/body entries: 31
- bottom total area: 7.0
- FF_bottom peak velocity: 9.422380571777467e-06
- Main_bottom peak velocity: 1.8805210856953636e-05
- target peak velocity: 0.001
- FF_bottom peak ratio to target: 0.009422380571777467
- Main_bottom peak ratio to target: 0.018805210856953636
- FF_bottom correlation with target: -0.1586800527889297
- Main_bottom correlation with target: -0.19358323086702695
- FF_bottom max absolute difference: 0.0009950769442875753
- Main_bottom max absolute difference: 0.0009901604296610458
- monitor source in legacy bottom history: particle velocity after G2P
- monitor source in force-velocity trace: grid velocity after kinematic constraint and particle velocity after G2P
- obvious reflection: no obvious bottom-level reflection in this short run; top/lateral reflection is not assessed in this bottom-only stage
- native trace counts: {'solver_core_calls': 1502, 'ul_explicit_usl_updating_calls': 1502, 'p2g_calls': 1502, 'compute_forces_calls': 1502, 'grid_update_calls': 1502, 'g2p_calls': 1502, 'velocity_gradient_calls': 1502, 'stress_update_calls': 1502, 'compliant_particle_traction_update_calls': 1502, 'calculate_interpolation_calls': 1502, 'grid_velocity_calls': 1502, 'particle_traction_calls': 1502, 'native_traction_calls': 1502, 'absorbing_constraint_calls': 1502, 'grid_update_stage_record_calls': 1502, 'kinematic_constraint_calls': 1502}

Result: PARTIAL
