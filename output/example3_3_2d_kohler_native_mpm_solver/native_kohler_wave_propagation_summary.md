# Native Kohler Wave Propagation Summary

- source code modified: NO
- diagnostic simulation_time override: 0.5
- dt: 1e-05
- expected steps: 50000
- recorded rows: 50001
- elapsed_wall_time_s: 118.38944387435913
- history_csv: `F:\dhh\GeoTaichi-dhh\output\example3_3_2d_kohler_native_mpm_solver\native_kohler_wave_propagation_history.csv`
- figure: `F:\dhh\GeoTaichi-dhh\output\example3_3_2d_kohler_native_mpm_solver\native_kohler_wave_propagation.png`

## Monitors

- FF_bottom particle_id: 0
- Main_bottom particle_id: 47
- FF_top particle_id: 38, nearest_distance_to_requested_point: 0.1767766952966369
- Main_top particle_id: 503, nearest_distance_to_requested_point: 0.1767766952966369

## Travel Time

- Cs: 12.649110640673518
- H: 5.0
- expected_H_over_Cs: 0.3952847075210474
- FF_top_first_arrival_time: 0.31408000000017333
- Main_top_first_arrival_time: 0.2291600000000884
- FF_top_threshold_used: 1.4149856042422472e-08
- Main_top_threshold_used: 1.6956074659901789e-07

## Peak Velocities

- input_peak: 0.001
- FF_bottom_peak: 9.422359653399326e-06
- Main_bottom_peak: 1.8805210856953636e-05
- FF_top_peak: 2.829971208484494e-07
- Main_top_peak: 3.3912149319803575e-06
- FF_bottom_peak_ratio_to_input: 0.009422359653399326
- Main_bottom_peak_ratio_to_input: 0.018805210856953636
- FF_top_peak_ratio_to_input: 0.0002829971208484494
- Main_top_peak_ratio_to_input: 0.0033912149319803575

## Trace Counts

- trace_counts: {'solver_core_calls': 50001, 'ul_explicit_usl_updating_calls': 50001, 'p2g_calls': 50001, 'compute_forces_calls': 50001, 'grid_update_calls': 50001, 'g2p_calls': 50001, 'velocity_gradient_calls': 50001, 'stress_update_calls': 50001, 'compliant_particle_traction_update_calls': 50001, 'calculate_interpolation_calls': 50001, 'grid_velocity_calls': 50001, 'particle_traction_calls': 50001, 'native_traction_calls': 50001, 'absorbing_constraint_calls': 50001, 'grid_update_stage_record_calls': 50001, 'kinematic_constraint_calls': 50001}

## Judgment

- The small bottom amplitude is not only a short-time issue; even over 0.5 s it remains below 5% of input.
