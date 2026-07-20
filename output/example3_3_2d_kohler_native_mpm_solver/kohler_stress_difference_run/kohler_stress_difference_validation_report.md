# Kohler Stress Difference Validation Report

- before_run: `F:\dhh\GeoTaichi-dhh\output\example3_3_2d_kohler_native_mpm_solver\new_geometry_corrected_dynamic_run`
- after_run: `F:\dhh\GeoTaichi-dhh\output\example3_3_2d_kohler_native_mpm_solver\kohler_stress_difference_run`
- source_modified: `F:\dhh\GeoTaichi-dhh\scripts\run_geometry_corrected_dynamic_native_mpm.py`
- modified function only: `update_corrected_particle_tractions()` lateral interface stress term.
- unchanged: geometry, particle generator, grid generator, GeoTaichi solver source, P2G, G2P, stress update, material, bottom input velocity, bottom traction conversion, timestep, and monitor points.
- simulation_time: 0.015 s
- input period T: 0.01 s
- before history columns: main=`main_top_vx`, free-field=`free_field_top_vx`
- after history columns: main=`main_model_top_vx`, free-field=`soil_column_top_vx`

## Formula Comparison

- Before: `sigma_ff*n+rho*C*(v_ff-v_main)`
- After: `(sigma_ff-sigma_main)*n+rho*C*(v_ff-v_main)`
- Main traction remains `traction`; free-field traction remains `-traction`.

## Metric Comparison

| case | phase | peak_mpm | peak_ref | peak_error | NRMSE | correlation | phase_difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main | before | 0.0522427111864 | 0.0996208384082 | -0.0473781272218 | 0.338782096095 | 0.47807003105 | 0.00127 |
| main | after | 0.0522728040814 | 0.0996208384082 | -0.0473480343268 | 0.339000559653 | 0.477021675885 | 0.00127 |
| free_field | before | 0.0530989915133 | 0.0996208384082 | -0.0465218468949 | 0.331540631649 | 0.508779807456 | 0.00119 |
| free_field | after | 0.0531861111522 | 0.0996208384082 | -0.046434727256 | 0.332408596912 | 0.504846835818 | 0.00121 |

## Action-Reaction Check

- before max residual magnitude: 0
- after max residual magnitude: 0
- after residual fail count (>1e-10): 0 of 60080
- action-reaction after status: PASS

## Side Force Magnitude Check

- before max main side-force magnitude: 0.05937288140871734
- after max main side-force magnitude: 0.05373652300966715
- max side-force growth ratio after/before: 0.90506847123941947
- sum abs side-force growth ratio after/before: 0.99902575260973947
- abnormal side-force growth judgment: NO

## Reflection Check

- before reflection indicator: 0.033749666064977646
- after reflection indicator: 0.033851316198706627
- reflection indicator change after-before: 0.00010165013372898102
- wave reflection improved: NO

## Required Outputs

- side formula after: `F:\dhh\GeoTaichi-dhh\output\example3_3_2d_kohler_native_mpm_solver\kohler_stress_difference_run\side_boundary_formula_after.md`
- side force balance after: `F:\dhh\GeoTaichi-dhh\output\example3_3_2d_kohler_native_mpm_solver\kohler_stress_difference_run\side_force_balance_after.csv`
- figure: `F:\dhh\GeoTaichi-dhh\output\example3_3_2d_kohler_native_mpm_solver\kohler_stress_difference_run\figure3_9_kohler_stress_difference.png`
- metrics compare: `F:\dhh\GeoTaichi-dhh\output\example3_3_2d_kohler_native_mpm_solver\kohler_stress_difference_run\kohler_stress_difference_metrics_compare.csv`

## Conclusion

- Formula modification status: PASS
- Action-reaction status: PASS
- Side force abnormal growth: NO
- Reflection improved: NO
- No tuning, curve scaling, monitor-point modification, geometry modification, or bottom-input modification was applied.
