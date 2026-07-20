from __future__ import annotations

import csv
import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "example3_3_free_field_shear_wave_mpm.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location("example3_3_free_field_shear_wave_mpm", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_result():
    module = load_example_module()
    module.setup_output_dirs()
    settings = module.Settings()
    static_state = module.run_static_equilibrium(settings)
    result = module.run_dynamic_shear_wave(settings, static_state)
    diagnostics = module.write_kohler_boundary_diagnostics(result, settings)
    histories = module.save_histories(result)
    return module, settings, result, diagnostics, histories


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_old_equivalent_lateral_boundary_is_disabled_and_scale_removed():
    _, _, _, diagnostics, _ = build_result()
    rows = read_rows(diagnostics["legacy_path"])
    assert rows
    assert all(row["disabled"] == "True" for row in rows)
    assert all(row["used_in_current_simulation"] == "False" for row in rows)
    removed_name = "main_lateral" + "_coupling_scale"
    assert removed_name not in (ROOT / "src" / "mpm" / "boundaries" / "KohlerStrictLateralFreeFieldBoundary.py").read_text(encoding="utf-8")


def test_free_field_columns_are_2d_independent_mpm_models():
    _, _, result, diagnostics, _ = build_result()
    manager = result.kohler_boundary_manager
    assert manager.left_free_field_column.material_points.shape[1] == 2
    assert manager.left_free_field_column.grid_nodes.shape[1] == 2
    rows = read_rows(diagnostics["independence_path"])
    for row in rows:
        assert row["has_2d_material_points"] == "True"
        assert row["has_2d_grid"] == "True"
        assert row["has_independent_mass"] == "True"
        assert row["has_independent_momentum"] == "True"
        assert row["has_independent_velocity"] == "True"
        assert row["has_independent_stress"] == "True"
        assert row["has_independent_deformation_gradient"] == "True"
        assert row["shares_memory_with_main_model"] == "False"
        assert row["status"] == "PASS"


def test_column_width_is_four_grid_cells_and_material_profile_matches():
    _, _, _, diagnostics, _ = build_result()
    for row in read_rows(diagnostics["geometry_path"]):
        assert int(row["expected_num_cells_width"]) == 4
        assert int(row["actual_num_cells_width"]) == 4
        assert math.isclose(float(row["actual_width"]), float(row["expected_width"]))
        assert row["material_profile_match"] == "True"
        assert row["status"] == "PASS"


def test_true_periodic_boundary_is_enforced():
    _, _, _, diagnostics, _ = build_result()
    for row in read_rows(diagnostics["periodic_mapping_path"]):
        assert row["shared_dof"] == "True"
        assert row["status"] == "PASS"
    for row in read_rows(diagnostics["periodic_path"])[:100]:
        assert float(row["velocity_difference_norm"]) < 1.0e-12
        assert float(row["momentum_difference_norm"]) < 1.0e-12
        assert row["status"] == "PASS"


def test_static_sigma_xx_sigma_xz_and_full_height_pairing():
    _, _, _, diagnostics, _ = build_result()
    static_rows = read_rows(diagnostics["static_path"])
    assert static_rows
    assert any(abs(float(row["main_sigma_xx_static"])) > 0.0 for row in static_rows)
    assert all(row["status"] == "PASS" for row in static_rows)
    for row in read_rows(diagnostics["pairing_path"]):
        assert row["material_match"] == "True"
        assert row["status"] == "PASS"
    for row in read_rows(diagnostics["coverage_path"]):
        assert float(row["coverage_ratio"]) == 1.0
        assert row["status"] == "PASS"


def test_eq30_components_sigma_dynamic_and_main_force_injection():
    _, settings, _, diagnostics, _ = build_result()
    rows = read_rows(diagnostics["traction_application_path"])
    assert rows
    for row in rows[:200]:
        normal = float(row["static_support_normal"]) + float(row["dynamic_stress_normal"]) + float(row["dashpot_normal"])
        shear = float(row["static_support_shear"]) + float(row["dynamic_stress_shear"]) + float(row["dashpot_shear"])
        assert math.isclose(float(row["total_traction_normal"]), normal, rel_tol=1.0e-12, abs_tol=1.0e-12)
        assert math.isclose(float(row["total_traction_shear"]), shear, rel_tol=1.0e-12, abs_tol=1.0e-12)
        assert row["traction_added_to_main_force"] == "True"
        assert row["status"] == "PASS"
    assert math.isclose(settings.quiet_shear_coeff, 10.0, rel_tol=1.0e-12)
    assert math.isclose(settings.quiet_normal_coeff, 17.320532, rel_tol=1.0e-6)
    for row in read_rows(diagnostics["force_injection_path"]):
        assert row["num_force_injected_points"] == row["num_coupled_boundary_points"]
        assert row["force_entered_momentum_update"] == "True"
        assert row["status"] == "PASS"


def test_update_order_no_feedback_and_corner_superposition():
    _, _, _, diagnostics, _ = build_result()
    for row in read_rows(diagnostics["update_order_path"]):
        assert row["left_ff_updated"] == "True"
        assert row["right_ff_updated"] == "True"
        assert row["dynamic_stress_computed"] == "True"
        assert row["full_height_lateral_traction_computed"] == "True"
        assert row["lateral_traction_injected_to_main"] == "True"
        assert row["main_model_updated"] == "True"
        assert row["main_to_ff_feedback_detected"] == "False"
        assert row["status"] == "PASS"
    for row in read_rows(diagnostics["corner_path"]):
        assert math.isclose(float(row["total_force_x"]), float(row["base_force_x"]) + float(row["lateral_force_x"]), rel_tol=1.0e-12)
        assert math.isclose(float(row["total_force_z"]), float(row["base_force_z"]) + float(row["lateral_force_z"]), rel_tol=1.0e-12)
        assert row["superposition_correct"] == "True"


def test_kernel_cycle_history_and_vtk_integrity():
    module, _, result, diagnostics, histories = build_result()
    for row in read_rows(diagnostics["kernel_update_path"])[:40]:
        assert all(
            row[key] == "True"
            for key in (
                "p2g_mass_done",
                "p2g_momentum_done",
                "internal_force_done",
                "external_force_done",
                "periodic_boundary_done",
                "grid_update_done",
                "g2p_done",
                "position_update_done",
                "deformation_gradient_update_done",
                "stress_update_done",
                "dynamic_stress_extracted",
            )
        )
        assert row["status"] == "PASS"
    for row in read_rows(histories["history_source_integrity"]):
        assert row["copied_from_other_history"] == "False"
        assert row["uses_reference_curve"] == "False"
        assert row["status"] == "PASS"
    vtk = module.write_explicit_ff_column_vtk(result)
    validation = module.validate_kohler_vtk_outputs(vtk)
    assert validation["validation_pass"] is True
    for row in read_rows(vtk["source_check_path"]):
        assert row["copied_from_main_model"] == "False"
        assert row["status"] == "PASS"
    for key in ("main_model_pvd", "left_free_field_column_pvd", "right_free_field_column_pvd"):
        root = ET.parse(vtk[key]).getroot()
        assert root.findall(".//DataSet")
