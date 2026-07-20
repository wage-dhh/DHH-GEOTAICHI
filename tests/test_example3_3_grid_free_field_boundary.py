from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "example3_3_flac3d_free_field_new.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location("example3_3_flac3d_free_field_new", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_grid_interface_pair_schema_and_layer_uniqueness():
    module = load_example_module()
    module.reset_output()
    mpm, _, grid_params, _ = module.build_fresh_geometry()
    pairs = module.build_grid_boundary_pairs(mpm, grid_params)
    module.write_csv(
        module.GRID_INTERFACE_PAIR_CSV,
        [*module.REQUIRED_PAIR_COLUMNS, "A_i", "grid_height", "unit_thickness"],
        pairs,
    )
    validation = module.validate_grid_boundary_pairs(pairs)

    rows = read_rows(module.GRID_INTERFACE_PAIR_CSV)
    assert rows
    for column in module.REQUIRED_PAIR_COLUMNS:
        assert column in rows[0]
    assert validation["pair_status"] == "PASS"
    assert validation["side_layer_unique"] is True
    assert validation["left_right_layer_match"] is True
    assert float(validation["max_abs_dz"]) < module.PAIR_TOLERANCE

    for side in ("left", "right"):
        layer_ids = [int(row["layer_id"]) for row in rows if row["side"] == side]
        assert sorted(layer_ids) == list(range(len(layer_ids)))
        assert len(layer_ids) == len(set(layer_ids))


def test_lateral_free_field_force_is_grid_level_and_balanced():
    module = load_example_module()
    pair = {
        "pair_id": 0,
        "side": "left",
        "layer_id": 0,
        "main_grid_id": 1,
        "ff_grid_id": 2,
        "A_i": 0.125,
    }
    main_v = np.zeros(3, dtype=float)
    ff_v = np.zeros(3, dtype=float)
    grid_force = np.zeros(3, dtype=float)
    main_v[1] = 1.0
    ff_v[2] = 3.0
    rows = module.assemble_lateral_free_field_grid_force([pair], main_v, ff_v, grid_force, time_value=0.0)

    expected = module.RHO * module.CS * pair["A_i"] * (ff_v[2] - main_v[1])
    assert math.isclose(rows[0]["Fx_main"], expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert math.isclose(rows[0]["Fx_ff"], -expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert math.isclose(rows[0]["balance_error"], 0.0, abs_tol=1.0e-30)
    assert math.isclose(grid_force[1], expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert math.isclose(grid_force[2], -expected, rel_tol=1.0e-12, abs_tol=1.0e-12)


def test_no_particle_particle_pair_path_in_fresh_grid_boundary_script():
    text = EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "particle_particle_pair" not in text
    assert "particle.force += Fx" not in text
    assert "assemble_lateral_free_field_grid_force" in text
