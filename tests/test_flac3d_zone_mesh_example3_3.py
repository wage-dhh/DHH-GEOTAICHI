from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mpm.generator.Flac3DZoneMesh import (  # noqa: E402
    EXAMPLE3_3_HISTORY_POINTS,
    BrickZone,
    WedgeZone,
    build_example3_3_zone_mesh,
)


def main() -> None:
    mesh = build_example3_3_zone_mesh()
    output_dir = ROOT / "output" / "flac3d_zone_mesh_example3_3"
    paths = mesh.write_diagnostics(output_dir, EXAMPLE3_3_HISTORY_POINTS)
    brick_count = sum(isinstance(zone, BrickZone) for zone in mesh.zones)
    wedge_count = sum(isinstance(zone, WedgeZone) for zone in mesh.zones)
    zmax_rows = mesh.zmax_by_xy()
    zmax_x2_y1 = next((row["z_max"] for row in zmax_rows if row["x"] == 2.0 and row["y"] == 1.0), None)
    target = (2.0, 1.0, 5.0)
    is_gp = mesh.find_gridpoint(target) is not None
    containing = mesh.containing_zones(target)

    print(f"total_gridpoints={len(mesh.gridpoints)}")
    print(f"total_zones={len(mesh.zones)}")
    print(f"brick_zone_count={brick_count}")
    print(f"wedge_zone_count={wedge_count}")
    print(f"x=2,y=1_z_max={zmax_x2_y1}")
    print(f"target_(2,1,5)_is_real_grid_point={is_gp}")
    print(f"target_(2,1,5)_inside_real_cell={bool(containing)}")
    print(f"target_(2,1,5)_containing_zone_ids={[zone.id for zone in containing]}")
    for name, path in paths.items():
        print(f"{name}={path}")

    assert len(mesh.gridpoints) > 0
    assert len(mesh.zones) > 0
    assert brick_count > 0
    assert wedge_count > 0


if __name__ == "__main__":
    main()

