"""Minimal FLAC3D zone-mesh parser for Example 3.3 geometry diagnostics.

This module is intentionally independent from the MPM solver.  It parses the
small subset of FLAC3D ``gen zone brick/wedge`` commands used by Dynamic
Analysis Example 3.3 and writes auditable mesh diagnostics.

Stage 1 scope:
    - parse and store gridpoints/zones/connectivity;
    - support 8-node brick zones and 6-node wedge zones as mesh data;
    - diagnose history point gridpoint/cell containment;
    - do not generate MPM material points or free-field grids.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class GridPoint:
    id: int
    x: float
    y: float
    z: float

    @property
    def coord(self) -> Point3:
        return (self.x, self.y, self.z)


@dataclass
class Zone:
    id: int
    zone_type: str
    node_ids: tuple[int, ...]
    command: str
    group: str = "main_grid"
    size: tuple[int, int, int] | None = None

    def contains(self, point: Point3, gridpoints: list[GridPoint], tol: float = 1.0e-9) -> bool:
        raise NotImplementedError


@dataclass
class BrickZone(Zone):
    zone_type: str = field(default="brick", init=False)

    def contains(self, point: Point3, gridpoints: list[GridPoint], tol: float = 1.0e-9) -> bool:
        coords = np.asarray([gridpoints[node_id].coord for node_id in self.node_ids], dtype=float)
        p = np.asarray(point, dtype=float)
        mins = coords.min(axis=0) - tol
        maxs = coords.max(axis=0) + tol
        return bool(np.all(p >= mins) and np.all(p <= maxs))


@dataclass
class WedgeZone(Zone):
    zone_type: str = field(default="wedge", init=False)

    def contains(self, point: Point3, gridpoints: list[GridPoint], tol: float = 1.0e-9) -> bool:
        vertices = np.asarray([gridpoints[node_id].coord for node_id in self.node_ids], dtype=float)
        p = np.asarray(point, dtype=float)
        if np.any(p < vertices.min(axis=0) - tol) or np.any(p > vertices.max(axis=0) + tol):
            return False
        # Standard triangular-prism decomposition.  This is sufficient for the
        # current diagnostics and avoids claiming FLAC3D shape-function parity.
        tetra_indices = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))
        return any(_point_in_tetra(p, vertices[list(indices)], tol) for indices in tetra_indices)


def _point_in_tetra(point: np.ndarray, tetra: np.ndarray, tol: float) -> bool:
    matrix = np.column_stack((tetra[1] - tetra[0], tetra[2] - tetra[0], tetra[3] - tetra[0]))
    det = float(np.linalg.det(matrix))
    if abs(det) < tol:
        return False
    bary = np.linalg.solve(matrix, point - tetra[0])
    weights = np.array([1.0 - bary.sum(), bary[0], bary[1], bary[2]], dtype=float)
    return bool(np.all(weights >= -tol) and np.all(weights <= 1.0 + tol))


class ZoneMesh:
    def __init__(self, merge_tolerance: float = 1.0e-9) -> None:
        self.merge_tolerance = merge_tolerance
        self.gridpoints: list[GridPoint] = []
        self.zones: list[Zone] = []
        self._point_index: dict[tuple[int, int, int], int] = {}

    @classmethod
    def from_commands(cls, commands: Iterable[str]) -> "ZoneMesh":
        mesh = cls()
        for command in commands:
            mesh.add_command(command)
        return mesh

    def add_command(self, command: str) -> None:
        command = " ".join(command.strip().split())
        if not command:
            return
        tokens = command.split()
        if len(tokens) < 5 or tokens[:2] != ["gen", "zone"]:
            raise ValueError(f"Unsupported FLAC3D command: {command}")
        kind = tokens[2]
        size = _parse_size(tokens)
        points = _parse_points(tokens)
        if kind == "brick":
            self._add_brick_command(command, size, points)
        elif kind == "wedge":
            self._add_wedge_command(command, size, points)
        else:
            raise ValueError(f"Unsupported zone kind {kind!r} in command: {command}")

    def _add_gridpoint(self, coord: Point3) -> int:
        key = tuple(int(round(value / self.merge_tolerance)) for value in coord)
        existing = self._point_index.get(key)
        if existing is not None:
            return existing
        node_id = len(self.gridpoints)
        self.gridpoints.append(GridPoint(node_id, float(coord[0]), float(coord[1]), float(coord[2])))
        self._point_index[key] = node_id
        return node_id

    def _add_brick_command(self, command: str, size: tuple[int, int, int], points: dict[str, Point3]) -> None:
        nx, ny, nz = size
        p0 = points.get("p0", (0.0, 0.0, 0.0))
        # Example 3.3 omits p1/p2/p3 for axis-aligned bricks.  For this subset,
        # FLAC coordinates are reconstructed as p0 + size lengths.
        p1 = points.get("p1", (p0[0] + nx, p0[1], p0[2]))
        p2 = points.get("p2", (p0[0], p0[1] + ny, p0[2]))
        p3 = points.get("p3", (p0[0], p0[1], p0[2] + nz))
        origin = np.asarray(p0, dtype=float)
        vx = (np.asarray(p1, dtype=float) - origin) / nx
        vy = (np.asarray(p2, dtype=float) - origin) / ny
        vz = (np.asarray(p3, dtype=float) - origin) / nz
        node_lattice: dict[tuple[int, int, int], int] = {}
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    coord = tuple((origin + i * vx + j * vy + k * vz).tolist())
                    node_lattice[(i, j, k)] = self._add_gridpoint(coord)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    node_ids = (
                        node_lattice[(i, j, k)],
                        node_lattice[(i + 1, j, k)],
                        node_lattice[(i + 1, j + 1, k)],
                        node_lattice[(i, j + 1, k)],
                        node_lattice[(i, j, k + 1)],
                        node_lattice[(i + 1, j, k + 1)],
                        node_lattice[(i + 1, j + 1, k + 1)],
                        node_lattice[(i, j + 1, k + 1)],
                    )
                    self.zones.append(BrickZone(id=len(self.zones), node_ids=node_ids, command=command, size=size))

    def _add_wedge_command(self, command: str, size: tuple[int, int, int], points: dict[str, Point3]) -> None:
        vertices = _example_wedge_vertices(command, size, points)
        node_ids = tuple(self._add_gridpoint(vertex) for vertex in vertices)
        self.zones.append(WedgeZone(id=len(self.zones), node_ids=node_ids, command=command, size=size))

    def nearest_gridpoint(self, point: Point3) -> tuple[GridPoint, float]:
        target = np.asarray(point, dtype=float)
        coords = np.asarray([gp.coord for gp in self.gridpoints], dtype=float)
        distances = np.linalg.norm(coords - target, axis=1)
        index = int(np.argmin(distances))
        return self.gridpoints[index], float(distances[index])

    def find_gridpoint(self, point: Point3, tol: float = 1.0e-9) -> GridPoint | None:
        nearest, distance = self.nearest_gridpoint(point)
        return nearest if distance <= tol else None

    def containing_zones(self, point: Point3, tol: float = 1.0e-9) -> list[Zone]:
        return [zone for zone in self.zones if zone.contains(point, self.gridpoints, tol)]

    def zmax_by_xy(self) -> list[dict[str, object]]:
        by_xy: dict[tuple[float, float], float] = {}
        for gp in self.gridpoints:
            key = (gp.x, gp.y)
            by_xy[key] = max(by_xy.get(key, -math.inf), gp.z)
        return [
            {"x": x, "y": y, "z_max": zmax}
            for (x, y), zmax in sorted(by_xy.items(), key=lambda item: (item[0][0], item[0][1]))
        ]

    def history_point_report(self, points: Iterable[Point3]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for point in points:
            real_gp = self.find_gridpoint(point)
            containing = self.containing_zones(point)
            nearest, distance = self.nearest_gridpoint(point)
            group = "main_grid" if (real_gp is not None or containing) and point[0] >= 0.0 and point[1] >= 0.0 else "outside"
            if point[0] < 0.0 or point[1] < 0.0:
                group = "free_field_pending"
            rows.append(
                {
                    "target_x": point[0],
                    "target_y": point[1],
                    "target_z": point[2],
                    "is_real_grid_point": real_gp is not None,
                    "is_inside_real_cell": bool(containing),
                    "group": group,
                    "containing_zone_ids": " ".join(str(zone.id) for zone in containing),
                    "nearest_gridpoint_id": nearest.id,
                    "nearest_x": nearest.x,
                    "nearest_y": nearest.y,
                    "nearest_z": nearest.z,
                    "distance_error": distance,
                    "status": "PASS" if real_gp is not None or containing else "FAIL",
                }
            )
        return rows

    def write_diagnostics(self, output_dir: str | Path, history_points: Iterable[Point3]) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "gridpoints": output / "gridpoints.csv",
            "zones": output / "zones.csv",
            "zone_connectivity": output / "zone_connectivity.csv",
            "geometry_zmax_by_xy": output / "geometry_zmax_by_xy.csv",
            "history_point_containment": output / "history_point_containment.csv",
        }
        _write_csv(paths["gridpoints"], ["id", "x", "y", "z"], [gp.__dict__ for gp in self.gridpoints])
        _write_csv(
            paths["zones"],
            ["id", "zone_type", "group", "size_x", "size_y", "size_z", "command"],
            [
                {
                    "id": zone.id,
                    "zone_type": zone.zone_type,
                    "group": zone.group,
                    "size_x": zone.size[0] if zone.size else "",
                    "size_y": zone.size[1] if zone.size else "",
                    "size_z": zone.size[2] if zone.size else "",
                    "command": zone.command,
                }
                for zone in self.zones
            ],
        )
        max_nodes = max(len(zone.node_ids) for zone in self.zones) if self.zones else 0
        connectivity_header = ["zone_id", "zone_type"] + [f"node_{i}" for i in range(max_nodes)]
        _write_csv(
            paths["zone_connectivity"],
            connectivity_header,
            [
                {
                    "zone_id": zone.id,
                    "zone_type": zone.zone_type,
                    **{f"node_{i}": zone.node_ids[i] if i < len(zone.node_ids) else "" for i in range(max_nodes)},
                }
                for zone in self.zones
            ],
        )
        _write_csv(paths["geometry_zmax_by_xy"], ["x", "y", "z_max"], self.zmax_by_xy())
        history_rows = self.history_point_report(history_points)
        _write_csv(
            paths["history_point_containment"],
            [
                "target_x",
                "target_y",
                "target_z",
                "is_real_grid_point",
                "is_inside_real_cell",
                "group",
                "containing_zone_ids",
                "nearest_gridpoint_id",
                "nearest_x",
                "nearest_y",
                "nearest_z",
                "distance_error",
                "status",
            ],
            history_rows,
        )
        return paths


def _parse_size(tokens: list[str]) -> tuple[int, int, int]:
    if "size" not in tokens:
        raise ValueError("Expected size in FLAC3D zone command")
    index = tokens.index("size")
    return (int(tokens[index + 1]), int(tokens[index + 2]), int(tokens[index + 3]))


def _parse_points(tokens: list[str]) -> dict[str, Point3]:
    points: dict[str, Point3] = {}
    i = 0
    while i < len(tokens):
        if re.fullmatch(r"p[0-7]", tokens[i]):
            points[tokens[i]] = (float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3]))
            i += 4
        else:
            i += 1
    return points


def _example_wedge_vertices(command: str, size: tuple[int, int, int], points: dict[str, Point3]) -> tuple[Point3, ...]:
    if all(f"p{i}" in points for i in range(6)):
        return tuple(points[f"p{i}"] for i in range(6))
    if "p0" not in points:
        raise ValueError(f"Wedge command needs p0 for Example 3.3 default inference: {command}")
    # Example 3.3's first wedge omits p1..p5.  FLAC3D infers them from the
    # current default brick axes.  Stage 1 records that inference explicitly.
    nx, ny, nz = size
    p0 = points["p0"]
    x0, y0, z0 = p0
    return (
        (x0, y0, z0),
        (x0 + nx, y0, z0),
        (x0, y0 + ny, z0),
        (x0, y0, z0 + nz),
        (x0 + nx, y0 + ny, z0),
        (x0, y0 + ny, z0 + nz),
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


EXAMPLE3_3_COMMANDS = [
    "gen zone brick size 6 3 2",
    "gen zone brick size 2 3 2 p0 0 0 2",
    "gen zone brick size 2 3 2 p0 4 0 2",
    "gen zone wedge size 1 3 2 p0 2 0 2",
    "gen zone wedge size 1 3 2 p0 4 3 2 p1 3 3 2 p2 4 0 2 p3 4 3 4 p4 3 0 2 p5 4 0 4",
]

EXAMPLE3_3_HISTORY_POINTS: list[Point3] = [
    (2.0, 1.0, 0.0),
    (2.0, 1.0, 5.0),
    (-1.0, -1.0, 0.0),
    (-1.0, -1.0, 5.0),
    (-1.0, 0.0, 0.0),
    (-1.0, 0.0, 5.0),
    (2.0, -1.0, 0.0),
    (2.0, -1.0, 5.0),
]


def build_example3_3_zone_mesh() -> ZoneMesh:
    return ZoneMesh.from_commands(EXAMPLE3_3_COMMANDS)

