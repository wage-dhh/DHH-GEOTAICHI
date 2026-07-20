"""2D x-z Kohler-style lateral free-field boundary components.

This module is intentionally separate from the 3D Example 3.3 validation path.
It reuses the validated plane-strain column mechanics while exposing 2D-specific
class names for the independent x-z example.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.mpm.boundaries.KohlerStrictLateralFreeFieldBoundary import (
    KohlerFreeFieldColumn2D,
    KohlerMaterial,
    KohlerStrictLateralFreeFieldBoundaryManager,
)


@dataclass
class Kohler2DLateralFreeFieldBoundaryManager(KohlerStrictLateralFreeFieldBoundaryManager):
    """Independent 2D x-z manager for Example 3.3 plane-strain validation."""

    mp_per_cell: int = 2


__all__ = [
    "KohlerMaterial",
    "KohlerFreeFieldColumn2D",
    "Kohler2DLateralFreeFieldBoundaryManager",
]

