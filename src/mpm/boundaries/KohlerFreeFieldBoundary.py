"""Compatibility imports for the strict Kohler lateral free-field boundary."""

from src.mpm.boundaries.KohlerStrictLateralFreeFieldBoundary import (  # noqa: F401
    KohlerCornerSuperposition,
    KohlerDynamicStressTraction,
    KohlerFreeFieldColumn2D,
    KohlerLateralDashpotCoupler,
    KohlerMainFFPairing,
    KohlerMaterial,
    KohlerPeriodicBoundary,
    KohlerStaticSupport,
    KohlerStrictLateralFreeFieldBoundaryManager,
)

KohlerFreeFieldBoundaryManager = KohlerStrictLateralFreeFieldBoundaryManager

