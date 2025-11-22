# engine/physics/__init__.py

from .mana_energy import (
    ManaEnergyParams,
    mana_purity,
    mana_energy_density,
)

from .mana_phase import (
    PhaseCode,
    PhaseProperties,
    PhaseThresholds,
    classify_phases,
)

from .mana_conversion import apply_phase_conversions

__all__ = [
    # energy
    "ManaEnergyParams",
    "mana_purity",
    "mana_energy_density",
    # phases
    "PhaseCode",
    "PhaseProperties",
    "PhaseThresholds",
    "classify_phases",
    # conversions
    "apply_phase_conversions",
]
