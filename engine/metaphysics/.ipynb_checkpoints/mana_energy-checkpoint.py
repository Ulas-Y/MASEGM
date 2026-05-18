# engine/physics/mana_energy.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from engine.math.b_calculus import xp
from engine.constants import C_MANA as default_C_mana, K_MANA as default_K_mana, eps as default_eps

@dataclass
class ManaEnergyParams:
    """
    Parameters for the mana energy law

        E_mana = K_mana * rho_mana * purity * C_mana^2

    where:
        - rho_mana is just the local mana density (mana.grid)
        - purity  = mana / (mana + matter)
        - C_mana is the "speed of light" of mana
        - K_mana is a tunable constant that sets the overall scale

    Keep these at 1.0 until you want to play with the actual numbers.
    """
    K_mana: float = default_K_mana               #float = 1.0       # conversion factor (mass -> energy)
    C_mana: float = default_C_mana               #float = 1.0       # max propagation speed in "mana-units"
    eps: float = default_eps                     #float = 1e-12     # numerical floor

    @property
    def C_mana_sq(self) -> float:
        return self.C_mana * self.C_mana


def mana_purity(
    mana_grid: Any,
    matter_grid: Any,
    eps: float = default_eps,
) -> Any:
    """
    Compute local purity:

        purity = mana / (mana + matter)

    with a tiny epsilon to avoid division by zero.
    """
    mana_grid = xp.asarray(mana_grid)
    matter_grid = xp.asarray(matter_grid)
    eps_arr = xp.asarray(eps)

    total = mana_grid + matter_grid
    total = xp.maximum(total, eps_arr)
    return mana_grid / total


def mana_energy_density(
    mana_grid: Any,
    purity: Any,
    phase: Any,
    params: ManaEnergyParams,
) -> Any:
    """
    Phase-aware mana energy density.
    High phases (especially Purinium) get massive energy density.
    """
    """
    Local mana energy density:

        E = K_mana * rho_mana * purity * C_mana^2

    Right now we treat rho_mana == mana_grid.
    If later you introduce a "mass per unit mana" you can factor it in here.
    """
    mana_grid = xp.asarray(mana_grid)
    purity = xp.asarray(purity)
    phase = xp.asarray(phase)

    k_mana = xp.asarray(params.K_mana)
    c_mana_sq = xp.asarray(params.C_mana_sq)

    base_energy = k * mana_grid * purity * c2
    
    # Apply phase multiplier
    mult = xp.asarray([PHASE_ENERGY_MULT.get(int(p), 1.0) for p in phase.flat])
    mult = mult.reshape(phase.shape)
    
    return base_energy * mult
