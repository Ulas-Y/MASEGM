# engine/physics/mana_energy.py

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


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
    K_mana: float = 1.0       # conversion factor (mass -> energy)
    C_mana: float = 1.0       # max propagation speed in "mana-units"
    eps: float = 1e-12        # numerical floor

    @property
    def C_mana_sq(self) -> float:
        return self.C_mana * self.C_mana


def mana_purity(
    mana_grid: np.ndarray,
    matter_grid: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute local purity:

        purity = mana / (mana + matter)

    with a tiny epsilon to avoid division by zero.
    """
    total = mana_grid + matter_grid
    total = np.maximum(total, eps)
    return mana_grid / total


def mana_energy_density(
    mana_grid: np.ndarray,
    purity: np.ndarray,
    params: ManaEnergyParams,
) -> np.ndarray:
    """
    Local mana energy density:

        E = K_mana * rho_mana * purity * C_mana^2

    Right now we treat rho_mana == mana_grid.
    If later you introduce a "mass per unit mana" you can factor it in here.
    """
    return params.K_mana * mana_grid * purity * params.C_mana_sq
