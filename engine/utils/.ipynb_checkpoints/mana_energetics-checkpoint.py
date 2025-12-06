import numpy as np
from engine.constants import K_MANA, C_MANA

def mana_energy_from_state(mana_grid: np.ndarray,
                           purity: np.ndarray) -> np.ndarray:
    """
    Compute mana energy density E_cell = K_MANA * mana * purity * C_MANA^2
    """
    return K_MANA * mana_grid * purity * (C_MANA ** 2)
