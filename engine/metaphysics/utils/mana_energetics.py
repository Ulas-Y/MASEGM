from engine.constants import K_MANA, C_MANA
from engine.math import b_calculus

xp = b_calculus.xp


def mana_energy_from_state(mana_grid,
                           purity):
    """
    Compute mana energy density E_cell = K_MANA * mana * purity * C_MANA^2
    """
    mana_array = xp.asarray(mana_grid)
    purity_array = xp.asarray(purity)

    k_mana = xp.asarray(K_MANA)
    c_mana = xp.asarray(C_MANA)

    return k_mana * mana_array * purity_array * xp.power(c_mana, 2)
