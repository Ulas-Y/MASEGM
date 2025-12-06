# engine/physics/mana_conversion.py

from __future__ import annotations

from typing import Sequence

import numpy as np

from .mana_phase import PhaseCode, PhaseProperties


def apply_phase_conversions(
    mana_grid: np.ndarray,
    matter_grid: np.ndarray,
    energy_grid: np.ndarray,
    purity: np.ndarray,
    phase: np.ndarray,
    phase_props: Sequence[PhaseProperties],
    dt: float,
) -> None:
    """
    Apply matter <-> mana <-> energy reactions based on phase.

    This is basically your old _apply_matter_conversion, but moved
    into its own module to keep PhaseTransitionRule cleaner.

    For now we keep it simple and conservative:
      - matter_to_mana is proportional to local matter
      - mana_to_energy is proportional to local mana
      - for Purinium, matter annihilation also dumps energy.
    """
    for code in PhaseCode:
        props = phase_props[int(code)]

        # skip phases that don't convert anything
        if props.matter_to_mana == 0.0 and props.mana_to_energy == 0.0:
            continue

        mask = (phase == int(code))
        if not np.any(mask):
            continue

        if props.matter_to_mana != 0.0:
            # matter annihilation -> mana (and sometimes energy)
            available = matter_grid[mask]
            converted = props.matter_to_mana * available * dt
            matter_grid[mask] -= converted
            mana_grid[mask] += converted

            if code == PhaseCode.PURINIUM:
                # Purinium dumps the converted rest-mass as mana-energy.
                # (You can multiply by a big factor later if you want
                #  more "star-like" behaviour.)
                energy_grid[mask] += converted

        if props.mana_to_energy != 0.0:
            # mana leaking into energy (radiation, heating, etc.)
            m_local = mana_grid[mask]
            leaked = props.mana_to_energy * m_local * dt
            mana_grid[mask] -= leaked
            energy_grid[mask] += leaked
