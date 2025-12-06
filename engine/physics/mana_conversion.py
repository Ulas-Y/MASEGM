# engine/physics/mana_conversion.py

from __future__ import annotations

from typing import Any, Sequence

from engine.math.b_calculus import xp

from .mana_phase import PhaseCode, PhaseProperties


def apply_phase_conversions(
    mana_grid: Any,
    matter_grid: Any,
    energy_grid: Any,
    purity: Any,
    phase: Any,
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
    mana_grid = xp.asarray(mana_grid)
    matter_grid = xp.asarray(matter_grid)
    energy_grid = xp.asarray(energy_grid)
    phase = xp.asarray(phase)
    dt_arr = xp.asarray(dt)

    for code in PhaseCode:
        props = phase_props[int(code)]

        # skip phases that don't convert anything
        if props.matter_to_mana == 0.0 and props.mana_to_energy == 0.0:
            continue

        mask = phase == xp.asarray(int(code))
        has_phase = mask.any()
        has_phase_bool = bool(has_phase.item() if hasattr(has_phase, "item") else has_phase)
        if not has_phase_bool:
            continue

        if props.matter_to_mana != 0.0:
            # matter annihilation -> mana (and sometimes energy)
            available = matter_grid[mask]
            matter_to_mana = xp.asarray(props.matter_to_mana)
            converted = matter_to_mana * available * dt_arr
            matter_grid[mask] = matter_grid[mask] - converted
            mana_grid[mask] = mana_grid[mask] + converted

            if code == PhaseCode.PURINIUM:
                # Purinium dumps the converted rest-mass as mana-energy.
                # (You can multiply by a big factor later if you want
                #  more "star-like" behaviour.)
                energy_grid[mask] = energy_grid[mask] + converted

        if props.mana_to_energy != 0.0:
            # mana leaking into energy (radiation, heating, etc.)
            m_local = mana_grid[mask]
            mana_to_energy = xp.asarray(props.mana_to_energy)
            leaked = mana_to_energy * m_local * dt_arr
            mana_grid[mask] = mana_grid[mask] - leaked
            energy_grid[mask] = energy_grid[mask] + leaked
