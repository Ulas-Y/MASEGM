# engine/physics/mana_phase.py

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

from typing import Tuple

import numpy as np


class PhaseCode(IntEnum):
    """
    Integer codes for mana phases.
    These are what get stored in mana.phase.
    """
    PARTICLES = 0
    PLASMA    = 1
    GAS       = 2
    LIQUID    = 3
    AETHER    = 4
    PURINIUM  = 5


@dataclass(frozen=True)
class PhaseProperties:
    """
    Behaviour modifiers for each mana phase.

    These are purely dimensionless multipliers that the engine
    uses to tweak local reactions.
    """
    name: str

    # how strongly this phase tends to "burn" mana to lower entropy (<0)
    # or amplify mana via self-purification (>0) around the 90% pivot.
    entropy_feedback: float

    # extra mana decay rate independent of entropy (useful for low purity)
    mana_decay: float = 0.0

    # matter -> mana conversion rate per unit matter per unit time
    matter_to_mana: float = 0.0

    # mana -> energy leakage rate (could be radiation, heat, etc.)
    mana_to_energy: float = 0.0

    # special extra self-purification factor used only for
    # high-purity phases (aether & purinium)
    purify_boost: float = 0.0


@dataclass
class PhaseThresholds:
    """
    Thresholds that define where each phase lives in
    (purity, energy_density) space.

    For now, only purity decides the basic phase, and
    energy_density is only used to distinguish
    'aether' vs 'purinium'.
    """
    # purity thresholds (fractions of 1.0)
    p_particle: float = 1e-3      # < 0.1%
    p_plasma: float   = 0.05      # 0.1%–5%
    p_gas: float      = 0.50      # 5%–50%
    p_liquid: float   = 0.95      # 50%–95%
    p_aether: float   = 0.999     # 95%–99.9%

    # pivot where feedback flips sign
    p_pivot: float = 0.90

    # how many times above the mean mana density a region must be
    # to be considered Purinium (instead of just very dense Aether)
    purinium_density_threshold: float = 5.0


def classify_phases(
    purity: np.ndarray,
    mana_grid: np.ndarray,
    thresholds: PhaseThresholds,
) -> np.ndarray:
    """
    Classify phases based on purity and relative mana density.

    Returns an integer numpy array of PhaseCode values.
    """
    phase = np.zeros_like(purity, dtype=np.int8)

    # 0: particles
    mask = purity < thresholds.p_particle
    phase[mask] = PhaseCode.PARTICLES

    # 1: plasma
    mask = (purity >= thresholds.p_particle) & (purity < thresholds.p_plasma)
    phase[mask] = PhaseCode.PLASMA

    # 2: gas
    mask = (purity >= thresholds.p_plasma) & (purity < thresholds.p_gas)
    phase[mask] = PhaseCode.GAS

    # 3: liquid (refined mana)
    mask = (purity >= thresholds.p_gas) & (purity < thresholds.p_liquid)
    phase[mask] = PhaseCode.LIQUID

    # 4: Aether by default for very high purity
    mask_high = (purity >= thresholds.p_liquid)
    phase[mask_high] = PhaseCode.AETHER

    # 5: Purinium = Aether + very high local density
    if np.any(mask_high):
        mean_mana = float(mana_grid.mean())
        if mean_mana > 0.0:
            dense = mana_grid > (
                thresholds.purinium_density_threshold * mean_mana
            )
            purinium_mask = mask_high & dense
            phase[purinium_mask] = PhaseCode.PURINIUM

    return phase


def default_phase_properties() -> Tuple[PhaseProperties, ...]:
    """
    Convenience helper: the same behaviours you already had,
    but factored out into a reusable place.
    Index order matches PhaseCode values.
    """
    return (
        PhaseProperties(
            name="particles",
            entropy_feedback=-0.3,
            mana_decay=0.2,
            matter_to_mana=0.0,
            mana_to_energy=0.1,
        ),
        PhaseProperties(
            name="plasma",
            entropy_feedback=-0.15,
            mana_decay=0.05,
            matter_to_mana=0.0,
            mana_to_energy=0.05,
        ),
        PhaseProperties(
            name="gas",
            entropy_feedback=-0.05,
            mana_decay=0.01,
            matter_to_mana=0.0,
            mana_to_energy=0.02,
        ),
        PhaseProperties(
            name="liquid",
            entropy_feedback=+0.10,
            mana_decay=0.0,
            matter_to_mana=0.01,
            mana_to_energy=0.01,
        ),
        PhaseProperties(
            name="aether",
            entropy_feedback=+0.40,
            mana_decay=0.0,
            matter_to_mana=0.05,
            mana_to_energy=0.0,
            purify_boost=0.2,
        ),
        PhaseProperties(
            name="purinium",
            entropy_feedback=+1.0,
            mana_decay=0.0,
            matter_to_mana=0.8,   # aggressive matter annihilation
            mana_to_energy=0.0,
            purify_boost=1.0,
        ),
    )
