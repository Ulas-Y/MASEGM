import numpy as np
from scipy.signal import convolve2d

from engine.rules.interaction_rules import InteractionRule
from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor



class ManaPhase:
    PARTICLES = 0
    ENERGY    = 1
    GAS       = 2
    REFINED   = 3
    AETHER    = 4
    PURINIUM  = 5


class PhaseTransitionRule(InteractionRule):
    """
    Implements your purity-based mana phases + entropy balancer/amplifier
    behaviour in a single interaction rule.

    - Computes local purity from mana + matter:
        p = mana / (mana + matter)
    - Classifies each cell into one of your phases.
    - Low/mid purity (< 0.9) = entropy balancer:
        * smooths mana slightly to prevent collapse
    - High purity (>= 0.9) = entropy amplifier:
        * locally boosts mana multiplicatively (B-style)
    - Purinium (>= 0.999) = entropy void:
        * freezes mana & damps energy locally
    """

    def __init__(
        self,
        high_purity_cutoff: float = 0.9,
        purinium_cutoff: float = 0.999,
        smooth_strength: float = 0.5,
        amp_strength: float = 3.0,
        purinium_damp: float = 5.0,
    ):
        self.high_purity_cutoff = high_purity_cutoff
        self.purinium_cutoff = purinium_cutoff
        self.smooth_strength = smooth_strength
        self.amp_strength = amp_strength
        self.purinium_damp = purinium_damp

        # small averaging kernel for "entropy balancing" smoothing
        self._smooth_kernel = np.array(
            [[0.0, 1.0, 0.0],
             [1.0, 4.0, 1.0],
             [0.0, 1.0, 0.0]], dtype=float
        )
        self._smooth_kernel /= self._smooth_kernel.sum()

    def _classify_phases(self, mana_grid, matter_grid):
        eps = 1e-12
        total = mana_grid + matter_grid + eps
        purity = mana_grid / total  # 0..1

        phase = np.full(mana_grid.shape, ManaPhase.GAS, dtype=np.uint8)

        p = purity

        phase[p < 1e-3]  = ManaPhase.PARTICLES
        phase[(p >= 1e-3)  & (p < 5e-2)]  = ManaPhase.ENERGY
        phase[(p >= 5e-2) & (p < 0.5)]    = ManaPhase.GAS
        phase[(p >= 0.5)  & (p < 0.95)]   = ManaPhase.REFINED
        phase[(p >= 0.95) & (p < self.purinium_cutoff)] = ManaPhase.AETHER
        phase[p >= self.purinium_cutoff]  = ManaPhase.PURINIUM

        return purity, phase

    def apply(self, mana: ManaField, matter: MatterField, energy: EnergyTensor, dt: float) -> None:
        m = mana.grid
        mat = matter.grid
        e = energy.grid

        purity, phase = self._classify_phases(m, mat)
        mana.phase = phase  # store for inspection / later use

        # ---- 1) Low/mid purity → entropy balancer: gentle smoothing ----
        balancer_mask = purity < self.high_purity_cutoff
        if np.any(balancer_mask):
            smoothed = convolve2d(m, self._smooth_kernel, mode="same", boundary="symm")
            # blend original & smoothed in low-purity regions
            blend = self.smooth_strength * dt
            blend = np.clip(blend, 0.0, 1.0)
            m[balancer_mask] = (1.0 - blend) * m[balancer_mask] + blend * smoothed[balancer_mask]

        # ---- 2) High purity (but not purinium) → entropy amplifier ----
        amp_mask = (purity >= self.high_purity_cutoff) & (purity < self.purinium_cutoff)
        if np.any(amp_mask):
            # amplifier factor grows with purity above cutoff
            # roughly exp(strength * (p - cutoff))
            local_p = purity[amp_mask]
            factor = np.exp(self.amp_strength * (local_p - self.high_purity_cutoff) * dt)
            m[amp_mask] *= factor

        # ---- 3) Purinium → entropy void & energy damp ----
        pur_mask = purity >= self.purinium_cutoff
        if np.any(pur_mask):
            # Freeze mana (no change here: we already modified m,
            # but we can slightly pull it back toward previous values if we cached them)
            # For now: just strong energy damping.
            damp = np.exp(-self.purinium_damp * dt)
            e[pur_mask] *= damp

        # Clamp mana to non-negative
        mana.ensure_positive()
