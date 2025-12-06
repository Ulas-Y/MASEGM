# engine/rules/phase_rules.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
from engine.utils.thermo_utils import mana_energy_from_state
from engine.constants import K_MANA, C_MANA  # if needed

import numpy as np

from .interaction_rules import InteractionRule


@dataclass(frozen=True)
class PhaseProperties:
    """Local behaviour modifiers for each mana phase.

    All coefficients are *dimensionless* multipliers that act on
    existing engine behaviour. Keep them small-ish (<< 1) to avoid
    numerical explosions.
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

    # special extra self-purification factor used only for Purinium
    purify_boost: float = 0.0


class PhaseTransitionRule(InteractionRule):
    """
    Classifies mana into phases by purity and applies:
    - entropy-driven amplification/decay around 90% purity
    - matter->mana conversion in high-purity phases
    - special behaviour for Purinium cores

    This is the "G-final" mana phase engine.
    """

    # purity thresholds (fraction of 1.0)
    P_PARTICLE = 1e-3   # < 0.1%
    P_PLASMA   = 0.05   # 0.1%–5%
    P_GAS      = 0.50   # 5%–50%
    P_LIQUID   = 0.95   # 50%–95%
    P_AETHER   = 0.999  # 95%–99.9%
    # >= P_AETHER and high local density => Purinium

    # pivot where feedback flips sign
    P_PIVOT = 0.90      # 90% purity

    def __init__(
        self,
        purinium_density_threshold: float = 5.0,
        eps: float = 1e-12,
    ) -> None:
        """
        purinium_density_threshold:
            minimum mana density (relative to mean) to label a cell as
            Purinium instead of just Aether.

        eps:
            numerical floor to avoid division by zero.
        """
        self.eps = eps
        self.purinium_density_threshold = purinium_density_threshold

        # phase behaviour table (tunable!)
        self._phases: Dict[str, PhaseProperties] = {
            "particles": PhaseProperties(
                name="particles",
                entropy_feedback=-0.3,
                mana_decay=0.2,
                matter_to_mana=0.0,
                mana_to_energy=0.1,
            ),
            "plasma": PhaseProperties(
                name="plasma",
                entropy_feedback=-0.15,
                mana_decay=0.05,
                matter_to_mana=0.0,
                mana_to_energy=0.05,
            ),
            "gas": PhaseProperties(
                name="gas",
                entropy_feedback=-0.05,
                mana_decay=0.01,
                matter_to_mana=0.0,
                mana_to_energy=0.02,
            ),
            "liquid": PhaseProperties(
                name="liquid",
                entropy_feedback=+0.10,
                mana_decay=0.0,
                matter_to_mana=0.01,
                mana_to_energy=0.01,
            ),
            "aether": PhaseProperties(
                name="aether",
                entropy_feedback=+0.40,
                mana_decay=0.0,
                matter_to_mana=0.05,
                mana_to_energy=0.0,
                purify_boost=0.2,
            ),
            "purinium": PhaseProperties(
                name="purinium",
                entropy_feedback=+1.0,
                mana_decay=0.0,
                matter_to_mana=0.8,   # aggressive matter annihilation
                mana_to_energy=0.0,
                purify_boost=1.0,
            ),
        }

        # these will be filled every step for inspection / plotting
        self.purity: np.ndarray | None = None
        self.phase: np.ndarray | None = None   # integer codes
        self.phase_names: Tuple[str, ...] = (
            "particles",
            "plasma",
            "gas",
            "liquid",
            "aether",
            "purinium",
        )

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def apply(self, mana, matter, energy, dt: float) -> None:
        """
        Called each world step. Mutates mana.grid, matter.grid, energy.grid
        in place according to purity / phase rules.
        """
        m = mana.grid
        mat = matter.grid
        e_grid = energy.grid  # not heavily used yet, but here for future

        # 1) compute purity field
        purity = self._compute_purity(m, mat)

        # 2) classify phases
        phase = self._classify_phases(purity, m)

        # 3) apply entropy-driven amplification / decay
        self._apply_entropy_feedback(m, purity, phase, dt)

        # 4) apply matter->mana conversion and purinium behaviour
        self._apply_matter_conversion(m, mat, e_grid, purity, phase, dt)

        # store for later inspection / plotting
        self.purity = purity
        self.phase = phase
        # attach to mana field so you can inspect in notebooks
        mana.purity = purity
        mana.phase = phase

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _compute_purity(self, mana_grid: np.ndarray,
                        matter_grid: np.ndarray) -> np.ndarray:
        total_mass = mana_grid + matter_grid
        total_mass = np.maximum(total_mass, self.eps)
        purity = mana_grid / total_mass
        return purity

    def _classify_phases(
        self,
        purity: np.ndarray,
        mana_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Returns an integer array with phase codes:
        0=particles, 1=plasma, 2=gas, 3=liquid, 4=aether, 5=purinium
        """
        phase = np.zeros_like(purity, dtype=np.int8)

        # 0: particles
        mask = purity < self.P_PARTICLE
        phase[mask] = 0

        # 1: plasma
        mask = (purity >= self.P_PARTICLE) & (purity < self.P_PLASMA)
        phase[mask] = 1

        # 2: gas
        mask = (purity >= self.P_PLASMA) & (purity < self.P_GAS)
        phase[mask] = 2

        # 3: liquid (refined mana)
        mask = (purity >= self.P_GAS) & (purity < self.P_LIQUID)
        phase[mask] = 3

        # 4: aether by default
        mask = (purity >= self.P_LIQUID)
        phase[mask] = 4

        # 5: purinium = aether + very high local density
        if np.any(mask):
            mean_mana = float(mana_grid.mean() + self.eps)
            dense = mana_grid > (self.purinium_density_threshold * mean_mana)
            purinium_mask = mask & dense
            phase[purinium_mask] = 5

        return phase

    def _apply_entropy_feedback(
        self,
        mana_grid: np.ndarray,
        purity: np.ndarray,
        phase: np.ndarray,
        dt: float,
    ) -> None:
        """
        Implements:
        - below 90% purity → mana "burns itself" to decrease entropy
        - above 90%         → mana self-amplifies

        Each phase has its own feedback strength.
        """
        # base feedback around the pivot
        delta_p = purity - self.P_PIVOT  # negative below, positive above

        # start with zero growth
        growth = np.zeros_like(mana_grid)

        for code, name in enumerate(self.phase_names):
            props = self._phases[name]
            mask = phase == code
            if not np.any(mask):
                continue

            # phase-specific feedback
            local = props.entropy_feedback * delta_p[mask]

            # extra boost for purinium / aether self-purification
            if name in ("aether", "purinium"):
                local += props.purify_boost * np.maximum(delta_p[mask], 0.0)

            # plus any unconditional mana decay (e.g. particles, plasma)
            if props.mana_decay != 0.0:
                local -= props.mana_decay

            growth[mask] = local

        # apply multiplicative update; clip to avoid negative densities
        factor = 1.0 + growth * dt
        factor = np.clip(factor, 0.0, 1e6)
        mana_grid *= factor

    def _apply_matter_conversion(
        self,
        mana_grid: np.ndarray,
        matter_grid: np.ndarray,
        energy_grid: np.ndarray,
        purity: np.ndarray,
        phase: np.ndarray,
        dt: float,
    ) -> None:
        """
        High-purity phases can:
        - convert matter to mana (aether, purinium)
        - optionally pump some of that into energy_grid
        """
        for code, name in enumerate(self.phase_names):
            props = self._phases[name]
            if props.matter_to_mana == 0.0 and props.mana_to_energy == 0.0:
                continue

            mask = phase == code
            if not np.any(mask):
                continue

            if props.matter_to_mana != 0.0:
                # matter annihilation -> mana
                available = matter_grid[mask]
                converted = props.matter_to_mana * available * dt
                matter_grid[mask] -= converted
                mana_grid[mask] += converted

                # for Purinium we could optionally add big energy spikes
                if name == "purinium":
                    # First: matter annihilation -> *mana* (we already did converted)
                    # Now: some fraction becomes energy via your formula

                    local_purity = purity[mask]
                    local_mana   = mana_grid[mask]

                    # full mana energy in those cells:
                    full_E = mana_energy_from_state(local_mana, local_purity)

                    # Only convert the *newly created* mana portion to energy:
                    # scale by converted / local_mana (safe with small epsilon)
                    ratio = np.where(local_mana > 0, converted / (local_mana + 1e-12), 0.0)
                    dE = full_E * ratio

                    energy_grid[mask] += dE

            if props.mana_to_energy != 0.0:
                # mana leaking into energy (radiation, heating, etc.)
                m_local = mana_grid[mask]
                leaked = props.mana_to_energy * m_local * dt
                mana_grid[mask] -= leaked
                energy_grid[mask] += leaked

class ManaPhaseRule(PhaseTransitionRule):
    """Backwards-compatible alias for older code."""
    pass
