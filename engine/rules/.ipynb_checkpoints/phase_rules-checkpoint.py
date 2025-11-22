# engine/rules/phase_rules.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Sequence

import numpy as np

from .interaction_rules import InteractionRule

from ..physics.mana_energy import (
    ManaEnergyParams,
    mana_purity,
    mana_energy_density,
)
from ..physics.mana_phase import (
    PhaseCode,
    PhaseProperties,
    PhaseThresholds,
    classify_phases,
    default_phase_properties,
)
from ..physics.mana_conversion import apply_phase_conversions


class PhaseTransitionRule(InteractionRule):
    """
    Classifies mana into phases by purity and applies:
      - entropy-driven amplification/decay around 90% purity
      - matter -> mana conversion in high-purity phases
      - special behaviour for Purinium cores

    This is the "G-final" mana phase engine, refactored to use the
    shared helpers in engine.physics.
    """

    def __init__(
        self,
        thresholds: PhaseThresholds | None = None,
        energy_params: ManaEnergyParams | None = None,
    ) -> None:
        # thresholds controlling purity/phase boundaries
        self.thresholds = thresholds or PhaseThresholds()
        # mana energy law params
        self.energy_params = energy_params or ManaEnergyParams()

        # phase behaviour table (index = PhaseCode value)
        self._phase_props: Tuple[PhaseProperties, ...] = default_phase_properties()

        # these will be filled every step for inspection / plotting
        self.purity: np.ndarray | None = None
        self.phase: np.ndarray | None = None   # integer codes

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def apply(self, mana, matter, energy, dt: float) -> None:
        """
        Called each world step. Mutates mana.grid, matter.grid, energy.grid
        in place according to purity / phase rules.
        """
        m_grid = mana.grid
        mat_grid = matter.grid
        e_grid = energy.grid

        # 1) compute purity field from mana + matter
        purity = mana_purity(
            m_grid,
            mat_grid,
            eps=self.energy_params.eps,
        )

        # 2) classify phases using purity + relative density
        phase = classify_phases(
            purity,
            m_grid,
            self.thresholds,
        )

        # 3) entropy-driven amplification / decay (Rule 4)
        self._apply_entropy_feedback(m_grid, purity, phase, dt)

        # 4) matter/mana/energy conversions (Rule 3)
        apply_phase_conversions(
            mana_grid=m_grid,
            matter_grid=mat_grid,
            energy_grid=e_grid,
            purity=purity,
            phase=phase,
            phase_props=self._phase_props,
            dt=dt,
        )

        # store for later inspection / plotting
        self.purity = purity
        self.phase = phase
        # attach to mana field so you can inspect in notebooks
        mana.purity = purity
        mana.phase = phase

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _apply_entropy_feedback(
        self,
        mana_grid: np.ndarray,
        purity: np.ndarray,
        phase: np.ndarray,
        dt: float,
    ) -> None:
        """
        Implements:
          - below ~90% purity → mana "burns itself" to decrease entropy
          - above ~90%        → mana self-amplifies

        Each phase has its own feedback strength taken from PhaseProperties.
        """
        p_pivot = self.thresholds.p_pivot

        # base feedback around the pivot
        delta_p = purity - p_pivot  # negative below, positive above

        # start with zero growth
        growth = np.zeros_like(mana_grid)

        for code in PhaseCode:
            props = self._phase_props[int(code)]
            mask = (phase == int(code))
            if not np.any(mask):
                continue

            local = props.entropy_feedback * delta_p[mask]

            # extra boost for high-purity phases to self-purify
            if props.purify_boost != 0.0:
                local += props.purify_boost * np.maximum(delta_p[mask], 0.0)

            # plus any unconditional mana decay (e.g. particles, plasma)
            if props.mana_decay != 0.0:
                local -= props.mana_decay

            growth[mask] = local

        # apply multiplicative update; clip to avoid negative densities
        factor = 1.0 + growth * dt
        factor = np.clip(factor, 0.0, 1e6)
        mana_grid *= factor
