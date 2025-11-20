import numpy as np
from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor

class InteractionRule:
    def apply(self, mana: ManaField, matter: MatterField, energy: EnergyTensor, dt: float) -> None:
        raise NotImplementedError


class ManaCondensesToMatter(InteractionRule):
    """
    Mana -> Matter conversion that depends on global entropy.

    Base behavior: d(matter) = rate * mana * dt.
    If entropy is low (more ordered / concentrated), the condensation is stronger.
    """

    def __init__(self, base_rate: float = 0.01, entropy_sensitivity: float = 2.0):
        self.base_rate = base_rate
        self.entropy_sensitivity = entropy_sensitivity
        self._last_entropy = None
        self._S_max = 1.0

    def set_entropy(self, S: float, S_max: float) -> None:
        self._last_entropy = S
        self._S_max = max(S_max, 1e-12)

    def apply(self, mana: ManaField, matter: MatterField, energy: EnergyTensor, dt: float) -> None:
        field = mana.grid

        rate = self.base_rate
        if self._last_entropy is not None:
            frac = self._last_entropy / self._S_max  # 0..1
            # low entropy -> bigger factor; high entropy -> closer to 1
            factor = np.exp(self.entropy_sensitivity * (1.0 - frac))
            rate = self.base_rate * factor

        delta = rate * field * dt

        mana.grid -= delta
        matter.grid += delta
