import numpy as np
from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor


class InteractionRule:
    """
    Base class for mana–matter–energy interaction rules.
    """

    def apply(
        self,
        mana: ManaField,
        matter: MatterField,
        energy: EnergyTensor,
        dt: float,
    ) -> None:
        raise NotImplementedError


class ManaCondensesToMatter(InteractionRule):
    """
    Simple prototype:
        - where mana is high, a bit of mana turns into matter
        - total "stuff" is approximately conserved

    Parameters
    ----------
    rate : float
        Conversion rate per unit time.
    """

    def __init__(self, rate: float = 0.01):
        self.rate = rate

    def apply(
        self,
        mana: ManaField,
        matter: MatterField,
        energy: EnergyTensor,
        dt: float,
    ) -> None:
        # amount converted is proportional to current mana
        converted = self.rate * dt * mana.grid

        # ensure we don't remove more mana than we have
        converted = np.minimum(converted, mana.grid)

        # mana decreases, matter and energy increase
        mana.grid -= converted
        matter.density += converted
        energy.energy += converted  # later: use different mapping
