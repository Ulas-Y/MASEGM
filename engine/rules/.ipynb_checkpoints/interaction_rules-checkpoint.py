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
