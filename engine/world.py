from dataclasses import dataclass, field
from typing import List

from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor
from engine.rules.mana_rules import ManaRule
from engine.rules.matter_rules import MatterRule
from engine.rules.interaction_rules import InteractionRule


@dataclass
class World:
    mana: ManaField
    matter: MatterField
    energy: EnergyTensor

    mana_rules: List[ManaRule] = field(default_factory=list)
    matter_rules: List<MatterRule] = field(default_factory=list)
    interaction_rules: List[InteractionRule] = field(default_factory=list)

    def step(self, dt: float) -> None:
        """Apply all rules once."""
        for rule in self.mana_rules:
            rule.apply(self.mana, dt)

        for rule in self.matter_rules:
            rule.apply(self.matter, dt)

        for rule in self.interaction_rules:
            rule.apply(self.mana, self.matter, self.energy, dt)
