from dataclasses import dataclass, field
from typing import List

from engine.rules.interaction_rules import InteractionRule
from engine.rules.mana_rules import ManaRule
from engine.rules.matter_rules import MatterRule
from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor



@dataclass
class World:
    mana: ManaField
    matter: MatterField
    energy: EnergyTensor

    mana_rules: List[ManaRule] = field(default_factory=list)
    matter_rules: List[MatterRule] = field(default_factory=list)
    interaction_rules: List[InteractionRule] = field(default_factory=list)

    def step(self, dt: float) -> None:
        """Advance the simulation by one explicit time step.

        The lifecycle is intentionally staged so rule side effects can build on
        each other in a predictable order:

        1. ``mana_rules`` run first and may mutate ``mana`` in-place.
        2. ``matter_rules`` run next and may mutate ``matter`` in-place using
           the already-updated mana state if they read from it internally.
        3. ``interaction_rules`` run last and may couple ``mana``, ``matter``,
           and ``energy`` in-place.

        Each ``apply`` call is expected to edit the provided fields directly
        (no copies) and should avoid resetting shared tensors another rule might
        rely on later in the step. Rules should be written to compose cleanly
        under this ordering and to assume that a small ``dt`` is used so the
        explicit updates remain stable; large ``dt`` values may produce
        integration artifacts because no internal sub-stepping is performed.
        """
        for rule in self.mana_rules:
            rule.apply(self.mana, dt)

        for rule in self.matter_rules:
            rule.apply(self.matter, dt)

        for rule in self.interaction_rules:
            rule.apply(self.mana, self.matter, self.energy, dt)
