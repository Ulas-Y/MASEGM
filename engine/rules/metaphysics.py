import numpy as np
from engine.fields.mana_field import ManaField
from engine.math.b_calculus import log_gradient  # Already used in b_advect

class MetaphysicsRule:
    """Base for metaphysical/physics ops on fields."""
    def apply(self, mana: ManaField, dt: float) -> None:
        raise NotImplementedError

class GravityAttraction(MetaphysicsRule):
    """Mana gravity: Attracts to high density via b_advect with negative strength."""
    def __init__(self, G: float = 0.1):  # Positive G for attraction magnitude
        self.G = G

    def apply(self, mana: ManaField, dt: float) -> None:
        # Call b_advect with -G for pull toward high (flips your internal -strength to positive v toward grad)
        mana.b_advect(-self.G, dt)  # Negative strength = attraction in your impl