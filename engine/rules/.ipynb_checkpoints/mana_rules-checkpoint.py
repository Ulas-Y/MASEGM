import numpy as np
from engine.fields.mana_field import ManaField
from engine.math.b_calculus import b_add  # using for scaling



class ManaRule:
    """Base class for rules that only touch mana."""

    def apply(self, mana: ManaField, dt: float) -> None:
        """Modify mana field in-place."""
        raise NotImplementedError


class ConstantManaSource(ManaRule):
    """
    Adds a constant mana amount to a single cell every step.
    """

    def __init__(self, y: int, x: int, rate: float):
        self.y = y
        self.x = x
        self.rate = rate

    def apply(self, mana: ManaField, dt: float) -> None:
        mana.add_mana(self.y, self.x, self.rate * dt)

class BScaleManaGrowth(ManaRule):
    """
    Global B-scale mana growth rule.

    Uses the fact that for f(x) = x^k, D_B f = e^k.
    We treat `k` as a global B-intensity parameter and apply:

        mana(t + dt) = mana(t) * exp(k * dt)

    This is a direct application of your B-derivative interpretation:
    exponential-type multiplicative growth on the B-scale.
    """

    def __init__(self, k: float):
        self.k = k  # B-growth intensity

    def apply(self, mana: ManaField, dt: float) -> None:
        # growth factor per time step:
        factor = float(np.exp(self.k * dt))
        mana.b_scale_mult(factor)
