from engine.fields.mana_field import ManaField


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
