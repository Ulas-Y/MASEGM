from engine.math.b_calculus import xp


class MatterField:
    """
    Represents a matter density field on a 2D grid.
    Mirrors the structure of ManaField for consistency.
    """

    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        self.grid = xp.full(shape, initial_value)

    def total_matter(self) -> float:
        return float(self.grid.sum())

    def add_matter(self, amount: float) -> None:
        amount_arr = xp.asarray(amount)
        self.grid = self.grid + amount_arr

    def remove_matter(self, amount: float) -> None:
        amount_arr = xp.asarray(amount)
        updated = self.grid - amount_arr
        self.grid = xp.maximum(updated, 0.0)
