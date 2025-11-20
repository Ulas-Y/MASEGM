import numpy as np

class EnergyTensor:
    """
    For now it's a scalar energy field, but we give it a `.grid`
    attribute to match ManaField and MatterField.
    """

    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        self.grid = np.full(shape, initial_value, dtype=float)

        # For backward compatibility:
        self.energy = self.grid

    def add_energy(self, y: int, x: int, amount: float) -> None:
        self.grid[y, x] += amount

    def total_energy(self) -> float:
        return float(self.grid.sum())
