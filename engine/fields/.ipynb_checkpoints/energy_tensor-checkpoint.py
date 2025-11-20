import numpy as np


class EnergyTensor:
    """
    Very simple placeholder "energy tensor":
    for now it's just a scalar energy field.

    Later you can upgrade this to a full tensor (T_ij).
    """

    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        self.energy = np.full(shape, initial_value, dtype=float)

    def add_energy(self, y: int, x: int, amount: float) -> None:
        self.energy[y, x] += amount

    def total_energy(self) -> float:
        return float(self.energy.sum())
