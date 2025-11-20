import numpy as np
from scipy.signal import convolve2d
from engine.math.b_calculus import b_add, b_mul  # we’ll use these first


class ManaField:
    """
    Represents a mana density field on a 2D grid.
    """
    
    _LAPLACIAN_KERNEL = np.array(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]]
    )
    
    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        base = max(initial_value, 0.0)
        self.grid = np.full(shape, base, dtype=float) + 1e-12
    
        # NEW: phase index per cell (will be managed by phase rules)
        # 0: particles, 1: energy, 2: gas, 3: refined, 4: aether, 5: purinium
        self.phase = np.zeros(shape, dtype=np.uint8)   
    
    def add_mana(self, y: int, x: int, amount: float) -> None:
        self.grid[y, x] += amount
    
    def remove_mana(self, y: int, x: int, amount: float) -> None:
        self.grid[y, x] = max(0.0, self.grid[y, x] - amount)
    
    def total_mana(self) -> float:
        return float(self.grid.sum())
    
    def diffuse(self, rate: float, dt: float) -> None:
        """
        Simple explicit diffusion step:
        ∂M/∂t = rate * ∇² M
        
        This is VERY basic and not numerically perfect,
        but good for prototyping.
        """
        lap = convolve2d(self.grid, self._LAPLACIAN_KERNEL, mode="same", boundary="symm")
        self.grid += rate * dt * lap
    
    def copy(self) -> "ManaField":
        mf = ManaField(self.shape)
        mf.grid = self.grid.copy()
        return mf
    
    def ensure_positive(self, eps: float = 1e-12) -> None:
        """
        Ensure that all mana values stay strictly positive
        (needed for B-calculus, which uses ln).
        """
        self.grid = np.maximum(self.grid, eps)
    
    def b_scale_mul(self, factor: float) -> None:
        """
        Apply a multiplicative scaling on the B-scale:
        
            mana_new = mana ⊕ factor = mana * factor
        
        Using your B-add definition.
        """
        self.ensure_positive()
        self.grid = b_add(self.grid, factor)  # which is just self.grid * factor
    
    def b_scale_power(self, exponent: float) -> None:
        """
        Apply B-multiplication-like behaviour:
        
            mana_new = mana ⊗ exponent = mana^exponent
        """
        self.ensure_positive()
        self.grid = b_mul(self.grid, exponent)  # which is self.grid ** exponent
