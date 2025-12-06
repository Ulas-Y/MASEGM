import copy
import numpy as np
from scipy.signal import convolve2d
from engine.math.b_calculus import log_gradient, log_laplacian  # you already had log_laplacian
from engine.math.b_calculus import divergence                  # NEW
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
    
    def b_diffuse(self, rate: float, dt: float) -> None:
        """
        B-diffusion: multiplicative analogue of diffusion.

        In log-space:
            log m_new = log m + rate * Δ log m * dt
        So in normal space:
            m_new = m * exp(rate * Δ log m * dt)
        """
        lap_log = log_laplacian(self.grid)
        self.grid *= np.exp(rate * lap_log * dt)
        self.ensure_positive()
    
    def b_advect(self, strength: float, dt: float) -> None:
        """
        B-style advection / transport of mana.

        We build a velocity field v from gradients of ln(m):

            gy, gx = grad ln m
            v = -strength * grad ln m

        Then evolve mana using a discrete continuity equation:

            ∂m/∂t + div(m v) = 0  → m_new = m_old - dt * div(m v)
        """
        if strength == 0.0:
            return

        # Use log-gradient so behaviour is multiplicative-scale aware
        gy, gx = log_gradient(self.grid)

        # Velocity field (you can flip the sign if you want flows from low→high)
        vy = -strength * gy
        vx = -strength * gx

        # Flux = m * v
        Fy = self.grid * vy
        Fx = self.grid * vx

        # Divergence of the flux
        divF = divergence(Fy, Fx)

        # Advection update
        self.grid -= dt * divF
        self.ensure_positive()
    
    def copy(self) -> "ManaField":
        mf = ManaField(self.shape)
        mf.grid = np.copy(self.grid)
        mf.phase = np.copy(self.phase)

        # Copy over any additional metadata without sharing references
        for attr, value in self.__dict__.items():
            if attr in {"grid", "phase", "shape"}:
                continue
            mf.__dict__[attr] = copy.deepcopy(value)

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
