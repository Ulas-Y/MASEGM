from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor
from engine.constants import C_MANA, K_MANA
from engine.math.b_calculus import log_laplacian, xp



class InteractionRule:
    def apply(self, mana: ManaField, matter: MatterField, energy: EnergyTensor, dt: float) -> None:
        raise NotImplementedError

class ManaCondensesToMatter(InteractionRule):
    """
    Mana -> Matter conversion that depends on global entropy.

    Base behavior: d(matter) = rate * mana * dt.
    If entropy is low (more ordered / concentrated), the condensation is stronger.
    """

    def __init__(self, base_rate: float = 0.01, entropy_sensitivity: float = 2.0):
        self.base_rate = base_rate
        self.entropy_sensitivity = entropy_sensitivity
        self._last_entropy = None
        self._S_max = 1.0

    def set_entropy(self, S: float, S_max: float) -> None:
        self._last_entropy = S
        self._S_max = max(S_max, 1e-12)

    def apply(self, mana: ManaField, matter: MatterField, energy: EnergyTensor, dt: float) -> None:
        be = xp
        mana_grid = be.asarray(mana.grid)
        matter_grid = be.asarray(matter.grid)

        rate = self.base_rate
        if self._last_entropy is not None:
            frac = self._last_entropy / self._S_max  # 0..1
            # low entropy -> bigger factor; high entropy -> closer to 1
            factor = be.exp(self.entropy_sensitivity * (1.0 - frac))
            rate = self.base_rate * factor

        delta = rate * mana_grid * dt

        mana.grid = mana_grid - delta
        matter.grid = matter_grid + delta

class EnergyCoupledBGrowth(InteractionRule):
    """
    Extra B-style mana growth driven by local energy.

    - Normalizes energy to [0, 1] across the grid.
    - Multiplies mana by exp(alpha * E_norm * dt).

    So hot zones (high energy) multiplicatively amplify mana,
    cold zones barely grow.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def apply(self, mana: ManaField, matter: MatterField, energy: EnergyTensor, dt: float) -> None:
        be = xp
        E = be.asarray(energy.grid)
        M = be.asarray(mana.grid)

        # shift to non-negative, normalize
        E_shift = E - E.min()
        maxE = E_shift.max()
        if maxE <= 0.0:
            return

        E_norm = E_shift / maxE  # 0..1
        factor = be.exp(self.alpha * E_norm * dt)
        mana.grid = M * factor
        mana.ensure_positive()

class ManaEnergyBackReaction(InteractionRule):
    """
    Energy back-reaction from mana structure.

    dE/dt = gamma * |Δ ln m| - decay * E

    - Sharp mana features (large |Δ ln m|) generate energy.
    - Energy decays elsewhere at rate 'decay'.
    """

    def __init__(self, gamma: float = 1.0, decay: float = 0.5):
        self.gamma = gamma
        self.decay = decay

    def apply(self, mana: ManaField, matter: MatterField, energy: EnergyTensor, dt: float) -> None:
        be = xp
        m = be.asarray(mana.grid)
        e = be.asarray(energy.grid)

        lap_log_m = be.asarray(log_laplacian(m))
        source = self.gamma * abs(lap_log_m)

        # reaction + decay
        energy.grid = e + (source - self.decay * e) * dt
        energy.ensure_nonnegative()

class EnergyToManaCondensation(InteractionRule):
    def __init__(self, rate: float = 0.01):
        self.rate = rate

    def apply(self, mana, matter, energy, dt: float) -> None:
        be = xp
        e = be.asarray(energy.grid)
        m = be.asarray(mana.grid)

        # where purity is already high, energy more easily forms mana
        purity = getattr(mana, "purity", None)
        if purity is None:
            # fallback: assume moderate purity everywhere
            purity = be.asarray(m * 0 + 0.5)
        else:
            purity = be.asarray(purity)

        # conversion is stronger at high purity
        conv = self.rate * purity * e * dt

        energy.grid = e - conv
        mana.grid   = m + conv / (K_MANA * (C_MANA ** 2))  # invert your formula


