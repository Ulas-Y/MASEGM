from .config import EngineConfig
from .math_utils import clamp, normalize_field
from .plotting import plot_scalar_field
from .thermo_utils import mana_entropy   # ← NEW

__all__ = [
    "EngineConfig",
    "clamp",
    "normalize_field",
    "plot_scalar_field",
    "mana_entropy",
]
