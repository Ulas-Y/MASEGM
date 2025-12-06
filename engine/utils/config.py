from dataclasses import dataclass
from ..constants import (
    ny as default_ny, 
    nx as default_nx, 
    dt as default_dt, 
    steps as default_steps,
)

@dataclass
class EngineConfig:
    """
    Basic config container. Expand as needed.
    """
    ny: int = default_ny #default was 100
    nx: int = default_nx #default was 100
    dt: float = default_dt #default is 0.1
    steps: int = default_steps #default is 100
