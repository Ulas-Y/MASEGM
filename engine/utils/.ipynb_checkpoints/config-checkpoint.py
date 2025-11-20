from dataclasses import dataclass


@dataclass
class EngineConfig:
    """
    Basic config container. Expand as needed.
    """
    ny: int = 100
    nx: int = 100
    dt: float = 0.1
    steps: int = 100
