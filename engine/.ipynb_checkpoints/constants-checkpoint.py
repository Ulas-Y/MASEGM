# engine/constants.py

K_MANA = 1.0      # unit mass / conversion constant and default is 1.0
C_MANA = 1.0e3    # "speed of light" for mana in sim units and default was 1.0e2

#below is copy pasted from main.py

base_diffusion = 0.5       #default 0.5
alpha = 1.0                #default 1.0
energy_diffusion = 0.3     #default 0.3
transport_strength = 0.3   #default 0.3 "tweak these"

#below is copy pasted from config.py

"""
Basic config container. Expand as needed.
"""
ny: int = 100
nx: int = 100
dt: float = 0.1
steps: int = 100