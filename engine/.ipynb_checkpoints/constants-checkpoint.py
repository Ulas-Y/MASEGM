# engine/constants.py

K_MANA = 1.0      # unit mass / conversion constant
C_MANA = 1.0e2    # "speed of light" for mana in sim units

#below is copy pasted from main.py

base_diffusion = 0.5       #default 0.5
alpha = 1.0                #default 1.0
energy_diffusion = 0.3     #default 0.3
transport_strength = 0.3   #default 0.3 "tweak this"

#below is copy pasted from config.py

"""
Basic config container. Expand as needed.
"""
ny: int = 100
nx: int = 100
dt: float = 0.1
steps: int = 100