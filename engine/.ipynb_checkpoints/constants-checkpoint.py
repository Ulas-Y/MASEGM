# engine/constants.py

K_MANA = 1.0      # unit mass / conversion constant and default is 1.0
C_MANA = 1.0e3    # "speed of light" for mana in sim units and default was 1.0e2

#below is copy pasted from main.py

base_diffusion = 0.5       #default 0.5 , makes diffusion speed or force stronger
alpha = 1.0                #default 1.0 , makes 
energy_diffusion = 0.3     #default 0.3 , makes energy diffusion stronger or weaker
transport_strength = 0.3   #default 0.3 , makes mana fluxes strength different like strong or weak "tweak these"
#diffusion_rate = base_diffusion * (1.0 + alpha * (1.0 - frac))
#below is copy pasted from config.py

"""
Basic config container. Expand as needed.
"""
ny: int = 100  #grid size in y axis
nx: int = 100  #grid size in x axis
dt: float = 0.1  #the dt for engine/config
steps: int = 100  #steps amount for engine/config

#below is for energy from main.py
energy_alpha = 1.5 #default is 1.5 and this changes energy fluctuations for energy alpha