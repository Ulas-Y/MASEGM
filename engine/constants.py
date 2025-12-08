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
ny: int = 1024  #grid size in y axis
nx: int = 1024  #grid size in x axis
dt: float = 0.1  #the dt for engine/config
steps: int = 100  #steps amount for engine/config

#below is for energy from main.py
energy_alpha = 1.5 #default is 1.5 and this changes energy fluctuations for energy alpha

#below is from metaphysics/mana_phase.py from phase_thresholds

    # purity thresholds (fractions of 1.0)
p_particle: float = 1e-3      # < 0.1%
p_plasma: float   = 0.05      # 0.1%–5%
p_gas: float      = 0.50      # 5%–50%
p_liquid: float   = 0.95      # 50%–95%
p_aether: float   = 0.999     # 95%–99.9%

"""
pivot where feedback of entropy 
flips sign and it starts 
generating entropy to increase mana 
instead of lessening entropy at cost of mana
"""

p_pivot: float = 0.90

    # how many times above the mean mana density a region must be
    # to be considered Purinium (instead of just very dense Aether)
purinium_density_threshold: float = 5.0

#below is from metaphysics/mana_phase.py from Phase Properties as just understanding how to modify some other things

    # how strongly this phase tends to "burn" mana to lower entropy (<0)
    # or amplify mana via self-purification (>0) around the 90% pivot.
    #entropy_feedback: float
    
    # extra mana decay rate independent of entropy (useful for low purity)
    #mana_decay: float = 0.0
    
    # matter -> mana conversion rate per unit matter per unit time
    #matter_to_mana: float = 0.0
    
    # mana -> energy leakage rate (could be radiation, heat, etc.)
    #mana_to_energy: float = 0.0
    
    # special extra self-purification factor used only for
    # high-purity phases (aether & purinium)
    #purify_boost: float = 0.0
 #these are all normal assumptions from different phases in mana_phase.py

#(
#        PhaseProperties(
#            name="particles",
#            entropy_feedback=-0.3,
#            mana_decay=0.2,
#            matter_to_mana=0.0,
#            mana_to_energy=0.1,
#        ),
#        PhaseProperties(
#            name="plasma",
#            entropy_feedback=-0.15,
#            mana_decay=0.05,
#            matter_to_mana=0.0,
#            mana_to_energy=0.05,
#        ),
#        PhaseProperties(
#            name="gas",
#            entropy_feedback=-0.05,
#            mana_decay=0.01,
#            matter_to_mana=0.0,
#            mana_to_energy=0.02,
#        ),
#        PhaseProperties(
#            name="liquid",
#            entropy_feedback=+0.10,
#            mana_decay=0.0,
#            matter_to_mana=0.01,
#            mana_to_energy=0.01,
#        ),
#        PhaseProperties(
#            name="aether",
#            entropy_feedback=+0.40,
#            mana_decay=0.0,
#            matter_to_mana=0.05,
#            mana_to_energy=0.0,
#            purify_boost=0.2,
#        ),
#        PhaseProperties(
#            name="purinium",
#            entropy_feedback=+1.0,
#            mana_decay=0.0,
#            matter_to_mana=0.8,   # aggressive matter annihilation
#            mana_to_energy=0.0,
#            purify_boost=1.0,
#        ),
#    )
