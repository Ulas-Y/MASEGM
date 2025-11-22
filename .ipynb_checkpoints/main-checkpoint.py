import numpy as np

from engine.rules.mana_rules import ConstantManaSource, BScaleManaGrowth
from engine.rules.interaction_rules import (
    ManaCondensesToMatter,
    EnergyCoupledBGrowth,
    ManaEnergyBackReaction,   # NEW
)

from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor
from engine.utils import EngineConfig, plot_scalar_field, mana_entropy, detect_ness
from engine.world import World
from engine.constants import C_MANA, K_MANA
from engine.rules.phase_rules import PhaseTransitionRule
from engine.physics.mana_phase import PhaseThresholds

def main(growth_k: float = 0.5, steps: int | None = None):
    cfg = EngineConfig(ny=100, nx=100, dt=0.1, steps=100)
    n_cells = cfg.nx * cfg.ny
    S_max = np.log(n_cells)
    
    
    if steps is not None:
        cfg.steps = steps
    
    mana = ManaField(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    matter = MatterField(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    energy = EnergyTensor(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    
    world = World(mana=mana, matter=matter, energy=energy)
    
    source = ConstantManaSource(cfg.ny // 2, cfg.nx // 2, rate=1.0)
    world.mana_rules.append(source)
    
    b_growth = BScaleManaGrowth(k=growth_k)
    world.mana_rules.append(b_growth)
    
    # NEW: phase transition rule (mana purity phases)
    
    # if you still use ManaCondensesToMatter, keep it too:
    # condense = ManaCondensesToMatter(...)
    # world.interaction_rules.append(condense)
    
    condense = ManaCondensesToMatter(base_rate=0.01, entropy_sensitivity=2.0)
    world.interaction_rules.append(condense)
    
    # ...
    # NEW: phase transition rule (mana purity phases)
    thresholds = PhaseThresholds()  # you can tweak values here later
    phase_rule = PhaseTransitionRule(thresholds=thresholds)
    world.interaction_rules.append(phase_rule)
    
    # remove the second PhaseTransitionRule(...) block entirely

    
    energy_growth = EnergyCoupledBGrowth(alpha=1.5)
    world.interaction_rules.append(energy_growth)
    
    # NEW: energy back-reaction from mana curvature
    back_react = ManaEnergyBackReaction(gamma=1.0, decay=0.5)
    world.interaction_rules.append(back_react)
    
    
    base_diffusion = 0.5       #default 0.5
    alpha = 1.0                #default 1.0
    energy_diffusion = 0.3     #default 0.3
    transport_strength = 0.3   #default 0.3 "tweak this"

    total_history = []
    entropy_history = []
    
    for step in range(cfg.steps):
        S = mana_entropy(world.mana.grid)
        entropy_history.append(S)
    
        frac = S / S_max
        diffusion_rate = base_diffusion * (1.0 + alpha * (1.0 - frac))
    
        condense.set_entropy(S, S_max)
    
        world.step(cfg.dt)
    
        max_dt_diffusion = 0.25 * (1.0 / C_MANA)  # CFL-like condition
        effective_dt = min(cfg.dt, max_dt_diffusion)

        world.mana.b_diffuse(diffusion_rate, effective_dt)
        world.energy.b_diffuse(energy_diffusion, cfg.dt)
    
        # NEW: transport (currents)
        world.mana.b_advect(transport_strength, cfg.dt)
    
        total_history.append(world.mana.total_mana())
    
    
    total = total_history[-1]
    S_final = entropy_history[-1]
    
    ness_mana, span_mana, mean_mana = detect_ness(total_history)
    ness_S, span_S, mean_S = detect_ness(entropy_history)
    
    print(f"Total mana: {total}")
    print(f"Mana entropy (final): {S_final}")
    
    print("NESS check (last window):")
    print(f"  Mana   -> is_ness={ness_mana}, span={span_mana}, mean={mean_mana}")
    print(f"  Entropy-> is_ness={ness_S}, span={span_S}, mean={mean_S}")
    
    print(f"Mana entropy (min,max): {min(entropy_history)}, {max(entropy_history)}")
    plot_scalar_field(world.mana.grid, title="Mana after world evolution")

    return world





if __name__ == "__main__":
    main()

def purinium_core_test(growth_k: float = 1.0, steps: int = 400,
                       return_world: bool = False):
    cfg = EngineConfig(ny=100, nx=100, dt=0.1, steps=steps)
    
    mana = ManaField(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    matter = MatterField(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    energy = EnergyTensor(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    
    world = World(mana=mana, matter=matter, energy=energy)
    
    source = ConstantManaSource(cfg.ny // 2, cfg.nx // 2, rate=1.0)
    world.mana_rules.append(source)
    
    b_growth = BScaleManaGrowth(k=growth_k)
    world.mana_rules.append(b_growth)
    
    phase_rule = PhaseTransitionRule(
        thresholds=PhaseThresholds(
            purinium_density_threshold=5.0,  # tune this if you like
        )
    )
    world.interaction_rules.append(phase_rule)
    
    # remove the second PhaseTransitionRule(...) block entirely
    
    for _ in range(cfg.steps):
        world.step(cfg.dt)
        world.mana.diffuse(0.5, cfg.dt)  # or your tuned diffusion
    
    # stats
    total_mana = world.mana.total_mana()
    phase = world.mana.phase
    purinium_cells = int((phase == 5).sum())
    aether_cells = int((phase == 4).sum())
    
    
    print(f"Total mana: {total_mana}")
    print(f"Purinium cells: {purinium_cells}, Aether cells: {aether_cells}")
    
    plot_scalar_field(world.mana.grid, title="Purinium core test: mana field")
    
    if return_world:
        return world
    