import numpy as np

from engine.rules.mana_rules import ConstantManaSource, BScaleManaGrowth
from engine.rules.interaction_rules import ManaCondensesToMatter
from engine.rules.phase_rules import PhaseTransitionRule
from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor
from engine.utils import EngineConfig, plot_scalar_field, mana_entropy, detect_ness
from engine.world import World


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

    condense = ManaCondensesToMatter(base_rate=0.01, entropy_sensitivity=2.0)
    world.interaction_rules.append(condense)

    # NEW: purity/phase behaviour rule
    phase_rule = PhaseTransitionRule(
        high_purity_cutoff=0.9,
        purinium_cutoff=0.999,
        smooth_strength=0.5,
        amp_strength=3.0,
        purinium_damp=5.0,
    )
    world.interaction_rules.append(phase_rule)

    
    base_diffusion = 0.5
    alpha = 1.0  # how strongly low entropy boosts diffusion

    total_history = []
    entropy_history = []

    for step in range(cfg.steps):
        # compute global entropy BEFORE this step’s diffusion
        S = mana_entropy(world.mana.grid)
        entropy_history.append(S)

        frac = S / S_max
        diffusion_rate = base_diffusion * (1.0 + alpha * (1.0 - frac))

        condense.set_entropy(S, S_max)   # <-- pass entropy into the rule

        world.step(cfg.dt)
        world.mana.b_diffuse(diffusion_rate, cfg.dt)  # <-- NEW

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





if __name__ == "__main__":
    main()
