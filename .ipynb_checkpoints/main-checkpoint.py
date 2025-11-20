from engine.rules.interaction_rules import ManaCondensesToMatter
from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor
from engine.rules.mana_rules import ConstantManaSource
from engine.utils import EngineConfig, plot_scalar_field
from engine.world import World


def main():
    cfg = EngineConfig(ny=100, nx=100, dt=0.1, steps=100)

    mana = ManaField(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    matter = MatterField(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    energy = EnergyTensor(shape=(cfg.ny, cfg.nx), initial_value=0.0)

    world = World(mana=mana, matter=matter, energy=energy)
    
    interaction = ManaCondensesToMatter(rate=0.02)
    world.interaction_rules.append(interaction)

    # Add rules
    source = ConstantManaSource(cfg.ny // 2, cfg.nx // 2, rate=1.0)
    world.mana_rules.append(source)

    diffusion_rate = 0.5

    for step in range(cfg.steps):
        world.step(cfg.dt)
        # simple global diffusion outside the rule system for now
        world.mana.diffuse(diffusion_rate, cfg.dt)

    print("Total mana:", world.mana.total_mana())
    plot_scalar_field(world.mana.grid, title="Mana after world evolution")


if __name__ == "__main__":
    main()
