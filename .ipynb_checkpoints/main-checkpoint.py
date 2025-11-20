from engine.rules.interaction_rules import ManaCondensesToMatter
from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor
from engine.rules.mana_rules import ConstantManaSource, BScaleManaGrowth
from engine.world import World
from engine.utils import EngineConfig, plot_scalar_field, mana_entropy


def main(growth_k: float = 0.5, steps: int | None = None):
    cfg = EngineConfig(ny=100, nx=100, dt=0.1, steps=100)
    if steps is not None:
        cfg.steps = steps
    
    mana = ManaField(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    matter = MatterField(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    energy = EnergyTensor(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    
    world = World(mana=mana, matter=matter, energy=energy)
    
    source = ConstantManaSource(cfg.ny // 2, cfg.nx // 2, rate=1.0)
    world.mana_rules.append(source)
    
    # use the parameter from the caller:
    b_growth = BScaleManaGrowth(k=growth_k)
    world.mana_rules.append(b_growth)
    
    diffusion_rate = 0.5
    
    for step in range(cfg.steps):
        world.step(cfg.dt)
        world.mana.diffuse(diffusion_rate, cfg.dt)
    
    total = world.mana.total_mana()
    S = mana_entropy(world.mana.grid)
    
    print(f"Total mana: {total}")
    print(f"Mana entropy: {S}")
    plot_scalar_field(world.mana.grid, title="Mana after world evolution")




if __name__ == "__main__":
    main()
