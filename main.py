from engine.fields.mana_field import ManaField
from engine.fields.matter_field import MatterField
from engine.fields.energy_tensor import EnergyTensor
from engine.rules.mana_rules import ConstantManaSource
from engine.utils import EngineConfig, plot_scalar_field


def main():
    cfg = EngineConfig(ny=100, nx=100, dt=0.1, steps=50)

    mana = ManaField(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    matter = MatterField(shape=(cfg.ny, cfg.nx), initial_value=0.0)
    energy = EnergyTensor(shape=(cfg.ny, cfg.nx), initial_value=0.0)

    # One simple rule: constant mana source at center
    source = ConstantManaSource(
        y=cfg.ny // 2,
        x=cfg.nx // 2,
        rate=1.0,
    )

    for step in range(cfg.steps):
        source.apply(mana, cfg.dt)

    print("Total mana:", mana.total_mana())
    plot_scalar_field(mana.grid, title="Mana after source rule")

    # Matter + energy unused yet, but they are ready.

if __name__ == "__main__":
    
    diffusion_rate = 0.5

    for step in range(cfg.steps):
        source.apply(mana, cfg.dt)
        mana.diffuse(diffusion_rate, cfg.dt)

    main()