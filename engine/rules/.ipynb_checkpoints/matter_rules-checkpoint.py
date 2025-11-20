from engine.fields.matter_field import MatterField


class MatterRule:
    """Base class for rules acting on matter only."""

    def apply(self, matter: MatterField, dt: float) -> None:
        raise NotImplementedError
