from .simulation import run_simulation, SimulationConfig


class SimulationException(Exception):
    """A class to distinguish exceptions thrown by the hivpy framework."""
    pass
