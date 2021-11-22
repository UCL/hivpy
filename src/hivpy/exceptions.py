"""Functionality shared between multiple parts of the framework."""


class SimulationException(Exception):
    """A class to distinguish exceptions thrown by the hivpy framework."""
    pass


class ConfigException(Exception):
    """A class to indicate errors related to reading config files."""
    pass
