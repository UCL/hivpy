"""Functionality shared between multiple parts of the framework."""


class HIVpyException(Exception):
    """Superclass for all exceptions thrown by HIVpy framework"""


class SimulationException(HIVpyException):
    """A class to distinguish exceptions thrown by the hivpy framework."""
    pass


class OutputException(HIVpyException):
    """A class for exceptions thrown by the Output module"""
    pass
