"""Custom exception hierarchy for the Node framework."""

class NodeError(Exception):
    """Base class for all Node framework exceptions."""
    pass

class ConfigurationError(NodeError):
    """Raised when there is an issue with configuration parsing or structure."""
    pass

class DimensionMismatchError(NodeError):
    """Raised when dimensions are incompatible during broadcast or reduction."""
    pass

class CacheError(NodeError):
    """Raised when cache operations fail."""
    pass
