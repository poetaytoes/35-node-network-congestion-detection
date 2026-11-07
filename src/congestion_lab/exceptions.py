"""Project-specific exception hierarchy."""


class CongestionLabError(Exception):
    """Base exception for expected project failures."""


class ConfigurationError(CongestionLabError, ValueError):
    """Raised when configuration loading or validation fails."""


class ContractViolation(CongestionLabError, AssertionError):
    """Raised when a scientific or pairing contract is violated."""


class ArtifactIntegrityError(CongestionLabError):
    """Raised when an artifact hash or lineage record is invalid."""
