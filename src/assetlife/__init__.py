from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

from . import (
    datasets,
    lifetime_models,
    policies,
    sampling,
    stochastic_processes,
)

with suppress(PackageNotFoundError):
    __version__ = version("assetlife")

__all__ = [
    "datasets",
    "lifetime_models",
    "policies",
    "sampling",
    "stochastic_processes",
]
