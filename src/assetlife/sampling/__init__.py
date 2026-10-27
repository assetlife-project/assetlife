"""Sampling utilities for lifetimes and stochastic processes."""

from ._sample_fit_data import (
    sample_failures_from_nhpp,
    sample_lifetimes_from_renewal_process,
)
from ._sample_processes import sample_process

__all__ = [
    "sample_failures_from_nhpp",
    "sample_lifetimes_from_renewal_process",
    "sample_process",
]
