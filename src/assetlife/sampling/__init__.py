"""Sampling utilities for lifetimes and stochastic processes."""

from ._sample_lifetimes import sample_lifetimes_from_renewal_process
from ._sample_processes import sample_process

__all__ = ["sample_lifetimes_from_renewal_process", "sample_process"]
