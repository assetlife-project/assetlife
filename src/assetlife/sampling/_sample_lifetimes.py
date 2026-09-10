"""Lifetime random variate sampling utilities."""

from collections.abc import Sequence
from typing import Literal, NamedTuple, overload

import numpy as np
from optype.numpy import Array1D

from assetlife.lifetime_models import LifetimeDistribution, ParametricLifetimeRegression
from assetlife.sampling._iterables import RenewalProcessIterable
from assetlife.stochastic_processes import RenewalProcess
from assetlife.typing import CoercibleFloat64_1D, Seed


class LifetimeFitArgs(NamedTuple):
    """Lifetime data sampled for model fitting."""

    time: Array1D[np.float64]
    event: Array1D[np.bool_] | None = None
    entry: Array1D[np.float64] | None = None


@overload
def sample_lifetimes_from_renewal_process(
    lifetime_model: LifetimeDistribution,
    nb_samples: int,
    time_window: tuple[float, float],
    covar: Literal[None],
    a0: CoercibleFloat64_1D | None = None,
    ar: CoercibleFloat64_1D | None = None,
    seed: Seed = None,
) -> LifetimeFitArgs: ...
@overload
def sample_lifetimes_from_renewal_process(
    lifetime_model: ParametricLifetimeRegression,
    nb_samples: int,
    time_window: tuple[float, float],
    covar: Sequence[float],
    a0: CoercibleFloat64_1D | None = None,
    ar: CoercibleFloat64_1D | None = None,
    seed: Seed = None,
) -> LifetimeFitArgs: ...
def sample_lifetimes_from_renewal_process(
    lifetime_model: LifetimeDistribution | ParametricLifetimeRegression,
    nb_samples: int,
    time_window: tuple[float, float],
    covar: Sequence[float] | None = None,
    a0: CoercibleFloat64_1D | None = None,
    ar: CoercibleFloat64_1D | None = None,
    seed: Seed = None,
) -> LifetimeFitArgs:
    """Sample lifetimes from a renewal process.

    Parameters
    ----------
    lifetime_model : LifetimeDistribution or ParametricLifetimeRegression
        Lifetime model to use. If a regression is used, ``covar`` must be set.
    nb_samples : int
        Number of samples.
    time_window : tuple of float
        Observation time window.
    covar : sequence of float, optional
        Covariate values used when ``lifetime_model`` is a regression.
    a0 : float or 1d array of floats, optional
        Initial ages.
    ar : float or 1d array of floats, optional
        Preventive ages of replacement.
    seed : int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, optional
        Random seed or random number generator.

    Returns
    -------
    out : LifetimeFitArgs
        Sampled ``time``, ``event`` and ``entry`` arrays.

    Notes
    -----
    It is currently not possible to pass different covariate values for each
    sampled observation. To do so, call this function with different covariate
    values and reconstruct the covariate array manually.
    """
    if isinstance(lifetime_model, ParametricLifetimeRegression):
        assert covar is not None
        renewal_process = RenewalProcess(lifetime_model.freeze(*covar))
    else:
        renewal_process = RenewalProcess(lifetime_model)

    iterable = RenewalProcessIterable(
        renewal_process, nb_samples, time_window, a0=a0, ar=ar, seed=seed
    )
    struct_array = np.concatenate(tuple(iterable))

    return LifetimeFitArgs(
        time=struct_array["time"],
        event=struct_array["event"],
        entry=struct_array["entry"],
    )
