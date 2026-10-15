"""Lifetime random variate sampling utilities."""

from collections.abc import Sequence
from typing import NamedTuple, overload

import numpy as np
import pandas as pd
from optype.numpy import Array1D

from assetlife.lifetime_models import LifetimeDistribution, ParametricLifetimeRegression
from assetlife.lifetime_models._base import ParametricLifetimeModel
from assetlife.lifetime_models._distributions import Exponential
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
    covar: None,
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
def sample_lifetimes_from_renewal_process(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
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
    seed : int, BitGenerator, Generator or RandomState, optional
        Random seed or random number generator, see ``np.random``.

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


def sample_failures_from_nhpp(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    lifetime_model: ParametricLifetimeModel,
    n_assets: int,
    n_failures_per_asset: int,
    start_observation: float = 0,
    end_observation: float = np.inf,
    covariates_labels: Sequence[str] | None = None,
    covariates_values: Sequence[Array1D[np.float64]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Ensure covariates make sense when using a regression
    is_regression = False
    if isinstance(lifetime_model, ParametricLifetimeRegression):
        if covariates_labels is None or covariates_values is None:
            error_msg = (
                "When using a Regression as lifetime model, please specify covariates."
            )
            raise ValueError(error_msg)
        if len(covariates_values) != len(covariates_labels):
            error_msg = "Make sure covariates labels and values have same lengths."
            raise ValueError(error_msg)
        is_regression = True

    # Sample the basis, with default Exponential
    exp = Exponential(rate=1)
    sample = exp.rvs(size=(n_assets, n_failures_per_asset))
    sample = np.cumsum(sample, axis=1)

    # Apply NHPP property, depending on ichf use of covariates or not
    if is_regression:
        assert covariates_values is not None
        sample = lifetime_model.ichf(sample, *covariates_values)
    else:
        sample = lifetime_model.ichf(sample)

    # Create indexes for assets
    indexes = np.arange(0, n_assets)[:, np.newaxis] * np.ones_like(sample)

    # Create Failure table and truncate on the observation window
    failures = pd.DataFrame(data={"time": sample.flatten(), "id": indexes.flatten()})
    failures = failures[
        (failures["time"] > start_observation) & (failures["time"] < end_observation)
    ]

    # Create Asset table and specify the observation for each asset
    samples_last = sample.max(axis=1)
    failures_end_observation = np.minimum(samples_last, end_observation)
    assets = pd.DataFrame(
        data={
            "id": np.arange(n_assets),
            "start": start_observation,
            "end": failures_end_observation,
        }
    )
    # Add the covariates if used
    if is_regression:
        assert covariates_labels is not None
        assert covariates_values is not None
        for label, value in zip(covariates_labels, covariates_values, strict=True):
            assets[label] = value

    return failures, assets
