"""Lifetime random variate sampling utilities."""

from collections.abc import Sequence
from typing import NamedTuple, overload

import numpy as np
import polars as pl
from optype.numpy import Array1D

from ._iterables import RenewalProcessIterable
from assetlife._typing import CoercibleFloat64_1D, Seed
from assetlife.lifetime_models import LifetimeDistribution, ParametricLifetimeRegression
from assetlife.lifetime_models._distributions import Exponential
from assetlife.stochastic_processes import RenewalProcess


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


@overload
def sample_failures_from_nhpp(
    lifetime_model: LifetimeDistribution,
    n_assets: int,
    n_failures_per_asset: int,
    time_window: tuple[float, float],
    covariates_labels: None,
    covariates_values: None,
) -> tuple[pl.DataFrame, pl.DataFrame]: ...
@overload
def sample_failures_from_nhpp(
    lifetime_model: ParametricLifetimeRegression,
    n_assets: int,
    n_failures_per_asset: int,
    time_window: tuple[float, float],
    covariates_labels: Sequence[str],
    covariates_values: Sequence[Array1D[np.float64]],
) -> tuple[pl.DataFrame, pl.DataFrame]: ...
def sample_failures_from_nhpp(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    lifetime_model: LifetimeDistribution | ParametricLifetimeRegression,
    n_assets: int,
    n_failures_per_asset: int,
    time_window: tuple[float, float],
    covariates_labels: Sequence[str] | None = None,
    covariates_values: Sequence[Array1D[np.float64]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sample failures data from a NHPP, used for fit

    Parameters
    ----------
    lifetime_model : parametric lifetime distribution or regression
        Lifetime Model used for intensity function of the NHPP
    n_assets : int
        Number of assets present in the sample
    n_failures_per_asset : int
        Number of failures to generate per asset (before censoring)
    time_window : tuple of float
        Observation time window.
    covariates_labels : sequence of str, optional
        Labels to use for covariates in the asset table. Only usefull if
        ``lifetime_model`` is a parametric regression. Defaults to None.
    covariates_values : sequence of 1d-array of float, optional
        Values of the covariates, in broadcastable shape. Only usefull if
        ``lifetime_model`` is a parametric regression. Defaults to None.

    Returns
    -------
    failures : polars.DataFrame
        Failure table, one row per failure, with columns ``id`` (asset id) and
        ``time``, as used by the NHPP fit.
    assets : polars.DataFrame
        Asset table, one row per asset, with columns ``id``, ``start`` and
        ``end`` (the observation window of the asset), and the covariates if
        any.

    Raises
    ------
    ValueError
        If covariates are missing with a regression

    Examples
    --------
    Sample failures of 10 assets, then fit a NHPP on them:

    .. code-block::

        failures, assets = sample_failures_from_nhpp(
            Weibull(2, 0.05), n_assets=10, n_failures_per_asset=5, time_window=(0, 30)
        )
        NonHomogeneousPoissonProcess(Weibull()).fit(failures, assets)
    """

    start_observation, end_observation = time_window[0], time_window[1]
    # Ensure covariates make sense when using a regression
    if isinstance(lifetime_model, ParametricLifetimeRegression):
        if covariates_labels is None or covariates_values is None:
            error_msg = (
                "When using a Regression as lifetime model, please specify covariates."
            )
            raise ValueError(error_msg)
        if len(covariates_values) != len(covariates_labels):
            error_msg = "Make sure covariates labels and values have same lengths."
            raise ValueError(error_msg)

    # Sample the basis, with default Exponential
    exp = Exponential(rate=1)
    sample = exp.rvs(size=(n_assets, n_failures_per_asset))
    sample = np.cumsum(sample, axis=1)

    # Apply NHPP property, depending on ichf use of covariates or not
    if isinstance(lifetime_model, ParametricLifetimeRegression):
        assert covariates_values is not None
        sample = lifetime_model.ichf(sample, *covariates_values)
    else:
        sample = lifetime_model.ichf(sample)

    # Asset id of each sampled failure: integers, as in the asset table, so
    # that the two tables can be joined
    ids = np.repeat(np.arange(n_assets), n_failures_per_asset)
    time = sample.ravel()

    # Create Failure table, truncated on the observation window
    observed = (time > start_observation) & (time < end_observation)
    failures = pl.DataFrame({"id": ids[observed], "time": time[observed]})

    # Create Asset table and specify the observation for each asset
    covariates: dict[str, Array1D[np.float64]] = {}
    if isinstance(lifetime_model, ParametricLifetimeRegression):
        assert covariates_labels is not None
        assert covariates_values is not None
        covariates = {
            # one value per asset, whatever the broadcastable shape given
            label: np.broadcast_to(np.ravel(value), (n_assets,))
            for label, value in zip(covariates_labels, covariates_values, strict=True)
        }
    assets = pl.DataFrame({
        "id": np.arange(n_assets),
        "start": np.full(n_assets, start_observation, dtype=np.float64),
        "end": np.minimum(sample.max(axis=1), end_observation),
        **covariates,
    })

    return failures, assets
