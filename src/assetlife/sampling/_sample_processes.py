"""Stochastic process sampling functions."""

from __future__ import annotations

from typing import (
    NamedTuple,
    overload,
)

import numpy as np
import optype.numpy as onp

from assetlife.stochastic_processes import (
    Kijima1Process,
    Kijima2Process,
    NonHomogeneousPoissonProcess,
    RenewalProcess,
    RenewalRewardProcess,
)
from assetlife.typing import CoercibleFloat64_1D, Seed

from ._iterables import (
    Kijima1ProcessIterable,
    Kijima2ProcessIterable,
    NonHomogeneousPoissonProcessIterable,
    RenewalProcessIterable,
    RenewalRewardProcessIterable,
)


class StochasticSample(NamedTuple):
    """
    Stochastic sample.

    Attributes
    ----------
    timeline : np.ndarray of float
        The timeline corresponding to the sample.
    events : np.ndarray of booleans
        Indicates if the event has been observed.
    preventive_renewals : np.ndarray of bool
        Indicates if the event is a preventive renewal.
    rewards : np.ndarray of float or None
        Optionally, the corresponding reward.


    Notes
    -----
    If several assets are encoded, assets are on axis 0 and timeline is on axis 1.
    """

    timeline: onp.ArrayND[np.float64]
    events: onp.ArrayND[np.bool_]
    preventive_renewals: onp.ArrayND[np.bool_]
    rewards: onp.ArrayND[np.float64] | None


@overload
def sample_process(
    process: RenewalProcess,
    nb_samples: int,
    time_window: tuple[float, float],
    *,
    a0: CoercibleFloat64_1D | None = None,
    ar: CoercibleFloat64_1D | None = None,
    seed: Seed = None,
) -> StochasticSample: ...
@overload
def sample_process(
    process: NonHomogeneousPoissonProcess[()],
    nb_samples: int,
    time_window: tuple[float, float],
    *,
    a0: CoercibleFloat64_1D | None = None,
    ar: CoercibleFloat64_1D | None = None,
    seed: Seed = None,
) -> StochasticSample: ...
@overload
def sample_process(
    process: Kijima1Process[()],
    nb_samples: int,
    time_window: tuple[float, float],
    *,
    a0: CoercibleFloat64_1D | None = None,
    ar: CoercibleFloat64_1D | None = None,
    seed: Seed = None,
) -> StochasticSample: ...
@overload
def sample_process(
    process: Kijima2Process[()],
    nb_samples: int,
    time_window: tuple[float, float],
    *,
    a0: CoercibleFloat64_1D | None = None,
    ar: CoercibleFloat64_1D | None = None,
    seed: Seed = None,
) -> StochasticSample: ...
@overload
def sample_process(
    process: RenewalRewardProcess,
    nb_samples: int,
    time_window: tuple[float, float],
    cf: CoercibleFloat64_1D,
    *,
    a0: CoercibleFloat64_1D | None = None,
    ar: CoercibleFloat64_1D | None = None,
    cp: CoercibleFloat64_1D | None = None,
    cf1: CoercibleFloat64_1D | None = None,
    cp1: CoercibleFloat64_1D | None = None,
    discounting_rate: float = 0.0,
    seed: Seed = None,
) -> StochasticSample: ...
def sample_process(
    process: RenewalProcess
    | RenewalRewardProcess
    | NonHomogeneousPoissonProcess[()]
    | Kijima1Process[()]
    | Kijima2Process[()],
    nb_samples: int,
    time_window: tuple[float, float],
    cf: CoercibleFloat64_1D | None = None,
    *,
    a0: CoercibleFloat64_1D | None = None,
    ar: CoercibleFloat64_1D | None = None,
    cp: CoercibleFloat64_1D | None = None,
    cf1: CoercibleFloat64_1D | None = None,
    cp1: CoercibleFloat64_1D | None = None,
    discounting_rate: float = 0.0,
    seed: Seed = None,
) -> StochasticSample:
    """Sample a stochastic process.

    Parameters
    ----------
    process : stochastic process
        Process to sample. Must be frozen if it has covariates.
    nb_samples : int
        Number of samples.
    time_window : tuple of float
        Observation time window.
    cf : float or 1d array of floats, optional
        Failure reward. Required for renewal reward processes.
    a0 : float or 1d array of floats, optional
        Initial ages.
    ar : float or 1d array of floats, optional
        Preventive ages of replacement.
    cp : float or 1d array of floats, optional
        Preventive replacement reward.
    cf1 : float or 1d array of floats, optional
        First failure reward.
    cp1 : float or 1d array of floats, optional
        First preventive replacement reward.
    discounting_rate : float, default=0.0
        Discounting rate applied to rewards.
    seed : int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, optional
        Random seed or random number generator.

    Returns
    -------
    out : StochasticSample
        Sampled timeline, events, preventive renewals and optional rewards.
    """
    if isinstance(process, RenewalProcess):
        iterable = RenewalProcessIterable(
            process, nb_samples, time_window, a0=a0, ar=ar, seed=seed
        )
    elif isinstance(process, RenewalRewardProcess):
        assert cf is not None
        iterable = RenewalRewardProcessIterable(
            process,
            nb_samples,
            time_window,
            cf,
            a0=a0,
            ar=ar,
            cp=cp,
            cf1=cf1,
            cp1=cp1,
            discounting_rate=discounting_rate,
            seed=seed,
        )
    elif isinstance(process, NonHomogeneousPoissonProcess):
        iterable = NonHomogeneousPoissonProcessIterable(
            process, nb_samples, time_window, a0=a0, ar=ar, seed=seed
        )
    elif isinstance(process, Kijima1Process):
        iterable = Kijima1ProcessIterable(
            process, nb_samples, time_window, a0=a0, ar=ar, seed=seed
        )
    else:
        iterable = Kijima2ProcessIterable(
            process, nb_samples, time_window, a0=a0, ar=ar, seed=seed
        )
    struct_array = np.concatenate(tuple(iterable))
    assert struct_array.dtype.names is not None  # typeguard

    nb_samples = struct_array["id"].max() + 1

    # unique values of timeline on axis 1
    timeline, col_timeline = np.unique(struct_array["timeline"], return_inverse=True)

    # construction of matrixes
    events = np.zeros((nb_samples, timeline.size), dtype=bool)
    events[struct_array["id"], col_timeline] = struct_array["event"]
    events = events.squeeze()

    preventive_renewals = np.zeros((nb_samples, timeline.size), dtype=bool)
    preventive_renewals[struct_array["id"], col_timeline] = ~struct_array["event"]
    preventive_renewals[:, -1] = False
    preventive_renewals = preventive_renewals.squeeze()

    rewards = None
    if "rewards" in struct_array.dtype.names:
        rewards = np.zeros((nb_samples, timeline.size), dtype=float)
        rewards[struct_array["id"], col_timeline] = struct_array["reward"]
        rewards = rewards.squeeze()

    return StochasticSample(
        timeline=timeline,
        events=events,
        preventive_renewals=preventive_renewals,
        rewards=rewards,
    )
