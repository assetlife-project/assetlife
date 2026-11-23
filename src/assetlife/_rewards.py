"""Reward and discounting utilities."""

import numpy as np

from assetlife._typing import CoercibleFloat64_1D, CoercibleFloat64_ND, Float64_ND


def compute_rewards(
    time: CoercibleFloat64_ND,
    *,
    cf: CoercibleFloat64_1D,
    cp: CoercibleFloat64_1D | None = None,
    ar: CoercibleFloat64_1D | None = None,
    a0: CoercibleFloat64_1D | None = None,
) -> Float64_ND:
    """
    Compute failure or preventive replacement rewards.

    Parameters
    ----------
    time : float or np.ndarray
        Event or replacement times.
    cf : float or 1d array
        Failure reward.
    cp : float or 1d array, optional
        Preventive replacement reward. Must be set with ``ar``.
    ar : float or 1d array, optional
        Preventive replacement age. Must be set with ``cp``.
    a0 : float or 1d array, optional
        Initial ages.

    Returns
    -------
    out : float or np.ndarray
        Reward values.

    Raises
    ------
    TypeError
        If only one of ``cp`` and ``ar`` is set.
    """
    age = time if a0 is None else time + a0
    # run-to-failure rewards
    if cp is None and ar is None:
        return np.ones_like(age) * np.float64(cf)
    # preventive age replacement rewards
    if cp is not None and ar is not None:
        return np.where(age < ar, cf, cp)
    msg = "Bad arguments. cp and ar must be set together."
    raise TypeError(msg)


def broadcast_rewards_args(
    *args: CoercibleFloat64_1D | None,
) -> tuple[int, ...]:
    """Return the broadcast shape of reward arguments."""
    return np.broadcast_shapes(
        *(np.asarray(arg).shape for arg in args if arg is not None)
    )


def exp_discounting_factor(time: CoercibleFloat64_ND, rate: float) -> Float64_ND:
    """
    Compute the exponential discounting factor.

    Parameters
    ----------
    time : float or np.ndarray
        Time values.
    rate : float
        Discounting rate.

    Returns
    -------
    out : float or np.ndarray
        Discounting factors.

    Raises
    ------
    ValueError
        If ``rate`` is negative (or NaN).
    """
    if not rate >= 0:  # also rejects NaN
        msg = f"Discounting rate must be non-negative, got {rate}"
        raise ValueError(msg)
    if rate > 0:
        return np.exp(-rate * time, dtype=np.float64)
    if isinstance(time, np.ndarray):
        return np.ones_like(time, dtype=np.float64)
    return np.float64(1)


def exp_discounting_annuity_factor(
    time: CoercibleFloat64_ND, rate: float
) -> Float64_ND:
    """
    Compute the exponential discounting annuity factor.

    Parameters
    ----------
    time : float or np.ndarray
        Time values.
    rate : float
        Discounting rate.

    Returns
    -------
    out : float or np.ndarray
        Annuity factors.

    Raises
    ------
    ValueError
        If ``rate`` is negative (or NaN).
    """
    if not rate >= 0:  # also rejects NaN
        msg = f"Discounting rate must be non-negative, got {rate}"
        raise ValueError(msg)
    if rate > 0:
        return (1 - np.exp(-rate * time, dtype=np.float64)) / rate
    if isinstance(time, np.ndarray):
        return time.astype(np.float64)
    return np.float64(time)
