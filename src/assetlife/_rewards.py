"""Reward and discounting utilities."""

import numpy as np

from assetlife.typing import CoercibleFloat64_1D, CoercibleFloat64_ND, Float64_ND


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
    """
    if a0 is not None:
        time = time + a0
    # run-to-failure rewards
    if cp is None and ar is None:
        return np.ones_like(time) * np.float64(cf)
    # preventive age replacement rewards
    elif cp is not None and ar is not None:
        return np.where(time < ar, cf, cp)
    else:
        raise TypeError("Bad arguments. cp and ar must be set together.")


def broadcast_rewards_args(
    *args: CoercibleFloat64_1D | None,
) -> tuple[int, ...]:
    """Return the broadcast shape of reward arguments."""
    return np.broadcast_shapes(
        *(np.asarray(arg).shape for arg in args if arg is not None)
    )


def discounting_factor(time: CoercibleFloat64_ND, rate: float) -> Float64_ND:
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
    """
    assert rate >= 0
    if rate != 0.0:
        return np.exp(-rate * time, dtype=np.float64)
    if isinstance(time, np.ndarray):
        return np.ones_like(time, dtype=np.float64)
    return np.float64(1)


def discounting_annuity_factor(time: CoercibleFloat64_ND, rate: float) -> Float64_ND:
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
    """
    assert rate >= 0
    if rate != 0.0:
        return (1 - np.exp(-rate * time, dtype=np.float64)) / rate
    if isinstance(time, np.ndarray):
        return time.astype(np.float64)
    return np.float64(time)


class ExponentialDiscounting:
    """
    Exponential discounting.

    Parameters
    ----------
    rate : float, default=0.0
        Discounting rate.
    """

    rate: float

    def __init__(self, rate: float = 0.0) -> None:
        self.rate = rate

    def factor(self, time: CoercibleFloat64_ND) -> Float64_ND:
        """
        Compute the discounting factor.

        Parameters
        ----------
        time : float or np.ndarray
            Time values.

        Returns
        -------
        out : float or np.ndarray
            Discounting factors.
        """
        if self.rate != 0.0:
            return np.exp(-self.rate * time, dtype=np.float64)
        if isinstance(time, np.ndarray):
            return np.ones_like(time, dtype=np.float64)
        return np.float64(1)

    def annuity_factor(self, time: CoercibleFloat64_ND) -> Float64_ND:
        """
        Compute the discounting annuity factor.

        Parameters
        ----------
        time : float or np.ndarray
            Time values.

        Returns
        -------
        out : float or np.ndarray
            Annuity factors.
        """
        if self.rate != 0.0:
            return (1 - np.exp(-self.rate * time, dtype=np.float64)) / self.rate
        if isinstance(time, np.ndarray):
            return time.astype(np.float64)
        return np.float64(time)
