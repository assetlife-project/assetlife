from collections.abc import Callable

import numpy as np
import optype.numpy as onp

from assetlife._typing import CoercibleFloat64_ND, Float64_ND

__all__ = [
    "laguerre_quadrature",
    "legendre_quadrature",
    "unweighted_laguerre_quadrature",
]

type ST = int | float
type NumpyST = np.floating | np.uint


def _control_bounds(*bounds: CoercibleFloat64_ND) -> None:
    for bound in bounds:
        if np.any(bound < 0):
            msg = "Bound values of the integral can't be lower than 0"
            raise ValueError(msg)


def legendre_quadrature(
    func: Callable[[CoercibleFloat64_ND], Float64_ND],
    lower_bound: CoercibleFloat64_ND,
    upper_bound: CoercibleFloat64_ND,
    args: tuple[object, ...] = (),
    deg: int = 10,
) -> np.float64 | onp.ArrayND[np.float64]:
    r"""Numerical integration of :math:`f(x)` over the interval :math:`[a,b]`

    Parameters
    ----------
    func : Callable
        A function of the form `y = func(x, a, b, c, ...)` taking floats or ndarrays
        as inputs and returning a np.float64 or an ndarray. `a, b, c, ...` are extra
        arguments that must be passed in the `args` parameter.
    lower_bound : float or ndarray
        The lower bound of the integration.
    upper_bound : float or ndarray
        The upper bound of the integration. Can't be `np.inf`.
    args : any
        Extra arguments used in the function call.
    deg : int, default is 10.
        Number of sample points and weights for the quadrature

    Notes
    -----
    The function computes if `y = func(x, a, b, c, ...)` has a compatible shape with
    `lower_bound` and `upper_bound` for
    `x.shape == np.broadcast_shapes(lower_bound, upper_bound)`. Otherwise, adapt
    `lower_bound`, `upper_bound` or change `func` implementation.d, upper_bound)`.

    Returns
    -------
    out : np.float64 or np.ndarray
        The output shape corresponds to a broadcast between `a`, `b` and `*args`.

    Raises
    ------
    ValueError
        If a bound is negative, if `upper_bound` is infinite, or if `lower_bound`
        is greater than `upper_bound`.
    """
    lower_bound = np.float64(lower_bound)  # (*a.shape,)
    upper_bound = np.float64(upper_bound)  # (*b.shape,)
    _control_bounds(lower_bound, upper_bound)
    lower_bound, upper_bound, *_ = np.broadcast_arrays(
        lower_bound, upper_bound, *(np.asarray(arg) for arg in args)
    )  # (*shape,)
    if np.any(upper_bound == np.inf):
        msg = "Bound values of Legendre quadrature must be finite"
        raise ValueError(msg)
    if np.any(lower_bound > upper_bound):
        msg = "Bound values a must be lower than values of b"
        raise ValueError(msg)
    x, w = np.polynomial.legendre.leggauss(deg)  # (deg,)
    x = np.expand_dims(
        x, axis=tuple(range(1, max(2, lower_bound.ndim + 1)))
    )  # (deg, 1, ..., 1)
    w = np.expand_dims(
        w, axis=tuple(range(1, max(2, lower_bound.ndim + 1)))
    )  # (deg, 1, ..., 1)
    p = (upper_bound - lower_bound) / 2  # (*shape,)
    m = (lower_bound + upper_bound) / 2  # (*shape,)
    u = p * x + m  # (deg, *shape)
    v = p * w  # (deg, *shape)
    fvalues = func(u)  # (deg, *shape)
    return np.sum(v * fvalues, axis=0).reshape(lower_bound.shape)  # (*shape,)


def laguerre_quadrature(
    func: Callable[[CoercibleFloat64_ND], Float64_ND],
    lower_bound: CoercibleFloat64_ND,
    args: tuple[object, ...] = (),
    deg: int = 10,
) -> np.float64 | onp.ArrayND[np.float64]:
    r"""Numerical integration of :math:`f(x) * exp(-x)` over :math:`[a, \infty]`.

    Parameters
    ----------
    func : Callable
        A function of the form `y = func(x, a, b, c, ...)` taking floats or ndarrays
        as inputs and returning a np.float64 or an ndarray. `a, b, c, ...` are extra
        arguments that must be passed in the `args` parameter.
    lower_bound : float or ndarray
        The lower bound of the integration.
    args : float or ndarray
        Extra arguments used in the function call.
    deg : int, default is 10.
        Number of sample points and weights for the quadrature

    Notes
    -----
    The function computes if `y = func(x, a, b, c, ...)` has a compatible shape with
    `lower_bound`. Otherwise, adapt `lower_bound` or change `func` implementation.

    Returns
    -------
    out : np.float64 or np.ndarray
    """
    lower_bound = np.float64(lower_bound)
    _control_bounds(lower_bound)
    lower_bound, *_ = np.broadcast_arrays(
        lower_bound, *(np.asarray(arg) for arg in args)
    )  # (*shape,)
    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    x = np.expand_dims(
        x, axis=tuple(range(1, max(2, lower_bound.ndim + 1)))
    )  # (deg, 1, ..., 1)
    w = np.expand_dims(
        w, axis=tuple(range(1, max(2, lower_bound.ndim + 1)))
    )  # (deg, 1, ..., 1)
    fvalues = func(x + lower_bound)  # (deg, *shape)
    exp_a = np.where(
        np.exp(-np.float64(lower_bound)) == 0, 1.0, np.exp(-np.float64(lower_bound))
    )  # (*shape,)
    return np.sum(w * fvalues * exp_a, axis=0).reshape(lower_bound.shape)  # (*shape,)


def unweighted_laguerre_quadrature(
    func: Callable[[CoercibleFloat64_ND], Float64_ND],
    lower_bound: CoercibleFloat64_ND,
    args: tuple[object, ...] = (),
    deg: int = 10,
) -> np.float64 | onp.ArrayND[np.float64]:
    r"""Numerical integration of :math:`f(x)` over the interval :math:`[a, \infty]`

    Parameters
    ----------
    func : Callable
        A function of the form `y = func(x, a, b, c, ...)` taking floats or ndarrays
        as inputs and returning a np.float64 or an ndarray. `a, b, c, ...` are extra
        arguments that must be passed in the `args` parameter.
    lower_bound : float or ndarray
        The lower bound of the integration.
    args : float or ndarray
        Extra arguments used in the function call.
    deg : int, default is 10.
        Number of sample points and weights for the quadrature

    Notes
    -----
    The function computes if `y = func(x, a, b, c, ...)` has a compatible shape with
    `lower_bound`. Otherwise, adapt `lower_bound` or change `func` implementation.

    Returns
    -------
    out : np.float64 or np.ndarray
    """

    lower_bound = np.float64(lower_bound)
    _control_bounds(lower_bound)
    lower_bound, *_ = np.broadcast_arrays(
        lower_bound, *(np.asarray(arg) for arg in args)
    )  # (*shape,)
    x, w = np.polynomial.laguerre.laggauss(deg)  # (deg,)
    x = np.expand_dims(
        x, axis=tuple(range(1, max(2, lower_bound.ndim + 1)))
    )  # (deg, 1, ..., 1)
    w = np.expand_dims(
        w, axis=tuple(range(1, max(2, lower_bound.ndim + 1)))
    )  # (deg, 1, ..., 1)
    fvalues = func(x + lower_bound)  # (deg, *shape)
    return np.sum(w * fvalues * np.exp(x), axis=0).reshape(
        lower_bound.shape
    )  # (*shape,)
