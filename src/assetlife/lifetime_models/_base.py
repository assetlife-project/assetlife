"""Parametric lifetime model base classes and fitting utilities."""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
    final,
    no_type_check,
    overload,
)

import matplotlib.pyplot as plt
import numpy as np
import optype.numpy as onp
from matplotlib.axes import Axes
from numpydoc import docscrape  # pyright: ignore[reportMissingTypeStubs]
from scipy import stats
from scipy.optimize import newton
from typing_extensions import override

from assetlife.base import FitConfig, MaximumLikelihoodOptimizer, ParametricModel
from assetlife.quadratures import legendre_quadrature, unweighted_laguerre_quadrature
from assetlife.typing import CoercibleFloat64_ND, CovarTs, Float64_ND


@overload
def _plot_probability_function(
    x: onp.Array1D[np.float64],
    y: onp.Array1D[np.float64],
    se: Literal[None],
    ci_bounds: Literal[None],
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes: ...
@overload
def _plot_probability_function(
    x: onp.Array1D[np.float64],
    y: onp.Array1D[np.float64],
    se: onp.Array1D[np.float64],
    ci_bounds: tuple[float, float],
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes: ...
@no_type_check
def _plot_probability_function(
    x: onp.Array1D[np.float64],
    y: onp.Array1D[np.float64],
    se: onp.Array1D[np.float64] | None = None,
    ci_bounds: tuple[float, float] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    if ax is None:
        ax = cast(Axes, plt.gca())
    ax.plot(x, y, **kwargs)
    if se is not None and ci_bounds is not None:
        alpha_ci = kwargs.get("alpha_ci", 0.95)
        assert isinstance(alpha_ci, float)
        z = stats.norm.ppf((1 + alpha_ci) / 2)
        yl = np.clip(y - z * se, ci_bounds[0], ci_bounds[1])
        yu = np.clip(y + z * se, ci_bounds[0], ci_bounds[1])
        drawstyle = kwargs.get("drawstyle", "default")
        step = drawstyle.split("-")[1] if "steps-" in drawstyle else None
        _ = ax.fill_between(
            x,
            yl,
            yu,
            facecolors=[ax.lines[-1].get_color()],
            step=step,
            alpha=0.25,
            label=f"IC-{alpha_ci}",
        )
        ax.legend()
    if kwargs.get("label") is not None:
        ax.legend()
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0.0, right=np.max(x))
    return ax


class ParametricLifetimeModel(ParametricModel, ABC, Generic[*CovarTs]):
    """Base class for parametric lifetime models.

    The class defines the common API for survival, hazard, cumulative hazard,
    density, moments, random sampling, plotting, conditioning, and freezing.
    Subclasses may implement only a minimal subset of probability functions,
    since default formulas derive some functions from others.
    """

    @property
    def args_shape(self) -> tuple[int, ...]:
        """Shape of additional model arguments."""
        return ()

    @abstractmethod
    def sf(self, time: CoercibleFloat64_ND, *args: *CovarTs) -> Float64_ND:
        """
        The survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            sf values at each given time(s).
        """
        if hasattr(self, "chf"):
            return np.exp(
                -self.chf(
                    time,
                    *args,
                )
            )
        elif hasattr(self, "pdf") and hasattr(self, "hf"):
            return np.divide(self.pdf(time, *args), self.hf(time, *args))
        else:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
                {class_name} must implement concrete sf function
                """
            )

    @abstractmethod
    def hf(self, time: CoercibleFloat64_ND, *args: *CovarTs) -> Float64_ND:
        """
        The hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            hf values at each given time(s).
        """
        if hasattr(self, "pdf") and hasattr(self, "sf"):
            return np.divide(self.pdf(time, *args), self.sf(time, *args))
        else:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
                {class_name} must implement concrete hf function.
                """
            )

    @abstractmethod
    def chf(self, time: CoercibleFloat64_ND, *args: *CovarTs) -> Float64_ND:
        """
        The cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            chf values at each given time(s).
        """
        if hasattr(self, "sf"):
            return -np.log(self.sf(time, *args))
        elif hasattr(self, "pdf") and hasattr(self, "hf"):
            return -np.log(self.pdf(time, *args) / self.hf(time, *args))
        else:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
                {class_name} must implement concrete chf or at least concrete
                hf function.
                """
            )

    @abstractmethod
    def pdf(self, time: CoercibleFloat64_ND, *args: *CovarTs) -> Float64_ND:
        """
        The probability density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            pdf values at each given time(s).
        """
        try:
            return self.sf(time, *args) * self.hf(time, *args)
        except NotImplementedError as err:
            class_name = type(self).__name__
            raise NotImplementedError(
                f"""
            {class_name} must implement pdf or the above functions
            """
            ) from err

    def cdf(self, time: CoercibleFloat64_ND, *args: *CovarTs) -> Float64_ND:
        """
        The cumulative distribution function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            cdf values at each given time(s).
        """
        return 1 - self.sf(time, *args)

    def ppf(self, probability: CoercibleFloat64_ND, *args: *CovarTs) -> Float64_ND:
        """
        The percent point function, inverse of the CDF.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            ppf values at each given probability value(s).
        """
        probability = np.asarray(probability)
        return self.isf(1 - probability, *args)

    def median(self, *args: *CovarTs) -> Float64_ND:
        """
        The median.

        Parameters
        ----------
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        return self.ppf(0.5, *args)

    def isf(self, probability: CoercibleFloat64_ND, *args: *CovarTs) -> Float64_ND:
        """
        The inverse survival function.

        Parameters
        ----------
        probability : float or np.ndarray
            Probability value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            isf values at each given probability value(s).
        """

        def func(x: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]:
            return np.asarray(self.sf(x, *args) - probability, dtype=np.float64)

        return newton(func, x0=np.asarray(probability, dtype=np.float64))

    def ichf(
        self,
        cumulative_hazard_rate: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        """
        Inverse cumulative hazard function.

        Parameters
        ----------
        cumulative_hazard_rate : float or np.ndarray
            Cumulative hazard rate value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            ichf values at each given cumulative hazard rate(s).
        """

        def func(x: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]:
            return np.asarray(self.chf(x, *args) - cumulative_hazard_rate)

        return np.float64(
            newton(
                func,
                x0=np.zeros_like(cumulative_hazard_rate),
            )
        )

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *args: *CovarTs,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> Float64_ND:
        """
        Random variate sampling.

        Parameters
        ----------
        size : int or tuple (m, n) of int
            Size of the generated sample.
        *args
            Any additional args.
        seed : optional int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, default is None
            If int or BitGenerator, seed for random number generator. If
            np.random.RandomState or np.random.Generator, use as given.

        Returns
        -------
        out : float or ndarray
            Sample values.
        """
        rng = np.random.default_rng(seed)
        shape = self.args_shape
        if size is not None:
            size = (size,) if isinstance(size, int) else tuple(size)
            shape = np.broadcast_shapes(shape, size)
        probability = rng.uniform(size=shape)
        return self.isf(probability, *args)

    def mean(self, *args: *CovarTs) -> Float64_ND:
        """
        The mean of the distribution.

        Parameters
        ----------
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        return self.moment(1, *args)

    def var(self, *args: *CovarTs) -> Float64_ND:
        """
        The variance of the distribution.

        Parameters
        ----------
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        return self.moment(2, *args) - self.moment(1, *args) ** 2

    def plot(
        self,
        fname: Literal["sf", "cdf", "chf", "hf", "pdf"],
        time: onp.Array1D[np.float64],
        *args: *CovarTs,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot function.

        Parameters
        ----------
        fname : str
            The function name to plot. Allowed names are sf, cdf, chf, hf, pdf.
        time : 1d array
            The timeline used for x-axis.
        *args
            Any additional args required to compute the function.
        ax : plt.Axes, optional
            An optional existing matplotlib.axes.
        **kwargs
            Extra arguments to configure the plot:
                - ci : bool, default is True if the model has fitting_results
                - alpha_ci :
                - any arguments allowed by matplotlib.plot
        """
        if kwargs.get("ci", False) is True:
            raise ValueError("ci is available for fitted models only.")
        y = getattr(self, fname)(time, *args)
        assert onp.is_array_1d(y)  # typeguards
        ci = kwargs.pop("ci", hasattr(self, "fitting_results"))
        if ci:
            time = np.asarray(time, dtype=float)
            se = estimate_se(self, fname, time, *args)
            if se is not None:
                ci_bounds = (0.0, np.inf)
                if fname in ("sf", "chf"):
                    ci_bounds = (0.0, 1.0)
                return _plot_probability_function(
                    time, y, se=se, ci_bounds=ci_bounds, ax=ax, **kwargs
                )
        return _plot_probability_function(time, y, ax=ax, **kwargs)

    def apply_condition(
        self,
        *,
        ar: CoercibleFloat64_ND | None = None,
        a0: CoercibleFloat64_ND | None = None,
    ) -> ParametricLifetimeModel[*CovarTs]:
        """
        Return a model with age replacement, left truncation, or both.

        Parameters
        ----------
        ar : float or ndarray, optional
            Age replacement threshold.
        a0 : float or ndarray, optional
            Initial age for left truncation.

        Returns
        -------
        out : ParametricLifetimeModel
            Conditioned lifetime model.
        """
        # Apply left truncation first for numerical stability
        if a0 is not None:
            left_truncated_model = _LeftTruncatedModel(self, a0)
            if ar is not None:
                # If both are applied, ar becomes ar - a0
                return _AgeReplacementModel(left_truncated_model, np.float64(ar) - a0)
            return left_truncated_model
        if ar is not None:
            return _AgeReplacementModel(self, ar)
        return self

    def freeze(self, *args: *CovarTs) -> _FrozenParametricLifetimeModel[*CovarTs]:
        """Return a model with additional arguments stored."""
        return _FrozenParametricLifetimeModel(self, *args)

    def ls_integrate(
        self,
        func: Callable[[CoercibleFloat64_ND], Float64_ND],
        a: CoercibleFloat64_ND,
        b: CoercibleFloat64_ND,
        *density_args: *CovarTs,
        func_args: tuple[CoercibleFloat64_ND, ...] = (),
        deg: int = 10,
    ) -> Float64_ND:
        """
        Lebesgue-Stieltjes integration.

        Parameters
        ----------
        func : Callable
            Function to integrate with respect to the lifetime distribution.
        a : float or ndarray
            Lower bound of the integration.
        b : float or ndarray
            Upper bound of the integration.
        *density_args
            Additional arguments required by the lifetime model.
        func_args : tuple, default=()
            Additional arguments required by ``func``.
        deg : int, default=10
            Number of sample points and weights for the quadrature.

        Returns
        -------
        out : np.ndarray
            Lebesgue-Stieltjes integration of ``func`` from ``a`` to ``b``.
        """

        def integrand(
            x: CoercibleFloat64_ND,
        ) -> Float64_ND:
            return func(x) * self.pdf(x, *density_args)

        arr_a, arr_b = np.broadcast_arrays(a, b)  # (), (n,) or (m, n)
        if np.any(arr_a > arr_b):
            raise ValueError("Bound values a must be lower than values of b")

        bmax = self.isf(1e-4, *density_args)
        a, b, bmax = np.broadcast_arrays(a, b, bmax)
        binf = np.isinf(b)
        b = np.where(binf, bmax, b)
        integration = legendre_quadrature(
            integrand, a, b, args=(*func_args, *density_args), deg=deg
        )
        return np.where(
            binf,
            integration
            + unweighted_laguerre_quadrature(
                integrand, b, args=(*func_args, *density_args), deg=deg
            ),
            integration,
        )

    def mrl(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        """
        The mean residual life function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            Function values at each given time(s).
        """

        def func(
            x: CoercibleFloat64_ND,
        ) -> Float64_ND:
            return np.float64(x) - np.float64(time)

        return self.ls_integrate(func, time, np.inf, *args) / self.sf(time, *args)

    def moment(
        self,
        n: int,
        *args: *CovarTs,
    ) -> Float64_ND:
        """
        The n-th order moment.

        Parameters
        ----------
        n : int
            Order of the moment, at least 1.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        if n < 1:
            raise ValueError("order of the moment must be at least 1")

        def func(
            x: CoercibleFloat64_ND,
        ) -> Float64_ND:
            return np.power(x, n, dtype=np.float64)

        # high degree of polynome to ensure high precision
        return self.ls_integrate(
            func,
            0.0,
            np.inf,
            *args,
            deg=100,
        )


P = ParamSpec("P")
T = TypeVar("T")


def document_args(
    *,
    base_cls: type,
    args_docstring: list[docscrape.Parameter],
    returns: list[docscrape.Parameter] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Extend method docstrings with argument documentation."""

    def decorator_extend_docstring(
        method: Callable[P, T],
    ) -> Callable[P, T]:
        base_doc = getattr(base_cls, method.__name__).__doc__
        numpy_docstring = docscrape.NumpyDocString(base_doc)
        new_parameters_docstring: list[docscrape.Parameter] = []
        for param in numpy_docstring["Parameters"]:  # pyright: ignore[reportUnknownVariableType]
            if param.name != "*args":  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                new_parameters_docstring.append(param)  # pyright: ignore[reportArgumentType]
            else:
                new_parameters_docstring += args_docstring
        numpy_docstring["Parameters"] = new_parameters_docstring
        if returns is not None:
            numpy_docstring["Returns"] = returns
        method.__doc__ = str(numpy_docstring)

        @functools.wraps(method)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return method(*args, **kwargs)

        return wrapper

    return decorator_extend_docstring


class _AgeReplacementModel(ParametricLifetimeModel[*CovarTs]):
    r"""
    Age replacement model.

    Lifetime model where the assets are replaced at age :math:`a_r`. This is
    equivalent to the model of :math:`\min(X,a_r)` where :math:`X` is a
    baseline lifetime and :math:`a_r` is the age of replacement.

    Parameters
    ----------
    baseline : any parametric lifetime model (frozen lifetime model works)
        The base lifetime model without conditional probabilities

    Attributes
    ----------
    baseline
    ar
    """

    baseline: ParametricLifetimeModel[*CovarTs]
    ar: CoercibleFloat64_ND

    def __init__(
        self,
        baseline: ParametricLifetimeModel[*CovarTs],
        ar: CoercibleFloat64_ND,
    ):
        super().__init__()
        self.baseline = baseline
        self.ar = ar

    @property
    @override
    def args_shape(self) -> tuple[int, ...]:
        return np.broadcast_shapes(np.asarray(self.ar).shape, self.baseline.args_shape)

    @override
    def sf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return np.where(time < self.ar, self.baseline.sf(time, *args), 0.0)

    @override
    def hf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return np.where(time < self.ar, self.baseline.hf(time, *args), 0.0)

    @override
    def chf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return np.where(time < self.ar, self.baseline.chf(time, *args), 0.0)

    @override
    def isf(
        self,
        probability: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return np.minimum(self.baseline.isf(probability, *args), self.ar)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return np.minimum(self.baseline.ichf(cumulative_hazard_rate, *args), self.ar)

    @override
    def pdf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return np.where(time < self.ar, self.baseline.pdf(time, *args), 0)

    @override
    def ls_integrate(
        self,
        func: Callable[[CoercibleFloat64_ND], Float64_ND],
        a: CoercibleFloat64_ND,
        b: CoercibleFloat64_ND,
        *density_args: *CovarTs,
        func_args: tuple[CoercibleFloat64_ND, ...] = (),
        deg: int = 10,
    ) -> Float64_ND:
        b = np.minimum(self.ar, b)
        integration = self.baseline.ls_integrate(
            func, a, b, *density_args, func_args=func_args, deg=deg
        )
        return integration + np.where(
            b == self.ar, func(self.ar) * self.baseline.sf(self.ar, *density_args), 0
        )

    @override
    def __repr__(self) -> str:
        a0 = getattr(self.baseline, "a0", None)
        return f"{self.baseline!r}.apply_condition(a0={a0!r}, ar={self.ar!r})"


class _LeftTruncatedModel(
    ParametricLifetimeModel[*CovarTs],
):
    r"""Left truncated model.

    Lifetime model where the assets have already reached the age :math:`a_0`.

    Parameters
    ----------
    baseline : any parametric lifetime model (frozen lifetime model works)
        The base lifetime model without conditional probabilities
    nb_params
    params
    params_names
    plot

    Attributes
    ----------
    baseline
    a0
    """

    baseline: ParametricLifetimeModel[*CovarTs]
    a0: CoercibleFloat64_ND

    def __init__(
        self,
        baseline: ParametricLifetimeModel[*CovarTs],
        a0: CoercibleFloat64_ND,
    ):
        super().__init__()
        self.baseline = baseline
        self.a0 = a0

    @property
    @override
    def args_shape(self) -> tuple[int, ...]:
        return np.broadcast_shapes(np.asarray(self.a0).shape, self.baseline.args_shape)

    @override
    def sf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return self.baseline.sf(time + self.a0, *args) / self.baseline.sf(
            self.a0, *args
        )

    @override
    def pdf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return super().pdf(time, *args)

    @override
    def isf(
        self,
        probability: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        cumulative_hazard_rate = -np.log(probability + 1e-6)  # avoid division by zero
        return self.ichf(cumulative_hazard_rate, *args)

    @override
    def chf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return self.baseline.chf(self.a0 + time, *args) - self.baseline.chf(
            self.a0, *args
        )

    @override
    def hf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return self.baseline.hf(self.a0 + time, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        return (
            self.baseline.ichf(
                cumulative_hazard_rate + self.baseline.chf(self.a0, *args), *args
            )
            - self.a0
        )

    @override
    def __repr__(self) -> str:
        ar = getattr(self.baseline, "ar", None)
        return f"{self.baseline!r}.apply_condition(a0={self.a0!r}, ar={ar!r})"


class _FrozenParametricLifetimeModel(ParametricLifetimeModel[()], Generic[*CovarTs]):
    """Parametric lifetime model with additional arguments stored."""

    args: tuple[*CovarTs]
    unfrozen: ParametricLifetimeModel[*CovarTs]

    def __init__(
        self,
        model: ParametricLifetimeModel[*CovarTs],
        *args: *CovarTs,
    ) -> None:
        super().__init__()
        self.unfrozen = model
        self.args = args

    @property
    @override
    def args_shape(self) -> tuple[int, ...]:
        return np.broadcast_shapes(
            self.unfrozen.args_shape, *(np.asarray(arg).shape for arg in self.args)
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def sf(self, time: CoercibleFloat64_ND) -> Float64_ND:
        return self.unfrozen.sf(time, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def hf(self, time: CoercibleFloat64_ND) -> Float64_ND:
        return self.unfrozen.hf(time, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def chf(self, time: CoercibleFloat64_ND) -> Float64_ND:
        return self.unfrozen.chf(time, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def pdf(self, time: CoercibleFloat64_ND) -> Float64_ND:
        return self.unfrozen.pdf(time, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def cdf(self, time: CoercibleFloat64_ND) -> Float64_ND:
        return self.unfrozen.cdf(time, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def ppf(self, probability: CoercibleFloat64_ND) -> Float64_ND:
        return self.unfrozen.ppf(probability, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def median(self) -> Float64_ND:
        return self.unfrozen.median(*self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def isf(self, probability: CoercibleFloat64_ND) -> Float64_ND:
        return self.unfrozen.isf(probability, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def ichf(self, cumulative_hazard_rate: CoercibleFloat64_ND) -> Float64_ND:
        return self.unfrozen.ichf(cumulative_hazard_rate, *self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def mean(self) -> Float64_ND:
        return self.unfrozen.mean(*self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def var(self) -> Float64_ND:
        return self.unfrozen.var(*self.args)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> Float64_ND:
        return self.unfrozen.rvs(
            size,
            *self.args,
            seed=seed,
        )

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=[])
    def ls_integrate(
        self,
        func: Callable[[CoercibleFloat64_ND], Float64_ND],
        a: CoercibleFloat64_ND,
        b: CoercibleFloat64_ND,
        *,
        func_args: tuple[CoercibleFloat64_ND, ...] = (),
        deg: int = 10,
    ) -> Float64_ND:
        return self.unfrozen.ls_integrate(
            func, a, b, *self.args, func_args=func_args, deg=deg
        )

    def __getattr__(self, key: str) -> Any:
        # __getattr__ needed to catch jac_<func> if it exists
        frozen_type = self.unfrozen.__class__.__name__
        try:
            attr = getattr(self.unfrozen, key)
        except AttributeError as err:
            raise AttributeError(
                f"Frozen {frozen_type} has no attribute {key}"
            ) from err

        def wrapper(*args: Any, **kwargs: Any):
            return attr(*(*args, *self.args), **kwargs)

        if inspect.ismethod(attr):
            return wrapper
        return attr

    @override
    def __repr__(self) -> str:
        return f"{self.unfrozen!r}.freeze({self.args!r})"


def estimate_se(
    model: ParametricLifetimeModel[*CovarTs],
    fname: str,
    time: onp.Array1D[np.float64],
    *args: *CovarTs,
) -> onp.Array1D[np.float64] | None:
    """

    References
    ----------
    .. [1] Meeker, W. Q., Escobar, L. A., & Pascual, F. G. (2022).
        Statistical methods for reliability data. John Wiley & Sons.
    """
    # [1] equation B.10 in Appendix
    if (
        model.fitting_results is not None
        and model.fitting_results.covariance_matrix is not None
    ):
        se = np.zeros_like(time)
        jac_f = getattr(model, f"jac_{fname}")(time[1:], *args)
        se[1:] = np.sqrt(
            np.einsum(
                "i...,ij,j...->...",
                jac_f,
                model.fitting_results.covariance_matrix,
                jac_f,
            )
        )
        return se
    return None


class FittableParametricLifetimeModel(ParametricLifetimeModel[*CovarTs], ABC):
    """Base class for parametric lifetime models that can be fitted."""

    @abstractmethod
    def jac_hf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> onp.ArrayND[np.float64]:
        """
        The Jacobian of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            The derivatives with respect to each parameter. If the result is
            an `np.ndarray`, the first dimension holds the number of parameters.
        """

    @abstractmethod
    def jac_chf(
        self, time: CoercibleFloat64_ND, *args: *CovarTs
    ) -> onp.ArrayND[np.float64]:
        """
        The Jacobian of the cumulative hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            The derivatives with respect to each parameter. If the result is
            an `np.ndarray`, the first dimension holds the number of parameters.
        """

    @abstractmethod
    def dhf(
        self, time: CoercibleFloat64_ND, *args: *CovarTs
    ) -> onp.ArrayND[np.float64]:
        """
        The derivative of the hazard function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            Function values at each given time(s).
        """

    def jac_sf(
        self, time: CoercibleFloat64_ND, *args: *CovarTs
    ) -> onp.ArrayND[np.float64]:
        """
        The Jacobian of the survival function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            The derivatives with respect to each parameter. If the result is
            an `np.ndarray`, the first dimension holds the number of parameters.
        """
        return -self.jac_chf(time, *args) * self.sf(time, *args)

    def jac_cdf(
        self, time: CoercibleFloat64_ND, *args: *CovarTs
    ) -> onp.ArrayND[np.float64]:
        """
        The Jacobian of the cumulative distribution function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            The derivatives with respect to each parameter. If the result is
            an `np.ndarray`, the first dimension holds the number of parameters.
        """
        return -self.jac_sf(time, *args)

    def jac_pdf(
        self, time: CoercibleFloat64_ND, *args: *CovarTs
    ) -> onp.ArrayND[np.float64]:
        """
        The Jacobian of the probability density function.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args
            Any additional args.

        Returns
        -------
        out : np.float64 or np.ndarray
            The derivatives with respect to each parameter. If the result is
            an `np.ndarray`, the first dimension holds the number of parameters.

        """
        jac = self.jac_hf(time, *args) * self.sf(time, *args) + self.jac_sf(
            time, *args
        ) * self.hf(time, *args)
        return jac

    @abstractmethod
    def init_likelihood(
        self,
        time: onp.Array1D[np.float64] | onp.Array[tuple[int, Literal[2]], np.float64],
        args: Sequence[onp.Array1D[np.float64]] | None = None,
        event: onp.Array1D[np.bool_] | None = None,
        entry: onp.Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> LifetimeLikelihood:
        r"""
        Initialize the lifetime likelihood used to fit the parameters.

        ``fit`` method is the preferred way to fit model parameters. However,
        users can also interact with the likelihood returned by
        ``init_likelihood`` to study the optimization process.

        This method implementation is usually composed of 3 steps:
            1. Initialize an object to store and preprocess lifetime values.
            2. Create a ``FitConfig`` instance depending on the model needs.
            3. Instantiate and return a ``LifetimeLikelihood``.

        ``init_likelihood`` is separated from ``fit`` in order to reuse existing
        likelihood parametrization in case of model composition. Any parameters
        initialization needed by the likelihood optimizer (e.g. ``x0`` or
        ``bounds`` as required in step 2.) are left to specific functions
        alongside concrete model implementations. These functions are invoked
        within ``init_likelihood``.

        Parameters
        ----------
        time : 1d array or 2d array
            Observed lifetime values. 1d array can handle complete and right censored
            lifetimes with ``event``. To add left censored or interval censored
            lifetimes, use 2d array.
        args : any ndarray or tuple of ndarray, default is None
            Additional arguments required by the model (e.g. covar).
        event : 1d array of bool, default is None
            Boolean indicators tagging lifetime values as right censored or complete.
        entry : 1d array, default is None
            Left truncations applied to lifetime values.
        **kwargs
            Extra arguments to control the parameters optimization. It can be:

                - those used by `scipy.optimize.minimize
                  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
                  to search for the parameters that minimize the negative
                  log-likelihood.
                - `covariance_method` to control the method used to estimate
                  parameters covariance. Values can be `"cs"`, `"2point"`,
                  `"exact"` or `False`. To skip parameters covariance
                  estimation, set `covariance_method` to `False`, otherwise the
                  default method associated to the model will be used. If
                  `covariance_method` is `"exact"` the `hess` must be passed
                  too.

        Returns
        -------
        out : LifetimeLikelihood instance
        """


@dataclass
class LifetimeData:
    """Preprocessed lifetime observations used for likelihood computation."""

    nb_observations: int = field(init=False)
    complete_time: onp.Array[tuple[int, Literal[1]], np.float64] = field(
        init=False, repr=False
    )
    censored_time: (
        onp.Array[tuple[int, Literal[1]], np.float64]
        | onp.Array[tuple[int, Literal[2]], np.float64]
    ) = field(init=False, repr=False)
    left_truncations: onp.Array[tuple[int, Literal[1]], np.float64] = field(
        init=False, repr=False
    )
    complete_time_args: tuple[onp.Array[tuple[int, Literal[1]], np.float64], ...] = (
        field(init=False, repr=False)
    )
    censored_time_args: tuple[onp.Array[tuple[int, Literal[1]], np.float64], ...] = (
        field(init=False, repr=False)
    )
    left_truncations_args: tuple[onp.Array[tuple[int, Literal[1]], np.float64], ...] = (
        field(init=False, repr=False)
    )

    def __init__(
        self,
        time: onp.Array1D[np.float64] | onp.Array[tuple[int, Literal[2]], np.float64],
        event: onp.Array1D[np.bool_] | None = None,
        entry: onp.Array1D[np.float64] | None = None,
        args: Sequence[onp.Array1D[np.float64]] = (),
    ) -> None:
        column_time = time[:, None]
        if column_time.shape[-1] == 2 and event is not None:
            raise ValueError("If time is given as intervals, event must be None")
        column_event = None
        if column_time.shape[-1] == 1:
            column_event = (
                event[:, None]
                if event is not None
                else np.ones_like(time, dtype=np.bool_)
            )
        column_entry = (
            entry[:, None]
            if entry is not None
            else np.zeros(len(time), dtype=np.float64)
        )
        if np.any(column_time <= column_entry):
            raise ValueError("All time values must be greater than entry values")
        column_args = tuple(arg[:, None] for arg in args)
        sizes = [
            len(x)
            for x in (column_time, column_event, column_entry, *column_args)
            if x is not None
        ]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"""
                All lifetime data must have the same number of values. Fields
                length are different. Got {tuple(sizes)}
                """
            )
        non_zero_entry = np.flatnonzero(column_entry)
        if column_event is not None:
            non_zero_event = np.flatnonzero(column_event)
            zero_event = np.flatnonzero(column_event == 0)
            self.nb_observations = len(time)
            self.complete_time = column_time[non_zero_event]
            self.censored_time = column_time[zero_event]
            self.left_truncations = column_entry[non_zero_entry]
            self.complete_time_args = tuple(arg[non_zero_event] for arg in column_args)
            self.censored_time_args = tuple(arg[zero_event] for arg in column_args)
            self.left_truncations_args = tuple(
                arg[non_zero_entry] for arg in column_args
            )
        else:
            complete_time_index = np.flatnonzero(column_time[:, 0] == column_time[:, 1])
            non_complete_time_index = np.flatnonzero(
                column_time[:, 0] != column_time[:, 1]
            )
            self.nb_observations = len(time)
            self.complete_time = column_time[:, 1][complete_time_index]
            self.censored_time = column_time[non_complete_time_index]
            self.left_truncations = column_entry[non_zero_entry]
            self.complete_time_args = tuple(
                arg[complete_time_index] for arg in column_args
            )
            self.censored_time_args = tuple(
                arg[non_complete_time_index] for arg in column_args
            )
            self.left_truncations_args = tuple(
                arg[non_zero_entry] for arg in column_args
            )


@final
class LifetimeLikelihood(
    MaximumLikelihoodOptimizer[
        FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]], LifetimeData
    ]
):
    """
    Maximum likelihood estimator from lifetime data.

    Parameters
    ----------
    model : FittableParametricLifetimeModel
        Model with initialized parameters.
    data : LifetimeData
        Preprocessed lifetime observations.
    config : FitConfig
        Configuration used by the optimizer.

    Attributes
    ----------
    model : FittableParametricLifetimeModel
        Model used by the likelihood.
    data : LifetimeData
        Preprocessed lifetime observations.
    config : FitConfig
        Configuration used by the optimizer.
    """

    data: LifetimeData

    def __init__(
        self,
        model: FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]],
        data: LifetimeData,
        config: FitConfig,
    ):
        self.model = model
        self.data = data
        self.config = config
        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log

    @property
    @override
    def nb_observations(self) -> int:
        return self.data.nb_observations

    @override
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        self.model.set_params(params)
        return (
            self._complete_time_contrib()
            + self._censored_time_contrib()
            + self._left_truncations_contrib()
        )

    def jac_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array1D[np.float64]:
        """
        Jacobian of the negative log likelihood.

        The Jacobian is computed with respect to parameters.

        Parameters
        ----------
        params : 1d array of floats
            Parameter values.

        Returns
        -------
        out : 1d array of floats
        """
        self.model.set_params(params)
        return (
            self._jac_complete_time_contrib()
            + self._jac_censored_time_contrib()
            + self._jac_left_truncations_contrib()
        )

    def _complete_time_contrib(self) -> float:
        if self.data.complete_time.size == 0.0:
            return 0.0
        res = -np.sum(
            np.log(
                self.model.pdf(self.data.complete_time, *self.data.complete_time_args)
            )
        )
        return res

    def _jac_complete_time_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.complete_time.size == 0:
            return np.zeros_like(self.model.get_params())
        jac = -self.model.jac_pdf(
            self.data.complete_time, *self.data.complete_time_args
        ) / self.model.pdf(self.data.complete_time, *self.data.complete_time_args)

        return np.sum(jac, axis=(1, 2))

    def _censored_time_contrib(self) -> float:
        if self.data.censored_time.size == 0:
            return 0.0
        if self.data.censored_time.shape[-1] > 1:
            # interval censored time
            return np.sum(
                -np.log(
                    10**-10
                    + self.model.cdf(
                        self.data.censored_time[:, 1], *self.data.censored_time_args
                    )
                    - self.model.cdf(
                        self.data.censored_time[:, 0], *self.data.censored_time_args
                    )
                ),
            )
        else:
            # right censored time
            return np.sum(
                self.model.chf(self.data.censored_time, *self.data.censored_time_args)
            )

    def _jac_censored_time_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.censored_time.size == 0:
            return np.zeros_like(self.model.get_params())
        if self.data.censored_time.shape[-1] > 1:
            # interval censored time
            jac_interval_censored = (
                self.model.jac_sf(
                    self.data.censored_time[:, 1], *self.data.censored_time_args
                )
                - self.model.jac_sf(
                    self.data.censored_time[:, 0], *self.data.censored_time_args
                )
            ) / (
                10**-10
                + self.model.cdf(
                    self.data.censored_time[:, 1], *self.data.censored_time_args
                )
                - self.model.cdf(
                    self.data.censored_time[:, 0], *self.data.censored_time_args
                )
            )

            return np.sum(jac_interval_censored, axis=(1, 2))
        else:
            # right censored time
            return np.sum(
                self.model.jac_chf(
                    self.data.censored_time, *self.data.censored_time_args
                ),
                axis=(1, 2),
            )

    def _left_truncations_contrib(self) -> float:
        if self.data.left_truncations.size == 0.0:
            return 0.0
        return -np.sum(
            self.model.chf(self.data.left_truncations, *self.data.left_truncations_args)
        )

    def _jac_left_truncations_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.left_truncations.size == 0.0:
            return np.zeros_like(self.model.get_params())
        jac = -self.model.jac_chf(
            self.data.left_truncations, *self.data.left_truncations_args
        )
        return np.sum(jac, axis=(1, 2))
