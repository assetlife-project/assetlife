"""Preventive age replacement policies."""

from __future__ import annotations

import textwrap
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast, override

import numpy as np
from scipy.optimize import newton

from ._base import BaseAgeReplacementPolicy, OneCycleExpectedCosts
from assetlife._rewards import exp_discounting_annuity_factor, exp_discounting_factor
from assetlife.lifetime_models import ParametricLifetimeModel
from assetlife.quadratures import legendre_quadrature
from assetlife.stochastic_processes import (
    NonHomogeneousPoissonProcess,
    RenewalProcess,
    RenewalRewardProcess,
)

if TYPE_CHECKING:
    import optype.numpy as onp

    from assetlife.typing import (
        CoercibleFloat64_1D,
        CoercibleFloat64_ND,
        Float64_1D,
        Float64_ND,
        Timeline,
    )

import functools
import inspect

R = TypeVar("R")
P = ParamSpec("P")


def check_impossible_replacements(
    ar: CoercibleFloat64_1D,
    a0: CoercibleFloat64_1D | None,
) -> None:
    """Warn when replacement ages are lower than initial ages."""
    # check ar is greater than a0 if a0 is provided
    if a0 is not None and np.any(ar < a0):
        warnings.warn(
            textwrap.dedent(
                """
                Some ages of replacement are inferior to assets ages.
                You may change ages of replacement.
                """
            ),
            stacklevel=2,
        )


def get_cf_cp(
    **costs: CoercibleFloat64_1D,
) -> tuple[Float64_1D, Float64_1D]:
    """
    Extract failure and preventive replacement costs.

    Raises
    ------
    TypeError
        If ``cf`` or ``cp`` is missing from ``costs``.
    """
    try:
        return np.float64(costs["cf"]), np.float64(costs["cp"])
    except KeyError as err:
        raise TypeError("Missing cf and cp values") from err


class OneCycleAgeReplacementPolicy(
    BaseAgeReplacementPolicy[ParametricLifetimeModel[()]]
):
    r"""One-cycle age replacement policy.

    Asset is replaced at age :math:`a_r` with cost :math:`c_p`, or upon failure
    with cost :math:`c_f`. Only one replacement cycle is considered.

    Parameters
    ----------
    lifetime_model : ParametricLifetimeModel
        Lifetime model representing durations between events.

    References
    ----------
    .. [1] Coolen-Schrijner, P., & Coolen, F. P. A. (2006). On optimality
        criteria for age replacement. Proceedings of the Institution of
        Mechanical Engineers, Part O: Journal of Risk and Reliability,
        220(1), 21-29
    """

    period_before_discounting: float

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        period_before_discounting: float = 1.0,
    ) -> None:
        super().__init__(lifetime_model)
        self.period_before_discounting = period_before_discounting

    def _expected_costs(self) -> OneCycleExpectedCosts:
        return OneCycleExpectedCosts(
            self.baseline,
            period_before_discounting=self.period_before_discounting,
        )

    @override
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        cf, cp = get_cf_cp(**costs)
        check_impossible_replacements(ar, a0)
        return self._expected_costs().expected_net_present_value(
            tf, nb_steps, a0=a0, ar=ar, cf=cf, cp=cp, discounting_rate=discounting_rate
        )

    @override
    def asymptotic_expected_net_present_value(
        self,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> Float64_1D:
        cf, cp = get_cf_cp(**costs)
        check_impossible_replacements(ar, a0)
        return self._expected_costs().asymptotic_expected_net_present_value(
            a0=a0, cp=cp, cf=cf, ar=ar, discounting_rate=discounting_rate
        )

    @override
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        cf, cp = get_cf_cp(**costs)
        check_impossible_replacements(ar, a0)
        return self._expected_costs().expected_equivalent_annual_cost(
            tf, nb_steps, a0=a0, ar=ar, cp=cp, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> Float64_1D:
        cf, cp = get_cf_cp(**costs)
        check_impossible_replacements(ar, a0)
        return self._expected_costs().asymptotic_expected_equivalent_annual_cost(
            a0=a0, ar=ar, cp=cp, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def compute_optimal_ar(
        self,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> float | onp.Array1D[np.float64]:
        cf, cp = get_cf_cp(**costs)

        x0 = np.minimum(cp / (cf - cp), 1)  # () or (m, 1)

        # x0 must have the same shape than eq(x0) (see scipy.newton doc)
        sf_x0 = self.baseline.sf(x0)
        if sf_x0.ndim == 2 and x0.ndim == 0:
            x0 = np.tile(x0, (sf_x0.shape[0], 1))

        def eq(a: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]:
            return np.asarray(
                exp_discounting_factor(a, rate=discounting_rate)
                / exp_discounting_annuity_factor(a, rate=discounting_rate)
                * (
                    (cf - cp) * self.baseline.hf(a)
                    - cp / exp_discounting_annuity_factor(a, rate=discounting_rate)
                )
            )

        return newton(eq, x0)


class AgeReplacementPolicy(BaseAgeReplacementPolicy[ParametricLifetimeModel[()]]):
    r"""Age replacement renewal policy.

    Asset is replaced at age :math:`a_r` with cost :math:`c_p`, or upon failure
    with cost :math:`c_f`.

    Parameters
    ----------
    lifetime_model : ParametricLifetimeModel
        Lifetime model representing durations between events.

    References
    ----------
    .. [1] Mazzuchi, T. A., Van Noortwijk, J. M., & Kallen, M. J. (2007).
        Maintenance optimization. Encyclopedia of Statistics in Quality and
        Reliability, 1000-1008.
    """

    @override
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        cf, cp = get_cf_cp(**costs)
        check_impossible_replacements(ar, a0)
        return RenewalRewardProcess(self.baseline).expected_total_reward(
            tf, nb_steps, a0=a0, ar=ar, cp=cp, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        cf, cp = get_cf_cp(**costs)
        check_impossible_replacements(ar, a0)
        return RenewalRewardProcess(self.baseline).expected_equivalent_annual_worth(
            tf, nb_steps, a0=a0, ar=ar, cp=cp, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def asymptotic_expected_net_present_value(
        self,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> Float64_1D:
        cf, cp = get_cf_cp(**costs)
        check_impossible_replacements(ar, a0)
        return RenewalRewardProcess(self.baseline).asymptotic_expected_total_reward(
            a0=a0, ar=ar, cp=cp, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> Float64_1D:
        cf, cp = get_cf_cp(**costs)
        check_impossible_replacements(ar, a0)
        return RenewalRewardProcess(
            self.baseline
        ).asymptotic_expected_equivalent_annual_worth(
            a0=a0, ar=ar, cp=cp, cf=cf, discounting_rate=discounting_rate
        )

    def annual_number_of_replacements(
        self,
        nb_years: int,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        """
        The expected annual number of replacements.

        Parameters
        ----------
        nb_years : int
            Number of years used to project annual replacements.
        ar : float or np.ndarray
            Ages of replacement.
        a0 : float or np.ndarray, optional
            Initial ages.

        Returns
        -------
        out : tuple of np.ndarray
            Timeline and corresponding values.
        """

        check_impossible_replacements(ar, a0)
        timeline, nb_renewals = RenewalProcess(self.baseline).renewal_function(
            nb_years, nb_years + 1, a0=a0, ar=ar
        )
        return timeline[1:], np.diff(nb_renewals, axis=0)

    def annual_number_of_failures(
        self,
        nb_years: int,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        """
        The expected annual number of replacements upon failure.

        Parameters
        ----------
        nb_years : int
            Number of years used to project annual replacements.
        ar : float or np.ndarray
            Ages of replacement.
        a0 : float or np.ndarray, optional
            Initial ages.

        Returns
        -------
        out : tuple of np.ndarray
            Timeline and corresponding values.
        """

        check_impossible_replacements(ar, a0)
        timeline, nb_events = RenewalProcess(self.baseline).expected_number_of_events(
            nb_years, nb_years + 1, a0=a0, ar=ar
        )
        return timeline[1:], np.diff(nb_events, axis=0)

    @override
    def compute_optimal_ar(
        self,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> float | onp.Array1D[np.float64]:

        cf, cp = get_cf_cp(**costs)
        x0 = np.minimum(
            np.float64(cp) / (cf - cp),
            1,
        )

        # x0 must have the same shape than eq(x0) (see scipy.newton doc)
        sf_x0 = self.baseline.sf(x0)
        if sf_x0.ndim == 2 and x0.ndim == 0:
            x0 = np.tile(x0, (sf_x0.shape[0], 1))

        def eq(
            a: onp.ArrayND[np.float64],
        ) -> onp.ArrayND[np.float64]:  # () or (m, 1)
            f = legendre_quadrature(
                lambda x: (
                    exp_discounting_factor(x, rate=discounting_rate)
                    * self.baseline.sf(x)
                ),
                0,
                a,
            )
            g = legendre_quadrature(
                lambda x: (
                    exp_discounting_factor(x, rate=discounting_rate)
                    * self.baseline.pdf(x)
                ),
                0,
                a,
            )
            return np.asarray(
                exp_discounting_factor(a, rate=discounting_rate)
                * ((cf - cp) * (self.baseline.hf(a) * f - g) - cp)
                / f**2
            )

        return newton(eq, x0)


def get_cr_cp(
    **costs: CoercibleFloat64_1D,
) -> tuple[CoercibleFloat64_1D, CoercibleFloat64_1D]:
    """
    Extract replacement and preventive replacement costs.

    Raises
    ------
    TypeError
        If ``cr`` or ``cp`` is missing from ``costs``.
    """
    try:
        return costs["cr"], costs["cp"]
    except KeyError as err:
        raise TypeError("Missing cr and cp values") from err


def not_implemented_a0[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        a0 = bound.arguments.get("a0")
        if a0 is not None:
            msg = (
                f"{func.__qualname__}:"
                " NHPP policies with initial ages will be covered in a future release"
            )
            raise ValueError(msg)
        return func(*args, **kwargs)

    return wrapper


def reshape_ar[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    sig = inspect.signature(func)
    if "ar" not in sig.parameters:
        msg = f"{func.__qualname__} doit avoir un paramètre nommé 'ar'"
        raise TypeError(msg)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = sig.bind(*args, **kwargs)
        bound.arguments["ar"] = np.asarray(
            bound.arguments["ar"], dtype=np.float64
        ).reshape(-1, 1)
        return cast("Callable[..., R]", func)(*bound.args, **bound.kwargs)

    return wrapper


class RecurrentEventAgeReplacementPolicy(
    BaseAgeReplacementPolicy[NonHomogeneousPoissonProcess[()]]
):
    r"""Age replacement policy for recurrent events.

    Parameters
    ----------
    baseline : NonHomogeneousPoissonProcess
        Underlying non-homogeneous Poisson process.
    """

    @reshape_ar
    def _one_cycle_n_failures(
        self,
        ar: CoercibleFloat64_ND,
        discounting_rate: float = 0.0,
    ) -> Float64_ND:

        return np.asarray(
            legendre_quadrature(
                lambda t: (
                    exp_discounting_factor(t, rate=discounting_rate)
                    * self.baseline.intensity(t)
                ),
                0,
                ar,
            ),
            dtype=float,
        )

    @staticmethod
    @reshape_ar
    def _one_cycle_n_renewals(
        ar: CoercibleFloat64_1D,
        discounting_rate: float = 0.0,
    ) -> Float64_ND:

        return exp_discounting_factor(ar, rate=discounting_rate)

    @not_implemented_a0
    @reshape_ar
    def _n_failures(
        self,
        timeline: Timeline,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,  # ruff: ignore[unused-method-argument]  # pyright: ignore[reportUnusedParameter]
        discounting_rate: float = 0.0,
    ) -> Float64_ND:

        return np.asarray(
            legendre_quadrature(
                lambda t: (
                    exp_discounting_factor(t, rate=discounting_rate)
                    * self.baseline.intensity(t % ar)
                ),
                0,
                timeline,
            ),
            dtype=float,
        )

    @staticmethod
    @not_implemented_a0
    @reshape_ar
    def _n_renewals(
        timeline: Timeline,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,  # ruff: ignore[unused-static-method-argument]  # pyright: ignore[reportUnusedParameter]
        discounting_rate: float = 0.0,
    ) -> Float64_ND:

        geometric_ratio = (
            1 - exp_discounting_factor((timeline // ar) * ar, rate=discounting_rate)
        ) / (1 - exp_discounting_factor(ar, rate=discounting_rate))
        return (
            RecurrentEventAgeReplacementPolicy._one_cycle_n_renewals(
                ar, discounting_rate
            )
            * geometric_ratio
        )

    @override
    @not_implemented_a0
    @reshape_ar
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        cr, cp = get_cr_cp(**costs)
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)
        return timeline, cp * self._n_renewals(
            timeline, ar, a0, discounting_rate
        ) + cr * self._n_failures(timeline, ar, a0, discounting_rate)

    @override
    @reshape_ar
    def asymptotic_expected_net_present_value(
        self,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> Float64_1D:
        cr, cp = get_cr_cp(**costs)
        one_cycle_npv = cp * self._one_cycle_n_renewals(
            ar, discounting_rate
        ) + cr * self._one_cycle_n_failures(ar, discounting_rate)
        return (1 / (1 - exp_discounting_factor(ar, discounting_rate))) * one_cycle_npv

    @override
    @not_implemented_a0
    @reshape_ar
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        timeline, npv = self.expected_net_present_value(
            tf, nb_steps, ar=ar, a0=a0, discounting_rate=discounting_rate, **costs
        )
        return (
            discounting_rate / (1 - exp_discounting_factor(timeline, discounting_rate))
        ) * npv

    @override
    @not_implemented_a0
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> Float64_1D:
        cr, cp = get_cr_cp(**costs)
        if discounting_rate == 0:
            return (cp + cr * self._one_cycle_n_failures(ar)) / ar
        _, npv = self.asymptotic_expected_net_present_value(
            ar=ar, a0=a0, discounting_rate=discounting_rate, **costs
        )
        return discounting_rate * npv

    @override
    def compute_optimal_ar(
        self,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> float | onp.Array1D[np.float64]:
        """
        Compute the optimal ages of replacement.

        Parameters
        ----------
        discounting_rate : float, default=0.0
            The discounting rate used for cost computations.
        **costs : floats or 1d arrays
            Required costs ``cr`` and ``cp``.

        Returns
        -------
        ar : float or np.ndarray
            Optimal ages of replacement.
        """
        cr, cp = get_cr_cp(**costs)

        x0 = np.atleast_2d(self.baseline.lifetime_model.mean())

        def eq(a: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]:
            if discounting_rate != 0:
                return np.asarray(
                    (1 - exp_discounting_factor(a, rate=discounting_rate))
                    / discounting_rate
                    * self.baseline.intensity(a)
                    - legendre_quadrature(
                        lambda t: (
                            exp_discounting_factor(t, rate=discounting_rate)
                            * self.baseline.intensity(t)
                        ),
                        0,
                        a,
                    )
                    - np.float64(cp) / cr,
                    dtype=float,
                )
            return np.asarray(
                a * self.baseline.intensity(a)
                - self.baseline.cumulative_intensity(a)
                - np.float64(cp) / cr,
                dtype=float,
            )

        return newton(eq, x0)
