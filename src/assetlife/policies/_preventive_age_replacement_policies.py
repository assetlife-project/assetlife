"""Preventive age replacement policies."""

from __future__ import annotations

import textwrap
import warnings
from typing import ParamSpec, TypeVar

import numpy as np
import optype.numpy as onp
from scipy.optimize import newton
from typing_extensions import override

from assetlife._rewards import discounting_annuity_factor, discounting_factor
from assetlife.lifetime_models import ParametricLifetimeModel
from assetlife.quadratures import legendre_quadrature
from assetlife.stochastic_processes import (
    NonHomogeneousPoissonProcess,
    RenewalProcess,
    RenewalRewardProcess,
)
from assetlife.typing import CoercibleFloat64_1D, Float64_1D, Timeline

from ._base import BaseAgeReplacementPolicy, OneCycleExpectedCosts

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
    """Extract failure and preventive replacement costs."""
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
    ):
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
        _sf_x0 = self.baseline.sf(x0)
        if _sf_x0.ndim == 2 and x0.ndim == 0:
            x0 = np.tile(x0, (_sf_x0.shape[0], 1))

        def eq(a: onp.ArrayND[np.float64]) -> onp.ArrayND[np.float64]:
            return np.asarray(
                discounting_factor(a, rate=discounting_rate)
                / discounting_annuity_factor(a, rate=discounting_rate)
                * (
                    (cf - cp) * self.baseline.hf(a)
                    - cp / discounting_annuity_factor(a, rate=discounting_rate)
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
        _sf_x0 = self.baseline.sf(x0)
        if _sf_x0.ndim == 2 and x0.ndim == 0:
            x0 = np.tile(x0, (_sf_x0.shape[0], 1))

        def eq(
            a: onp.ArrayND[np.float64],
        ) -> onp.ArrayND[np.float64]:  # () or (m, 1)
            f = legendre_quadrature(
                lambda x: (
                    discounting_factor(x, rate=discounting_rate) * self.baseline.sf(x)
                ),
                0,
                a,
            )
            g = legendre_quadrature(
                lambda x: (
                    discounting_factor(x, rate=discounting_rate) * self.baseline.pdf(x)
                ),
                0,
                a,
            )
            return np.asarray(
                discounting_factor(a, rate=discounting_rate)
                * ((cf - cp) * (self.baseline.hf(a) * f - g) - cp)
                / f**2
            )

        return newton(eq, x0)


def get_cr_cp(
    **costs: CoercibleFloat64_1D,
) -> tuple[CoercibleFloat64_1D, CoercibleFloat64_1D]:
    """Extract replacement and preventive replacement costs."""
    try:
        return costs["cr"], costs["cp"]
    except KeyError as err:
        raise TypeError("Missing cr and cp values") from err


class NonHomogeneousPoissonAgeReplacementPolicy(
    BaseAgeReplacementPolicy[NonHomogeneousPoissonProcess[()]]
):
    r"""Age replacement policy for non-homogeneous Poisson processes.

    Parameters
    ----------
    baseline : NonHomogeneousPoissonProcess
        Underlying non-homogeneous Poisson process.
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
        raise NotImplementedError("implementation will come in a future release")

    @override
    def asymptotic_expected_net_present_value(
        self,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> Float64_1D:
        raise NotImplementedError("implementation will come in a future release")

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
        raise NotImplementedError("implementation will come in a future release")

    @override
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> Float64_1D:
        cr, cp = get_cr_cp(**costs)
        if a0 is not None:
            raise ValueError(
                "NHPP policies with initial ages will be covered in a future release"
            )

        if discounting_rate == 0.0:
            asymptotic_eeac = (
                cp
                + cr * legendre_quadrature(lambda t: self.baseline.intensity(t), 0, ar)
            ) / np.float64(ar)
        else:
            asymptotic_eeac = (
                discounting_rate
                * (
                    cp * discounting_factor(ar, rate=discounting_rate)
                    + cr
                    * legendre_quadrature(
                        lambda t: (
                            discounting_factor(t, rate=discounting_rate)
                            * self.baseline.intensity(t)
                        ),
                        0,
                        ar,
                    )
                )
                / (1 - discounting_factor(ar, rate=discounting_rate))
            )
        return np.squeeze(asymptotic_eeac)  # () or (m,)

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
                    (1 - discounting_factor(a, rate=discounting_rate))
                    / discounting_rate
                    * self.baseline.intensity(a)
                    - legendre_quadrature(
                        lambda t: (
                            discounting_factor(t, rate=discounting_rate)
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
