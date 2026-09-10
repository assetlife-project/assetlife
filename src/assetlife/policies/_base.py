"""Base classes and cost helpers for maintenance policies."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, overload

import numpy as np
import optype.numpy as onp

from assetlife._rewards import (
    compute_rewards,
    discounting_annuity_factor,
    discounting_factor,
)
from assetlife.base import ParametricModel
from assetlife.lifetime_models import ParametricLifetimeModel
from assetlife.typing import (
    CoercibleFloat64_1D,
    CoercibleFloat64_ND,
    Float64_1D,
    Float64_ND,
    Timeline,
)


class OneCycleExpectedCosts:
    """Expected cost computations for one-cycle policies."""

    lifetime_model: ParametricLifetimeModel[()]
    period_before_discounting: float

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        period_before_discounting: float = 1.0,
    ) -> None:
        if period_before_discounting <= 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting
        self.lifetime_model = lifetime_model

    @overload
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]: ...
    @overload
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]: ...
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        """Compute the expected net present value on a finite timeline."""
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)
        etc = np.asarray(
            self.lifetime_model.apply_condition(a0=a0, ar=ar).ls_integrate(
                lambda x: (
                    compute_rewards(x, a0=a0, cp=cp, cf=cf)
                    * discounting_factor(x, discounting_rate)
                ),
                np.zeros_like(timeline),
                timeline,
                func_args=tuple(arg for arg in (cf, a0, cp, ar) if arg is not None),
                deg=15,
            ),
            dtype=float,
        )  # (nb_steps,) or (m, nb_steps)
        return timeline, etc  # (nb_steps,) and (nb_steps,)/(m, nb_steps)

    @overload
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]: ...
    @overload
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]: ...
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        """Compute the expected equivalent annual cost on a finite timeline."""
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)
        value = self._expected_equivalent_annual_cost(
            timeline, ar=ar, a0=a0, cf=cf, cp=cp, discounting_rate=discounting_rate
        )
        assert onp.is_array_1d(value) or onp.is_array_2d(value)
        return timeline, value  # (nb_steps,) or (m, nb_steps)

    @overload
    def asymptotic_expected_net_present_value(
        self,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D: ...
    @overload
    def asymptotic_expected_net_present_value(
        self,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D: ...
    def asymptotic_expected_net_present_value(
        self,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D:
        """Compute the asymptotic expected net present value."""
        # reward partial expectation
        return np.squeeze(
            self.lifetime_model.apply_condition(a0=a0, ar=ar).ls_integrate(
                lambda x: (
                    compute_rewards(x, a0=a0, cf=cf, cp=cp, ar=ar)
                    * discounting_factor(x, rate=discounting_rate)
                ),
                0.0,
                np.inf,
                func_args=tuple(arg for arg in (cf, a0, cp, ar) if arg is not None),
            )
        )

    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D: ...
    @overload
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D: ...
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D:
        """Compute the asymptotic expected equivalent annual cost."""
        value = self._expected_equivalent_annual_cost(
            np.array(np.inf),
            a0=a0,
            ar=ar,
            cp=cp,
            cf=cf,
            discounting_rate=discounting_rate,
        )
        assert onp.is_array_1d(value) or isinstance(value, np.float64)  # typeguard
        return value

    def _expected_equivalent_annual_cost(
        self,
        timeline: onp.ArrayND[np.float64],
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D | onp.Array2D[np.float64]:
        def f(x: CoercibleFloat64_ND) -> Float64_ND:
            # avoid zero division + 1e-6
            return (
                compute_rewards(x, a0=a0, cf=cf, cp=cp, ar=ar)
                * discounting_factor(x, rate=discounting_rate)
                / (discounting_annuity_factor(x, rate=discounting_rate) + 1e-6)
            )

        if timeline.ndim == 1:
            timeline = timeline.reshape(-1, 1)
        conditional_model = self.lifetime_model.apply_condition(a0=a0, ar=ar)
        q0 = conditional_model.cdf(self.period_before_discounting) * f(
            np.asarray(self.period_before_discounting, dtype=float)
        )  # () or (m, 1)
        a = np.full_like(
            timeline, self.period_before_discounting
        )  # (nb_steps,) or (m, nb_steps)

        # change first value of lower bound to compute the integral
        a[timeline < self.period_before_discounting] = 0.0  # (nb_steps,)
        # a = np.where(timeline < self.period_before_discounting, 0., a)  # (nb_steps,)
        integral = conditional_model.ls_integrate(
            f,
            a,
            timeline,
            func_args=tuple(arg for arg in (cf, a0, cp, ar) if arg is not None),
            deg=100,
        )  # (nb_steps,) or (m, nb_steps) if q0: (), or (m, nb_steps) if q0 : (m, 1)
        mask = np.broadcast_to(
            timeline < self.period_before_discounting, integral.shape
        )  # (), (nb_steps,) or (m, nb_steps)
        q0 = np.broadcast_to(q0, integral.shape)  # (nb_steps,) or (m, nb_steps)
        integral = np.squeeze(np.where(mask, q0, q0 + integral))
        if integral.ndim == 0:
            return np.float64(integral)
        return integral


M = TypeVar("M", bound=ParametricModel)


class BaseRunToFailurePolicy(ABC, Generic[M]):
    """Base class for run-to-failure policies."""

    baseline: M

    def __init__(
        self,
        baseline: M,
    ):
        self.baseline = baseline

    @abstractmethod
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        r"""
        The expected net present value.

        .. math::

            z(t) = \mathbb{E}(Z_t) = \int_{0}^{\infty}\mathbb{E}(Z_t~|~X_1 = x)dF(x)

        where :

        - :math:`t` is the time
        - :math:`X_1 \sim F` is the random lifetime of the first asset
        - :math:`Z_t` are the random costs at each time :math:`t`
        - :math:`\delta` is the discounting rate

        It is computed by solving the renewal equation.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretize the time.
        cf : float or 1d array
            The cost of failure.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        discounting_rate : float, default is 0.
            The discounting rate used for cost computations.

        Returns
        -------
        out : tuple of np.ndarray
            Timeline and corresponding values.
        """

    @abstractmethod
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        r"""
        The expected equivalent annual cost.

        .. math::

            q(t) = \dfrac{\delta z(t)}{1 - e^{-\delta t}}

        where :

        - :math:`t` is the time.
        - :math:`z(t)` is the expected net present value at time :math:`t`.
        - :math:`\delta` is the discounting rate.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretize the time.
        cf : float or 1d array
            The cost of failure.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        discounting_rate : float, default is 0.
            The discounting rate used for cost computations.

        Returns
        -------
        out : tuple of np.ndarray
            Timeline and corresponding values.
        """

    @abstractmethod
    def asymptotic_expected_net_present_value(
        self,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D:
        r"""
        The asymptotic expected net present value.

        .. math::

            \lim_{t\to\infty} z(t)

        Parameters
        ----------
        cf : float or 1d array
            The cost of failure.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        discounting_rate : float, default is 0.
            The discounting rate used for cost computations.

        Returns
        -------
        ndarray
            The asymptotic expected values.
        """

    @abstractmethod
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D:
        r"""
        The asymptotic expected equivalent annual cost.

        .. math::

            \lim_{t\to\infty} q(t)

        Parameters
        ----------
        cf : float or 1d array
            The cost of failure.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        discounting_rate : float, default is 0.
            The discounting rate used for cost computations.

        Returns
        -------
        ndarray
            The asymptotic expected values.
        """


class BaseAgeReplacementPolicy(ABC, Generic[M]):
    """Base class for age replacement policies."""

    baseline: M

    def __init__(self, baseline: M):
        self.baseline = baseline

    @abstractmethod
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
        r"""
        The expected net present value.

        .. math::

            z(t) = \mathbb{E}(Z_t) = \int_{0}^{\infty}\mathbb{E}(Z_t~|~X_1 = x)dF(x)

        where :

        - :math:`t` is the time
        - :math:`X_1 \sim F` is the random lifetime of the first asset
        - :math:`Z_t` are the random costs at each time :math:`t`
        - :math:`\delta` is the discounting rate

        It is computed by solving the renewal equation.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretize the time.
        ar : float or 1d array
            Preventive ages of replacement.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        discounting_rate : float, default is 0.
            The discounting rate used for cost computations.
        **costs : floats or 1d arrays
            Required costs, such as ``cp``, ``cf`` and/or ``cr``.

        Returns
        -------
        out : tuple of np.ndarray
            Timeline and corresponding values.
        """

    @abstractmethod
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
        r"""
        The expected equivalent annual cost.

        .. math::

            q(t) = \dfrac{\delta z(t)}{1 - e^{-\delta t}}

        where :

        - :math:`t` is the time.
        - :math:`z(t)` is the expected net present value at time :math:`t`.
        - :math:`\delta` is the discounting rate.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretize the time.
        ar : float or 1d array
            Preventive ages of replacement.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        discounting_rate : float, default is 0.
            The discounting rate used for cost computations.
        **costs : floats or 1d arrays
            Required costs, such as ``cp``, ``cf`` and/or ``cr``.

        Returns
        -------
        out : tuple of np.ndarray
            Timeline and corresponding values.
        """

    @abstractmethod
    def asymptotic_expected_net_present_value(
        self,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> Float64_1D:
        r"""
        The asymptotic expected net present value.

        .. math::

            \lim_{t\to\infty} z(t)

        Parameters
        ----------
        ar : float or 1d array
            Preventive ages of replacement.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        discounting_rate : float, default is 0.
            The discounting rate used for cost computations.
        **costs : floats or 1d arrays
            Required costs, such as ``cp``, ``cf`` and/or ``cr``.

        Returns
        -------
        ndarray
            The asymptotic expected values.
        """

    @abstractmethod
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> Float64_1D:
        r"""
        The asymptotic expected equivalent annual cost.

        .. math::

            \lim_{t\to\infty} q(t)

        Parameters
        ----------
        ar : float or 1d array
            Preventive ages of replacement.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        discounting_rate : float, default is 0.
            The discounting rate used for cost computations.
        **costs : floats or 1d arrays
            Required costs, such as ``cp``, ``cf`` and/or ``cr``.

        Returns
        -------
        ndarray
            The asymptotic expected values.
        """

    @abstractmethod
    def compute_optimal_ar(
        self,
        discounting_rate: float = 0.0,
        **costs: CoercibleFloat64_1D,
    ) -> float | onp.Array1D[np.float64]:
        """
        Compute the optimal ages of replacement.

        Parameters
        ----------
        discounting_rate : float, default is 0.
            The discounting rate used for cost computations.
        **costs : floats or 1d arrays
            Required costs, such as ``cp``, ``cf`` and/or ``cr``.

        Returns
        -------
        out : float or 1d array
            Optimal ages of replacement.
        """
