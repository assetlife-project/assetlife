"""Renewal and renewal reward processes."""

from collections.abc import Callable
from typing import ParamSpec, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp

from assetlife._rewards import (
    ExponentialDiscounting,
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

FT: TypeAlias = Callable[[CoercibleFloat64_ND], Float64_ND]


class RenewalEquationSolver:
    """
    Renewal equation solver.
    """

    lifetime_model: ParametricLifetimeModel[()]
    first_lifetime_model: ParametricLifetimeModel[()] | None
    func: FT
    func1: FT | None

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        func: FT,
        first_lifetime_model: ParametricLifetimeModel[()] | None = None,
        func1: FT | None = None,
    ) -> None:
        self.lifetime_model = lifetime_model
        self.func = func
        if first_lifetime_model:
            assert func1 is not None
        self.first_lifetime_model = first_lifetime_model
        self.func1 = func1

    def solve(
        self, tf: float, nb_steps: int, discounting_rate: float = 0.0
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        """
        Solve the renewal equation on a finite timeline.

        Parameters
        ----------
        tf : float
            Final time.
        nb_steps : int
            Number of steps used to discretize the time.
        discounting_rate : float, default=0.0
            Exponential discounting rate.

        Returns
        -------
        out : tuple of np.ndarray
            Timeline and corresponding values.
        """

        discounting = ExponentialDiscounting(discounting_rate)
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (s,)
        tm = 0.5 * (timeline[1:] + timeline[:-1])  # (s-1,)
        f = np.asarray(
            self.lifetime_model.cdf(timeline.reshape(-1, 1))
        )  # (s,) or (s, m)
        fm = np.asarray(
            self.lifetime_model.cdf(tm.reshape(-1, 1))
        )  # (s-1,) or (s-1, m)
        y = np.asarray(self.func(timeline.reshape(-1, 1)))  # (s,) or (s, m)
        d = np.asarray(discounting.factor(timeline))  # (s,)
        z = np.empty(y.shape)  # (s,) or (s, m)
        u = d.reshape(-1, 1) * np.insert(f[1:] - fm, 0, 1, axis=0)
        v = d[:-1].reshape(-1, 1) * np.insert(np.diff(fm, axis=0), 0, 1, axis=0)
        q0 = 1 / (1 - d[0] * fm[0])
        z[0] = y[0]
        z[1] = q0 * (y[1] + z[0] * u[1])
        for n in range(2, len(f)):
            z[n] = q0 * (y[n] + z[0] * u[n] + np.sum(z[1:n][::-1] * v[1:n], axis=0))

        if self.first_lifetime_model is not None and self.func1 is not None:
            f1 = np.asarray(self.first_lifetime_model.cdf(timeline.reshape(-1, 1)))
            f1m = np.asarray(self.first_lifetime_model.cdf(tm.reshape(-1, 1)))
            y1 = np.asarray(self.func1(timeline.reshape(-1, 1)))
            z1 = np.empty(y1.shape)
            u1 = d.reshape(-1, 1) * np.insert(f1[1:] - f1m, 0, 1, axis=0)
            v1 = d[:-1].reshape(-1, 1) * np.insert(np.diff(f1m, axis=0), 0, 1, axis=0)
            z1[0] = y1[0]
            z1[1] = y1[1] + z[0] * u1[1] + z[1] * d[0] * f1m[0]
            for n in range(2, len(f1)):
                z1[n] = (
                    y1[n]
                    + z[0] * u1[n]
                    + z[n] * d[0] * f1m[0]
                    + np.sum(z[1:n][::-1] * v1[1:n], axis=0)
                )
            assert onp.is_array_2d(z1)
            return timeline, np.squeeze(z1)
        assert onp.is_array_2d(z)
        return timeline, np.squeeze(z)


R = TypeVar("R")
P = ParamSpec("P")


class RenewalProcess(ParametricModel):
    """
    Renewal process.

    Parameters
    ----------
    lifetime_model : ParametricLifetimeModel
        Lifetime model representing durations between events.
    first_lifetime_model : ParametricLifetimeModel, optional
        Lifetime model for the first renewal in a delayed renewal process.
        Defaults to ``lifetime_model``.
    """

    lifetime_model: ParametricLifetimeModel[()]
    first_lifetime_model: ParametricLifetimeModel[()]
    _different_first_lifetime_model: bool

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        first_lifetime_model: ParametricLifetimeModel[()] | None = None,
    ) -> None:
        super().__init__()
        self.lifetime_model = lifetime_model
        if first_lifetime_model is None:
            self._different_first_lifetime_model = False
            self.first_lifetime_model = self.lifetime_model
        else:
            self._different_first_lifetime_model = True
            self.first_lifetime_model = first_lifetime_model

    def renewal_function(
        self,
        tf: float,
        nb_steps: int,
        *,
        a0: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        r"""
        The renewal function.

        It gives the expected total number of renewals :math:`m`.
        It is computed  by solving the renewal equation:

        .. math::

            m(t) = F_1(t) + \int_0^t m(t-x) \mathrm{d}F(x)

        where:

        - :math:`F` is the cumulative distribution function of the time to failure :math:`X`.
        - :math:`F_1` is the cumulative distribution function of the first time to failure :math:`X_1`.

        If ``ar`` is given, :math:`F` becomes :math:`F_{a_r}` defined by :math:`T = \text{min}(X,~a_r) \sim F_{a_r}`.
        The same applies for :math:`X_1`. :math:`F_1` becomes :math:`F_{1_{a_r}}` defined by :math:`T_1 = \text{min}(X_1,~a_r) \sim F_{a_r}`.

        If ``a0`` is given, :math:`F_1` becomes :math:`\mathbb{P}(X \leq t |~ X > a_0)`.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretize the time.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        ar : float or 1d array, optional
            Preventive ages of replacements.

        Returns
        -------
        out : tuple of np.ndarray
            Timeline and corresponding values.

        References
        ----------
        .. [1] Rausand, M., Barros, A., & Hoyland, A. (2020). System Reliability
            Theory: Models, Statistical Methods, and Applications. John Wiley &
            Sons.
        """

        renewal_equation_solver = RenewalEquationSolver(
            self.lifetime_model.apply_condition(ar=ar),
            self.first_lifetime_model.apply_condition(ar=ar, a0=a0).cdf,
        )
        return renewal_equation_solver.solve(tf, nb_steps)

    def renewal_density(
        self,
        tf: float,
        nb_steps: int,
        *,
        a0: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        r"""The renewal density.

        It is the derivative :math:`\omega` of the renewal function :math:`m`.
        See the :py:meth:`~assetlife.stochastic_processes.RenewalProcess.renewal_function`.

        .. math::

            \omega(t) = m'(t) = f_1(t) + \int_0^t \omega(t-x) \mathrm{d}F(x)

        where:

        - :math:`F` is the cumulative distribution function of the time to failure :math:`X`.
        - :math:`f_1` is the probability density function of the first time to failure :math:`X_1`.

        If ``ar`` is given, :math:`F` becomes :math:`F_{a_r}` defined by :math:`T = \text{min}(X,~a_r) \sim F_{a_r}`.
        The same applies for :math:`X_1`. :math:`F_1` becomes :math:`F_{1_{a_r}}` defined by :math:`T_1 = \text{min}(X_1,~a_r) \sim F_{1_{a_r}}`.

        If ``a0`` is given, :math:`F_1` becomes :math:`\mathbb{P}(X \leq t |~ X > a_0)`.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretize the time.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        ar : float or 1d array, optional
            Preventive ages of replacements.

        Returns
        -------
        tuple of np.ndarray
            Timeline and corresponding values.

        References
        ----------
        .. [1] Rausand, M., Barros, A., & Hoyland, A. (2020). System Reliability
            Theory: Models, Statistical Methods, and Applications. John Wiley &
            Sons.
        """
        renewal_equation_solver = RenewalEquationSolver(
            self.lifetime_model.apply_condition(ar=ar),
            self.first_lifetime_model.apply_condition(ar=ar, a0=a0).pdf,
        )
        return renewal_equation_solver.solve(tf, nb_steps)

    def expected_number_of_events(
        self,
        tf: float,
        nb_steps: int,
        *,
        a0: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        r"""
        The expected number of observed events.

        Here, events are assets failures, i.e. only the assets failures are counted
        (not the preventive replacements at ``ar``).

        The function is noted :math:`m_e` and computed by solving :

        .. math::

            m_e(t) = F(\text{min}(t,~a_r)) + \int_0^{t}m_e(t-x)dF_{a_r}(x)

        where:

        - :math:`F` is the cumulative distribution function of the time to failure :math:`X`.
        - :math:`F_{a_r}` is the cumulative distribution of :math:`T = \text{min}(X,~a_r)`.

        If ``a0`` or ``first_lifetime_model`` is given, instead, we compute :math:`m_e^{\text{delayed}}` by solving:

        .. math::

            m_e^{\text{delayed}}(t) = F_1(\text{min}(t,~a_r)) + \int_0^{t}m_e(t-x)dF_{1_{a_r}}(x)

        where:

        - :math:`F_1` is the cumulative distribution function of the first time to failure :math:`X_1`.
        - :math:`F_{1_{a_r}}` is the cumulative distribution of :math:`T_1 = \text{min}(X_1,~a_r)`.

        .. note::

            If ``ar`` is ``None``, :math:`a_r = \infty`.

            This function is complementary to :py:meth:`~assetlife.stochastic_processes.RenewalProcess.expected_number_of_preventive_renewals`
            i.e. :math:`m(t) = m_e(t) + m_p(t)`.

            See also :py:meth:`~assetlife.stochastic_processes.RenewalProcess.renewal_function`.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretize the time.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        ar : float or 1d array, optional
            Preventive ages of replacements.

        Returns
        -------
        out : tuple of np.ndarray
            Timeline and corresponding values.

        Notes
        -----
        Preventive replacements are not considered as events. Only renewals are. Thus,
        they are not counted.
        """

        def F(t: CoercibleFloat64_ND) -> Float64_ND:
            _ar = np.float64(ar) if ar is not None else np.inf
            return self.lifetime_model.cdf(np.minimum(t, _ar))

        def F1(
            t: CoercibleFloat64_ND,
        ) -> Float64_ND:
            left_truncated_model = self.first_lifetime_model.apply_condition(a0=a0)
            _ar = np.float64(ar) if ar is not None else np.inf
            _a0 = np.float64(a0) if a0 is not None else 0.0
            return left_truncated_model.cdf(np.minimum(t, _ar - _a0))

        if self._different_first_lifetime_model or a0 is not None:
            renewal_equation_solver = RenewalEquationSolver(
                self.lifetime_model.apply_condition(ar=ar),
                F,
                self.first_lifetime_model.apply_condition(a0=a0, ar=ar),
                F1,
            )
        else:
            renewal_equation_solver = RenewalEquationSolver(
                self.lifetime_model.apply_condition(ar=ar),
                F,
            )

        return renewal_equation_solver.solve(tf, nb_steps)

    def expected_number_of_preventive_renewals(
        self,
        tf: float,
        nb_steps: int,
        *,
        ar: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        r"""
        The expected number of preventive renewals.

        The function is noted :math:`m_p` and computed by solving :

        .. math::

            m_p(t) = \mathbb{1}_{t > a_r} \cdot (1 - F(a_r)) + \int_0^{t}m_p(t-x)dF_{a_r}(x)

        where:

        - :math:`F` is the cumulative distribution function of the time to failure :math:`X`.
        - :math:`F_{a_r}` is the cumulative distribution of :math:`T = \text{min}(X,~a_r)`.

        If ``a0`` or ``first_lifetime_model`` is given, instead, we compute :math:`m_p^{\text{delayed}}` by solving:

        .. math::

            m_p^{\text{delayed}}(t) = \mathbb{1}_{t > a_r} \cdot (1 - F_1(a_r)) + \int_0^{t}m_p(t-x)dF_{1_{a_r}}(x)

        where:

        - :math:`F_1` is the cumulative distribution function of the first time to failure :math:`X_1`.
        - :math:`F_{1_{a_r}}` is the cumulative distribution of :math:`T_1 = \text{min}(X_1,~a_r)`.

        .. note::

            If ``ar`` is ``None``, :math:`a_r = \infty`.

            This function is complementary to :py:meth:`~assetlife.stochastic_processes.RenewalProcess.expected_number_of_events`
            i.e. :math:`m(t) = m_e(t) + m_p(t)`.

            See also :py:meth:`~assetlife.stochastic_processes.RenewalProcess.renewal_function`.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretize the time.
        ar : float or 1d array
            Preventive ages of replacements.
        a0 : float or 1d array, optional
            Initial ages of the assets.

        Returns
        -------
        out : tuple of np.ndarray
            Timeline and corresponding values.

        """

        def F(t: CoercibleFloat64_ND) -> Float64_ND:
            return (1 - self.lifetime_model.cdf(ar)) * (t > ar)

        def F1(t: CoercibleFloat64_ND) -> Float64_ND:
            _a0 = np.float64(a0) if a0 is not None else 0.0
            first_ar = ar - _a0
            return (
                1 - self.first_lifetime_model.apply_condition(a0=a0).cdf(first_ar)
            ) * (t > first_ar)

        if self._different_first_lifetime_model or a0 is not None:
            renewal_equation_solver = RenewalEquationSolver(
                self.lifetime_model.apply_condition(ar=ar),
                F,
                self.first_lifetime_model.apply_condition(a0=a0, ar=ar),
                F1,
            )
        else:
            renewal_equation_solver = RenewalEquationSolver(
                self.lifetime_model.apply_condition(ar=ar),
                F,
            )

        return renewal_equation_solver.solve(tf, nb_steps)


class RenewalRewardProcess(RenewalProcess):
    """
    Renewal reward process.

    Parameters
    ----------
    lifetime_model : ParametricLifetimeModel
        Lifetime model representing durations between events.
    first_lifetime_model : ParametricLifetimeModel, optional
        Lifetime model for the first renewal in a delayed renewal process.
        Defaults to ``lifetime_model``.
    """

    @overload
    def expected_total_reward(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        cf1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]: ...
    @overload
    def expected_total_reward(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D,
        ar: CoercibleFloat64_1D,
        cf1: CoercibleFloat64_1D | None = None,
        cp1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]: ...
    def expected_total_reward(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        cf1: CoercibleFloat64_1D | None = None,
        cp1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        r"""The expected total reward.

        The renewal equation solved to compute the expected reward is:

        .. math::

            z(t) = \int_0^t E[Y | X = x] e^{-\delta x} \mathrm{d}F(x) + \int_0^t z(t-x)
            e^{-\delta x}\mathrm{d}F(x)

        where:

        - :math:`z` is the expected total reward.
        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model.
        - :math:`X` the interarrival random variable.
        - :math:`Y` the associated reward.
        - :math:`D` the exponential discount factor.

        If the renewal reward process is delayed, the expected total reward is
        modified as:

        .. math::

            z_1(t) = \int_0^t E[Y_1 | X_1 = x] e^{-\delta x} \mathrm{d}F_1(x) + \int_0^t
            z(t-x) e^{-\delta x} \mathrm{d}F_1(x)

        where:

        - :math:`z_1` is the expected total reward with delay.
        - :math:`F_1` is the cumulative distribution function of the lifetime
          model for the first renewal.
        - :math:`X_1` the interarrival random variable of the first renewal.
        - :math:`Y_1` the associated reward of the first renewal.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretize the time.
        cf : float or 1d array
            The cost of failure.
        cp : float or 1d array, optional
            The cost of preventive replacement. Must be set with ar.
        ar : float or 1d array, optional
            Preventive ages of replacements. Must be set with cp.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        cf1 : float or 1d array, optional
            The cost of first failure. If not set, defaults to cf.
        cp1 : float or 1d array, optional
            The cost of the first preventive replacement. Must be set with ar.
            If not set, defaults to cp.
        discounting_rate : float, default is 0.
            The discounting rate to apply for reward computations.

        Returns
        -------
        tuple of np.ndarray
            Timeline and corresponding values.

        """

        if (cp is None) != (ar is None):
            raise TypeError("cp and ar must be set together.")
        if cp1 is not None and cp is None:
            raise TypeError("cp1 can only be set when cp and ar are set.")

        def F(t: CoercibleFloat64_ND) -> Float64_ND:
            return self.lifetime_model.apply_condition(ar=ar).ls_integrate(
                lambda x: (
                    compute_rewards(x, cf=cf, a0=a0, cp=cp, ar=ar)
                    * discounting_factor(x, discounting_rate)
                ),
                np.zeros_like(t),
                np.asarray(t),
                func_args=tuple(arg for arg in (cf, a0, cp, ar) if arg is not None),
                deg=15,
            )

        def F1(t: CoercibleFloat64_ND) -> Float64_ND:
            return self.first_lifetime_model.apply_condition(a0=a0, ar=ar).ls_integrate(
                lambda x: (
                    compute_rewards(
                        x,
                        cf=cf1 if cf1 is not None else cf,
                        a0=a0,
                        cp=cp1 if cp1 is not None else cp,
                        ar=ar,
                    )
                    * discounting_factor(x, discounting_rate)
                ),
                np.zeros_like(t),
                np.asarray(t),
                func_args=tuple(
                    arg
                    for arg in (
                        cf1 if cf1 is not None else cf,
                        a0,
                        cp1 if cp1 is not None else cp,
                        ar,
                    )
                    if arg is not None
                ),
                deg=15,
            )

        if self._different_first_lifetime_model or a0 is not None:
            renewal_equation_solver = RenewalEquationSolver(
                self.lifetime_model.apply_condition(ar=ar),
                F,
                self.first_lifetime_model.apply_condition(a0=a0, ar=ar),
                F1,
            )
        else:
            renewal_equation_solver = RenewalEquationSolver(
                self.lifetime_model.apply_condition(ar=ar),
                F,
            )

        return renewal_equation_solver.solve(
            tf, nb_steps, discounting_rate=discounting_rate
        )

    @overload
    def asymptotic_expected_total_reward(
        self,
        *,
        cf: CoercibleFloat64_1D,
        cf1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D: ...
    @overload
    def asymptotic_expected_total_reward(
        self,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D,
        ar: CoercibleFloat64_1D,
        cf1: CoercibleFloat64_1D | None = None,
        cp1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D: ...
    def asymptotic_expected_total_reward(
        self,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        cf1: CoercibleFloat64_1D | None = None,
        cp1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D:
        r"""Asymptotic expected total reward.

        The asymptotic expected total reward is:

        .. math::

            z^\infty = \lim_{t\to \infty} z(t) = \dfrac{E\left[Y e^{-\delta X}\right]}{1-E\left[e^{-\delta X}\right]}

        where:

        - :math:`X` the interarrival random variable.
        - :math:`Y` the associated reward.
        - :math:`D` the exponential discount factor.

        If the renewal reward process is delayed, the asymptotic expected total
        reward is modified as:

        .. math::

            z_1^\infty = E\left[Y_1 e^{-\delta X_1}\right] + z^\infty E\left[e^{-\delta X_1}\right]

        where:

        - :math:`X_1` the interarrival random variable of the first renewal.
        - :math:`Y_1` the associated reward of the first renewal.

        Parameters
        ----------
        cf : float or 1d array
            The cost of failure.
        cp : float or 1d array, optional
            The cost of preventive replacement. Must be set with ar.
        ar : float or 1d array, optional
            Preventive ages of replacements. Must be set with cp.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        cf1 : float or 1d array, optional
            The cost of first failure. If not set, defaults to cf.
        cp1 : float or 1d array, optional
            The cost of the first preventive replacement. Must be set with ar.
            If not set, defaults to cp.
        discounting_rate : float, default is 0.
            The discounting rate to apply for reward computations.

        Returns
        -------
        ndarray
            The asymptotic expected total reward of the process.
        """

        if (cp is None) != (ar is None):
            raise TypeError("cp and ar must be set together.")
        if cp1 is not None and cp is None:
            raise TypeError("cp1 can only be set when cp and ar are set.")

        lf = self.lifetime_model.apply_condition(ar=ar).ls_integrate(
            lambda x: discounting_factor(x, discounting_rate),
            0,
            np.inf,
            deg=100,
        )  # () or (m, 1)
        if discounting_rate == 0.0:
            return np.full_like(np.squeeze(lf), np.inf)
        ly = self.lifetime_model.apply_condition(ar=ar).ls_integrate(
            lambda x: (
                compute_rewards(x, cf=cf, a0=a0, cp=cp, ar=ar)
                * discounting_factor(x, discounting_rate)
            ),
            0,
            np.inf,
            func_args=tuple(arg for arg in (cf, a0, cp, ar) if arg is not None),
            deg=100,
        )  # () or (m, 1)
        z = np.squeeze(ly / (1 - lf))  # () or (m,)

        if self.first_lifetime_model:
            # Apply delay for the first renewal with a0
            # If no a0 are given, will result in the same solution
            lf1 = np.squeeze(
                self.first_lifetime_model.apply_condition(a0=a0, ar=ar).ls_integrate(
                    lambda x: discounting_factor(x, discounting_rate),
                    0.0,
                    np.inf,
                    deg=100,
                )
            )  # () or (m,)
            ly1 = np.squeeze(
                self.first_lifetime_model.apply_condition(a0=a0, ar=ar).ls_integrate(
                    lambda x: (
                        compute_rewards(
                            x,
                            cf=cf1 if cf1 is not None else cf,
                            a0=a0,
                            cp=cp1 if cp1 is not None else cp,
                            ar=ar,
                        )
                        * discounting_factor(x, discounting_rate)
                    ),
                    0.0,
                    np.inf,
                    func_args=tuple(
                        arg
                        for arg in (
                            cf1 if cf1 is not None else cf,
                            a0,
                            cp1 if cp1 is not None else cp,
                            ar,
                        )
                        if arg is not None
                    ),
                    deg=100,
                )
            )  # () or (m,)
            z = ly1 + z * lf1
        return z

    @overload
    def expected_equivalent_annual_worth(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        cf1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]: ...
    @overload
    def expected_equivalent_annual_worth(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D,
        ar: CoercibleFloat64_1D,
        cf1: CoercibleFloat64_1D | None = None,
        cp1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]: ...
    def expected_equivalent_annual_worth(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        cf1: CoercibleFloat64_1D | None = None,
        cp1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        """Expected equivalent annual worth.

        Gives the equivalent annual worth of the expected total reward of the
        process at each point of the timeline.

        The equivalent annual worth at time :math:`t` is equal to the expected
        total reward :math:`z` divided by the annuity factor :math:`AF(t)`.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretize the time.
        cf : float or 1d array
            The cost of failure.
        cp : float or 1d array, optional
            The cost of preventive replacement. Must be set with ar.
        ar : float or 1d array, optional
            Preventive ages of replacements. Must be set with cp.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        cf1 : float or 1d array, optional
            The cost of first failure. If not set, defaults to cf.
        cp1 : float or 1d array, optional
            The cost of the first preventive replacement. Must be set with ar.
            If not set, defaults to cp.
        discounting_rate : float, default is 0.
            The discounting rate to apply for reward computations.

        Returns
        -------
        tuple of np.ndarray
            Timeline and corresponding values.
        """
        if (cp is None) != (ar is None):
            raise TypeError("cp and ar must be set together.")
        if cp1 is not None and cp is None:
            raise TypeError("cp1 can only be set when cp and ar are set.")

        if cp is not None and ar is not None:
            timeline, z = self.expected_total_reward(
                tf,
                nb_steps,
                cf=cf,
                a0=a0,
                cp=cp,
                ar=ar,
                cf1=cf1,
                cp1=cp1,
                discounting_rate=discounting_rate,
            )
        else:
            timeline, z = self.expected_total_reward(
                tf,
                nb_steps,
                cf=cf,
                a0=a0,
                cf1=cf1,
                discounting_rate=discounting_rate,
            )
        af = discounting_annuity_factor(timeline, discounting_rate)  # (nb_steps,)
        if z.ndim == 2:
            af = af.reshape(-1, 1)  # (nb_steps, 1)
        q0 = compute_rewards(
            0.0, cf=cf, a0=a0, cp=cp, ar=ar
        ) * self.lifetime_model.apply_condition(a0=a0).pdf(0.0)
        # () or (m,)
        q = z / (af + 1e-6)  # # (nb_steps,) or (nb_steps, m) avoid zero division
        eeac = np.where(af == 0, q0, q)  # (nb_steps,) or (m, nb_steps)
        return timeline, eeac

    @overload
    def asymptotic_expected_equivalent_annual_worth(
        self,
        *,
        cf: CoercibleFloat64_1D,
        cf1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D: ...
    @overload
    def asymptotic_expected_equivalent_annual_worth(
        self,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D,
        ar: CoercibleFloat64_1D,
        cf1: CoercibleFloat64_1D | None = None,
        cp1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D: ...
    def asymptotic_expected_equivalent_annual_worth(
        self,
        *,
        cf: CoercibleFloat64_1D,
        cp: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        cf1: CoercibleFloat64_1D | None = None,
        cp1: CoercibleFloat64_1D | None = None,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D:
        """Asymptotic expected equivalent annual worth.

        Parameters
        ----------
        cf : float or 1d array
            The cost of failure.
        cp : float or 1d array, optional
            The cost of preventive replacement. Must be set with ar.
        ar : float or 1d array, optional
            Preventive ages of replacements. Must be set with cp.
        a0 : float or 1d array, optional
            Initial ages of the assets.
        cf1 : float or 1d array, optional
            The cost of first failure. If not set, defaults to cf.
        cp1 : float or 1d array, optional
            The cost of the first preventive replacement. Must be set with ar.
            If not set, defaults to cp.
        discounting_rate : float, default is 0.
            The discounting rate to apply for reward computations.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual worth.
        """
        if (cp is None) != (ar is None):
            raise TypeError("cp and ar must be set together.")
        if cp1 is not None and cp is None:
            raise TypeError("cp1 can only be set when cp and ar are set.")

        if discounting_rate == 0.0:
            ls = self.lifetime_model.apply_condition(ar=ar).ls_integrate(
                lambda x: compute_rewards(x, cf=cf, a0=a0, cp=cp, ar=ar),
                0.0,
                np.inf,
                func_args=tuple(arg for arg in (cf, a0, cp, ar) if arg is not None),
                deg=100,
            )
            mean = self.lifetime_model.apply_condition(ar=ar).mean()
            return ls / mean
        if cp is not None and ar is not None:
            res = discounting_rate * self.asymptotic_expected_total_reward(
                cf=cf,
                a0=a0,
                cp=cp,
                ar=ar,
                cf1=cf1,
                cp1=cp1,
                discounting_rate=discounting_rate,
            )
        else:
            res = discounting_rate * self.asymptotic_expected_total_reward(
                cf=cf,
                a0=a0,
                cf1=cf1,
                discounting_rate=discounting_rate,
            )
        assert onp.is_array_1d(res) or isinstance(res, np.float64)  # typeguard
        return res
