"""Run-to-failure maintenance policies."""

from __future__ import annotations

from typing_extensions import override

import numpy as np
import optype.numpy as onp

from ._base import BaseRunToFailurePolicy, OneCycleExpectedCosts
from assetlife.lifetime_models._base import ParametricLifetimeModel
from assetlife.stochastic_processes._renewal_processes import RenewalRewardProcess
from assetlife.typing import CoercibleFloat64_1D, Float64_1D, Timeline


class OneCycleRunToFailurePolicy(BaseRunToFailurePolicy[ParametricLifetimeModel[()]]):
    r"""One-cycle run-to-failure policy.

    Asset is replaced upon failure with cost :math:`c_f`. Only one replacement
    cycle is considered.

    Parameters
    ----------
    lifetime_model : ParametricLifetimeModel
        Lifetime model representing durations between events.
    """

    period_before_discounting: float

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[()],
        period_before_discounting: float = 1.0,
    ) -> None:
        super().__init__(lifetime_model)
        self.period_before_discounting = period_before_discounting

    @property
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
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        return self._expected_costs.expected_net_present_value(
            tf, nb_steps, a0=a0, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        return self._expected_costs.expected_equivalent_annual_cost(
            tf, nb_steps, a0=a0, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def asymptotic_expected_net_present_value(
        self,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D:
        return self._expected_costs.asymptotic_expected_net_present_value(
            a0=a0, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D:
        return self._expected_costs.asymptotic_expected_equivalent_annual_cost(
            a0=a0, cf=cf, discounting_rate=discounting_rate
        )


class RunToFailurePolicy(BaseRunToFailurePolicy[ParametricLifetimeModel[()]]):
    r"""Run-to-failure renewal policy.

    Asset is replaced upon failure with cost :math:`c_f`.

    Parameters
    ----------
    lifetime_model : ParametricLifetimeModel
        Lifetime model representing durations between events.

    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in
        the Engineering and Informational Sciences, 22(1), 53-74.
    """

    @property
    def _stochastic_reward_process(self) -> RenewalRewardProcess:
        return RenewalRewardProcess(self.baseline)

    @override
    def expected_net_present_value(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        return self._stochastic_reward_process.expected_total_reward(
            tf, nb_steps, a0=a0, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def expected_equivalent_annual_cost(
        self,
        tf: float,
        nb_steps: int,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> tuple[Timeline, onp.Array1D[np.float64] | onp.Array2D[np.float64]]:
        return self._stochastic_reward_process.expected_equivalent_annual_worth(
            tf, nb_steps, a0=a0, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def asymptotic_expected_net_present_value(
        self,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D:
        return self._stochastic_reward_process.asymptotic_expected_total_reward(
            a0=a0, cf=cf, discounting_rate=discounting_rate
        )

    @override
    def asymptotic_expected_equivalent_annual_cost(
        self,
        *,
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
    ) -> Float64_1D:
        return (
            self._stochastic_reward_process.asymptotic_expected_equivalent_annual_worth(
                a0=a0, cf=cf, discounting_rate=discounting_rate
            )
        )
