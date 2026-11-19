from __future__ import annotations

from typing import TYPE_CHECKING, final, override

import numpy as np

from ._datasets import NHPPData
from assetlife._optimization import FitConfig, MaximumLikelihoodOptimizer
from assetlife.lifetime_models import (
    FittableParametricLifetimeModel,
)
from assetlife.typing import (
    CoercibleFloat64_ND,
)

if TYPE_CHECKING:
    import optype.numpy as onp


@final
class NHPPLikelihood(
    MaximumLikelihoodOptimizer[
        FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]], NHPPData
    ]
):
    def __init__(
        self,
        model: FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]],
        data: NHPPData,
        config: FitConfig,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log

    @property
    @override
    def nb_observations(self) -> int:
        n = self.data.failures_time.size
        if self.data.partial_observations is not None:
            n += self.data.partial_observations.failures_count.size
        return n

    @override
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        self.model.set_params(params)
        return (
            self._exact_events_contrib()
            + self._observation_period_contrib()
            + self._partial_observation_contrib()
        )

    def jac_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array1D[np.float64]:
        self.model.set_params(params)
        return (
            self._jac_exact_events_contrib()
            + self._jac_observation_period_contrib()
            + self._jac_partial_observation_contrib()
        )

    def _exact_events_contrib(self) -> float:
        return -np.sum(
            np.log(self.model.hf(self.data.failures_time, *self.data.failures_covars))
        )

    def _jac_exact_events_contrib(self) -> onp.ArrayND[np.float64]:
        jac = -self.model.jac_hf(
            self.data.failures_time, *self.data.failures_covars
        ) / self.model.hf(self.data.failures_time, *self.data.failures_covars)
        return np.sum(jac, axis=1)

    def _observation_period_contrib(self) -> float:
        return np.sum(
            self.model.chf(self.data.observations_end, *self.data.observations_covars)
            - self.model.chf(
                self.data.observations_start, *self.data.observations_covars
            )
        )

    def _jac_chf(
        self,
        time: onp.Array1D[np.float64],
        covars: tuple[onp.Array1D[np.float64], ...],
    ) -> onp.Array2D[np.float64]:
        # chf(0) = 0 whatever the parameters, so its jacobian is 0. It is not
        # evaluated at 0, where some models give nan (e.g. log(0) * 0)
        positive = time > 0
        if positive.all():
            return self.model.jac_chf(time, *covars)
        nb_params = self.model.get_params().size
        if not positive.any():
            return np.zeros((nb_params, time.size), dtype=np.float64)
        jac_positive = self.model.jac_chf(
            time[positive], *(covar[positive] for covar in covars)
        )
        # dtype of jac_positive may be complex (complex-step covariance)
        jac = np.zeros((nb_params, time.size), dtype=jac_positive.dtype)
        jac[:, positive] = jac_positive
        return jac

    def _jac_observation_period_contrib(self) -> onp.ArrayND[np.float64]:
        jac = self._jac_chf(
            self.data.observations_end, self.data.observations_covars
        ) - self._jac_chf(self.data.observations_start, self.data.observations_covars)
        return np.sum(jac, axis=1)

    def _partial_observation_contrib(self) -> float:
        partial = self.data.partial_observations
        if partial is None:
            return 0.0
        return np.sum(
            -partial.failures_count
            * np.log(
                self.model.chf(partial.end, *partial.covars)
                - self.model.chf(partial.start, *partial.covars)
            )
        )

    def _jac_partial_observation_contrib(self) -> onp.ArrayND[np.float64]:
        partial = self.data.partial_observations
        if partial is None:
            return np.zeros_like(self.model.get_params(), dtype=np.float64)
        a = self._jac_chf(partial.end, partial.covars) - self._jac_chf(
            partial.start, partial.covars
        )
        b = self.model.chf(partial.end, *partial.covars) - self.model.chf(
            partial.start, *partial.covars
        )
        jac = -partial.failures_count * (a / b)
        return np.sum(jac, axis=1)
