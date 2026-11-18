from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, final, override

import numpy as np

from assetlife._optimization import FitConfig, MaximumLikelihoodOptimizer
from assetlife.lifetime_models import (
    FittableParametricLifetimeModel,
)
from assetlife.typing import (
    CoercibleFloat64_ND,
)

if TYPE_CHECKING:
    import optype.numpy as onp
    import pandas as pd


class PartialObservations(NamedTuple):
    """Numbers of events observed per time window, with their covariates."""

    failures_count: onp.Array1D[np.float64]
    start: onp.Array1D[np.float64]
    end: onp.Array1D[np.float64]
    covars: tuple[onp.Array1D[np.float64], ...]


class NHPPData:
    """NHPP data container for NHPPLikelihood."""

    failures_time: onp.Array1D[np.float64]
    failures_covars: tuple[onp.Array1D[np.float64], ...]
    observations_start: onp.Array1D[np.float64]
    observations_end: onp.Array1D[np.float64]
    observations_covars: tuple[onp.Array1D[np.float64], ...]
    covariates: list[str]
    partial_observations: PartialObservations | None

    def __init__(
        self,
        failures: pd.DataFrame,
        assets: pd.DataFrame,
        covariates: list[str] | None = None,
        partial_observations: pd.DataFrame | None = None,
    ) -> None:
        if covariates is None:
            covariates = []
        self.covariates = covariates

        assets_covariates = assets[["id", *self.covariates]]

        # Get covariates for each failure time, using asset id as primary key
        failures_merged = failures.merge(assets_covariates, how="left", on="id")
        self.failures_time = failures_merged["time"].to_numpy(dtype=np.float64)
        self.failures_covars = tuple(
            failures_merged[covar].to_numpy(dtype=np.float64)
            for covar in self.covariates
        )

        # Same for observation periods
        self.observations_start = assets["start"].to_numpy(dtype=np.float64)
        self.observations_end = assets["end"].to_numpy(dtype=np.float64)
        self.observations_covars = tuple(
            assets[covar].to_numpy(dtype=np.float64) for covar in self.covariates
        )

        # If partial observations are given, merge covariates with each time period
        # and count
        if partial_observations is None:
            self.partial_observations = None
        else:
            partial_observations_merged = partial_observations.merge(
                assets_covariates, how="left", on="id"
            )
            self.partial_observations = PartialObservations(
                failures_count=partial_observations_merged["count"].to_numpy(
                    dtype=np.float64
                ),
                start=partial_observations_merged["start"].to_numpy(dtype=np.float64),
                end=partial_observations_merged["end"].to_numpy(dtype=np.float64),
                covars=tuple(
                    partial_observations_merged[covar].to_numpy(dtype=np.float64)
                    for covar in self.covariates
                ),
            )


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

    def _jac_observation_period_contrib(self) -> onp.ArrayND[np.float64]:
        jac = self.model.jac_chf(
            self.data.observations_end, *self.data.observations_covars
        ) - self.model.jac_chf(
            self.data.observations_start, *self.data.observations_covars
        )
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
        a = self.model.jac_chf(partial.end, *partial.covars) - self.model.jac_chf(
            partial.start, *partial.covars
        )
        b = self.model.chf(partial.end, *partial.covars) - self.model.chf(
            partial.start, *partial.covars
        )
        jac = -partial.failures_count * (a / b)
        return np.sum(jac, axis=1)
