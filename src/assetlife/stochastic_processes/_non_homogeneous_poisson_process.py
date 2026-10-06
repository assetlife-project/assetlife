"""Non-homogeneous Poisson process models."""

from __future__ import annotations

from typing import Any, Generic, Self
from typing_extensions import override

import numpy as np
import optype.numpy as onp
import pandas as pd

from assetlife.base import (
    FitConfig,
    FittingResults,
    MaximumLikelihoodOptimizer,
    ParametricModel,
)
from assetlife.lifetime_models import (
    FittableParametricLifetimeModel,
    ParametricLifetimeModel,
    init_distrib_params_from_lifetimes,
    init_regression_params_from_lifetimes,
)
from assetlife.lifetime_models._distributions import (
    Gamma,
    LifetimeDistribution,
    get_distrib_params_bounds,
)
from assetlife.lifetime_models._parametric_regressions import (
    ParametricLifetimeRegression,
    get_regression_params_bounds,
)
from assetlife.typing import (
    CoercibleFloat64_ND,
    CovarTs,
    Float64_ND,
)


class NHPPData:
    failures_time: onp.Array1D[np.float64]
    failures_covars: tuple[Any, ...]  # TODO: fix
    observations_start: onp.Array1D[np.float64]
    observations_end: onp.Array1D[np.float64]
    observations_covars: tuple[Any, ...]  # TODO: fix
    covariates: list[str]
    has_partial: bool
    partial_observations_count = onp.Array1D[np.int64] | None
    partial_observations_start = onp.Array1D[np.float64] | None
    partial_observations_end = onp.Array1D[np.float64] | None
    partial_observations_covars = tuple[Any, ...] | None  # TODO: fix

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

        failures_merged = failures.merge(assets_covariates, how="left", on="id")
        self.failures_time = failures_merged["time"].to_numpy(dtype=np.float64)
        self.failures_covars = tuple(
            failures_merged[covar] for covar in self.covariates
        )

        self.observations_start = assets["start"].to_numpy(dtype=np.float64)
        self.observations_end = assets["end"].to_numpy(dtype=np.float64)
        self.observations_covars = tuple(
            assets[covar].to_numpy() for covar in self.covariates
        )

        if partial_observations is None:
            self.has_partial = False
            self.partial_observations_count = None
            self.partial_observations_start = None
            self.partial_observations_end = None
            self.partial_observations_covars = None
        else:
            self.has_partial = True
            partial_observations_merged = partial_observations.merge(
                assets_covariates, how="left", on="id"
            )
            self.partial_observations_count = partial_observations_merged[
                "count"
            ].to_numpy(dtype=np.int64)
            self.partial_observations_start = partial_observations_merged[
                "start"
            ].to_numpy(dtype=np.float64)
            self.partial_observations_end = partial_observations_merged["end"].to_numpy(
                dtype=np.float64
            )
            self.partial_observations_covars = tuple(
                partial_observations_merged[covar].to_numpy()
                for covar in self.covariates
            )


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
    def nb_observations(self) -> int:
        n = self.data.failures_time.size
        if self.data.has_partial:
            n += self.data.partial_observations_count.size  # TODO: typing for None case
        return n

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
        if not self.data.has_partial:
            return 0.0

        # TODO : typing for None case
        return np.sum(
            -self.data.partial_observations_count
            * np.log(
                self.model.chf(
                    self.data.partial_observations_end,
                    *self.data.partial_observations_covars,
                )
                - self.model.chf(
                    self.data.partial_observations_start,
                    *self.data.partial_observations_covars,
                )
            )
        )

    def _jac_partial_observation_contrib(self) -> onp.ArrayND[np.float64]:
        if not self.data.has_partial:
            return np.zeros_like(self.model.get_params(), dtype=np.float64)

        # TODO : typing for None case
        a = self.model.jac_chf(
            self.data.partial_observations_end,
            *self.data.partial_observations_covars,
        ) - self.model.jac_chf(
            self.data.partial_observations_start, *self.data.partial_observations_covars
        )
        b = self.model.chf(
            self.data.partial_observations_end, *self.data.partial_observations_covars
        ) - self.model.chf(
            self.data.partial_observations_start, *self.data.partial_observations_covars
        )
        jac = -self.data.partial_observations_count * (a / b)
        return np.sum(jac, axis=1)


def init_nhpp_likelihood(
    model: FittableParametricLifetimeModel,
    failures: pd.DataFrame,
    assets: pd.DataFrame,
    covariates: list[str] | None = None,
    partial_observations: pd.DataFrame | None = None,
    **kwargs: Any,
) -> NHPPLikelihood:
    data = NHPPData(failures, assets, covariates, partial_observations)

    if isinstance(model, LifetimeDistribution):
        fresh_model = type(model)()
        if (covariates is not None) and len(covariates) > 0:
            msg = "No covariates can be given for fit when using a distribution."
            raise ValueError(msg)
        x0 = kwargs.get(
            "x0", init_distrib_params_from_lifetimes(fresh_model, data.failures_time)
        )
        config = FitConfig(x0)
        config.scipy_minimize_options["bounds"] = kwargs.get(
            "bounds", get_distrib_params_bounds(fresh_model)
        )
        config.covariance_method = kwargs.get(
            "covariance_method", "2point" if isinstance(fresh_model, Gamma) else "cs"
        )
    elif isinstance(model, ParametricLifetimeRegression):
        if (covariates is None) or len(covariates) == 0:
            msg = "Covariates must be given for fit when using a regression."
            raise ValueError(msg)
        fresh_model = type(model)(
            type(model.baseline)(), coefficients=(0.0,) * len(covariates)
        )
        x0 = kwargs.get(
            "x0", init_regression_params_from_lifetimes(fresh_model, data.failures_time)
        )
        config = FitConfig(x0)
        config.scipy_minimize_options["bounds"] = kwargs.get(
            "bounds", get_regression_params_bounds(fresh_model)
        )
        config.covariance_method = kwargs.get(
            "covariance_method",
            "2point" if isinstance(fresh_model.baseline, Gamma) else "cs",
        )
    else:
        msg = f"Cannot initiate NHPP likelihood with the model {model}, expected Parametric Distribution or Regression."
        raise TypeError(msg)

    config.scipy_minimize_options["method"] = kwargs.get("method", "L-BFGS-B")
    return NHPPLikelihood(fresh_model, data, config)


class NonHomogeneousPoissonProcess(ParametricModel, Generic[*CovarTs]):
    fitting_results: FittingResults | None
    lifetime_model: ParametricLifetimeModel[*CovarTs]  # not accurate is case of fit

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[*CovarTs],
    ) -> None:
        super().__init__()
        self.lifetime_model = lifetime_model

    def intensity(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        """
        The intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.lifetime_model.hf(time, *args)

    def cumulative_intensity(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        """
        The cumulative intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.lifetime_model.chf(time, *args)

    def fit(
        self,
        failures: pd.DataFrame,
        assets: pd.DataFrame,
        covariates: list[str] | None = None,
        partial_observations: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> Self:
        optimizer = init_nhpp_likelihood(
            self.lifetime_model,
            failures,
            assets,
            covariates,
            partial_observations,
            **kwargs,
        )  # TODO: typing for non-fittable case
        fitting_results = optimizer.optimize()
        if isinstance(self.lifetime_model, ParametricLifetimeRegression):
            if (covariates is None) or len(covariates) == 0:
                msg = "Covariates must be given for fit when using a regression."
                raise ValueError(msg)
            self.lifetime_model.covar_effect.set_params(
                [0.0] * len(covariates)
            )  # modify nb coef inplace
        self.set_params(fitting_results.optimal_params)
        self.fitting_results = fitting_results
        return self


class FrozenNonHomogeneousPoissonProcess(
    NonHomogeneousPoissonProcess[()], Generic[*CovarTs]
):
    """Non-homogeneous Poisson process with additional arguments stored."""

    unfrozen: NonHomogeneousPoissonProcess[*CovarTs]
    args: tuple[*CovarTs]

    def __init__(
        self,
        nhpp: NonHomogeneousPoissonProcess[*CovarTs],
        *args: *CovarTs,
    ) -> None:
        super().__init__(nhpp.lifetime_model.freeze(*args))
        self.unfrozen = nhpp
        self.args = args

    @override
    def intensity(self, time: CoercibleFloat64_ND) -> Float64_ND:
        """
        The intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.lifetime_model.hf(time)

    @override
    def cumulative_intensity(self, time: CoercibleFloat64_ND) -> Float64_ND:
        """
        The cumulative intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.lifetime_model.chf(time)
