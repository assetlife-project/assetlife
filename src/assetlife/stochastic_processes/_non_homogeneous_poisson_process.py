"""Non-homogeneous Poisson process models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, override

from ._datasets import NHPPData
from ._likelihoods import NHPPLikelihood
from assetlife._optimization import FitConfig
from assetlife.base import FittingResults, ParametricModel
from assetlife.lifetime_models._distributions import (
    Gamma,
    LifetimeDistribution,
)
from assetlife.lifetime_models._parametric_regressions import (
    ParametricLifetimeRegression,
)

if TYPE_CHECKING:
    import pandas as pd

    from assetlife.lifetime_models import (
        ParametricLifetimeModel,
    )
    from assetlife.typing import (
        CoercibleFloat64_ND,
        CovarTs,
        Float64_ND,
    )


class NonHomogeneousPoissonProcess[*CovarTs](ParametricModel):
    """Non Homogeneous Poisson Process (NHPP)

    Parameters
    ----------
    lifetime_model : ParametricLifetimeModel
        Lifetime model that captures the NHPP intensity (lambda function) as its
        hazard function.
    fitting_results: FittingResults | None
        Fitting Results after fitting the NHPP with failure data.
    """

    lifetime_model: ParametricLifetimeModel[*CovarTs]  # not accurate is case of fit
    fitting_results: FittingResults | None

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

    def init_likelihood(
        self,
        failures: pd.DataFrame,
        assets: pd.DataFrame,
        covariates: list[str] | None = None,
        partial_observations: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> NHPPLikelihood:
        data = NHPPData(failures, assets, covariates, partial_observations)

        if isinstance(self.lifetime_model, LifetimeDistribution):
            if (covariates is not None) and len(covariates) > 0:
                msg = "No covariates can be given for fit when using a distribution."
                raise ValueError(msg)
            fresh_model = type(self.lifetime_model)()
            covariance_method = kwargs.get(
                "covariance_method",
                "2point" if isinstance(fresh_model, Gamma) else "cs",
            )
        elif isinstance(self.lifetime_model, ParametricLifetimeRegression):
            if (covariates is None) or len(covariates) == 0:
                msg = "Covariates must be given for fit when using a regression."
                raise ValueError(msg)
            fresh_model = type(self.lifetime_model)(
                type(self.lifetime_model.baseline)(),
                coefficients=(0.0,) * len(covariates),
            )
            covariance_method = kwargs.get(
                "covariance_method",
                "2point" if isinstance(fresh_model.baseline, Gamma) else "cs",
            )
        else:
            msg = (
                "Cannot initiate NHPP likelihood with the model "
                f"{self.lifetime_model}, expected Parametric Distribution or "
                "Regression."
            )
            raise TypeError(msg)

        x0 = kwargs.get("x0", fresh_model._init_params(data.failures_time))  # ruff: ignore[private-member-access]
        config = FitConfig(x0)
        config.scipy_minimize_options["bounds"] = kwargs.get(
            "bounds", fresh_model.get_params_bounds()
        )
        config.covariance_method = covariance_method
        config.scipy_minimize_options["method"] = kwargs.get("method", "L-BFGS-B")
        return NHPPLikelihood(fresh_model, data, config)

    def fit(
        self,
        failures: pd.DataFrame,
        assets: pd.DataFrame,
        covariates: list[str] | None = None,
        partial_observations: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the NHPP parametric model with failure observation history.

        Parameters
        ----------
        failures : pd.DataFrame
            Table of failure observed times per asset
        assets : pd.DataFrame
            Table of assets properties, including observation window and possible
            covariates
        covariates : list[str] | None, optional
            List of columns of the asset table to use as covariates. Defaults to None.
        partial_observations : pd.DataFrame | None, optional
            Optional partial observations, a number of events per time window.
            Defaults to None.

        Returns
        -------
        Self
            The fitted process.

        Raises
        ------
        ValueError
            If covariates are given with a distribution, or missing with a
            regression.

        Examples
        --------
        >>> failures
            id  time
        0   1   3.2
        1   1   7.8
        2   2   5.0
        3   3   2.1
        4   3   4.6
        5   3   9.3
        >>> assets
            id  start   end  covar_1  covar_2
        0   1    0.0  10.0      0.5       10
        1   2    0.0  12.0      1.2       20
        2   3    0.0  11.5      0.8       15
        3   4    0.0   8.0      2.0       30
        >>> covariates
        ['covar_1', 'covar_2']
        >>> partial_observations
            id  start  end  count
        0   4    0.0  4.0      1
        1   4    4.0  8.0      2
        """
        optimizer = self.init_likelihood(
            failures,
            assets,
            covariates,
            partial_observations,
            **kwargs,
        )
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

    def freeze(self, *args: *CovarTs) -> FrozenNonHomogeneousPoissonProcess[*CovarTs]:
        """
        Return a process with additional arguments stored.

        Parameters
        ----------
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenNonHomogeneousPoissonProcess
        """
        return FrozenNonHomogeneousPoissonProcess(self, *args)


class FrozenNonHomogeneousPoissonProcess[*CovarTs](NonHomogeneousPoissonProcess[()]):
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
