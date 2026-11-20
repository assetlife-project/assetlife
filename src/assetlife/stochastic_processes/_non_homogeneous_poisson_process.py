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
    from collections.abc import Sequence

    from narwhals.typing import IntoDataFrame

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

    def _init_likelihood(self, data: NHPPData, **kwargs: Any) -> NHPPLikelihood:
        if isinstance(self.lifetime_model, LifetimeDistribution):
            if data.nb_covars > 0:
                msg = "No covariates can be given for fit when using a distribution."
                raise ValueError(msg)
            fresh_model = type(self.lifetime_model)()
            covariance_method = kwargs.get(
                "covariance_method",
                "2point" if isinstance(fresh_model, Gamma) else "cs",
            )
        elif isinstance(self.lifetime_model, ParametricLifetimeRegression):
            if data.nb_covars == 0:
                msg = "Covariates must be given for fit when using a regression."
                raise ValueError(msg)
            fresh_model = type(self.lifetime_model)(
                type(self.lifetime_model.baseline)(),
                coefficients=(0.0,) * data.nb_covars,
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

        x0 = (
            kwargs["x0"]
            if "x0" in kwargs
            else fresh_model._init_params(data.failures_time)  # ruff: ignore[private-member-access]
        )
        config = FitConfig(x0)
        config.scipy_minimize_options["bounds"] = kwargs.get(
            "bounds", fresh_model.get_params_bounds()
        )
        config.covariance_method = covariance_method
        config.scipy_minimize_options["method"] = kwargs.get("method", "L-BFGS-B")
        return NHPPLikelihood(fresh_model, data, config)

    def _fit(self, data: NHPPData, **kwargs: Any) -> Self:
        likelihood = self._init_likelihood(data, **kwargs)
        fitting_results = likelihood.optimize()
        if isinstance(self.lifetime_model, ParametricLifetimeRegression):
            self.lifetime_model.covar_effect.set_params(
                [0.0] * data.nb_covars
            )  # resize coefficients
        self.set_params(fitting_results.optimal_params)
        self.fitting_results = fitting_results
        return self

    def fit(
        self,
        failures: IntoDataFrame,
        assets: IntoDataFrame,
        covar_column_names: Sequence[str] = (),
        *,
        partial_observations: IntoDataFrame | None = None,
        **kwargs: Any,
    ) -> Self:
        """Estimate the process parameters from failure observation history.

        Parameters
        ----------
        failures : DataFrame
            Failure times, one row per failure, as any eager dataframe supported
            by narwhals (polars, pandas, pyarrow...). Columns are ``id``, the
            asset id, and ``time``, the failure time as float. Each id must be
            an asset id of ``assets``. Other columns are ignored.
        assets : DataFrame
            Assets, one row per asset, as any eager dataframe. Columns are
            ``id``, the unique asset id, and ``start`` and ``end``, the bounds of
            the observation window of the asset as floats. Assets may have no
            failure. Other columns are ignored, except covariates selected with
            ``covar_column_names``.
        covar_column_names : sequence of str, default ()
            Names of the covariate columns in ``assets``. Each covariate must be
            a column of floats. Covariates are required if the lifetime model is
            a regression, and forbidden if it is a distribution.
        partial_observations : DataFrame, optional
            Numbers of failures observed per time window, without failure times,
            as any eager dataframe. Columns are ``id``, the asset id, ``start``
            and ``end``, the bounds of the time window as floats, and ``count``,
            the number of failures observed in the window. Each id must be an
            asset id of ``assets`` and each window must be within the
            observation window of its asset. Other columns are ignored.
        **kwargs
            Extra keyword arguments passed to `scipy.optimize.minimize
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_,
            which searches for the parameters minimizing the negative
            log-likelihood.

            The additional keyword ``covariance_method`` controls how the
            covariance matrix of the estimated parameters is computed. Allowed
            values are ``"cs"``, ``"2point"``, ``"exact"`` or ``False``. If
            ``False``, the covariance is not estimated. If ``"exact"``, ``hess``
            must also be provided. If not set, the model's default method is
            used.

        Returns
        -------
        self : NonHomogeneousPoissonProcess
            The fitted process. Estimated parameters are set in place.

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
        >>> partial_observations
            id  start  end  count
        0   4    0.0  4.0      1
        1   4    4.0  8.0      2
        >>> nhpp.fit(
        ...     failures,
        ...     assets,
        ...     ["covar_1", "covar_2"],
        ...     partial_observations=partial_observations,
        ... )
        """
        return self._fit(
            NHPPData(
                failures,
                assets,
                covar_column_names=covar_column_names,
                partial_observations=partial_observations,
            ),
            **kwargs,
        )

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
