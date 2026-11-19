from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import optype.numpy as onp
from narwhals.typing import IntoDataFrame

from ._covar_effect import LinearCovarEffect
from ._datasets import CoxData
from ._likelihoods import (
    BreslowPartialLifetimeLikelihood,
    CoxPartialLifetimeLikelihood,
    EfronPartialLifetimeLikelihood,
    _risk_set_sums,
)
from assetlife._optimization import FitConfig
from assetlife.base import FittingResults
from assetlife.typing import CoercibleFloat64_1D


class CoxEstimation(NamedTuple):
    timeline: onp.Array1D[np.float64]
    values: onp.Array1D[np.float64]
    se: onp.Array1D[np.float64] | None = None


class SemiParametricProportionalHazard:
    """
    Cox semi-parametric proportional hazard.
    """

    fitting_results: FittingResults
    _covar_effect: LinearCovarEffect
    _data: CoxData
    _sf0_estimation: CoxEstimation

    def __init__(
        self,
        dataset: IntoDataFrame,
        covar_column_names: Sequence[str],
        *,
        field_column_names: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit the model to lifetime observations.

        As the model is semi-parametric, it is defined by its data: it is
        fitted when instantiated.

        Parameters
        ----------
        dataset : DataFrame
            Lifetime observations, one row per observation, as any eager
            dataframe supported by narwhals (polars, pandas, pyarrow...).
            Lifetimes are complete or right-censored, given as a ``time`` column
            of floats. An optional ``event`` column of booleans tells whether
            each lifetime is a failure (``True``) or right-censored
            (``False``). If ``event`` is missing, all lifetimes are complete.
            An optional ``entry`` column of floats holds left truncation values
            (``0.`` if missing). Other columns are ignored, except covariates
            selected with ``covar_column_names``.
        covar_column_names : sequence of str
            Names of the covariate columns in ``dataset``. At least one
            covariate is required. Each covariate must be a column of floats.
        field_column_names : mapping of str to str, optional
            Column name of each lifetime field in ``dataset``, as
            ``{field: column_name}``, e.g. ``{"time": "duration"}``. Fields are
            ``"time"``, ``"event"`` and ``"entry"``. Fields not given are
            looked up under their own name.
        **kwargs
            Extra keyword arguments passed to `scipy.optimize.minimize
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_,
            which searches for the coefficients minimizing the negative partial
            log-likelihood.

            The additional keyword ``covariance_method`` controls how the
            covariance matrix of the estimated coefficients is computed. If not
            set, ``"exact"`` is used.
        """
        self._data = CoxData(
            dataset, covar_column_names, field_column_names=field_column_names
        )
        self._covar_effect = LinearCovarEffect(*(0.0,) * self._data.nb_covars)
        fitting_results = self._init_likelihood(**kwargs).optimize()
        self._covar_effect.set_params(fitting_results.optimal_params)
        self.fitting_results = fitting_results
        timeline = self._data.ordered_event_time.copy()
        self._sf0_estimation = CoxEstimation(timeline=timeline, values=self._sf0())

    def _init_likelihood(self, **kwargs: Any) -> CoxPartialLifetimeLikelihood:
        """Partial likelihood handling the ties of the event times.

        Efron if more than 3 failures share an event time, Breslow if at most 3
        do and some pairs exist, the plain Cox partial likelihood otherwise.
        """
        x0 = kwargs.get("x0", np.zeros(self._data.nb_covars, dtype=float))
        config = FitConfig(x0)
        config.scipy_minimize_options["method"] = kwargs.get("method", "trust-exact")
        config.covariance_method = kwargs.get("covariance_method", "exact")
        event_count = self._data.event_count
        if (event_count > 3).any():
            return EfronPartialLifetimeLikelihood(
                self._covar_effect, self._data, config
            )
        if 2 in event_count:
            return BreslowPartialLifetimeLikelihood(
                self._covar_effect, self._data, config
            )
        return CoxPartialLifetimeLikelihood(self._covar_effect, self._data, config)

    def get_params(self) -> onp.Array1D[np.float64]:
        return self._covar_effect.get_params()

    def _chf0(self) -> onp.ArrayND[np.float64]:
        """
        The baseline cumulative hazard function estimation.

        Returns
        -------
        out : np.ndarray
            The estimated values at each ordered event time.
        """
        return np.cumsum(
            self._data.event_count[:, None]
            / _risk_set_sums(self._covar_effect, self._data)
        )

    def _sf0(self) -> onp.ArrayND[np.float64]:
        """
        The baseline survival function estimation.

        Returns
        -------
        out : np.ndarray
            The estimated values at each ordered event time.
        """
        return np.exp(-self._chf0())

    def sf(self, *covar: CoercibleFloat64_1D, se: bool = True) -> CoxEstimation:
        """
        The survival function estimations.

        Parameters
        ----------
        covar: np.array
            array with covariates values
        se : bool, default True
            If True, the standard errors are returned in addition to timeline
            and sf values.

        Returns
        -------
        out : tuple of timeline, values, optionally se. Default is None
            A timeline, corresponding sf values and optionnaly the standard
            errors. If the estimations does not exist yet, returns None.
        """
        g = self._covar_effect.g(*covar)
        g = g[:, None] if g.ndim == 1 else g
        if se and self.fitting_results.covariance_matrix is not None:
            return CoxEstimation(
                self._sf0_estimation.timeline,
                self._sf0_estimation.values**g,
                se=self._sf0_estimation.values**g
                * np.sqrt(
                    self._q1_q2_sum(
                        *covar, covariance_matrix=self.fitting_results.covariance_matrix
                    )
                ),
            )
        return CoxEstimation(
            self._sf0_estimation.timeline,
            self._sf0_estimation.values**g,
        )

    def _q1_q2_sum(
        self,
        *covar: CoercibleFloat64_1D,
        covariance_matrix: onp.ArrayND[np.float64],
    ) -> onp.ArrayND[np.float64]:
        """
        Klein and Moeschberger: Survival Analysis Techniques for Censored and
        Truncated Data (p. 284).
        """
        psi_values = _risk_set_sums(self._covar_effect, self._data)
        psi_order_1 = _risk_set_sums(self._covar_effect, self._data, order=1)
        d_j_on_psi = self._data.event_count[:, None] / psi_values

        q3 = np.cumsum(
            (
                (psi_order_1 / psi_values)[None, :, :]
                - np.column_stack(covar)[:, None, :]
            )
            * d_j_on_psi[None, :, :],
            axis=1,
        )  # [m: new sample for inference, t: timeline, p]
        q2 = np.squeeze(
            np.matmul(
                q3[:, :, None, :],
                np.matmul(
                    covariance_matrix[None, None, :, :],
                    q3[:, :, :, None],
                ),
            )
        )  # [m, t]
        q1 = np.cumsum(d_j_on_psi * (1 / psi_values))
        return q1 + q2
