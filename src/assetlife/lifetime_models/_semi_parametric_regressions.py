from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, final, override

import numpy as np
import optype.numpy as onp

from ._parametric_regressions import LinearCovarEffect
from assetlife._optimization import FitConfig, MaximumLikelihoodOptimizer
from assetlife.base import FittingResults
from assetlife.typing import CoercibleFloat64_1D


class CoxData:
    time: onp.Array[tuple[int, Literal[1]], np.float64]
    covar: tuple[onp.Array[tuple[int, Literal[1]], np.float64], ...]
    event: onp.Array[tuple[int, Literal[1]], np.bool_] | None
    entry: onp.Array[tuple[int, Literal[1]], np.float64] | None

    ordered_event_time: onp.Array1D[np.float64]
    event_count: onp.Array1D[np.int64]
    risk_set: onp.Array2D[np.bool_]
    death_set: onp.Array2D[np.bool_]
    ordered_event_covar: tuple[onp.Array[tuple[int, Literal[1]], np.float64], ...]

    def __init__(
        self,
        time: onp.Array1D[np.float64],
        covar: Sequence[onp.Array1D[np.float64]],
        event: onp.Array1D[np.bool_] | None = None,
        entry: onp.Array1D[np.float64] | None = None,
    ) -> None:
        self.time = time[:, None]
        self.event = (
            event[:, None]
            if event is not None
            else np.ones_like(self.time, dtype=np.bool_)
        )
        self.entry = (
            entry[:, None]
            if entry is not None
            else np.zeros_like(self.time, dtype=np.float64)
        )
        self.covar = tuple(c[:, None] for c in covar)
        sizes = [len(x) for x in (self.time, self.event, self.entry, *self.covar)]

        if len(set(sizes)) != 1:
            msg = (
                "All lifetime data must have the same number of values. Fields "
                f"length are different. Got {tuple(sizes)}."
            )
            raise ValueError(msg)
        (
            self.ordered_event_time,  # uncensored sorted untied times
            ordered_event_index,
            self.event_count,
        ) = np.unique(
            self.time[self.event == 1],
            return_index=True,
            return_counts=True,
        )
        # here risk_set is mask array on time
        # left truncated & right censored
        self.risk_set = np.logical_and(
            (
                np.vstack([self.entry[:, 0]] * len(self.ordered_event_time))
                < np.hstack([self.ordered_event_time[:, None]] * len(self.time))
            ),
            (
                np.hstack([self.ordered_event_time[:, None]] * len(self.time))
                <= np.vstack([self.time[:, 0]] * len(self.ordered_event_time))
            ),
        )

        self.death_set = np.vstack(
            [self.time[:, 0] * self.event[:, 0]] * len(self.ordered_event_time)
        ) == np.hstack([self.ordered_event_time[:, None]] * len(self.time))

        self.ordered_event_covar = tuple(
            c[self.event[:, 0] == 1][ordered_event_index] for c in self.covar
        )


def _psi(
    covar_effect: LinearCovarEffect,
    data: CoxData,
    on: Literal["risk", "death"] = "risk",
    order: Literal[0, 1, 2] = 0,
) -> onp.ArrayND[np.float64]:
    """
    Psi formula used for likelihood computations.

    Parameters
    ----------
    covar_effect : LinearCovarEffect
        The covariates effect, holding the parameter values to use.
    data : CoxData
        The data of the Cox model.
    on : {"risk", "death"}, default is "risk"
        The set on which the sum is applied.
    order : {0, 1, 2}, default is 0
        Order of the derivatives with respect to params.

    Returns
    -------
    out : np.ndarray
        Psi formulation, of shape (m, 1) if order is 0, (m, p) if order
        is 1 and (m, p, p) if order is 2.

    Raises
    ------
    ValueError
        If ``on`` or ``order`` is not one of the allowed values.
    """
    if on == "risk":
        i_set = data.risk_set
    elif on == "death":
        i_set = data.death_set
    else:
        msg = f"on must be 'risk' or 'death', got {on!r}"
        raise ValueError(msg)

    if order == 0:
        # output shape: [m]
        return np.dot(i_set, covar_effect.g(*data.covar)).astype(np.float64)
    if order == 1:
        # output shape: [m, p]
        return np.dot(
            i_set,
            np.column_stack(data.covar) * covar_effect.g(*data.covar),
        )
    if order == 2:
        # output shape: [m, p, p]
        return np.tensordot(
            i_set[:, :None],
            np.column_stack(data.covar)[:, None]
            * np.column_stack(data.covar)[:, :, None]
            * np.asarray(covar_effect.g(*data.covar))[:, :, None],
            axes=1,
        ).astype(np.float64)
    msg = f"order must be 0, 1 or 2, got {order!r}"
    raise ValueError(msg)


class CoxPartialLifetimeLikelihood(
    MaximumLikelihoodOptimizer[LinearCovarEffect, CoxData]
):
    model: LinearCovarEffect
    data: CoxData
    config: FitConfig

    def __init__(
        self,
        model: LinearCovarEffect,
        data: CoxData,
        config: FitConfig,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config

        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log
        if "hess" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["hess"] = self.hess_negative_log

    @property
    @override
    def nb_observations(self) -> int:
        return len(self.data.time)

    @override
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        self.model.set_params(params)
        return -(
            np.log(self.model.g(*self.data.ordered_event_covar)).sum()
            - np.log(_psi(self.model, self.data)).sum()
        )

    def jac_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array1D[np.float64]:
        self.model.set_params(params)  # changes model params

        return -(
            np.column_stack(self.data.ordered_event_covar).sum(axis=0)
            - (_psi(self.model, self.data, order=1) / _psi(self.model, self.data)).sum(
                axis=0
            )
        )

    def hess_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array2D[np.float64]:
        self.model.set_params(params)  # changes model params

        psi_order_0 = _psi(self.model, self.data)
        psi_order_1 = _psi(self.model, self.data, order=1)

        hessian_part_1 = _psi(self.model, self.data, order=2) / psi_order_0[:, :, None]

        hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
            psi_order_1 / psi_order_0
        )[:, :, None]

        return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)


@final
class BreslowPartialLifetimeLikelihood(CoxPartialLifetimeLikelihood):
    s_j: onp.ArrayND[np.float64]

    def __init__(
        self,
        model: LinearCovarEffect,
        data: CoxData,
        config: FitConfig,
    ) -> None:
        super().__init__(model, data, config)

        self.s_j = np.dot(self.data.death_set, np.column_stack(self.data.covar))

    @override
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        self.model.set_params(params)  # changes model params

        return -(
            np.log(self.model.g(*np.unstack(self.s_j, axis=-1))).sum()
            - (
                self.data.event_count[:, None] * np.log(_psi(self.model, self.data))
            ).sum()
        )

    @override
    def jac_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array1D[np.float64]:
        self.model.set_params(params)  # changes model params

        return -(
            self.s_j.sum(axis=0)
            - (
                self.data.event_count[:, None]
                * (_psi(self.model, self.data, order=1) / _psi(self.model, self.data))
            ).sum(axis=0)
        )

    @override
    def hess_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array2D[np.float64]:
        self.model.set_params(params)  # changes model params

        psi_order_0 = _psi(self.model, self.data)
        psi_order_1 = _psi(self.model, self.data, order=1)

        hessian_part_1 = _psi(self.model, self.data, order=2) / psi_order_0[:, :, None]

        hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
            psi_order_1 / psi_order_0
        )[:, :, None]

        return (self.data.event_count[:, None, None] * hessian_part_1).sum(axis=0) - (
            self.data.event_count[:, None, None] * hessian_part_2
        ).sum(axis=0)


@final
class EfronPartialLifetimeLikelihood(CoxPartialLifetimeLikelihood):
    s_j: onp.ArrayND[np.float64]
    discount_rates: onp.ArrayND[np.float64]
    discount_rates_mask: onp.ArrayND[np.bool_]
    scipy_method = "trust-exact"

    def __init__(
        self,
        model: LinearCovarEffect,
        data: CoxData,
        config: FitConfig,
    ) -> None:
        super().__init__(model, data, config)
        self.s_j = np.dot(self.data.death_set, np.column_stack(self.data.covar))
        self.discount_rates = (
            np.vstack(
                [np.arange(np.max(self.data.event_count))] * len(self.data.event_count)
            )
            / self.data.event_count[:, None]
        )
        self.discount_rates_mask = np.where(self.discount_rates < 1, True, False)

    def _psi_efron(
        self,
        order: Literal[0, 1, 2] = 0,
    ) -> onp.ArrayND[np.float64]:
        """
        Psi formula for Efron method.

        Parameters
        ----------
        order : {0, 1, 2}, default is 0
            Order of the derivatives with respect to params.

        Returns
        -------
        out : np.ndarray
            Psi formulation for Efron method, of shape (m, max(d_j)) if order
            is 0, (m, max(d_j), p) if order is 1 and (m, max(d_j), p, p) if
            order is 2.

        Raises
        ------
        ValueError
            If ``order`` is not 0, 1 or 2.
        """

        if order == 0:
            # output shape: [m, max(d_j)]
            return (
                _psi(self.model, self.data, order=order) * self.discount_rates_mask
                - _psi(self.model, self.data, on="death", order=order)
                * self.discount_rates
                * self.discount_rates_mask
            )
        if order == 1:
            # output shape: [m, max(d_j), p]
            return (
                _psi(self.model, self.data, order=1)[:, None, :]
                * self.discount_rates_mask[:, :, None]
                - _psi(self.model, self.data, on="death", order=1)[:, None, :]
                * (self.discount_rates * self.discount_rates_mask)[:, :, None]
            )
        if order == 2:
            # output shape: [m, max(d_j), p, p]
            return (
                _psi(self.model, self.data, order=2)[:, None, :]
                * self.discount_rates_mask[:, :, None, None]
                - _psi(self.model, self.data, on="death", order=2)[:, None, :]
                * (self.discount_rates * self.discount_rates_mask)[:, :, None, None]
            )
        msg = f"order must be 0, 1 or 2, got {order!r}"
        raise ValueError(msg)

    @override
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        self.model.set_params(params)  # changes model params

        # .sum(axis=1, keepdims=True) --> sum on alpha to d_j
        # .sum() --> sum on j
        # using where in np.log allows to avoid 0. masked elements
        m = self._psi_efron()
        return -(
            np.log(self.model.g(*np.unstack(self.s_j, axis=-1))).sum()
            - np
            .log(m, out=np.zeros_like(m), where=(m != 0))
            .sum(axis=1, keepdims=True)
            .sum()
        )

    @override
    def jac_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array1D[np.float64]:
        self.model.set_params(params)  # changes model params
        # .sum(axis=1) --> sum on alpha to d_j
        # .sum(axis=0) --> sum on j
        # using where in np.divide allows to avoid 0. masked elements
        a = self._psi_efron(order=1)
        b = self._psi_efron()[:, :, None]
        return -(
            self.s_j.sum(axis=0)
            - np
            .divide(a, b, out=np.zeros_like(a), where=(b != 0))
            .sum(axis=1)
            .sum(axis=0)
        )

    @override
    def hess_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array2D[np.float64]:
        self.model.set_params(params)  # changes model params

        psi_order_0 = self._psi_efron()
        psi_order_1 = self._psi_efron(order=1)

        # .sum(axis=1) --> sum on alpha to d_j
        # using where in np.divide allows to avoid 0. masked elements
        a = self._psi_efron(order=2)
        b = psi_order_0[:, :, None, None]
        hessian_part_1 = np.divide(a, b, out=np.zeros_like(a), where=(b != 0)).sum(
            axis=1
        )

        # .sum(axis=1) --> sum on alpha to d_j
        # using where in np.divide allows to avoid 0. masked elements
        b = psi_order_0[:, :, None]
        hessian_part_2 = (
            np.divide(psi_order_1, b, out=np.zeros_like(psi_order_1), where=(b != 0))[
                :, :, None, :
            ]
            * (
                np.divide(
                    psi_order_1, b, out=np.zeros_like(psi_order_1), where=(b != 0)
                )
            )[:, :, :, None]
        )
        hessian_part_2 = hessian_part_2.sum(axis=1)

        return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)


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
    _likelihood: CoxPartialLifetimeLikelihood
    _sf0_estimation: CoxEstimation

    def __init__(
        self,
        time: onp.Array1D[np.float64],
        covar: onp.Array1D[np.float64] | Sequence[onp.Array1D[np.float64]],
        event: onp.Array1D[np.bool_] | None = None,
        entry: onp.Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> None:
        nb_covar = 1 if not isinstance(covar, Sequence) else len(covar)
        self._covar_effect = LinearCovarEffect(*(0.0,) * nb_covar)
        covar = (covar,) if not isinstance(covar, Sequence) else covar
        self._likelihood = self.init_likelihood(time, covar, event, entry, **kwargs)
        fitting_results = self._likelihood.optimize()
        self._covar_effect.set_params(fitting_results.optimal_params)
        self.fitting_results = fitting_results
        timeline = self._likelihood.data.ordered_event_time.copy()
        self._sf0_estimation = CoxEstimation(timeline=timeline, values=self._sf0())

    def init_likelihood(
        self,
        time: onp.Array1D[np.float64],
        covar: Sequence[onp.Array1D[np.float64]],
        event: onp.Array1D[np.bool_] | None = None,
        entry: onp.Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> CoxPartialLifetimeLikelihood:
        covar_effect = LinearCovarEffect(*(0.0,) * len(covar))
        x0 = kwargs.get("x0", np.zeros(len(covar), dtype=float))
        config = FitConfig(x0)
        config.scipy_minimize_options["method"] = kwargs.get("method", "trust-exact")
        config.covariance_method = kwargs.get("covariance_method", "exact")
        cox_data = CoxData(time, covar, event=event, entry=entry)
        _, event_count = np.unique(time[event == 1], return_counts=True)
        if (event_count > 3).any():  # efron
            return EfronPartialLifetimeLikelihood(self._covar_effect, cox_data, config)
        if (event_count <= 3).all() and (2 in event_count):
            return BreslowPartialLifetimeLikelihood(covar_effect, cox_data, config)
        return CoxPartialLifetimeLikelihood(covar_effect, cox_data, config)

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
            self._likelihood.data.event_count[:, None]
            / _psi(self._covar_effect, self._likelihood.data)
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
        psi_values = _psi(self._covar_effect, self._likelihood.data)
        psi_order_1 = _psi(self._covar_effect, self._likelihood.data, order=1)
        d_j_on_psi = self._likelihood.data.event_count[:, None] / psi_values

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
