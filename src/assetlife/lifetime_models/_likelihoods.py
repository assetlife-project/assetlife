from typing import Literal, final, override

import numpy as np
import optype.numpy as onp

from ._base import FittableParametricLifetimeModel
from ._covar_effect import LinearCovarEffect
from ._datasets import CoxData, LifetimeData
from assetlife._optimization import FitConfig, MaximumLikelihoodOptimizer
from assetlife.typing import CoercibleFloat64_ND


@final
class LifetimeLikelihood(
    MaximumLikelihoodOptimizer[
        FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]], LifetimeData
    ]
):
    """
    Maximum likelihood estimator from lifetime data.

    Parameters
    ----------
    model : FittableParametricLifetimeModel
        Model with initialized parameters.
    data : LifetimeData
        Preprocessed lifetime observations.
    config : FitConfig
        Configuration used by the optimizer.

    Attributes
    ----------
    model : FittableParametricLifetimeModel
        Model used by the likelihood.
    data : LifetimeData
        Preprocessed lifetime observations.
    config : FitConfig
        Configuration used by the optimizer.
    """

    data: LifetimeData

    def __init__(
        self,
        model: FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]],
        data: LifetimeData,
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
        return self.data.nb_observations

    @override
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        self.model.set_params(params)
        return (
            self._complete_time_contrib()
            + self._censored_time_contrib()
            + self._left_truncations_contrib()
        )

    def jac_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array1D[np.float64]:
        """
        Jacobian of the negative log likelihood.

        The Jacobian is computed with respect to parameters.

        Parameters
        ----------
        params : 1d array of floats
            Parameter values.

        Returns
        -------
        out : 1d array of floats
        """
        self.model.set_params(params)
        return (
            self._jac_complete_time_contrib()
            + self._jac_censored_time_contrib()
            + self._jac_left_truncations_contrib()
        )

    def _complete_time_contrib(self) -> float:
        if self.data.complete_time.size == 0:
            return 0.0
        return -np.sum(
            np.log(
                self.model.pdf(self.data.complete_time, *self.data.complete_time_covars)
            )
        )

    def _jac_complete_time_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.complete_time.size == 0:
            return np.zeros_like(self.model.get_params())
        jac = -self.model.jac_pdf(
            self.data.complete_time, *self.data.complete_time_covars
        ) / self.model.pdf(self.data.complete_time, *self.data.complete_time_covars)

        return np.sum(jac, axis=(1, 2))

    def _censored_time_contrib(self) -> float:
        if self.data.censored_time.size == 0:
            return 0.0
        if self.data.censored_time.shape[-1] > 1:
            # interval censored time
            return np.sum(
                -np.log(
                    10**-10
                    + self.model.cdf(
                        self.data.censored_time[:, 1], *self.data.censored_time_covars
                    )
                    - self.model.cdf(
                        self.data.censored_time[:, 0], *self.data.censored_time_covars
                    )
                ),
            )
        # right censored time
        return np.sum(
            self.model.chf(self.data.censored_time, *self.data.censored_time_covars)
        )

    def _jac_censored_time_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.censored_time.size == 0:
            return np.zeros_like(self.model.get_params())
        if self.data.censored_time.shape[-1] > 1:
            # interval censored time
            jac_interval_censored = (
                self.model.jac_sf(
                    self.data.censored_time[:, 1], *self.data.censored_time_covars
                )
                - self.model.jac_sf(
                    self.data.censored_time[:, 0], *self.data.censored_time_covars
                )
            ) / (
                10**-10
                + self.model.cdf(
                    self.data.censored_time[:, 1], *self.data.censored_time_covars
                )
                - self.model.cdf(
                    self.data.censored_time[:, 0], *self.data.censored_time_covars
                )
            )

            return np.sum(jac_interval_censored, axis=(1, 2))
        # right censored time
        return np.sum(
            self.model.jac_chf(
                self.data.censored_time, *self.data.censored_time_covars
            ),
            axis=(1, 2),
        )

    def _left_truncations_contrib(self) -> float:
        if self.data.left_truncations.size == 0:
            return 0.0
        return -np.sum(
            self.model.chf(
                self.data.left_truncations, *self.data.left_truncations_covars
            )
        )

    def _jac_left_truncations_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.left_truncations.size == 0:
            return np.zeros_like(self.model.get_params())
        jac = -self.model.jac_chf(
            self.data.left_truncations, *self.data.left_truncations_covars
        )
        return np.sum(jac, axis=(1, 2))


def _risk_set_sums(
    covar_effect: LinearCovarEffect,
    data: CoxData,
    on: Literal["risk", "death"] = "risk",
    order: Literal[0, 1, 2] = 0,
) -> onp.ArrayND[np.float64]:
    """Sums of covariates effect terms over a set of observations.

    At each ordered event time t_j, sums over the observations of the risk set
    (or of the death set) the terms g(x_i), x_i g(x_i) and x_i x_i^T g(x_i),
    where g(x) = exp(beta^T x) is the covariates effect. These are the
    S^(0), S^(1) and S^(2) statistics of the Cox model, i.e. the sum of g and
    its first and second derivatives with respect to beta. They are used by the
    partial likelihoods, and by the Breslow estimation of the baseline.

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
        Sums of shape (m, 1) if order is 0, (m, p) if order is 1 and (m, p, p)
        if order is 2, with m the number of ordered event times and p the
        number of covariates.

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
            i_set,
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
        return self.data.nb_observations

    @override
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        self.model.set_params(params)
        return -(
            np.log(self.model.g(*self.data.ordered_event_covar)).sum()
            - np.log(_risk_set_sums(self.model, self.data)).sum()
        )

    def jac_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array1D[np.float64]:
        self.model.set_params(params)  # changes model params

        return -(
            np.column_stack(self.data.ordered_event_covar).sum(axis=0)
            - (
                _risk_set_sums(self.model, self.data, order=1)
                / _risk_set_sums(self.model, self.data)
            ).sum(axis=0)
        )

    def hess_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array2D[np.float64]:
        self.model.set_params(params)  # changes model params

        psi_order_0 = _risk_set_sums(self.model, self.data)
        psi_order_1 = _risk_set_sums(self.model, self.data, order=1)

        hessian_part_1 = (
            _risk_set_sums(self.model, self.data, order=2) / psi_order_0[:, :, None]
        )

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
                self.data.event_count[:, None]
                * np.log(_risk_set_sums(self.model, self.data))
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
                * (
                    _risk_set_sums(self.model, self.data, order=1)
                    / _risk_set_sums(self.model, self.data)
                )
            ).sum(axis=0)
        )

    @override
    def hess_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array2D[np.float64]:
        self.model.set_params(params)  # changes model params

        psi_order_0 = _risk_set_sums(self.model, self.data)
        psi_order_1 = _risk_set_sums(self.model, self.data, order=1)

        hessian_part_1 = (
            _risk_set_sums(self.model, self.data, order=2) / psi_order_0[:, :, None]
        )

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
                _risk_set_sums(self.model, self.data, order=order)
                * self.discount_rates_mask
                - _risk_set_sums(self.model, self.data, on="death", order=order)
                * self.discount_rates
                * self.discount_rates_mask
            )
        if order == 1:
            # output shape: [m, max(d_j), p]
            return (
                _risk_set_sums(self.model, self.data, order=1)[:, None, :]
                * self.discount_rates_mask[:, :, None]
                - _risk_set_sums(self.model, self.data, on="death", order=1)[:, None, :]
                * (self.discount_rates * self.discount_rates_mask)[:, :, None]
            )
        if order == 2:
            # output shape: [m, max(d_j), p, p]
            return (
                _risk_set_sums(self.model, self.data, order=2)[:, None, :]
                * self.discount_rates_mask[:, :, None, None]
                - _risk_set_sums(self.model, self.data, on="death", order=2)[:, None, :]
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
