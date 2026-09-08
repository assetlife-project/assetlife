"""Lifetime regression

Notes
-----
This module contains two parametric lifetime regressions.
ProportionalHazard is not Cox regression (Cox is semiparametric).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import Any, Literal, Self, final

import numpy as np
import optype.numpy as onp
from numpydoc import docscrape  # pyright: ignore[reportMissingTypeStubs]
from scipy.optimize import Bounds
from typing_extensions import override

from assetlife.base import FitConfig, FittingResults, ParametricModel
from assetlife.typing import CoercibleFloat64_ND, Float64_ND

from ._base import (
    FittableParametricLifetimeModel,
    LifetimeData,
    LifetimeLikelihood,
    ParametricLifetimeModel,
    document_args,
)
from ._distributions import (
    Gamma,
    LifetimeDistribution,
    get_distrib_params_bounds,
    init_distrib_params_from_lifetimes,
)


class LinearCovarEffect(ParametricModel):
    """
    Covariates effect.

    Parameters
    ----------
    coefficients : tuple of float, default is (None,)
        Coefficients of the covariates effect.
    """

    def __init__(self, *coefficients: float):
        super().__init__(*coefficients)

    def g(self, *covar: CoercibleFloat64_ND) -> Float64_ND:
        """
        Returns the covariates effect.

        Parameters
        ----------
        covar : float or np.ndarray
            The covariate values

        Returns
        -------
        out : np.float64 or np.ndarray
        """
        nb_coef = self.get_params().size
        if len(covar) != nb_coef:
            raise ValueError(
                f"""
                Invalid number of covar. Got {nb_coef} coefficients but {len(covar)} covariates are given.
                """
            )
        broadcasted_covar = np.broadcast_arrays(*covar)
        stack_covar = np.stack(broadcasted_covar, axis=-1)
        return np.exp(np.sum(stack_covar * self.get_params(), axis=-1))

    def jac_g(self, *covar: CoercibleFloat64_ND) -> onp.ArrayND[np.float64]:
        """
        Returns the jacobian of the covariates effect.

        Parameters
        ----------
        covar : float or np.ndarray
            The covariate values

        Returns
        -------
        out : np.ndarray
        """
        g = self.g(*covar)
        broadcasted_covar = np.broadcast_arrays(*covar)
        stack_covar = np.stack(broadcasted_covar, axis=0)
        return stack_covar * g

    @override
    def __repr__(self) -> str:
        return f"LinearCovarEffect({self.get_params().item()!r})"


_covar_docstring = [
    docscrape.Parameter(
        "covar",
        "float or np.ndarray",
        [
            "Covariates values.",
            "float can only be valid if the regression has one coefficients.",
            "Otherwise it must be a ndarray of shape `(nb_coef,)` or `(m, nb_coef)`.",
        ],
    ),
]


class ParametricLifetimeRegression(
    FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]], ABC
):
    """
    Base class for lifetime regression.
    """

    baseline: LifetimeDistribution
    covar_effect: LinearCovarEffect
    fitting_results: FittingResults | None

    def __init__(
        self,
        baseline: LifetimeDistribution,
        coefficients: Sequence[float] = (),
    ):
        super().__init__()
        self.covar_effect = LinearCovarEffect(*coefficients)
        self.baseline = baseline

    def get_coefficients(self) -> onp.Array1D[np.float64]:
        """
        Returns the coefficients values.

        Returns
        -------
        out : ndarray
        """
        return self.covar_effect.get_params()

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def sf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        return super().sf(time, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def isf(
        self,
        probability: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        cumulative_hazard_rate = -np.log(
            np.clip(probability, 0, 1 - np.finfo(float).resolution)
        )
        return self.ichf(cumulative_hazard_rate, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def cdf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        return super().cdf(time, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def pdf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        return super().pdf(time, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def ppf(
        self,
        probability: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        return super().ppf(probability, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def median(self, *covar: CoercibleFloat64_ND) -> Float64_ND:
        return super().median(*covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def mean(self, *covar: CoercibleFloat64_ND) -> Float64_ND:
        return super().mean(*covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def var(self, *covar: CoercibleFloat64_ND) -> Float64_ND:
        return super().var(*covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_sf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> onp.ArrayND[np.float64]:
        return super().jac_sf(time, *covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_cdf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> onp.ArrayND[np.float64]:
        return super().jac_cdf(time, *covar)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_pdf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> onp.ArrayND[np.float64]:
        return super().jac_pdf(time, *covar)

    @override
    @document_args(base_cls=ParametricLifetimeModel, args_docstring=_covar_docstring)
    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *covar: CoercibleFloat64_ND,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> Float64_ND:
        return super().rvs(
            size,
            *covar,
            seed=seed,
        )

    @override
    def init_likelihood(
        self,
        time: onp.Array1D[np.float64] | onp.Array[tuple[int, Literal[2]], np.float64],
        args: Sequence[onp.Array1D[np.float64]] | None = None,
        event: onp.Array1D[np.bool_] | None = None,
        entry: onp.Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> LifetimeLikelihood:
        assert args is not None
        fresh_regression = type(self)(
            type(self.baseline)(), coefficients=(0.0,) * len(args)
        )  # init new regression object with appropriate number of covar
        lifetime_data = LifetimeData(time, event, entry, args)
        x0 = kwargs.get(
            "x0", init_regression_params_from_lifetimes(fresh_regression, lifetime_data)
        )
        fresh_regression.set_params(x0)
        config = FitConfig(x0)
        config.scipy_minimize_options["bounds"] = kwargs.get(
            "bounds", get_regression_params_bounds(fresh_regression)
        )
        config.scipy_minimize_options["method"] = kwargs.get("method", "L-BFGS-B")
        config.covariance_method = kwargs.get(
            "covariance_method",
            "2point" if isinstance(fresh_regression.baseline, Gamma) else "cs",
        )
        return LifetimeLikelihood(fresh_regression, lifetime_data, config)

    def fit(
        self,
        time: onp.Array1D[np.float64] | onp.Array[tuple[int, Literal[2]], np.float64],
        covar: onp.Array1D[np.float64] | Sequence[onp.Array1D[np.float64]],
        event: onp.Array1D[np.bool_] | None = None,
        entry: onp.Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> Self:
        if not isinstance(covar, Sequence):
            covar = (covar,)
        optimizer = self.init_likelihood(
            time, args=covar, event=event, entry=entry, **kwargs
        )
        self.fitting_results = optimizer.optimize()
        self.covar_effect.set_params([0.0] * len(covar))  # modify nb coef inplace
        self.set_params(self.fitting_results.optimal_params)

        return self


def init_regression_params_from_lifetimes(
    model: ParametricLifetimeRegression, data: LifetimeData
) -> onp.Array1D[np.float64]:
    param0 = np.zeros_like(model.get_params(), dtype=np.float64)
    param0[-model.baseline.get_params().size :] = init_distrib_params_from_lifetimes(
        model.baseline, data
    )
    return param0


def get_regression_params_bounds(model: ParametricLifetimeRegression) -> Bounds:
    nb_coefficients = model.covar_effect.get_params().size
    lb = np.concatenate(
        (
            np.full(nb_coefficients, -np.inf),
            get_distrib_params_bounds(
                model.baseline
            ).lb,  # baseline has _params_bounds according to typing
        )
    )
    ub = np.concatenate(
        (
            np.full(nb_coefficients, np.inf),
            get_distrib_params_bounds(model.baseline).ub,
        )
    )
    return Bounds(lb, ub)


@final
class ParametricProportionalHazard(ParametricLifetimeRegression):
    r"""
    Proportional Hazard regression.

    The cumulative hazard function :math:`H` is linked to the multiplier
    function :math:`g` by the relation:

    .. math::

        H(t, x) = g(\beta, x) H_0(t) = e^{\beta \cdot x} H_0(t)

    where :math:`x` is a vector of covariates, :math:`\beta` is the coefficient
    vector of the effect of covariates, :math:`H_0` is the baseline cumulative
    hazard function [1]_.

    |

    Parameters
    ----------
    baseline : FittableParametricLifetimeModel
        Any lifetime model that can be fitted.
    coefficients : tuple of floats (values can be None), default is (None,)
        Coefficients values of the covariate effects.

    Attributes
    ----------
    baseline : FittableParametricLifetimeModel
        The regression baseline model (lifetime model).
    covar_effect : _CovarEffect
        The regression covariate effect.
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.).
        If the model is not fitted, the value is None.

    References
    ----------
    .. [1] Sun, J. (2006). The statistical analysis of interval-censored failure
        time data (Vol. 3, No. 1). New York: springer.

    See Also
    --------
    regression.AFT : Accelerated Failure Time regression.

    """

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def hf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        return self.covar_effect.g(*covar) * self.baseline.hf(time)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def chf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        return self.covar_effect.g(*covar) * self.baseline.chf(time)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def ichf(
        self,
        cumulative_hazard_rate: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        return self.baseline.ichf(cumulative_hazard_rate / self.covar_effect.g(*covar))

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def dhf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> onp.ArrayND[np.float64]:
        return self.covar_effect.g(*covar) * self.baseline.dhf(time)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_hf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> onp.ArrayND[np.float64]:
        ndtime, *ndcovar = np.broadcast_arrays(time, *covar)
        u = self.baseline.hf(ndtime) * self.covar_effect.jac_g(
            *ndcovar
        )  # (nb_coef, ...)
        v = self.covar_effect.g(*ndcovar) * self.baseline.jac_hf(ndtime)  # (p, ...)
        return np.concatenate((u, v), axis=0)  # (p + nb_coef, ...)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_chf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> onp.ArrayND[np.float64]:
        ndtime, *ndcovar = np.broadcast_arrays(time, *covar)
        u = self.baseline.chf(ndtime) * self.covar_effect.jac_g(
            *ndcovar
        )  # (nb_coef, ...)
        v = self.covar_effect.g(*ndcovar) * self.baseline.jac_chf(ndtime)
        return np.concatenate((u, v), axis=0)  # (p + nb_coef, ...)

    @override
    def __repr__(self) -> str:
        return f"ParametricProportionalHazard({self.baseline!r}, {self.get_coefficients().tolist()!r})"


@final
class ParametricAcceleratedFailureTime(ParametricLifetimeRegression):
    r"""
    Accelerated failure time regression.

    The cumulative hazard function :math:`H` is linked to the multiplier
    function :math:`g` by the relation:

    .. math::

        H(t, x) = H_0\left(\dfrac{t}{g(\beta, x)}\right) = H_0(t e^{- \beta
        \cdot x})

    where :math:`x` is a vector of covariates, :math:`\beta` is the coefficient
    vector of the effect of covariates, :math:`H_0` is the baseline cumulative
    hazard function [1]_.

    |

    Parameters
    ----------
    baseline : FittableParametricLifetimeModel
        Any lifetime model that can be fitted.
    coefficients : tuple of floats (values can be None), default is (None,)
        Coefficients values of the covariate effects.

    Attributes
    ----------
    baseline : FittableParametricLifetimeModel
        The regression baseline model (lifetime model).
    covar_effect : _CovarEffect
        The regression covariate effect.
    fitting_results : FittingResults, default is None
        An object containing fitting results (AIC, BIC, etc.).
        If the model is not fitted, the value is None.

    References
    ----------
    .. [1] Kalbfleisch, J. D., & Prentice, R. L. (2011). The statistical
        analysis of failure time data. John Wiley & Sons.

    See Also
    --------
    regression.ProportionalHazard : proportional hazard regression
    """

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def hf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        t0 = time / self.covar_effect.g(*covar)
        return self.baseline.hf(t0) / self.covar_effect.g(*covar)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def chf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        t0 = time / self.covar_effect.g(*covar)
        return self.baseline.chf(t0)

    @override
    @document_args(
        base_cls=ParametricLifetimeRegression, args_docstring=_covar_docstring
    )
    def ichf(
        self,
        cumulative_hazard_rate: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> Float64_ND:
        return self.covar_effect.g(*covar) * self.baseline.ichf(cumulative_hazard_rate)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def dhf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> onp.ArrayND[np.float64]:
        t0 = time / self.covar_effect.g(*covar)
        return self.baseline.dhf(t0) / self.covar_effect.g(*covar) ** 2

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_hf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> onp.ArrayND[np.float64]:
        ndtime, *ndcovar = np.broadcast_arrays(time, *covar)
        g = self.covar_effect.g(*ndcovar)
        jac_g = self.covar_effect.jac_g(*ndcovar)  # (nb_coef, ...)
        t0 = ndtime / g
        baseline_jac_hf_t0 = self.baseline.jac_hf(t0)  # (p, ...)
        baseline_hf_t0 = self.baseline.hf(t0)
        baseline_dhf_t0 = self.baseline.dhf(t0)
        return np.concatenate(
            (
                -jac_g
                / g**2
                * (
                    baseline_hf_t0 + t0 * baseline_dhf_t0
                ),  # (nb_coef, ...) necessary to concatenate
                baseline_jac_hf_t0 / g,  # (p, ...)
            ),
            axis=0,
        )  # (p + nb_coef, ...)

    @override
    @document_args(
        base_cls=FittableParametricLifetimeModel, args_docstring=_covar_docstring
    )
    def jac_chf(
        self,
        time: CoercibleFloat64_ND,
        *covar: CoercibleFloat64_ND,
    ) -> onp.ArrayND[np.float64]:
        ndtime, *ndcovar = np.broadcast_arrays(time, *covar)
        g = self.covar_effect.g(*ndcovar)
        jac_g = self.covar_effect.jac_g(*ndcovar)  # (nb_coef, ...)
        t0 = ndtime / g
        baseline_jac_chf_t0 = self.baseline.jac_chf(t0)  # (p, ...)
        baseline_hf_t0 = self.baseline.hf(t0)
        return np.concatenate(
            (
                -jac_g / g * t0 * baseline_hf_t0,  #  (nb_coef, ...)
                baseline_jac_chf_t0,  # (p, ...)
            ),
            axis=0,
        )  # (p + nb_coef, ...)

    @override
    def __repr__(self) -> str:
        return f"ParametricAcceleratedFailureTime({self.baseline!r}, {self.get_coefficients().tolist()!r})"
