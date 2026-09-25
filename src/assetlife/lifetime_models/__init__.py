"""
``assetlife.lifetime_model``
=========================

The lifetime_model module exposes various stochastic models to modelize
lifetime data. Internal operations are computed using NumPy and Scipy.

- NumPy: https://github.com/numpy/numpy
- Scipy: https://github.com/scipy/scipy

Objects present in assetlife.lifetime_model are listed below.

Lifetime distributions
----------------------

    Exponential
    Weibull
    Gompertz
    Gamma
    LogLogistic
    MinimumDistribution
    EquilibriumDistribution


Lifetime regressions
--------------------

    ParametricProportionalHazard
    ParametricAcceleratedFailureTime

    
Semiparametric lifetime regression
----------------------------------

    SemiParametricProportionalHazard


Nonparametric models
--------------------

    KaplanMeier
    ECDF
    NelsonAalen


Init parameters methods
-----------------------

    init_distrib_params_from_lifetimes
    init_regression_params_from_lifetimes
"""

from ._base import (
    FittableParametricLifetimeModel,
    ParametricLifetimeModel,
)
from ._distributions import (
    Exponential,
    Gamma,
    Gompertz,
    LifetimeDistribution,
    LogLogistic,
    Weibull,
    init_distrib_params_from_lifetimes
)
from ._equilibrium_distribution import EquilibriumDistribution
from ._minimum_distribution import MinimumDistribution
from ._non_parametric_models import ECDF, KaplanMeier, NelsonAalen
from ._parametric_regressions import (
    LinearCovarEffect,
    ParametricAcceleratedFailureTime,
    ParametricLifetimeRegression,
    ParametricProportionalHazard,
    init_regression_params_from_lifetimes
)
from ._semi_parametric_regressions import SemiParametricProportionalHazard

__all__: list[str] = []
__all__ += [
    "FittableParametricLifetimeModel",
    "ParametricLifetimeModel",
]
__all__ += [
    "EquilibriumDistribution",
    "Exponential",
    "Gamma",
    "Gompertz",
    "LifetimeDistribution",
    "LogLogistic",
    "Weibull",
]
__all__ += ["EquilibriumDistribution"]
__all__ += ["MinimumDistribution"]
__all__ += [
    "LinearCovarEffect",
    "ParametricAcceleratedFailureTime",
    "ParametricLifetimeRegression",
    "ParametricProportionalHazard",
]
__all__ += ["ECDF", "KaplanMeier", "NelsonAalen"]
__all__ += ["SemiParametricProportionalHazard"]
__all__ += ["init_distrib_params_from_lifetimes","init_regression_params_from_lifetimes"]