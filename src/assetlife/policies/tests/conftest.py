from typing import Any

import numpy as np
import optype.numpy as onp
import pytest

from assetlife.lifetime_models import (
    Gamma,
    Gompertz,
    LogLogistic,
    ParametricAcceleratedFailureTime,
    ParametricProportionalHazard,
    Weibull,
)

DISTRIBUTIONS = [
    # Exponential(0.00795203), Exponential does not work for chosen cp/cf
    Weibull(2, 0.05),
    Gompertz(0.01, 0.1),
    Gamma(2, 0.05),
    LogLogistic(3, 0.05),
    # Weibull(3.46597395, 0.01227849),
    # Gompertz(0.00865741, 0.06062632),
    # Gamma(5.3571091, 0.06622822),
    # LogLogistic(3.92614064, 0.0133325),
]


COEFFICIENTS = (np.log(2), np.log(2))


NB_ASSETS = 3

covars = [np.array([0.0, 0.2, 0.4]), np.array([0.1, 0.3, 0.5])]
# covars = [np.linspace(0.0, 0.5, num=NB_ASSETS)] * len(COEFFICIENTS)

REGRESSIONS = [
    ParametricProportionalHazard(d, COEFFICIENTS).freeze(*covars) for d in DISTRIBUTIONS
] + [
    ParametricAcceleratedFailureTime(d, COEFFICIENTS).freeze(*covars)
    for d in DISTRIBUTIONS
]


@pytest.fixture(
    params=DISTRIBUTIONS + REGRESSIONS,
    ids=[repr(model) for model in DISTRIBUTIONS + REGRESSIONS],
)
def lifetime_model(request: pytest.FixtureRequest) -> Any:
    yield request.param


# @pytest.fixture(params=[0.0, 0.04], ids=lambda rate: f"discounting_rate:{rate}")
@pytest.fixture(params=[0.0], ids=lambda rate: f"discounting_rate:{rate}")
def discounting_rate(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture
def cp() -> Any:
    return np.ones((NB_ASSETS,), dtype=np.float64)


@pytest.fixture
def cf(cp: onp.Array1D[np.float64]) -> Any:
    return cp + np.array([5, 10, 20], dtype=np.float64)
