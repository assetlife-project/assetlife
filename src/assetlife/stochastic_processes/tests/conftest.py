from typing import Any

import numpy as np
import pytest

from assetlife.lifetime_models import (
    Exponential,
    Gamma,
    Gompertz,
    LogLogistic,
    ParametricAcceleratedFailureTime,
    ParametricProportionalHazard,
    Weibull,
)

DISTRIBUTIONS = [
    Exponential(0.00795203),
    Weibull(3.46597395, 0.01227849),
    Gompertz(0.00865741, 0.06062632),
    Gamma(5.3571091, 0.06622822),
    LogLogistic(3.92614064, 0.0133325),
]

COEFFICIENTS = (np.log(2), np.log(2))
NB_ASSETS = 3

covar = [np.linspace(0.0, 0.5, num=NB_ASSETS)] * len(COEFFICIENTS)

REGRESSIONS = [
    ParametricProportionalHazard(d, COEFFICIENTS).freeze(*covar) for d in DISTRIBUTIONS
] + [
    ParametricAcceleratedFailureTime(d, COEFFICIENTS).freeze(*covar)
    for d in DISTRIBUTIONS
]


@pytest.fixture(
    params=DISTRIBUTIONS + REGRESSIONS,
    ids=[repr(model) for model in DISTRIBUTIONS + REGRESSIONS],
)
def lifetime_model(request: pytest.FixtureRequest) -> Any:
    yield request.param
