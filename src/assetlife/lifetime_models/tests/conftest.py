from typing import Any

import numpy as np
import optype.numpy as onp
import pytest

from assetlife.datasets import load_insulator_string, load_power_transformer
from assetlife.lifetime_models import (
    Exponential,
    Gamma,
    Gompertz,
    LogLogistic,
    ParametricAcceleratedFailureTime,
    ParametricProportionalHazard,
    Weibull,
)


@pytest.fixture
def power_transformer_data() -> onp.Array1D[np.void]:
    return load_power_transformer()


@pytest.fixture
def insulator_string_data() -> onp.Array1D[np.void]:
    return load_insulator_string()


DISTRIBUTIONS = [
    Exponential(0.00795203),
    Weibull(3.46597395, 0.01227849),
    Gompertz(0.00865741, 0.06062632),
    Gamma(5.3571091, 0.06622822),
    LogLogistic(3.92614064, 0.0133325),
]


COEFFICIENTS = (np.log(2), np.log(2))


REGRESSIONS = [ParametricProportionalHazard(d, COEFFICIENTS) for d in DISTRIBUTIONS] + [
    ParametricAcceleratedFailureTime(d, COEFFICIENTS) for d in DISTRIBUTIONS
]


@pytest.fixture(params=DISTRIBUTIONS, ids=[repr(d) for d in DISTRIBUTIONS])
def distribution(request: pytest.FixtureRequest) -> Any:
    yield request.param


@pytest.fixture(params=REGRESSIONS, ids=[repr(r) for r in REGRESSIONS])
def regression(request: pytest.FixtureRequest) -> Any:
    yield request.param
