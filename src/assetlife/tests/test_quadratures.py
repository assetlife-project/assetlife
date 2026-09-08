from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from optype.numpy import Array1D, ArrayND

from assetlife.quadratures import laguerre_quadrature, legendre_quadrature
from assetlife.typing import CoercibleFloat64_ND, Float64_ND

M = 3  # nb assets
N = 10  # nb points


@pytest.fixture(
    params=[
        2.0 * np.ones((), dtype=np.float64),
        2.0 * np.ones((1,), dtype=np.float64),
        2.0 * np.ones((N,), dtype=np.float64),
        2.0 * np.ones((1, N), dtype=np.float64),
        2.0 * np.ones((M, 1), dtype=np.float64),
        2.0 * np.ones((M, N), dtype=np.float64),
    ],
    ids=lambda a: f"a:{a.shape}",
)
def lower_bound(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture(
    params=[
        8.0 * np.ones((), dtype=np.float64),
        8.0 * np.ones((1,), dtype=np.float64),
        8.0 * np.ones((N,), dtype=np.float64),
        8.0 * np.ones((1, N), dtype=np.float64),
        8.0 * np.ones((M, 1), dtype=np.float64),
        8.0 * np.ones((M, N), dtype=np.float64),
    ],
    ids=lambda b: f"b:{b.shape}",
)
def upper_bound(request: pytest.FixtureRequest) -> Any:
    return request.param


def f(x: CoercibleFloat64_ND) -> Float64_ND:
    return np.float64(x)


def g(x: CoercibleFloat64_ND, z: Array1D[np.float64]) -> Float64_ND:
    return np.float64(x) * z


def test_laguerre_quadrature(lower_bound: ArrayND[np.float64]):

    # integral_a^inf x*exp(-x)dx = (a + 1)*exp(-a)
    def expected_intg(a: CoercibleFloat64_ND) -> Float64_ND:
        return (a + 1) * np.exp(-np.float64(a))

    integration = laguerre_quadrature(f, lower_bound)
    assert integration.shape == np.broadcast_shapes(lower_bound.shape)
    assert_allclose(integration, expected_intg(lower_bound))


def test_laguerre_quadrature_with_args(lower_bound: ArrayND[np.float64]):
    # integral_a^inf x*exp(-x)dx = (a + 1)*exp(-a)
    def expected_intg(a: CoercibleFloat64_ND) -> Float64_ND:
        return (a + 1) * np.exp(-np.float64(a))

    z = np.full((N,), 3.0)
    lower_bound, _ = np.broadcast_arrays(lower_bound, z)
    integration = laguerre_quadrature(lambda x: g(x, z), lower_bound, args=(z,))
    assert integration.shape == np.broadcast_shapes(lower_bound.shape, z.shape)
    assert_allclose(integration, expected_intg(lower_bound) * z)


def test_legendre_quadrature(
    lower_bound: ArrayND[np.float64], upper_bound: ArrayND[np.float64]
):

    # integral_a^b xdx = (1/2)*(b^2 - a^2)
    def expected_intg(a: CoercibleFloat64_ND, b: CoercibleFloat64_ND) -> Float64_ND:
        return np.float64(0.5 * (b**2 - a**2))

    integration = legendre_quadrature(f, lower_bound, upper_bound)
    assert integration.shape == np.broadcast_shapes(
        lower_bound.shape, upper_bound.shape
    )
    assert_allclose(integration, expected_intg(lower_bound, upper_bound))


def test_legendre_quadrature_with_args(
    lower_bound: ArrayND[np.float64], upper_bound: ArrayND[np.float64]
):
    # integral_a^b xdx = (1/2)*(b^2 - a^2)
    def expected_intg(a: CoercibleFloat64_ND, b: CoercibleFloat64_ND) -> Float64_ND:
        return np.float64(0.5 * (b**2 - a**2))

    z = np.full((N,), 3.0)
    integration = legendre_quadrature(
        lambda x: g(x, z), lower_bound, upper_bound, args=(z,)
    )
    assert integration.shape == np.broadcast_shapes(
        lower_bound.shape, upper_bound.shape, z.shape
    )
    assert_allclose(integration, expected_intg(lower_bound, upper_bound) * z)
