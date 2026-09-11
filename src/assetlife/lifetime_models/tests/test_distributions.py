import numpy as np
import optype.numpy as onp
import pytest
from numpy.testing import assert_allclose

from .utils import generate_shapes, shape_id
from assetlife.lifetime_models import LifetimeDistribution
from assetlife.typing import CoercibleFloat64_ND, Float64_ND


@pytest.mark.parametrize(
    "method",
    ["sf", "hf", "chf", "cdf", "pdf", "dhf", "isf", "ichf", "ppf"],
)
@pytest.mark.parametrize(
    "shape",
    generate_shapes(max_ndim=2, nb_args=1),
    ids=shape_id,
)
def test_prob_func_broadcasting(
    distribution: LifetimeDistribution,
    method: str,
    shape: tuple[int] | tuple[int, int],
):
    assert getattr(distribution, method)(np.ones(shape) * 0.5).shape == shape


@pytest.mark.parametrize(
    "method",
    ["sf", "hf", "chf", "cdf", "pdf", "isf", "ichf", "ppf"],
)
@pytest.mark.parametrize(
    "a0_shape, shape",
    generate_shapes(max_ndim=2, nb_args=2),
    ids=shape_id,
)
def test_a0_prob_func_broadcasting(
    distribution: LifetimeDistribution,
    method: str,
    a0_shape: tuple[int] | tuple[int, int],
    shape: tuple[int] | tuple[int, int],
):
    assert getattr(distribution.apply_condition(a0=np.ones(a0_shape) * 0.3), method)(
        np.ones(shape) * 0.5
    ).shape == np.broadcast_shapes(a0_shape, shape)


@pytest.mark.parametrize(
    "method",
    ["sf", "hf", "chf", "cdf", "pdf", "isf", "ichf", "ppf"],
)
@pytest.mark.parametrize(
    "ar_shape, shape",
    generate_shapes(max_ndim=2, nb_args=2),
    ids=shape_id,
)
def test_ar_prob_func_broadcasting(
    distribution: LifetimeDistribution,
    method: str,
    ar_shape: tuple[int] | tuple[int, int],
    shape: tuple[int] | tuple[int, int],
):
    assert getattr(distribution.apply_condition(a0=np.ones(ar_shape) * 0.3), method)(
        np.ones(shape) * 0.5
    ).shape == np.broadcast_shapes(ar_shape, shape)


@pytest.mark.parametrize(
    "method",
    ["jac_sf", "jac_chf", "jac_cdf", "jac_pdf"],
)
@pytest.mark.parametrize(
    "time_shape",
    generate_shapes(max_ndim=2, nb_args=1),
    ids=shape_id,
)
def test_jac_func_broadcasting(
    distribution: LifetimeDistribution,
    method: str,
    time_shape: tuple[int] | tuple[int, int],
):
    assert (
        getattr(distribution, method)(np.ones(time_shape)).shape
        == (distribution.get_params().size,) + time_shape
    )


@pytest.mark.parametrize(
    "size",
    generate_shapes(max_ndim=2, nb_args=1),
    ids=shape_id,
)
def test_rvs_broadcasting(
    distribution: LifetimeDistribution,
    size: tuple[int] | tuple[int, int],
):
    assert distribution.rvs(size, seed=1).shape == size


def test_mean_broadcasting(distribution: LifetimeDistribution):
    assert distribution.mean().shape == ()


def test_var_broadcasting(distribution: LifetimeDistribution):
    assert distribution.var().shape == ()


def test_sf(distribution: LifetimeDistribution):
    assert_allclose(
        distribution.sf(np.full((3, 5), distribution.median())),
        np.full((3, 5), 0.5),
        rtol=1e-3,
    )


def test_isf(distribution: LifetimeDistribution):
    assert_allclose(
        distribution.isf(np.full((3, 5), 0.5)),
        np.full((3, 5), distribution.median()),
    )


def test_fit(
    distribution: LifetimeDistribution,
    power_transformer_data: onp.Array1D[np.void],
):
    expected_params = distribution.get_params().copy()
    distribution = distribution.fit(
        power_transformer_data["time"],
        event=power_transformer_data["event"],
        entry=power_transformer_data["entry"],
    )
    assert_allclose(distribution.get_params(), expected_params, rtol=1e-3)


def test_negative_log(
    distribution: LifetimeDistribution,
    power_transformer_data: onp.Array1D[np.void],
):
    likelihood = distribution.init_likelihood(
        power_transformer_data["time"],
        event=power_transformer_data["event"],
        entry=power_transformer_data["entry"],
    )
    assert isinstance(likelihood.negative_log(distribution.get_params()), float)


def test_jac_negative_log(
    distribution: LifetimeDistribution,
    power_transformer_data: onp.Array1D[np.void],
):
    likelihood = distribution.init_likelihood(
        power_transformer_data["time"],
        event=power_transformer_data["event"],
        entry=power_transformer_data["entry"],
    )
    params = distribution.get_params()
    assert likelihood.jac_negative_log(params).shape == (params.size,)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    generate_shapes(max_ndim=2, nb_args=2),
    ids=shape_id,
)
def test_ls_integrate_broadcasting(
    distribution: LifetimeDistribution,
    a_shape: tuple[int] | tuple[int, int],
    b_shape: tuple[int] | tuple[int, int],
):
    a = np.ones(a_shape) * 2.0
    b = np.ones(b_shape) * 8.0
    integration = distribution.ls_integrate(
        np.ones_like,
        a,
        b,
        deg=100,
    )
    assert integration.shape == np.broadcast_shapes(a_shape, b_shape)


def test_ls_integrate(distribution: LifetimeDistribution):
    a = np.ones((3, 5)) * 2
    b = np.ones((3, 5)) * 8
    integration = distribution.ls_integrate(
        np.ones_like,
        a,
        b,
        deg=100,
    )
    assert_allclose(
        integration,
        distribution.cdf(b) - distribution.cdf(a),
    )

    def func(x: CoercibleFloat64_ND) -> Float64_ND:
        return np.float64(x)

    integration = distribution.ls_integrate(
        func,
        np.zeros_like(a),
        np.full_like(b, np.inf),
        deg=100,
    )
    assert_allclose(
        integration, np.full(integration.shape, distribution.mean()), rtol=1e-3
    )


@pytest.mark.parametrize(
    "a0_shape, a_shape, b_shape",
    generate_shapes(max_ndim=2, nb_args=3),
    ids=shape_id,
)
def test_a0_ls_integrate_broadcasting(
    distribution: LifetimeDistribution,
    a0_shape: tuple[int] | tuple[int, int],
    a_shape: tuple[int] | tuple[int, int],
    b_shape: tuple[int] | tuple[int, int],
):
    integration = distribution.apply_condition(a0=np.ones(a0_shape) * 0.3).ls_integrate(
        np.ones_like,
        np.ones(a_shape) * 2.0,
        np.ones(b_shape) * 8.0,
    )
    assert integration.shape == np.broadcast_shapes(a0_shape, a_shape, b_shape)


@pytest.mark.parametrize(
    "ar_shape, a_shape, b_shape",
    generate_shapes(max_ndim=2, nb_args=3),
    ids=shape_id,
)
def test_ar_ls_integrate_broadcasting(
    distribution: LifetimeDistribution,
    ar_shape: tuple[int] | tuple[int, int],
    a_shape: tuple[int] | tuple[int, int],
    b_shape: tuple[int] | tuple[int, int],
):
    integration = distribution.apply_condition(ar=np.ones(ar_shape) * 3.0).ls_integrate(
        np.ones_like,
        np.ones(a_shape) * 2.0,
        np.ones(b_shape) * 8.0,
    )
    assert integration.shape == np.broadcast_shapes(ar_shape, a_shape, b_shape)


def test_moment_broadcasting(distribution: LifetimeDistribution):
    assert distribution.moment(n=1).shape == ()
    assert distribution.moment(n=2).shape == ()
