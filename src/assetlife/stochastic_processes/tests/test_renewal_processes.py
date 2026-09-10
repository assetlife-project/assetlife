import numpy as np
import pytest

from assetlife.lifetime_models import (
    EquilibriumDistribution,
    LifetimeDistribution,
    ParametricLifetimeModel,
)
from assetlife.stochastic_processes import RenewalProcess, RenewalRewardProcess


@pytest.mark.parametrize("ar", [True, False])
def test_renewal_density(lifetime_model: ParametricLifetimeModel[()], ar: bool):
    model = lifetime_model.apply_condition(ar=lifetime_model.isf(0.75) if ar else None)
    model1 = EquilibriumDistribution(lifetime_model).apply_condition(
        ar=lifetime_model.isf(0.75) if ar else None
    )
    renewal_process = RenewalProcess(model, model1)
    timeline, renewal_density = renewal_process.renewal_density(100, 200)
    assert timeline.shape == (200,)
    assert (
        renewal_density.shape == (200,)
        if isinstance(lifetime_model, LifetimeDistribution)
        else (200, 3)
    )
    np.testing.assert_allclose(
        renewal_density[-1], 1 / lifetime_model.mean(), rtol=1e-3
    )


@pytest.mark.parametrize("ar", [True, False])
def test_expected_total_reward(lifetime_model: ParametricLifetimeModel[()], ar: bool):
    model = lifetime_model.apply_condition(ar=lifetime_model.isf(0.75) if ar else None)
    renewal_reward_process = RenewalRewardProcess(model)
    timeline_m, m = renewal_reward_process.renewal_function(100, 200)
    assert timeline_m.shape == (200,)
    assert (
        m.shape == (200,)
        if isinstance(lifetime_model, LifetimeDistribution)
        else (200, 3)
    )

    timeline_z, z = renewal_reward_process.expected_total_reward(100, 200, cf=1.0)
    assert timeline_z.shape == (200,)
    assert (
        z.shape == (200,)
        if isinstance(lifetime_model, LifetimeDistribution)
        else (200, 3)
    )
    np.testing.assert_allclose(m, z, rtol=1e-3)


@pytest.mark.parametrize("ar", [True, False])
def test_renewal_reward_process_vec(
    lifetime_model: ParametricLifetimeModel[()], ar: bool
):
    cf0 = 1
    n = 3
    cf = cf0 / n
    model = lifetime_model.apply_condition(ar=lifetime_model.isf(0.75) if ar else None)

    rrp0 = RenewalRewardProcess(model)
    rrp = RenewalRewardProcess(model)

    timeline_z, z = rrp.expected_total_reward(
        100, 200, cf=np.full((n,), cf), discounting_rate=0.04
    )  # (3, nb_steps)
    assert timeline_z.shape == (200,)
    assert z.shape == (200, n)
    timeline_z0, z0 = rrp0.expected_total_reward(
        100, 200, cf=cf0, discounting_rate=0.04
    )  # (nb_steps,)
    assert timeline_z0.shape == (200,)
    assert (
        z0.shape == (200,)
        if isinstance(lifetime_model, LifetimeDistribution)
        else (200, 3)
    )
    assert z.shape == (200, n)
    if isinstance(lifetime_model, LifetimeDistribution):
        np.testing.assert_allclose(z0, z.sum(axis=1), rtol=1e-3)
    else:
        np.testing.assert_allclose(z0, n * z, rtol=1e-3)
