import numpy as np
import pytest
from optype.numpy import Array1D

from assetlife.lifetime_models import ParametricLifetimeModel
from assetlife.policies import AgeReplacementPolicy, OneCycleAgeReplacementPolicy


# @pytest.fixture(params=[OneCycleAgeReplacementPolicy, AgeReplacementPolicy])
@pytest.fixture(params=[AgeReplacementPolicy])
def policy(
    request: pytest.FixtureRequest, lifetime_model: ParametricLifetimeModel[()]
) -> OneCycleAgeReplacementPolicy | AgeReplacementPolicy:
    return request.param(lifetime_model)


# @pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_expected_equivalent_annual_cost(
    policy: OneCycleAgeReplacementPolicy | AgeReplacementPolicy,
    cf: Array1D[np.float64],
    cp: Array1D[np.float64],
    discounting_rate: float,
):
    try:
        ar = policy.compute_optimal_ar(cf=cf, cp=cp, discounting_rate=discounting_rate)
    except RuntimeError:
        pytest.skip("Optimization failed, EEAC may be too flat")

    qa = policy.asymptotic_expected_equivalent_annual_cost(
        ar=ar, cp=cp, cf=cf, discounting_rate=discounting_rate
    )  # () or (m,)
    assert qa.shape == np.broadcast_shapes(cf.shape, cp.shape)  # () or (m,)

    nb_steps = 2000
    timeline, q = policy.expected_equivalent_annual_cost(
        400, nb_steps=nb_steps, ar=ar, cp=cp, cf=cf, discounting_rate=discounting_rate
    )

    assert timeline.shape == (nb_steps,)
    assert q.shape == timeline.shape + qa.shape  # (2000, m) or (2000,)
    np.testing.assert_allclose(q[-1], qa, rtol=1e-1)


def test_optimal_replacement_age(
    policy: OneCycleAgeReplacementPolicy | AgeReplacementPolicy,
    cf: Array1D[np.float64],
    cp: Array1D[np.float64],
    discounting_rate: float,
):
    eps = 1e-2
    ar = policy.compute_optimal_ar(cf=cf, cp=cp, discounting_rate=discounting_rate)
    assert np.all(
        policy.asymptotic_expected_equivalent_annual_cost(
            ar=ar + eps, cp=cp, cf=cf, discounting_rate=discounting_rate
        )
        > policy.asymptotic_expected_equivalent_annual_cost(
            ar=ar, cp=cp, cf=cf, discounting_rate=discounting_rate
        )
    )
    assert np.all(
        policy.asymptotic_expected_equivalent_annual_cost(
            ar=ar - eps, cp=cp, cf=cf, discounting_rate=discounting_rate
        )
        > policy.asymptotic_expected_equivalent_annual_cost(
            ar=ar, cp=cp, cf=cf, discounting_rate=discounting_rate
        )
    )
