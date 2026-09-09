import numpy as np
import pytest
from optype.numpy import Array1D

from assetlife.lifetime_models import ParametricLifetimeModel
from assetlife.policies import OneCycleRunToFailurePolicy, RunToFailurePolicy


@pytest.fixture(params=[OneCycleRunToFailurePolicy, RunToFailurePolicy])
def policy(
    request: pytest.FixtureRequest, lifetime_model: ParametricLifetimeModel[()]
) -> OneCycleRunToFailurePolicy | RunToFailurePolicy:
    return request.param(lifetime_model)


def test_expected_equivalent_annual_cost(
    policy: OneCycleRunToFailurePolicy | RunToFailurePolicy,
    cf: Array1D[np.float64],
    discounting_rate: float,
):
    qa = policy.asymptotic_expected_equivalent_annual_cost(
        cf=cf, discounting_rate=discounting_rate
    )
    assert qa.shape == cf.shape

    nb_steps = 2000
    timeline, q = policy.expected_equivalent_annual_cost(
        400, nb_steps=nb_steps, cf=cf, discounting_rate=discounting_rate
    )
    assert timeline.shape == (nb_steps,)
    assert q.shape == timeline.shape + qa.shape
    np.testing.assert_allclose(q[-1], qa, rtol=1e-1)
