import numpy as np
import pytest

from assetlife._base import ParametricModel


class ModelA(ParametricModel):
    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)


class ModelB(ParametricModel):
    baseline: ModelA

    def __init__(self, model: ModelA, *coef: float) -> None:
        super().__init__(*coef)
        self.baseline = model


def test_model_composition() -> None:
    model_a = ModelA(1, 2)
    assert model_a.get_params() == pytest.approx(np.array([1, 2], dtype=np.float64))

    model_b = ModelB(model_a, 3, 4, 5)
    assert model_b.get_params() == pytest.approx(
        np.array([3, 4, 5, 1, 2], dtype=np.float64)
    )

    model_a.set_params(np.array([2, 3]))
    assert model_a.get_params() == pytest.approx(np.array([2, 3], dtype=np.float64))

    model_b.set_params(np.array([2, 3, 4, 5, 6]))
    assert model_b.get_params() == pytest.approx(
        np.array([2, 3, 4, 5, 6], dtype=np.float64)
    )
    assert model_b.baseline.get_params() == pytest.approx(
        np.array([5, 6], dtype=np.float64)
    )

    model_b.baseline.set_params(np.array([1, 2]))
    assert model_b.get_params() == pytest.approx(
        np.array([2, 3, 4, 1, 2], dtype=np.float64)
    )
