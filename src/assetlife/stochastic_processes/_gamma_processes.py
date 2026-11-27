import numpy as np
from scipy.special import gammaincc

from assetlife._base import ParametricModel
from assetlife._typing import (
    CoercibleFloat64_ND,
    Float64_ND,
)


class GammaProcess(ParametricModel):
    def __init__(
        self, rate: float | None, shape_rate: float | None, shape_power: float | None
    ) -> None:
        super().__init__(rate, shape_rate, shape_power)

    def shape_function(self, time: CoercibleFloat64_ND) -> Float64_ND:
        _, shape_rate, shape_power = self.get_params()
        return shape_rate * time**shape_power

    def crossing_probability_function(
        self, time: CoercibleFloat64_ND, threshold: CoercibleFloat64_ND
    ) -> Float64_ND:
        rate, _, _ = self.get_params()
        return gammaincc(self.shape_function(time), rate * threshold)

    def conditional_crossing_probability_function(
        self,
        current_state: CoercibleFloat64_ND,
        time: CoercibleFloat64_ND,
        threshold: CoercibleFloat64_ND,
    ) -> Float64_ND:
        return self.crossing_probability_function(
            time, (np.float64(threshold) - np.float64(current_state))
        )
