from collections.abc import Mapping, Sequence
from typing import Any, Self, final, override

import numpy as np
import optype.numpy as onp
from narwhals.typing import IntoDataFrame
from optype.numpy import ArrayND
from scipy.optimize import Bounds

from ._base import (
    FittableParametricLifetimeModel,
)
from ._likelihoods import LifetimeData, LifetimeLikelihood
from assetlife.typing import CoercibleFloat64_ND, CovarTs


@final
class MinimumDistribution(FittableParametricLifetimeModel[*CovarTs]):
    r"""
    Series structure of n identical and independent components.

    The hazard function of the system is given by:

    .. math::

        h(t) = n \cdot  h_0(t)

    where :math:`h_0` is the baseline hazard function of the components.

    Parameters
    ----------
    baseline : lifetime distribution or regression
        Lifetime model.

    Examples
    --------

    Computing the survival (or reliability) function for 3 structures of 3,6 and
    9 identical and idependent components:

    .. code-block::

        model = MinimumDistribution(Weibull(2, 0.05))
        t = np.arange(0, 10, 0.1)
        n = np.array([3, 6, 9]).reshape(-1, 1)
        model.sf(t, n)
    """

    baseline: FittableParametricLifetimeModel[*CovarTs]
    n: int

    def __init__(
        self,
        baseline: FittableParametricLifetimeModel[*CovarTs],
        n: int,
    ) -> None:
        super().__init__()
        self.n = n
        self.baseline = baseline

    @override
    def sf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().sf(time, *args)

    @override
    def pdf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return super().pdf(time, *args)

    @override
    def hf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.n * self.baseline.hf(time, *args)

    @override
    def chf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.n * self.baseline.chf(time, *args)

    @override
    def ichf(
        self,
        cumulative_hazard_rate: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.baseline.ichf(cumulative_hazard_rate / self.n, *args)

    @override
    def dhf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> ArrayND[np.float64]:
        return self.n * self.baseline.dhf(time, *args)

    @override
    def jac_chf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> ArrayND[np.float64]:
        return self.n * self.baseline.jac_chf(time, *args)

    @override
    def jac_hf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> ArrayND[np.float64]:
        return self.n * self.baseline.jac_chf(time, *args)

    @override
    def get_params_bounds(self) -> Bounds:
        return self.baseline.get_params_bounds()

    @override
    def _init_params(self, data: LifetimeData) -> onp.Array1D[np.float64]:
        return self.baseline._init_params(data)  # ruff: ignore[private-member-access]

    @override
    def _init_likelihood(
        self,
        data: LifetimeData,
        **kwargs: Any,
    ) -> LifetimeLikelihood:
        likelihood = self.baseline._init_likelihood(data, **kwargs)  # ruff: ignore[private-member-access]
        likelihood.model = MinimumDistribution(likelihood.model, self.n)
        return likelihood

    def fit(
        self,
        data: IntoDataFrame,
        covar_columns: Sequence[str] = (),
        *,
        columns: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Self:
        return self._fit(
            LifetimeData(data, columns=columns, covar_columns=covar_columns), **kwargs
        )
