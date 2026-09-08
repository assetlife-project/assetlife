from typing import final

import numpy as np
from optype.numpy import ArrayND
from scipy.optimize import newton
from typing_extensions import override

from assetlife.quadratures import legendre_quadrature
from assetlife.typing import CoercibleFloat64_ND, CovarTs

from ._base import ParametricLifetimeModel


@final
class EquilibriumDistribution(ParametricLifetimeModel[*CovarTs]):
    r"""
    Equilibrium distribution.

    The equilibirum distribution is the distribution that makes the renewal process
    stationnary.

    Parameters
    ----------
    baseline : any parametric lifetime model
        Lifetime model.

    References
    ----------
    .. [1] Ross, S. M. (1996). Stochastic stochastic_process. New York: Wiley.
    """

    baseline: ParametricLifetimeModel[*CovarTs]

    def __init__(
        self,
        baseline: ParametricLifetimeModel[*CovarTs],
    ):
        super().__init__()
        self.baseline = baseline

    @override
    def cdf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return legendre_quadrature(
            lambda x: np.asarray(self.baseline.sf(x, *args), dtype=float), 0, time
        ) / self.baseline.mean(*args)

    @override
    def sf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return 1 - self.cdf(time, *args)

    @override
    def pdf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.baseline.sf(time, *args) / self.baseline.mean(*args)

    @override
    def hf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return 1 / self.baseline.mrl(time, *args)

    @override
    def chf(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return -np.log(self.sf(time, *args))

    @override
    def isf(
        self,
        probability: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        def func(x: ArrayND[np.float64]) -> ArrayND[np.float64]:
            return np.asarray(self.sf(x, *args) - probability)

        return np.float64(
            newton(
                func,
                x0=np.asarray(self.baseline.isf(probability, *args)),
                args=args,
            )
        )

    @override
    def ichf(
        self,
        cumulative_hazard_rate: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> np.float64 | ArrayND[np.float64]:
        return self.isf(np.exp(-np.float64(cumulative_hazard_rate)), *args)
