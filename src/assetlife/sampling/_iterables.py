"""Iterable objects for stochastic process sampling."""

from typing import final
from typing_extensions import override

from ._base import StochasticDataIterable, StochasticDataIterator
from ._iterators import (
    Kijima1ProcessIterator,
    Kijima2ProcessIterator,
    NonHomogeneousPoissonProcessIterator,
    RenewalProcessIterator,
    RenewalRewardProcessIterator,
)
from assetlife.stochastic_processes import (
    Kijima1Process,
    Kijima2Process,
    NonHomogeneousPoissonProcess,
    RenewalProcess,
    RenewalRewardProcess,
)
from assetlife.typing import CoercibleFloat64_1D, Seed


@final
class RenewalProcessIterable(StochasticDataIterable[RenewalProcess]):
    """Iterable sampler for renewal processes."""

    iterator_cls = RenewalProcessIterator


@final
class RenewalRewardProcessIterable(StochasticDataIterable[RenewalRewardProcess]):
    """Iterable sampler for renewal reward processes."""

    iterator_cls = RenewalRewardProcessIterator

    def __init__(
        self,
        process: RenewalRewardProcess,
        nb_samples: int,
        time_window: tuple[float, float],
        cf: CoercibleFloat64_1D,
        *,
        a0: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        cp: CoercibleFloat64_1D | None = None,
        cf1: CoercibleFloat64_1D | None = None,
        cp1: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        seed: Seed = None,
    ):
        super().__init__(process, nb_samples, time_window, a0=a0, ar=ar, seed=seed)
        self.cf = cf
        self.cp = cp
        self.cf1 = cf1
        self.cp1 = cp1
        self.discounting_rate = discounting_rate

    @override
    def __iter__(self) -> StochasticDataIterator[RenewalRewardProcess]:
        return RenewalRewardProcessIterator(
            self.process,
            self.nb_samples,
            self.time_window,
            self.cf,
            a0=self.a0,
            ar=self.ar,
            cp=self.cp,
            cf1=self.cf1,
            cp1=self.cp1,
            discounting_rate=self.discounting_rate,
            seed=self.seed,
        )


@final
class NonHomogeneousPoissonProcessIterable(
    StochasticDataIterable[NonHomogeneousPoissonProcess[()]]
):
    """Iterable sampler for non-homogeneous Poisson processes."""

    iterator_cls = NonHomogeneousPoissonProcessIterator


@final
class Kijima1ProcessIterable(StochasticDataIterable[Kijima1Process[()]]):
    """Iterable sampler for Kijima I processes."""

    iterator_cls = Kijima1ProcessIterator


@final
class Kijima2ProcessIterable(StochasticDataIterable[Kijima2Process[()]]):
    """Iterable sampler for Kijima II processes."""

    iterator_cls = Kijima2ProcessIterator
