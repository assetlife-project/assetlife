"""Iterators for stochastic process sampling."""

from abc import ABC, abstractmethod
from typing import TypeVar, final
from typing_extensions import override

import numpy as np
import optype.numpy as onp
from numpy.lib import recfunctions as rfn

from ._base import StochasticDataIterator
from assetlife._rewards import compute_rewards, discounting_factor
from assetlife.lifetime_models._base import (
    ParametricLifetimeModel,
)
from assetlife.stochastic_processes import (
    Kijima1Process,
    Kijima2Process,
    NonHomogeneousPoissonProcess,
    RenewalProcess,
)
from assetlife.stochastic_processes._renewal_processes import RenewalRewardProcess
from assetlife.typing import CoercibleFloat64_1D, Seed


@final
class RenewalProcessIterator(StochasticDataIterator[RenewalProcess]):
    """Iterator for renewal process samples."""

    @property
    @override
    def lifetime_model(self) -> ParametricLifetimeModel[()]:
        return (
            self.process.first_lifetime_model.apply_condition(a0=self.ages)
            if self.replacement_cycle == 0
            else self.process.lifetime_model
        )

    @override
    def update_current_ages(
        self,
        time: onp.ArrayND[np.float64],
    ) -> None:
        """
        In a Renewal process, ages are reset to 0 after each iteration.
        """
        self.ages = np.zeros(self.sample_shape, dtype=np.float64)


@final
class RenewalRewardProcessIterator(StochasticDataIterator[RenewalRewardProcess]):
    """Iterator for renewal reward process samples."""

    cf: CoercibleFloat64_1D
    cp: CoercibleFloat64_1D | None
    cf1: CoercibleFloat64_1D | None
    cp1: CoercibleFloat64_1D | None
    discounting_rate: float

    def __init__(
        self,
        process: RenewalRewardProcess,
        nb_samples: int,
        time_window: tuple[float, float],
        cf: CoercibleFloat64_1D,
        a0: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        cp: CoercibleFloat64_1D | None = None,
        cf1: CoercibleFloat64_1D | None = None,
        cp1: CoercibleFloat64_1D | None = None,
        discounting_rate: float = 0.0,
        seed: Seed = None,
    ) -> None:
        super().__init__(process, nb_samples, time_window, a0, ar, seed)
        self.cf = cf
        self.cp = cp
        self.cf1 = cf1
        self.cp1 = cp1
        self.discounting_rate = discounting_rate

    @property
    @override
    def lifetime_model(self) -> ParametricLifetimeModel[()]:
        return (
            self.process.first_lifetime_model.apply_condition(a0=self.ages)
            if self.replacement_cycle == 0
            else self.process.lifetime_model
        )

    @override
    def update_current_ages(
        self,
        time: onp.ArrayND[np.float64],
    ) -> None:
        """
        In a Renewal process, ages are reset to 0 after each iteration.
        """
        self.ages = np.zeros(self.sample_shape, dtype=np.float64)

    @override
    def __next__(self) -> onp.ArrayND[np.void]:
        struct_array = super().__next__()
        struct_array = rfn.append_fields(
            struct_array,
            "reward",
            np.asarray(
                compute_rewards(
                    struct_array["time"],
                    cp=self.cp1
                    if self.cp1 is not None and self.replacement_cycle == 0
                    else self.cp,
                    cf=self.cf1 is not None and self.replacement_cycle == 0,
                    ar=self.ar,
                )
            )
            * discounting_factor(struct_array["timeline"], rate=self.discounting_rate),
            (np.dtype(np.float64),),
            usemask=False,
            asrecarray=False,
        )
        return struct_array


@final
class NonHomogeneousPoissonProcessIterator(
    StochasticDataIterator[NonHomogeneousPoissonProcess[()]]
):
    """Iterator for non-homogeneous Poisson process samples."""

    ages: onp.ArrayND[np.float64]

    @property
    @override
    def lifetime_model(self) -> ParametricLifetimeModel[()]:
        # Apply a Left truncation based on current ages on the model
        # self.ages is always 1d in LeftTruncatedModel
        return self.process.lifetime_model.apply_condition(a0=self.ages)

    @override
    def update_current_ages(
        self,
        time: onp.ArrayND[np.float64],
    ):
        """
        In a NHPP, ages are reset to 0 only when a replacement is made
        """
        # Update asset ages
        self.ages += time

        if self.ar is not None:
            self.ages[self.ages >= self.ar] = 0


# narrowed typevar
KPT = TypeVar(
    "KPT",
    Kijima1Process[()],
    Kijima2Process[()],
)


class VirtualAgeProcessIterator(StochasticDataIterator[KPT], ABC):
    """Base iterator for virtual age process samples."""

    ages: onp.ArrayND[np.float64]
    virtual_ages: onp.ArrayND[np.float64]

    def __init__(
        self,
        process: KPT,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        seed: Seed = None,
    ) -> None:
        super().__init__(
            process,
            nb_samples,
            time_window,
            ar=ar,
            a0=a0,
            seed=seed,
        )
        self.virtual_ages = self.ages.copy()

    @property
    @override
    def lifetime_model(self) -> ParametricLifetimeModel[()]:
        # Apply a Left truncation based on current ages on the model
        # self.ages is always 1d in LeftTruncatedModel
        return self.process.lifetime_model.apply_condition(a0=self.virtual_ages)

    @override
    def __next__(self) -> onp.ArrayND[np.void]:
        struct_array = super().__next__()
        struct_array = rfn.append_fields(
            struct_array,
            "virtual_age",
            self.virtual_ages[self.time_window_observer.observed_step],
            (np.dtype(np.float64),),
            usemask=False,
            asrecarray=False,
        )
        return struct_array

    @abstractmethod
    def update_virtual_ages(self, time: onp.ArrayND[np.float64]):
        """Update virtual ages after a sampled time step."""

    @override
    def update_current_ages(
        self,
        time: onp.ArrayND[np.float64],
    ):
        """
        In a Kijima Process, the concept of age is virtual, and depends on the q parameter of the process
        """
        # Update asset ages
        self.update_virtual_ages(time)
        self.ages += time

        if self.ar is not None:
            self.virtual_ages[self.ages >= self.ar] = 0
            self.ages[self.ages >= self.ar] = 0


@final
class Kijima1ProcessIterator(VirtualAgeProcessIterator[Kijima1Process[()]]):
    """Iterator for Kijima I process samples."""

    virtual_ages: onp.ArrayND[np.float64]

    @override
    def update_virtual_ages(self, time: onp.ArrayND[np.float64]):
        self.virtual_ages += self.process.q * time


@final
class Kijima2ProcessIterator(VirtualAgeProcessIterator[Kijima2Process[()]]):
    """Iterator for Kijima II process samples."""

    virtual_ages: onp.ArrayND[np.float64]

    @override
    def update_virtual_ages(self, time: onp.ArrayND[np.float64]):
        self.virtual_ages = self.process.q * (self.virtual_ages + time)
