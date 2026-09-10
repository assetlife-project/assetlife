"""Base classes for stochastic process sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import optype.numpy as onp
from numpy.lib import recfunctions as rfn
from typing_extensions import override

from assetlife.lifetime_models import ParametricLifetimeModel
from assetlife.stochastic_processes import (
    Kijima1Process,
    Kijima2Process,
    NonHomogeneousPoissonProcess,
    RenewalProcess,
    RenewalRewardProcess,
)
from assetlife.typing import CoercibleFloat64_1D, Seed

PT = TypeVar(
    "PT",
    RenewalProcess,
    RenewalRewardProcess,
    Kijima1Process[()],
    Kijima2Process[()],
    NonHomogeneousPoissonProcess[()],
)


class StochasticDataIterable(Iterable[onp.ArrayND[np.void]], Generic[PT], ABC):
    """Base iterable for stochastic process samples."""

    iterator_cls: type[StochasticDataIterator[PT]]
    process: PT
    nb_samples: int
    time_window: tuple[float, float]
    a0: CoercibleFloat64_1D | None
    ar: CoercibleFloat64_1D | None
    seed: Seed

    def __init__(
        self,
        process: PT,
        nb_samples: int,
        time_window: tuple[float, float],
        *,
        a0: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        seed: Seed = None,
    ):
        self.process = process

        t0, tf = time_window
        if t0 < 0 or tf < 0 or t0 > tf:
            raise ValueError(
                f"Incorrect time window. Got {time_window}.\n",
                "Values must be positive. ",
                "First value can't lower than second value.",
            )
        self.time_window = t0, tf

        self.a0 = a0
        self.ar = ar
        self.nb_samples = nb_samples
        self.seed = seed

    @override
    def __iter__(self) -> StochasticDataIterator[PT]:
        return self.iterator_cls(
            self.process,
            self.nb_samples,
            self.time_window,
            self.a0,
            self.ar,
            self.seed,
        )


class TimeWindowObserver:
    """Track sampled events inside an observation time window."""

    t0: float
    tf: float
    _timeline: onp.ArrayND[np.float64]
    _crossed_t0_counter: onp.ArrayND[np.int64]
    _crossed_tf_counter: onp.ArrayND[np.int64]

    def __init__(self, t0: float, tf: float, sample_shape: tuple[int, ...]):
        self.t0, self.tf = t0, tf
        self._timeline = np.zeros(sample_shape)
        self._crossed_t0_counter = np.zeros(sample_shape, dtype=np.int64)
        self._crossed_tf_counter = np.zeros(sample_shape, dtype=np.int64)

    @property
    def timeline(self) -> onp.ArrayND[np.float64]:
        """Current simulated timeline."""
        return self._timeline

    def add_to_timeline(self, time: onp.ArrayND[np.float64]) -> None:
        """Add sampled times to the current timeline."""
        self._timeline += time
        # update counters
        self._crossed_t0_counter[self.timeline > self.t0] += 1
        self._crossed_tf_counter[self.timeline > self.tf] += 1
        self._timeline[self.just_crossed_tf] = self.tf

    @property
    def just_crossed_t0(self):
        """Whether samples have just crossed the start of the window."""
        return self._crossed_t0_counter == 1

    @property
    def just_crossed_tf(self):
        """Whether samples have just crossed the end of the window."""
        return self._crossed_tf_counter == 1

    @property
    def observed_step(self):
        """Whether samples are observed in the time window."""
        return np.logical_and(
            self._crossed_t0_counter >= 1, self._crossed_tf_counter <= 1
        )

    @property
    def all_finished(self):
        """Whether all samples have reached the end of the window."""
        return np.all(self._crossed_tf_counter >= 1)


@dataclass
class SampleStep:
    """One simulated step before conversion to structured arrays."""

    residual_time: onp.ArrayND[np.float64]
    event: onp.ArrayND[np.bool_]
    entry: onp.ArrayND[np.float64]

    def apply_time_window(self, time_window: TimeWindowObserver) -> None:
        """Adjust the step to the observation time window."""
        last_date = time_window.timeline - self.residual_time
        installation_date = np.where(last_date < 0, 0, last_date) - self.entry

        self.entry = np.where(
            time_window.just_crossed_t0,
            time_window.t0 - installation_date,
            self.entry,
        )

        self.residual_time = np.where(
            time_window.just_crossed_t0,
            time_window.timeline - time_window.t0,
            self.residual_time,
        )

        self.residual_time = np.where(
            time_window.just_crossed_tf,
            self.residual_time - (time_window.timeline - time_window.tf),
            self.residual_time,
        )
        self.event = np.where(time_window.just_crossed_tf, False, self.event)

    def as_struct(
        self,
        time_window: TimeWindowObserver,
    ) -> onp.ArrayND[np.void]:
        """Convert observed samples to a structured array."""

        observed_step = time_window.observed_step
        struct_arr = np.zeros(
            observed_step.sum(),
            dtype=np.dtype(
                [
                    ("timeline", np.float64),
                    ("time", np.float64),
                    ("event", np.bool_),
                    ("entry", np.float64),
                    ("id", np.int64),
                ]
            ),
        )

        struct_arr["timeline"] = time_window.timeline[observed_step]
        struct_arr["time"] = (
            self.residual_time[observed_step] + self.entry[observed_step]
        )
        struct_arr["event"] = self.event[observed_step]
        struct_arr["entry"] = self.entry[observed_step]
        struct_arr["id"] = np.where(observed_step)[0]

        return struct_arr

    @staticmethod
    def add_field(
        struct_arr: onp.ArrayND[np.void],
        new_label: str,
        new_values: onp.ArrayND[np.float64],
    ) -> onp.ArrayND[np.void]:
        """Add a float field to a structured array."""
        return rfn.append_fields(
            struct_arr,
            new_label,
            new_values,
            (np.dtype(np.float64),),
            usemask=False,
            asrecarray=False,
        )


def _get_rvs_shape(
    nb_samples: int,
    process_shape: tuple[int, ...],
    a0: CoercibleFloat64_1D | None,
    ar: CoercibleFloat64_1D | None,
) -> tuple[int, ...]:
    a0_shape = np.array(a0).shape
    ar_shape = np.array(ar).shape
    broadcasted_shape = np.broadcast_shapes(a0_shape, ar_shape, process_shape)
    return (nb_samples, *broadcasted_shape)


class StochasticDataIterator(Iterator[onp.ArrayND[np.void]], Generic[PT], ABC):
    """Base iterator for stochastic process samples."""

    process: PT
    ar: onp.Array1D[np.float64] | None
    sample_shape: tuple[int, ...]
    ages: onp.ArrayND[np.float64]
    time_window_observer: TimeWindowObserver
    replacement_cycle: int
    random_generator: np.random.Generator

    def __init__(
        self,
        process: PT,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: CoercibleFloat64_1D | None = None,
        ar: CoercibleFloat64_1D | None = None,
        seed: Seed = None,
    ) -> None:
        self.process = process
        self.sample_shape = _get_rvs_shape(
            nb_samples, process.lifetime_model.args_shape, a0, ar
        )
        self.ar = (
            np.broadcast_to(np.asarray(ar), self.sample_shape).copy().astype(np.float64)
            if ar is not None
            else None
        )
        self.ages = (
            np.broadcast_to(a0, self.sample_shape).copy().astype(np.float64)
            if a0 is not None
            else np.zeros(self.sample_shape)
        )

        self.time_window_observer = TimeWindowObserver(
            time_window[0], time_window[1], sample_shape=self.sample_shape
        )

        self.replacement_cycle = 0
        self.random_generator = np.random.default_rng(seed)

    @property
    @abstractmethod
    def lifetime_model(self) -> ParametricLifetimeModel[()]:
        """
        Get the appropriate lifetime model to use at each iteration of the process.
        """

    @abstractmethod
    def update_current_ages(
        self,
        time: onp.ArrayND[np.float64],
    ) -> None:
        """
        Update ages at each iteration of the process.
        """

    def generate_sample_step(self) -> SampleStep:
        """Generate one simulated sample step."""
        residual_time = np.asarray(
            self.lifetime_model.rvs(self.sample_shape, seed=self.random_generator)
        )
        event = np.ones_like(residual_time, dtype=np.bool_)
        entry = self.ages.copy()

        observed_time = residual_time.copy()

        if self.ar is not None:
            preventive_replacements = (self.ages + residual_time) >= self.ar
            observed_time[preventive_replacements] = (
                self.ar[preventive_replacements] - self.ages[preventive_replacements]
            )
            event = ~preventive_replacements

        self.time_window_observer.add_to_timeline(observed_time)
        sample_step = SampleStep(observed_time, event, entry)
        sample_step.apply_time_window(self.time_window_observer)

        self.update_current_ages(residual_time)
        self.replacement_cycle += 1
        return sample_step

    @override
    def __next__(self) -> onp.ArrayND[np.void]:
        """Return the next observed structured sample step."""
        if not self.time_window_observer.all_finished:
            struct_array = self.generate_sample_step().as_struct(
                self.time_window_observer
            )
            while (
                struct_array.size == 0
            ):  # skip cycles while arrays are empty (if t0 != 0.)
                struct_array = self.generate_sample_step().as_struct(
                    self.time_window_observer
                )
                if self.time_window_observer.all_finished and struct_array.size > 0:
                    return struct_array
            return struct_array
        raise StopIteration
