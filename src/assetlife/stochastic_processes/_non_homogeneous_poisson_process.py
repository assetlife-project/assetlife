"""Non-homogeneous Poisson process models."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import field
from typing import Any, Generic, Self, no_type_check

import numpy as np
import optype.numpy as onp
from typing_extensions import override

from assetlife.base import FittingResults, ParametricModel
from assetlife.lifetime_models import (
    FittableParametricLifetimeModel,
    ParametricLifetimeModel,
)
from assetlife.typing import CoercibleFloat64_ND, CovarTs, Float64_ND


class NonHomogeneousPoissonProcess(ParametricModel, Generic[*CovarTs]):
    """
    Non-homogeneous Poisson process.

    Parameters
    ----------
    lifetime_model : ParametricLifetimeModel
        Lifetime model defining the process intensity.
    """

    fitting_results: FittingResults | None
    lifetime_model: ParametricLifetimeModel[*CovarTs]  # not accurate is case of fit

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[*CovarTs],
    ):
        super().__init__()
        self.lifetime_model = lifetime_model

    def intensity(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        """
        The intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.lifetime_model.hf(time, *args)

    def cumulative_intensity(
        self,
        time: CoercibleFloat64_ND,
        *args: *CovarTs,
    ) -> Float64_ND:
        """
        The cumulative intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.lifetime_model.chf(time, *args)

    def freeze(self, *args: *CovarTs) -> FrozenNonHomogeneousPoissonProcess[*CovarTs]:
        """
        Return a process with additional arguments stored.

        Parameters
        ----------
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenNonHomogeneousPoissonProcess
        """
        return FrozenNonHomogeneousPoissonProcess(self, *args)

    def fit(
        self,
        ages_at_events: onp.Array1D[np.float64],
        events_assets_ids: Sequence[str],
        first_ages: onp.Array1D[np.float64] | None = None,
        last_ages: onp.Array1D[np.float64] | None = None,
        lifetime_model_args: onp.Array1D[Any]
        | onp.Array2D[Any]
        | tuple[onp.Array1D[Any] | onp.Array2D[Any], ...]
        | None = None,
        assets_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Estimate process parameters from recurrent failure data.

        Parameters
        ----------
        ages_at_events : 1d array of floats
            Ages of each asset when events occurred.
        events_assets_ids : sequence of hashable
            Asset ids corresponding to ``ages_at_events``.
        first_ages : 1d array of floats, optional
            Asset ages before the observation period. If set, ``assets_ids`` is
            required and must have the same length.
        last_ages : 1d array of floats, optional
            Asset ages at the end of the observation period. If set,
            ``assets_ids`` is required and must have the same length.
        lifetime_model_args : tuple of np.ndarray, optional
            Additional arguments needed by the lifetime model. If set,
            ``assets_ids`` is required. For 1d arrays, the size must equal the
            length of ``assets_ids``. For 2d arrays, the first axis length must
            equal the length of ``assets_ids``.
        assets_ids : sequence of hashable, optional
            Unique asset ids corresponding to values in ``first_ages``,
            ``last_ages`` and/or ``lifetime_model_args``.

        Returns
        -------
        Self
            The current object with estimated parameters set in place.

        Examples
        --------

        Ages of assets AB2 and CX13 at each event.

        >>> from assetlife.lifetime_models import Weibull
        >>> from assetlife.stochastic_processes import NonHomogeneousPoissonProcess
        >>> nhpp = NonHomogeneousPoissonProcess(Weibull())
        >>> nhpp.fit(
            np.array([11., 13., 21., 25., 27.]),
            ("AB2", "CX13", "AB2", "AB2", "CX13"),
        )

        With additional information and lifetime model args.

        >>> from assetlife.lifetime_models import ParametricProportionalHazard
        >>> nhpp = NonHomogeneousPoissonProcess(ParametricProportionalHazard())
        >>> nhpp.fit(
            np.array([11., 13., 21., 25., 27.]),
            ("AB2", "CX13", "AB2", "AB2", "CX13"),
            first_ages = np.array([10., 12.]),
            last_ages = np.array([35., 60.]),
            lifetime_model_args=(np.array([[1.2, 5.5], [37.2, 22.2]]),)
        )
        """
        warnings.warn(
            "Fit method of NHPP will change in a future release", DeprecationWarning
        )
        assert isinstance(self.lifetime_model, FittableParametricLifetimeModel)

        nhpp_data = NHPPData(
            ages_at_events,
            events_assets_ids,
            first_ages=first_ages,
            last_ages=last_ages,
            model_args=lifetime_model_args,
            assets_ids=assets_ids,
        )
        time, event, entry, args = nhpp_data.to_lifetime_data()
        optimizer = self.lifetime_model.init_likelihood(
            time, args, event, entry, **kwargs
        )
        fitting_results = optimizer.optimize()
        self.set_params(fitting_results.optimal_params)
        self.fitting_results = fitting_results
        return self


class FrozenNonHomogeneousPoissonProcess(
    NonHomogeneousPoissonProcess[()], Generic[*CovarTs]
):
    """Non-homogeneous Poisson process with additional arguments stored."""

    unfrozen: NonHomogeneousPoissonProcess[*CovarTs]
    args: tuple[*CovarTs]

    def __init__(
        self,
        nhpp: NonHomogeneousPoissonProcess[*CovarTs],
        *args: *CovarTs,
    ):
        super().__init__(nhpp.lifetime_model.freeze(*args))
        self.unfrozen = nhpp
        self.args = args

    @override
    def intensity(self, time: CoercibleFloat64_ND) -> Float64_ND:
        """
        The intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.lifetime_model.hf(time)

    @override
    def cumulative_intensity(self, time: CoercibleFloat64_ND) -> Float64_ND:
        """
        The cumulative intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.lifetime_model.chf(time)


class NHPPData:
    """Preprocessed recurrent event data for NHPP fitting."""

    ages_at_events: onp.Array1D[np.float64]
    events_assets_ids: onp.Array1D[np.uint32]
    first_ages: onp.Array1D[np.float64] | None
    last_ages: onp.Array1D[np.float64] | None
    model_args: (
        onp.Array1D[Any]
        | onp.Array2D[Any]
        | tuple[onp.Array1D[Any] | onp.Array2D[Any], ...]
        | None
    )
    assets_ids: onp.Array1D[np.uint32] | None

    first_age_index: onp.Array1D[np.int64] = field(repr=False, init=False)
    last_age_index: onp.Array1D[np.int64] = field(repr=False, init=False)

    def __init__(
        self,
        ages_at_events: onp.Array1D[np.float64],
        events_assets_ids: Sequence[str],
        first_ages: onp.Array1D[np.float64] | None = None,
        last_ages: onp.Array1D[np.float64] | None = None,
        model_args: onp.Array1D[Any]
        | onp.Array2D[Any]
        | tuple[onp.Array1D[Any] | onp.Array2D[Any], ...]
        | None = None,
        assets_ids: Sequence[str] | None = None,
    ) -> None:

        # convert inputs to arrays
        self.ages_at_events = np.asarray(ages_at_events, dtype=np.float64)
        self.events_assets_ids = np.unique(
            np.asarray(events_assets_ids), return_inverse=True
        )[1].astype(np.uint32)
        self.assets_ids = None
        if assets_ids is not None:
            self.assets_ids = np.unique(np.asarray(assets_ids), return_inverse=True)[
                1
            ].astype(np.uint32)
        self.first_ages = first_ages
        self.last_ages = last_ages
        self.model_args = model_args
        self._sanity_checks()

        # sort fields
        sort_ind = np.lexsort((self.ages_at_events, self.events_assets_ids))
        self.events_assets_ids = self.events_assets_ids[sort_ind]
        self.ages_at_events = self.ages_at_events[sort_ind]

        # number of age value per asset id
        nb_ages_per_asset = np.unique_counts(self.events_assets_ids).counts
        # index of the first ages and last ages in ages
        self.first_age_index = np.where(
            np.roll(self.events_assets_ids, 1) != self.events_assets_ids
        )[0]
        self.last_age_index = np.append(
            self.first_age_index[1:] - 1, len(self.events_assets_ids) - 1
        )

        if self.assets_ids is not None:
            # sort fields
            sort_ind = np.argsort(self.assets_ids)
            self.assets_ids = self.assets_ids[sort_ind]
            self.first_ages = (
                self.first_ages[sort_ind]
                if self.first_ages is not None
                else self.first_ages
            )
            self.last_ages = (
                self.last_ages[sort_ind]
                if self.last_ages is not None
                else self.last_ages
            )
            self.model_args = (
                tuple(arg[sort_ind] for arg in self.model_args)
                if self.model_args is not None
                else self.model_args
            )

            if self.first_ages is not None and np.any(
                self.ages_at_events[self.first_age_index]
                <= self.first_ages[nb_ages_per_asset != 0]
            ):
                raise ValueError(
                    "Each first_ages value must be lower than all of its corresponding ages values"
                )
            if self.last_ages is not None and np.any(
                self.ages_at_events[self.last_age_index]
                >= self.last_ages[nb_ages_per_asset != 0]
            ):
                raise ValueError(
                    "Each last_ages value must be greater than all of its corresponding ages values"
                )

    def _sanity_checks(self) -> None:
        # control shapes
        if self.events_assets_ids.ndim != 1:
            raise ValueError(
                "Invalid array shape for events_assets_ids. Expected 1d-array"
            )
        if self.ages_at_events.ndim != 1:
            raise ValueError("Invalid array shape for ages. Expected 1d-array")
        if len(self.events_assets_ids) != len(self.ages_at_events):
            raise ValueError(
                "Shape of events_assets_ids and ages must be equal. Expected equal length 1d-arrays"
            )
        if self.assets_ids is not None:
            if self.assets_ids.ndim != 1:
                raise ValueError(
                    "Invalid array shape for assets_ids. Expected 1d-array"
                )
            if self.first_ages is not None:
                if self.first_ages.ndim != 1:
                    raise ValueError(
                        "Invalid array shape for start_ages. Expected 1d-array"
                    )
                if len(self.first_ages) != len(self.assets_ids):
                    raise ValueError(
                        "Shape of assets_ids and start_ages must be equal. Expected equal length 1d-arrays"
                    )
            if self.last_ages is not None:
                if self.last_ages.ndim != 1:
                    raise ValueError(
                        "Invalid array shape for last_ages. Expected 1d-array"
                    )
                if len(self.last_ages) != len(self.assets_ids):
                    raise ValueError(
                        "Shape of assets_ids and last_ages must be equal. Expected equal length 1d-arrays"
                    )
            if bool(self.model_args):
                for arg in self.model_args:
                    arg = np.atleast_2d(np.asarray(arg, dtype=np.float64))
                    if arg.ndim > 2:
                        raise ValueError(
                            "Invalid arg shape in model_args. onp.Arrays must be 0, 1 or 2d"
                        )
                    try:
                        _ = arg.reshape((len(self.assets_ids), -1))
                    except ValueError as err:
                        raise ValueError(
                            """
                            Invalid arg shape in model_args. onp.Arrays must
                            coherent with the number of assets given by
                            assets_ids
                            """
                        ) from err
        else:
            if self.first_ages is not None:
                raise ValueError(
                    "If first_ages is given, corresponding asset ids must be given in assets_ids"
                )
            if self.last_ages is not None:
                raise ValueError(
                    "If last_ages is given, corresponding asset ids must be given in assets_ids"
                )
            if bool(self.model_args):
                raise ValueError(
                    "If model_args is given, corresponding asset ids must be given in assets_ids"
                )

    @no_type_check
    def to_lifetime_data(
        self,
    ) -> tuple[
        onp.Array1D[np.float64],
        onp.Array1D[np.bool_],
        onp.Array1D[np.float64],
        tuple[onp.Array1D[np.float64], ...],
    ]:
        """Return lifetime data arrays used by lifetime likelihood fitting."""
        event = np.ones_like(self.ages_at_events, dtype=np.bool_)
        # insert_index = np.cumsum(nb_ages_per_asset)
        # insert_index = last_age_index + 1
        if self.last_ages is not None:
            time = np.insert(
                self.ages_at_events, self.last_age_index + 1, self.last_ages
            )
            event = np.insert(event, self.last_age_index + 1, False)
            _ids = np.insert(
                self.events_assets_ids, self.last_age_index + 1, self.assets_ids
            )
            if self.first_ages is not None:
                entry = np.insert(
                    self.ages_at_events,
                    np.insert((self.last_age_index + 1)[:-1], 0, 0),
                    self.first_ages,
                )
            else:
                entry = np.insert(self.ages_at_events, self.first_age_index, 0.0)
        else:
            time = self.ages_at_events.copy()
            _ids = self.events_assets_ids.copy()
            if self.first_ages is not None:
                entry = np.roll(self.ages_at_events, 1)
                entry[self.first_age_index] = self.first_ages
            else:
                entry = np.roll(self.ages_at_events, 1)
                entry[self.first_age_index] = 0.0
        model_args = (
            tuple(np.take(arg, _ids) for arg in self.model_args)
            if self.model_args is not None
            else ()
        )
        return time, event, entry, model_args
