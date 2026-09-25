"""Non-homogeneous Poisson process models."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import field
from typing import Any, Generic, Self, no_type_check
from typing_extensions import override

import numpy as np
import optype.numpy as onp

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
        ...     np.array([11.0, 13.0, 21.0, 25.0, 27.0]),
        ...     ("AB2", "CX13", "AB2", "AB2", "CX13"),
        ... )

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