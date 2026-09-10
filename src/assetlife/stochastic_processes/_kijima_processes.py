"""Kijima virtual age processes."""

from __future__ import annotations

from typing import Generic

import numpy as np

from assetlife.base import ParametricModel
from assetlife.lifetime_models._base import ParametricLifetimeModel
from assetlife.typing import CovarTs


class Kijima1Process(ParametricModel, Generic[*CovarTs]):
    """
    Kijima I virtual age process.

    Parameters
    ----------
    lifetime_model : ParametricLifetimeModel
        Lifetime model used to generate interarrival times.
    q : float, default=np.nan
        Restoration factor.
    """

    lifetime_model: ParametricLifetimeModel[*CovarTs]

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[*CovarTs],
        q: float = np.nan,
    ):
        super().__init__(q)
        self.lifetime_model = lifetime_model

    @property
    def q(self) -> np.float64:
        """Restoration factor."""
        return self.get_params()[0]

    def freeze(self, *args: *CovarTs) -> FrozenKijima1Process[*CovarTs]:
        """
        Return a process with additional arguments stored.

        Parameters
        ----------
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenKijima1Process
        """
        return FrozenKijima1Process(self, *args)


class FrozenKijima1Process(Kijima1Process[()], Generic[*CovarTs]):
    """Kijima I process with additional arguments stored."""

    unfrozen: Kijima1Process[*CovarTs]
    args: tuple[*CovarTs]

    def __init__(
        self,
        kijima_process: Kijima1Process[*CovarTs],
        *args: *CovarTs,
    ) -> None:
        super().__init__(kijima_process.lifetime_model.freeze(*args), kijima_process.q)
        self.unfrozen = kijima_process
        self.args = args


class Kijima2Process(ParametricModel, Generic[*CovarTs]):
    """
    Kijima II virtual age process.

    Parameters
    ----------
    lifetime_model : ParametricLifetimeModel
        Lifetime model used to generate interarrival times.
    q : float, default=np.nan
        Restoration factor.
    """

    lifetime_model: ParametricLifetimeModel[*CovarTs]

    def __init__(
        self,
        lifetime_model: ParametricLifetimeModel[*CovarTs],
        q: float = np.nan,
    ):
        super().__init__(q)
        self.lifetime_model = lifetime_model

    @property
    def q(self) -> np.float64:
        """Restoration factor."""
        return self.get_params()[0]

    def freeze(self, *args: *CovarTs) -> FrozenKijima2Process[*CovarTs]:
        """
        Return a process with additional arguments stored.

        Parameters
        ----------
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenKijima2Process
        """
        return FrozenKijima2Process(self, *args)


class FrozenKijima2Process(Kijima2Process[()], Generic[*CovarTs]):
    """Kijima II process with additional arguments stored."""

    unfrozen: Kijima2Process[*CovarTs]
    args: tuple[*CovarTs]

    def __init__(
        self,
        kijima_process: Kijima2Process[*CovarTs],
        *args: *CovarTs,
    ) -> None:
        super().__init__(kijima_process.lifetime_model.freeze(*args), kijima_process.q)
        self.unfrozen = kijima_process
        self.args = args
