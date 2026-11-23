"""Lifetime model of a series structure of identical components."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self, final, override

import numpy as np
import optype.numpy as onp
from narwhals.typing import IntoDataFrame
from optype.numpy import ArrayND
from scipy.optimize import Bounds

from ._base import (
    FittableParametricLifetimeModel,
)
from ._datasets_validation import validate_lifetime_data
from ._likelihoods import LifetimeLikelihood
from assetlife._typing import CoercibleFloat64_ND, CovarTs


@final
class MinimumDistribution(FittableParametricLifetimeModel[*CovarTs]):
    r"""
    Series structure of n identical and independent components.

    The structure fails at the first failure of its components. Its hazard
    function is:

    .. math::

        h(t) = n \cdot h_0(t)

    where :math:`h_0` is the hazard function of the components.

    Parameters
    ----------
    baseline : FittableParametricLifetimeModel
        Lifetime model of the components, a distribution or a regression.
    n : int
        Number of components.

    Examples
    --------
    Survival function of a structure of 3 identical and independent
    components:

    .. code-block::

        model = MinimumDistribution(Weibull(2, 0.05), 3)
        t = np.arange(0, 10, 0.1)
        model.sf(t)
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
        return self.n * self.baseline.jac_hf(time, *args)

    @override
    def get_params_bounds(self) -> Bounds:
        return self.baseline.get_params_bounds()

    @override
    def _init_params(self, time: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]:
        return self.baseline._init_params(time)  # ruff: ignore[private-member-access]

    @override
    def _set_nb_covars(self, nb_covars: int) -> None:
        self.baseline._set_nb_covars(nb_covars)  # ruff: ignore[private-member-access]

    @override
    def _default_covariance_method(self) -> Literal["cs", "2point"]:
        return self.baseline._default_covariance_method()  # ruff: ignore[private-member-access]

    def fit(
        self,
        dataset: IntoDataFrame,
        covar_column_names: Sequence[str] = (),
        *,
        field_column_names: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Estimate the model parameters from lifetime data.

        Parameters
        ----------
        dataset : DataFrame
            Lifetime observations, one row per observation, as any eager dataframe
            supported by narwhals (polars, pandas, pyarrow...). Lifetimes are given
            either as:

            - a ``time`` column of floats: complete or right-censored lifetimes. An
              optional ``event`` column of booleans tells whether each lifetime is a
              failure (``True``) or right-censored (``False``). If ``event`` is
              missing, all lifetimes are complete.
            - ``time_inf`` and ``time_sup`` columns of floats: interval-censored
              lifetimes ``[time_inf, time_sup]``. A complete lifetime is encoded as
              ``time_inf == time_sup``, a right-censored lifetime as
              ``[x, np.inf]`` and a left-censored lifetime as ``[entry, x]``. In
              this format, ``event`` must not be given.

            An optional ``entry`` column of floats holds left truncation values
            (``0.`` if missing). Other columns are ignored, except covariates
            selected with ``covar_column_names``.
        covar_column_names : sequence of str, default ()
            Names of the covariate columns in ``dataset``, used if ``baseline``
            is a regression model. Each covariate must be a column of floats.
        field_column_names : mapping of str to str, optional
            Column name of each lifetime field in ``dataset``, as
            ``{field: column_name}``, e.g. ``{"time": "duration", "event": "failed"}``.
            Fields are the lifetime variables ``"time"``, ``"time_inf"``,
            ``"time_sup"``, ``"event"`` and ``"entry"``. Fields not given are looked
            up under their own name.
        **kwargs
            Extra keyword arguments passed to `scipy.optimize.minimize
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_,
            which searches for the parameters minimizing the negative
            log-likelihood.

            The additional keyword ``covariance_method`` controls how the
            covariance matrix of the estimated parameters is computed. Allowed
            values are ``"cs"``, ``"2point"``, ``"exact"`` or ``False``. If
            ``False``, the covariance is not estimated. If ``"exact"``, ``hess``
            must also be provided. If not set, the model's default method is
            used.

        Returns
        -------
        self : MinimumDistribution
            The fitted model. Estimated parameters are set in place.
        """
        data = validate_lifetime_data(
            dataset, covar_column_names, field_column_names=field_column_names
        )
        return LifetimeLikelihood.from_model(self, data, **kwargs).fit(self)
