from collections.abc import Mapping, Sequence
from typing import Any, Literal, final, override

import narwhals as nw
import numpy as np
import optype.numpy as onp
from narwhals.typing import IntoDataFrame

from ._base import FittableParametricLifetimeModel
from assetlife._optimization import FitConfig, MaximumLikelihoodOptimizer
from assetlife.typing import CoercibleFloat64_ND

type _ColumnArray[T: np.generic] = onp.Array[tuple[int, Literal[1]], T]


def _to_column_array[T: np.generic](
    df: nw.DataFrame[Any], name: str, dtype: type[T]
) -> onp.Array2D[T]:
    """Column ``name`` as an ``(n, 1)`` array."""
    return df.select(nw.col(name)).to_numpy().astype(dtype)


class LifetimeData:
    """Observation schema for lifetimes.

    Used to compute the likelihood.

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
          ``[x, np.inf]`` and a left-censored lifetime as ``[0., x]``. In this
          format, ``event`` must not be given.

        An optional ``entry`` column of floats holds left truncation values
        (``0.`` if missing). Other columns are ignored, except covariates
        selected with ``covar_columns``.
    columns : mapping of str to str, optional
        Column names of the lifetime fields in ``dataset``, as
        ``{field: column_name}``, e.g. ``{"time": "duration", "event": "failed"}``.
        Valid fields are ``"time"``, ``"time_inf"``, ``"time_sup"``, ``"event"``
        and ``"entry"``. Fields not given are looked up under their own name.
    covar_columns : sequence of str, default ()
        Names of the covariate columns in ``dataset``, used by regression
        models. Each covariate must be a column of floats.

    Raises
    ------
    TypeError
        If ``dataset`` is not an eager dataframe.
    ValueError
        If several fields map to the same column, if lifetimes are not given
        either as ``time`` or as both ``time_inf`` and ``time_sup``, or if
        ``event`` is given with interval-censored lifetimes.

    Examples
    --------
    Right-censored and left-truncated lifetimes:

    >>> import numpy as np
    >>> import polars as pl
    >>> from assetlife.lifetime_models._base import LifetimeData
    >>> df = pl.DataFrame({
    ...     "time": [1.0, 2.0, 3.0, 4.0],
    ...     "event": [True, False, True, True],
    ...     "entry": [0.0, 0.0, 1.5, 0.0],
    ... })
    >>> data = LifetimeData(df)
    >>> data.nb_observations
    4
    >>> print(data.complete_time.ravel())
    [1. 3. 4.]
    >>> print(data.censored_time.ravel())
    [2.]
    >>> print(data.left_truncations.ravel())
    [1.5]

    Columns with other names are mapped with ``columns``:

    >>> df = pl.DataFrame({"duration": [5.0, 8.0], "failed": [True, False]})
    >>> data = LifetimeData(df, columns={"time": "duration", "event": "failed"})
    >>> print(data.complete_time.ravel())
    [5.]
    >>> print(data.censored_time.ravel())
    [8.]

    Interval-censored lifetimes:

    >>> df = pl.DataFrame({
    ...     "time_inf": [2.0, 0.0, 4.0],
    ...     "time_sup": [2.0, 3.0, np.inf],
    ... })
    >>> data = LifetimeData(df)
    >>> print(data.complete_time.ravel())
    [2.]
    >>> print(data.censored_time)
    [[ 0.  3.]
     [ 4. inf]]

    Covariates for regression models:

    >>> df = pl.DataFrame({
    ...     "time": [1.0, 2.0, 3.0],
    ...     "event": [True, False, True],
    ...     "pHCl": [0.5, 0.7, 0.9],
    ...     "HNO3": [1.0, 1.1, 1.2],
    ... })
    >>> data = LifetimeData(df, covar_columns=["pHCl", "HNO3"])
    >>> len(data.complete_time_covars)
    2
    >>> print(data.complete_time_covars[0].ravel())
    [0.5 0.9]
    >>> print(data.censored_time_covars[1].ravel())
    [1.1]
    """

    nb_observations: int
    nb_covars: int
    complete_time: _ColumnArray[np.float64]
    censored_time: (
        _ColumnArray[np.float64] | onp.Array[tuple[int, Literal[2]], np.float64]
    )
    left_truncations: _ColumnArray[np.float64]
    complete_time_covars: tuple[_ColumnArray[np.float64], ...]
    censored_time_covars: tuple[_ColumnArray[np.float64], ...]
    left_truncations_covars: tuple[_ColumnArray[np.float64], ...]

    def __init__(
        self,
        dataset: IntoDataFrame,
        *,
        columns: Mapping[str, str] | None = None,
        covar_columns: Sequence[str] = (),
    ) -> None:
        try:
            df = nw.from_native(dataset, eager_only=True)
        except TypeError:
            msg = f"Expected a DataFrame, got {type(dataset).__name__}"
            raise TypeError(msg) from None
        columns = columns or {}
        if len(set(columns.values())) != len(columns):
            msg = f"Several fields map to the same column: {dict(columns)}"
            raise ValueError(msg)
        df = df.rename({column: field for field, column in columns.items()})
        names = set(df.columns)

        bounds = {"time_inf", "time_sup"} & names
        if ("time" in names) == bool(bounds) or len(bounds) == 1:
            msg = "Expected either a 'time' column, or 'time_inf' and 'time_sup'"
            raise ValueError(msg)
        if bounds and "event" in names:
            msg = "'event' must not be given with interval-censored lifetimes"
            raise ValueError(msg)

        n = len(df)
        entry = (
            _to_column_array(df, "entry", np.float64)
            if "entry" in names
            else np.zeros((n, 1), dtype=np.float64)
        )
        covars = tuple(_to_column_array(df, name, np.float64) for name in covar_columns)
        self.nb_covars = len(covars)

        if bounds:
            lower = _to_column_array(df, "time_inf", np.float64)
            upper = _to_column_array(df, "time_sup", np.float64)
            complete = (lower == upper).ravel()
            self.complete_time = upper[complete]
            self.censored_time = np.column_stack((lower, upper))[~complete]
        else:
            time = _to_column_array(df, "time", np.float64)
            complete = (
                _to_column_array(df, "event", np.bool_).ravel()
                if "event" in names
                else np.ones(n, dtype=np.bool_)
            )
            self.complete_time = time[complete]
            self.censored_time = time[~complete]

        truncated = (entry != 0).ravel()
        self.nb_observations = n
        self.left_truncations = entry[truncated]
        self.complete_time_covars = tuple(covar[complete] for covar in covars)
        self.censored_time_covars = tuple(covar[~complete] for covar in covars)
        self.left_truncations_covars = tuple(covar[truncated] for covar in covars)


@final
class LifetimeLikelihood(
    MaximumLikelihoodOptimizer[
        FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]], LifetimeData
    ]
):
    """
    Maximum likelihood estimator from lifetime data.

    Parameters
    ----------
    model : FittableParametricLifetimeModel
        Model with initialized parameters.
    data : LifetimeData
        Preprocessed lifetime observations.
    config : FitConfig
        Configuration used by the optimizer.

    Attributes
    ----------
    model : FittableParametricLifetimeModel
        Model used by the likelihood.
    data : LifetimeData
        Preprocessed lifetime observations.
    config : FitConfig
        Configuration used by the optimizer.
    """

    data: LifetimeData

    def __init__(
        self,
        model: FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]],
        data: LifetimeData,
        config: FitConfig,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log

    @property
    @override
    def nb_observations(self) -> int:
        return self.data.nb_observations

    @override
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        self.model.set_params(params)
        return (
            self._complete_time_contrib()
            + self._censored_time_contrib()
            + self._left_truncations_contrib()
        )

    def jac_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array1D[np.float64]:
        """
        Jacobian of the negative log likelihood.

        The Jacobian is computed with respect to parameters.

        Parameters
        ----------
        params : 1d array of floats
            Parameter values.

        Returns
        -------
        out : 1d array of floats
        """
        self.model.set_params(params)
        return (
            self._jac_complete_time_contrib()
            + self._jac_censored_time_contrib()
            + self._jac_left_truncations_contrib()
        )

    def _complete_time_contrib(self) -> float:
        if self.data.complete_time.size == 0:
            return 0.0
        return -np.sum(
            np.log(
                self.model.pdf(self.data.complete_time, *self.data.complete_time_covars)
            )
        )

    def _jac_complete_time_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.complete_time.size == 0:
            return np.zeros_like(self.model.get_params())
        jac = -self.model.jac_pdf(
            self.data.complete_time, *self.data.complete_time_covars
        ) / self.model.pdf(self.data.complete_time, *self.data.complete_time_covars)

        return np.sum(jac, axis=(1, 2))

    def _censored_time_contrib(self) -> float:
        if self.data.censored_time.size == 0:
            return 0.0
        if self.data.censored_time.shape[-1] > 1:
            # interval censored time
            return np.sum(
                -np.log(
                    10**-10
                    + self.model.cdf(
                        self.data.censored_time[:, 1], *self.data.censored_time_covars
                    )
                    - self.model.cdf(
                        self.data.censored_time[:, 0], *self.data.censored_time_covars
                    )
                ),
            )
        # right censored time
        return np.sum(
            self.model.chf(self.data.censored_time, *self.data.censored_time_covars)
        )

    def _jac_censored_time_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.censored_time.size == 0:
            return np.zeros_like(self.model.get_params())
        if self.data.censored_time.shape[-1] > 1:
            # interval censored time
            jac_interval_censored = (
                self.model.jac_sf(
                    self.data.censored_time[:, 1], *self.data.censored_time_covars
                )
                - self.model.jac_sf(
                    self.data.censored_time[:, 0], *self.data.censored_time_covars
                )
            ) / (
                10**-10
                + self.model.cdf(
                    self.data.censored_time[:, 1], *self.data.censored_time_covars
                )
                - self.model.cdf(
                    self.data.censored_time[:, 0], *self.data.censored_time_covars
                )
            )

            return np.sum(jac_interval_censored, axis=(1, 2))
        # right censored time
        return np.sum(
            self.model.jac_chf(
                self.data.censored_time, *self.data.censored_time_covars
            ),
            axis=(1, 2),
        )

    def _left_truncations_contrib(self) -> float:
        if self.data.left_truncations.size == 0:
            return 0.0
        return -np.sum(
            self.model.chf(
                self.data.left_truncations, *self.data.left_truncations_covars
            )
        )

    def _jac_left_truncations_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.left_truncations.size == 0:
            return np.zeros_like(self.model.get_params())
        jac = -self.model.jac_chf(
            self.data.left_truncations, *self.data.left_truncations_covars
        )
        return np.sum(jac, axis=(1, 2))
