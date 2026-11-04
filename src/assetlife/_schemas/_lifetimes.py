from collections.abc import Mapping, Sequence
from typing import Any, Literal

import narwhals as nw
import numpy as np
import optype.numpy as onp
from narwhals.typing import IntoDataFrame

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
