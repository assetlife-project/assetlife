from collections.abc import Mapping, Sequence
from typing import Any, Literal
from typing_extensions import TypeIs

import narwhals as nw
import numpy as np
import optype.numpy as onp
from narwhals.typing import IntoDataFrame

from assetlife._dataset_validation import (
    check_columns_exist,
    check_no_missing_values,
    check_not_empty,
    check_numeric,
    to_eager_df,
)

type _ColumnArray[T: np.generic] = onp.Array[tuple[int, Literal[1]], T]


def _is_column_array[T: np.generic](
    arr: onp.Array2D[T],
) -> TypeIs[_ColumnArray[T]]:
    """Whether a 2D array has a single column.

    Narrows the ``(int, int)`` shape returned by narwhals ``to_numpy`` to
    ``(int, 1)``, which an ``assert`` on ``arr.shape`` cannot do.
    """
    return arr.shape[1] == 1


def _to_column_array(df: nw.DataFrame[Any], name: str) -> _ColumnArray[np.float64]:
    """Column ``name`` as an ``(n, 1)`` array of floats.

    Raises
    ------
    ValueError
        If the selection does not give a single column (not expected, as
        exactly one column is selected).
    """
    arr = df.select(nw.col(name)).to_numpy().astype(np.float64)
    if not _is_column_array(arr):
        msg = f"Expected a single column for {name!r}, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


type _Mask = onp.Array1D[np.bool_]
type _IntervalArray = onp.Array[tuple[int, Literal[2]], np.float64]


def _read_entry(df: nw.DataFrame[Any]) -> _ColumnArray[np.float64]:
    """Left truncation values, ``0.`` if ``entry`` is not a column of ``df``.

    Raises
    ------
    ValueError
        If values are not finite and non-negative.
    """
    if "entry" not in df.columns:
        return np.zeros((len(df), 1), dtype=np.float64)
    entry = _to_column_array(df, "entry")
    if np.any(~np.isfinite(entry) | (entry < 0)):
        msg = "'entry' values must be finite and >= 0"
        raise ValueError(msg)
    return entry


def _read_covars(
    df: nw.DataFrame[Any], names: Sequence[str]
) -> tuple[_ColumnArray[np.float64], ...]:
    """Covariates ``names`` of ``df``.

    Raises
    ------
    ValueError
        If values are not finite.
    """
    covars = tuple(_to_column_array(df, name) for name in names)
    for name, covar in zip(names, covars, strict=True):
        if np.any(~np.isfinite(covar)):
            msg = f"{name!r} values must be finite"
            raise ValueError(msg)
    return covars


def _read_right_censored(
    df: nw.DataFrame[Any], entry: _ColumnArray[np.float64]
) -> tuple[_Mask, _ColumnArray[np.float64], _ColumnArray[np.float64]]:
    """Mask of complete lifetimes, complete and right-censored times.

    Raises
    ------
    ValueError
        If ``time`` is not finite and greater than ``entry``, or ``event`` is
        not 0 or 1.
    """
    time = _to_column_array(df, "time")
    if np.any(~np.isfinite(time) | (time <= entry)):
        msg = "'time' values must be finite and > 'entry'"
        raise ValueError(msg)
    if "event" not in df.columns:
        complete = np.ones(len(df), dtype=np.bool_)
        return complete, time, time[~complete]
    event = df.get_column("event").to_numpy()
    if np.any(~np.isin(event, (0, 1))):
        msg = "'event' values must be booleans or 0/1"
        raise ValueError(msg)
    complete = event.astype(np.bool_)
    return complete, time[complete], time[~complete]


def _read_interval_censored(
    df: nw.DataFrame[Any], entry: _ColumnArray[np.float64]
) -> tuple[_Mask, _ColumnArray[np.float64], _IntervalArray]:
    """Mask of complete lifetimes, complete times and censoring intervals.

    Raises
    ------
    ValueError
        If ``time_inf`` is not finite or ``entry <= time_inf <= time_sup`` does
        not hold, or ``time_sup`` is not greater than ``entry``.
    """
    lower = _to_column_array(df, "time_inf")
    upper = _to_column_array(df, "time_sup")
    if np.any(~np.isfinite(lower) | (lower < entry) | (lower > upper)):
        msg = "'time_inf' must be finite and entry <= time_inf <= time_sup"
        raise ValueError(msg)
    if np.any(upper <= entry):
        msg = "'time_sup' must be > 'entry'"
        raise ValueError(msg)
    complete = (lower == upper).ravel()
    return complete, upper[complete], np.column_stack((lower, upper))[~complete]


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
          ``[x, np.inf]`` and a left-censored lifetime as ``[entry, x]``. In
          this format, ``event`` must not be given.

        An optional ``entry`` column of floats holds left truncation values
        (``0.`` if missing). Other columns are ignored, except covariates
        selected with ``covar_column_names``.
    field_column_names : mapping of str to str, optional
        Column name of each lifetime field in ``dataset``, as
        ``{field: column_name}``, e.g. ``{"time": "duration", "event": "failed"}``.
        Fields are the lifetime variables ``"time"``, ``"time_inf"``,
        ``"time_sup"``, ``"event"`` and ``"entry"``. Fields not given are looked
        up under their own name.
    covar_column_names : sequence of str, default ()
        Names of the covariate columns in ``dataset``, used by regression
        models. Each covariate must be a column of floats.

    Raises
    ------
    TypeError
        If ``dataset`` is not an eager dataframe or a used column, except
        ``event``, is not numeric.
    ValueError
        If a used column is missing or has missing values, if ``dataset`` is
        empty, if ``event`` is given with interval-censored lifetimes, or if
        values are invalid: ``entry`` must be finite and non-negative, ``event``
        must be 0 or 1, ``time`` must be finite and greater than ``entry``,
        ``time_inf`` must be finite and satisfy ``entry <= time_inf <=
        time_sup``, ``time_sup`` must be greater than ``entry`` and covariates
        must be finite.
    """

    nb_observations: int
    nb_covars: int
    complete_time: _ColumnArray[np.float64]
    censored_time: _ColumnArray[np.float64] | _IntervalArray
    left_truncations: _ColumnArray[np.float64]
    complete_time_covars: tuple[_ColumnArray[np.float64], ...]
    censored_time_covars: tuple[_ColumnArray[np.float64], ...]
    left_truncations_covars: tuple[_ColumnArray[np.float64], ...]

    def __init__(
        self,
        dataset: IntoDataFrame,
        *,
        field_column_names: Mapping[str, str] | None = None,
        covar_column_names: Sequence[str] = (),
    ) -> None:
        df = to_eager_df(dataset)
        fields = field_column_names or {}
        df = df.rename({col: f for f, col in fields.items() if col in df.columns})
        covar_names = list(covar_column_names)
        interval = "time" not in df.columns
        if interval and "event" in df.columns:
            msg = "'event' must not be given with interval-censored lifetimes"
            raise ValueError(msg)
        columns = [
            *(["time_inf", "time_sup"] if interval else ["time"]),
            *(["entry"] if "entry" in df.columns else []),
            *covar_names,
        ]
        check_columns_exist(df, columns)
        check_not_empty(df)
        check_numeric(df, columns)
        check_no_missing_values(df, columns)

        entry = _read_entry(df)
        covars = _read_covars(df, covar_names)
        complete, self.complete_time, self.censored_time = (
            _read_interval_censored(df, entry)
            if interval
            else _read_right_censored(df, entry)
        )

        truncated = (entry != 0).ravel()
        self.nb_observations = len(df)
        self.nb_covars = len(covars)
        self.left_truncations = entry[truncated]
        self.complete_time_covars = tuple(covar[complete] for covar in covars)
        self.censored_time_covars = tuple(covar[~complete] for covar in covars)
        self.left_truncations_covars = tuple(covar[truncated] for covar in covars)
