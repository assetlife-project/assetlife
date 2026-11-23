"""Validation of lifetime datasets into the containers used by lifetime models.

Each ``validate_*`` function checks a dataframe and returns a container of numpy
arrays: ``LifetimeData`` for parametric models, ``CoxData`` for the Cox
semi-parametric regression and ``NonParametricData`` for the Kaplan-Meier and
Nelson-Aalen estimators. ``validate_complete_time``, used by the empirical
cumulative distribution function, returns the times as a 1d array. Containers
hold arrays only and know nothing about dataframes.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from typing_extensions import TypeIs

import narwhals as nw
import numpy as np
import optype.numpy as onp
from narwhals.typing import IntoDataFrame

from assetlife._dataframe_validation import (
    check_columns_exist,
    check_no_missing_values,
    check_not_empty,
    check_numeric,
    to_eager_df,
)

type _ColumnArray[T: np.generic] = onp.Array[tuple[int, Literal[1]], T]
type _Mask = onp.Array1D[np.bool_]
type _IntervalArray = onp.Array[tuple[int, Literal[2]], np.float64]


# Datasets containers


@dataclass(frozen=True)
class LifetimeData:
    """Lifetime observations, as used by the likelihood of parametric models.

    Built by ``validate_lifetime_data``.

    Attributes
    ----------
    nb_observations : int
        Number of observations.
    nb_covars : int
        Number of covariates.
    complete_time : np.ndarray
        Complete lifetimes, of shape ``(n_complete, 1)``.
    censored_time : np.ndarray
        Right-censored lifetimes, of shape ``(n_censored, 1)``, or censoring
        intervals ``[time_inf, time_sup]``, of shape ``(n_censored, 2)``.
    left_truncations : np.ndarray
        Non-zero left truncation values, of shape ``(n_truncated, 1)``.
    complete_time_covars : tuple of np.ndarray
        Covariates of the complete lifetimes, each of shape ``(n_complete, 1)``.
    censored_time_covars : tuple of np.ndarray
        Covariates of the censored lifetimes, each of shape ``(n_censored, 1)``.
    left_truncations_covars : tuple of np.ndarray
        Covariates of the left truncated lifetimes, each of shape
        ``(n_truncated, 1)``.
    """

    nb_observations: int
    nb_covars: int
    complete_time: _ColumnArray[np.float64]
    censored_time: _ColumnArray[np.float64] | _IntervalArray
    left_truncations: _ColumnArray[np.float64]
    complete_time_covars: tuple[_ColumnArray[np.float64], ...]
    censored_time_covars: tuple[_ColumnArray[np.float64], ...]
    left_truncations_covars: tuple[_ColumnArray[np.float64], ...]


@dataclass(frozen=True)
class CoxData:
    """Lifetime observations, as used by the Cox semi-parametric regression.

    Built by ``validate_cox_data``. Used to compute the partial likelihoods and
    the baseline estimations.

    Attributes
    ----------
    nb_observations : int
        Number of observations.
    nb_covars : int
        Number of covariates.
    covar : tuple of np.ndarray
        Covariates of every observation, each of shape ``(n, 1)``.
    ordered_event_time : np.ndarray
        Sorted distinct failure times, of shape ``(m,)``.
    event_count : np.ndarray
        Number of failures at each ordered event time, of shape ``(m,)``.
    risk_set : np.ndarray
        Mask of observations at risk at each ordered event time
        (``entry < t <= time``), of shape ``(m, n)``.
    death_set : np.ndarray
        Mask of observations failing at each ordered event time, of shape
        ``(m, n)``.
    ordered_event_covar : tuple of np.ndarray
        Covariates of the first failure at each ordered event time, each of
        shape ``(m, 1)``.
    """

    nb_observations: int
    nb_covars: int
    covar: tuple[_ColumnArray[np.float64], ...]
    ordered_event_time: onp.Array1D[np.float64]
    event_count: onp.Array1D[np.int64]
    risk_set: onp.Array2D[np.bool_]
    death_set: onp.Array2D[np.bool_]
    ordered_event_covar: tuple[_ColumnArray[np.float64], ...]


@dataclass(frozen=True)
class NonParametricData:
    """Numbers at risk and of failures, as used by non-parametric models.

    Built by ``validate_non_parametric_data``. Used by the Kaplan-Meier and
    Nelson-Aalen estimators.

    Attributes
    ----------
    timeline : np.ndarray
        Sorted distinct observed times, failures and censorings, of shape
        ``(m,)``.
    nb_at_risk : np.ndarray
        Number of observations at risk at each time ``t`` of the timeline, i.e.
        such that ``entry < t <= time``, of shape ``(m,)``.
    nb_failures : np.ndarray
        Number of failures at each time of the timeline, of shape ``(m,)``.
    """

    timeline: onp.Array1D[np.float64]
    nb_at_risk: onp.Array1D[np.int64]
    nb_failures: onp.Array1D[np.int64]


def observed_time(data: LifetimeData) -> onp.Array1D[np.float64]:
    """Finite observed times of ``data``, flattened, to initialize parameters.

    Complete times, right-censored times and finite bounds of censoring
    intervals. Infinite upper bounds (right-censored lifetimes given as
    ``[x, inf]``) are dropped, as they would make statistics like the mean or
    the standard deviation infinite or NaN.
    """
    time = np.concatenate((data.complete_time.ravel(), data.censored_time.ravel()))
    return time[np.isfinite(time)]


# Column readers: read one lifetime field of a checked dataframe


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


def _read_time(
    df: nw.DataFrame[Any], entry: _ColumnArray[np.float64]
) -> _ColumnArray[np.float64]:
    """Observed times ``time`` of ``df``.

    Raises
    ------
    ValueError
        If ``time`` is not finite and greater than ``entry``.
    """
    time = _to_column_array(df, "time")
    if np.any(~np.isfinite(time) | (time <= entry)):
        msg = "'time' values must be finite and > 'entry'"
        raise ValueError(msg)
    return time


def _read_event(df: nw.DataFrame[Any]) -> _Mask:
    """Mask of complete lifetimes, all True if ``event`` is not a column of ``df``.

    Raises
    ------
    ValueError
        If ``event`` is not 0 or 1.
    """
    if "event" not in df.columns:
        return np.ones(len(df), dtype=np.bool_)
    event = df.get_column("event").to_numpy()
    if np.any(~np.isin(event, (0, 1))):
        msg = "'event' values must be booleans or 0/1"
        raise ValueError(msg)
    return event.astype(np.bool_)


def _read_right_censored(
    df: nw.DataFrame[Any], entry: _ColumnArray[np.float64]
) -> tuple[_Mask, _ColumnArray[np.float64], _ColumnArray[np.float64]]:
    """Mask of complete lifetimes, complete and right-censored times."""
    time = _read_time(df, entry)
    complete = _read_event(df)
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


def _read_flat_right_censored(
    df: nw.DataFrame[Any],
) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], _Mask]:
    """Entries, times and mask of complete lifetimes, as 1d arrays."""
    entry = _read_entry(df)
    time = _read_time(df, entry)
    return entry.ravel(), time.ravel(), _read_event(df)


# Dataframe preparation


def _rename_fields(
    dataset: IntoDataFrame, field_column_names: Mapping[str, str] | None
) -> nw.DataFrame[Any]:
    """``dataset`` as an eager dataframe, with columns renamed to their field."""
    df = to_eager_df(dataset)
    fields = field_column_names or {}
    return df.rename({col: f for f, col in fields.items() if col in df.columns})


def _check_columns(df: nw.DataFrame[Any], columns: Sequence[str]) -> None:
    """Check that ``columns`` exist, are numeric, have no missing values and
    that ``df`` has rows."""
    check_columns_exist(df, columns)
    check_not_empty(df)
    check_numeric(df, columns)
    check_no_missing_values(df, columns)


def _entry_column(df: nw.DataFrame[Any]) -> list[str]:
    """``["entry"]`` if ``entry`` is a column of ``df``, else ``[]``."""
    return ["entry"] if "entry" in df.columns else []


# Validation functions


def validate_lifetime_data(
    dataset: IntoDataFrame,
    covar_column_names: Sequence[str] = (),
    *,
    field_column_names: Mapping[str, str] | None = None,
) -> LifetimeData:
    """Validate lifetime observations for parametric models.

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
        Names of the covariate columns in ``dataset``, used by regression
        models. Each covariate must be a column of floats.
    field_column_names : mapping of str to str, optional
        Column name of each lifetime field in ``dataset``, as
        ``{field: column_name}``, e.g. ``{"time": "duration", "event": "failed"}``.
        Fields are the lifetime variables ``"time"``, ``"time_inf"``,
        ``"time_sup"``, ``"event"`` and ``"entry"``. Fields not given are looked
        up under their own name.

    Returns
    -------
    LifetimeData

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
    df = _rename_fields(dataset, field_column_names)
    covar_names = list(covar_column_names)
    interval = "time" not in df.columns
    if interval and "event" in df.columns:
        msg = "'event' must not be given with interval-censored lifetimes"
        raise ValueError(msg)
    time_columns = ["time_inf", "time_sup"] if interval else ["time"]
    _check_columns(df, [*time_columns, *_entry_column(df), *covar_names])

    entry = _read_entry(df)
    covars = _read_covars(df, covar_names)
    complete, complete_time, censored_time = (
        _read_interval_censored(df, entry)
        if interval
        else _read_right_censored(df, entry)
    )
    truncated = (entry != 0).ravel()
    return LifetimeData(
        nb_observations=len(df),
        nb_covars=len(covars),
        complete_time=complete_time,
        censored_time=censored_time,
        left_truncations=entry[truncated],
        complete_time_covars=tuple(covar[complete] for covar in covars),
        censored_time_covars=tuple(covar[~complete] for covar in covars),
        left_truncations_covars=tuple(covar[truncated] for covar in covars),
    )


def validate_cox_data(
    dataset: IntoDataFrame,
    covar_column_names: Sequence[str],
    *,
    field_column_names: Mapping[str, str] | None = None,
) -> CoxData:
    """Validate lifetime observations for the Cox semi-parametric regression.

    Parameters
    ----------
    dataset : DataFrame
        Lifetime observations, one row per observation, as any eager dataframe
        supported by narwhals (polars, pandas, pyarrow...). Lifetimes are
        complete or right-censored, given as a ``time`` column of floats. An
        optional ``event`` column of booleans tells whether each lifetime is a
        failure (``True``) or right-censored (``False``). If ``event`` is
        missing, all lifetimes are complete. An optional ``entry`` column of
        floats holds left truncation values (``0.`` if missing). Other columns
        are ignored, except covariates selected with ``covar_column_names``.
    covar_column_names : sequence of str
        Names of the covariate columns in ``dataset``. At least one covariate
        is required. Each covariate must be a column of floats.
    field_column_names : mapping of str to str, optional
        Column name of each lifetime field in ``dataset``, as
        ``{field: column_name}``, e.g. ``{"time": "duration", "event": "failed"}``.
        Fields are the lifetime variables ``"time"``, ``"event"`` and
        ``"entry"``. Fields not given are looked up under their own name.

    Returns
    -------
    CoxData

    Raises
    ------
    TypeError
        If ``dataset`` is not an eager dataframe or a used column, except
        ``event``, is not numeric.
    ValueError
        If no covariate is given, if a used column is missing or has missing
        values, if ``dataset`` is empty or has no failure, or if values are
        invalid: ``entry`` must be finite and non-negative, ``event`` must be 0
        or 1, ``time`` must be finite and greater than ``entry`` and covariates
        must be finite.
    """
    covar_names = list(covar_column_names)
    if not covar_names:
        msg = "At least one covariate is required"
        raise ValueError(msg)
    df = _rename_fields(dataset, field_column_names)
    _check_columns(df, ["time", *_entry_column(df), *covar_names])

    entry, time, complete = _read_flat_right_censored(df)
    if not complete.any():
        msg = "At least one failure is required ('event' all False)"
        raise ValueError(msg)
    covar = _read_covars(df, covar_names)

    # sorted distinct failure times, index of their first occurrence and
    # number of failures at each of them
    ordered_event_time, ordered_event_index, event_count = np.unique(
        time[complete], return_index=True, return_counts=True
    )
    event_time = ordered_event_time[:, None]
    return CoxData(
        nb_observations=len(df),
        nb_covars=len(covar_names),
        covar=covar,
        ordered_event_time=ordered_event_time,
        event_count=event_count,
        risk_set=(entry[None, :] < event_time) & (event_time <= time[None, :]),
        death_set=complete[None, :] & (time[None, :] == event_time),
        ordered_event_covar=tuple(c[complete][ordered_event_index] for c in covar),
    )


def validate_non_parametric_data(
    dataset: IntoDataFrame,
    *,
    field_column_names: Mapping[str, str] | None = None,
) -> NonParametricData:
    """Validate lifetime observations for non-parametric models.

    Parameters
    ----------
    dataset : DataFrame
        Lifetime observations, one row per observation, as any eager dataframe
        supported by narwhals (polars, pandas, pyarrow...). Lifetimes are
        complete or right-censored, given as a ``time`` column of floats. An
        optional ``event`` column of booleans tells whether each lifetime is a
        failure (``True``) or right-censored (``False``). If ``event`` is
        missing, all lifetimes are complete. An optional ``entry`` column of
        floats holds left truncation values (``0.`` if missing). Other columns
        are ignored.
    field_column_names : mapping of str to str, optional
        Column name of each lifetime field in ``dataset``, as
        ``{field: column_name}``, e.g. ``{"time": "duration", "event": "failed"}``.
        Fields are the lifetime variables ``"time"``, ``"event"`` and
        ``"entry"``. Fields not given are looked up under their own name.

    Returns
    -------
    NonParametricData

    Raises
    ------
    TypeError
        If ``dataset`` is not an eager dataframe or a used column, except
        ``event``, is not numeric.
    ValueError
        If a used column is missing or has missing values, if ``dataset`` is
        empty, or if values are invalid: ``entry`` must be finite and
        non-negative, ``event`` must be 0 or 1 and ``time`` must be finite and
        greater than ``entry``.
    """
    df = _rename_fields(dataset, field_column_names)
    _check_columns(df, ["time", *_entry_column(df)])

    entry, time, complete = _read_flat_right_censored(df)
    timeline, timeline_index = np.unique(time, return_inverse=True)
    # entry < t <= time: (time >= t) minus (entry >= t), as entry < time
    nb_at_risk = np.searchsorted(
        np.sort(entry), timeline, side="left"
    ) - np.searchsorted(np.sort(time), timeline, side="left")
    nb_failures = np.bincount(
        timeline_index, weights=complete, minlength=timeline.size
    ).astype(np.int64)
    return NonParametricData(
        timeline=timeline, nb_at_risk=nb_at_risk, nb_failures=nb_failures
    )


def validate_complete_time(
    dataset: IntoDataFrame,
    *,
    field_column_names: Mapping[str, str] | None = None,
) -> onp.Array1D[np.float64]:
    """Validate observed times, all taken as complete lifetimes.

    Only the ``time`` column is read: ``event``, ``entry`` and other columns
    are ignored. Used by the empirical cumulative distribution function.

    Parameters
    ----------
    dataset : DataFrame
        Lifetime observations, one row per observation, as any eager dataframe
        supported by narwhals (polars, pandas, pyarrow...), with a ``time``
        column of floats.
    field_column_names : mapping of str to str, optional
        Column name of the ``"time"`` field in ``dataset``, as
        ``{"time": column_name}``. If not given, ``time`` is looked up under
        its own name.

    Returns
    -------
    np.ndarray
        Observed times, of shape ``(n,)``.

    Raises
    ------
    TypeError
        If ``dataset`` is not an eager dataframe or ``time`` is not numeric.
    ValueError
        If ``time`` is missing or has missing values, if ``dataset`` is empty,
        or if ``time`` values are not finite and positive.
    """
    df = _rename_fields(dataset, field_column_names)
    _check_columns(df, ["time"])
    return _read_time(df, np.zeros((len(df), 1), dtype=np.float64)).ravel()
