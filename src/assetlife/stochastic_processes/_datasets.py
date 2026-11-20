from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from assetlife._dataset_validation import (
    check_columns_exist,
    check_hashable,
    check_no_missing_values,
    check_not_empty,
    check_numeric,
    to_eager_df,
)

if TYPE_CHECKING:
    import narwhals as nw
    import optype.numpy as onp
    from narwhals.typing import IntoDataFrame

_RESERVED_NAMES = frozenset({"id", "time", "start", "end", "count"})
# asset window bounds, renamed to not clash with failures or partial columns
_ASSET_START = "__asset_start__"
_ASSET_END = "__asset_end__"


def _read_dataframe(
    dataset: IntoDataFrame, name: str, columns: Sequence[str]
) -> nw.DataFrame[Any]:
    """``dataset`` restricted to ``id`` and ``columns``, checked.

    Columns must exist and have no missing values, ``id`` must be hashable,
    ``columns`` must be numeric and ``dataset`` must have rows, else a
    TypeError or ValueError is raised.
    """
    df = to_eager_df(dataset, name)
    check_columns_exist(df, ["id", *columns], name)
    check_not_empty(df, name)
    check_hashable(df, ["id"], name)
    check_numeric(df, columns, name)
    check_no_missing_values(df, ["id", *columns], name)
    return df.select("id", *columns)


def _join_assets(
    df: nw.DataFrame[Any], assets: nw.DataFrame[Any], name: str
) -> nw.DataFrame[Any]:
    """``df`` with the window bounds and covariates of its asset.

    Raises
    ------
    ValueError
        If ids of ``df`` are not asset ids.
    """
    joined = df.join(assets, on="id", how="left")
    if joined.get_column(_ASSET_START).is_null().any():
        msg = f"Ids of {name} must be asset ids"
        raise ValueError(msg)
    return joined


def _to_float_1d(df: nw.DataFrame[Any], name: str) -> onp.Array1D[np.float64]:
    return df.get_column(name).to_numpy().astype(np.float64)


class PartialObservations(NamedTuple):
    """Numbers of events observed per time window, with their covariates."""

    failures_count: onp.Array1D[np.float64]
    start: onp.Array1D[np.float64]
    end: onp.Array1D[np.float64]
    covars: tuple[onp.Array1D[np.float64], ...]


def _read_assets(assets: IntoDataFrame, covars: Sequence[str]) -> nw.DataFrame[Any]:
    """Checked assets, with ``start`` and ``end`` renamed for joins.

    Raises
    ------
    ValueError
        If asset ids are not unique, ``start`` is not finite and non-negative,
        ``end`` is not finite and greater than ``start``, or covariates are not
        finite.
    """
    df = _read_dataframe(assets, "assets", ["start", "end", *covars])
    if df.get_column("id").is_duplicated().any():
        msg = "Asset ids must be unique"
        raise ValueError(msg)
    start, end = _to_float_1d(df, "start"), _to_float_1d(df, "end")
    if np.any(~np.isfinite(start) | (start < 0)):
        msg = "'start' values must be finite and >= 0"
        raise ValueError(msg)
    if np.any(~np.isfinite(end) | (end <= start)):
        msg = "'end' must be finite and > 'start'"
        raise ValueError(msg)
    for name in covars:
        if np.any(~np.isfinite(_to_float_1d(df, name))):
            msg = f"{name!r} values must be finite"
            raise ValueError(msg)
    return df.rename({"start": _ASSET_START, "end": _ASSET_END})


def _read_failures(
    failures: IntoDataFrame, assets: nw.DataFrame[Any]
) -> nw.DataFrame[Any]:
    """Checked failures, joined with ``assets``.

    Raises
    ------
    ValueError
        If a failure time is not within the observation window of its asset.
    """
    df = _join_assets(
        _read_dataframe(failures, "failures", ["time"]), assets, "failures"
    )
    time = _to_float_1d(df, "time")
    if np.any(
        (time <= _to_float_1d(df, _ASSET_START)) | (time > _to_float_1d(df, _ASSET_END))
    ):
        msg = "Failure 'time' must be within its asset window: start < time <= end"
        raise ValueError(msg)
    return df


def _read_partial_observations(
    partial_observations: IntoDataFrame,
    assets: nw.DataFrame[Any],
    covars: Sequence[str],
) -> PartialObservations:
    """Checked partial observations, with the covariates of their asset.

    Raises
    ------
    ValueError
        If ``0 <= start < end < inf`` does not hold, or ``count`` is not finite
        and non-negative.
    """
    name = "partial_observations"
    df = _join_assets(
        _read_dataframe(partial_observations, name, ["start", "end", "count"]),
        assets,
        name,
    )
    start, end, count = (_to_float_1d(df, c) for c in ("start", "end", "count"))
    if np.any((start < 0) | ~np.isfinite(end) | (end <= start)):
        msg = "Partial observations must satisfy 0 <= 'start' < 'end' < inf"
        raise ValueError(msg)
    if np.any(~np.isfinite(count) | (count < 0)):
        msg = "'count' values must be finite and >= 0"
        raise ValueError(msg)
    return PartialObservations(
        failures_count=count,
        start=start,
        end=end,
        covars=tuple(_to_float_1d(df, c) for c in covars),
    )


class NHPPData:
    """Observation schema for non-homogeneous Poisson processes.

    Used to compute the likelihood.

    Parameters
    ----------
    failures : DataFrame
        Failure times, one row per failure, as any eager dataframe supported by
        narwhals (polars, pandas, pyarrow...). Columns are ``id``, the asset id,
        and ``time``, the failure time as float. Other columns are ignored.
    assets : DataFrame
        Assets, one row per asset, as any eager dataframe. Columns are ``id``,
        the unique asset id, and ``start`` and ``end``, the bounds of the
        observation window of the asset as floats. Other columns are ignored,
        except covariates selected with ``covar_column_names``.
    covar_column_names : sequence of str, default ()
        Names of the covariate columns in ``assets``, used by regression models.
        Each covariate must be a column of floats.
    partial_observations : DataFrame, optional
        Numbers of failures observed per time window, without failure times, as
        any eager dataframe. Columns are ``id``, the asset id, ``start`` and
        ``end``, the bounds of the time window as floats, and ``count``, the
        number of failures observed in the window. Other columns are ignored.

    Raises
    ------
    TypeError
        If a dataset is not an eager dataframe, if an ``id`` column is not
        hashable, or if another used column is not numeric.
    ValueError
        If a covariate uses a reserved name (``id``, ``time``, ``start``,
        ``end``, ``count``), if a column is missing, if a dataset is empty or
        has missing values, if asset ids are not unique, if ids of ``failures``
        or ``partial_observations`` are not asset ids, or if values are invalid:
        ``start`` must be finite and non-negative, ``end`` finite and greater
        than ``start``, failure ``time`` within the observation window of its
        asset (``start < time <= end``), partial observations windows such
        that ``0 <= start < end < inf``, ``count`` finite and non-negative, and
        covariates finite.
    """

    failures_time: onp.Array1D[np.float64]
    failures_covars: tuple[onp.Array1D[np.float64], ...]
    observations_start: onp.Array1D[np.float64]
    observations_end: onp.Array1D[np.float64]
    observations_covars: tuple[onp.Array1D[np.float64], ...]
    nb_covars: int
    partial_observations: PartialObservations | None

    def __init__(
        self,
        failures: IntoDataFrame,
        assets: IntoDataFrame,
        *,
        covar_column_names: Sequence[str] = (),
        partial_observations: IntoDataFrame | None = None,
    ) -> None:
        covars = list(covar_column_names)
        reserved = sorted(set(covars) & _RESERVED_NAMES)
        if reserved:
            msg = f"Reserved names cannot be used as covariates: {reserved}"
            raise ValueError(msg)

        assets_df = _read_assets(assets, covars)
        self.observations_start = _to_float_1d(assets_df, _ASSET_START)
        self.observations_end = _to_float_1d(assets_df, _ASSET_END)
        self.observations_covars = tuple(_to_float_1d(assets_df, c) for c in covars)
        self.nb_covars = len(covars)

        failures_df = _read_failures(failures, assets_df)
        self.failures_time = _to_float_1d(failures_df, "time")
        self.failures_covars = tuple(_to_float_1d(failures_df, c) for c in covars)

        self.partial_observations = (
            None
            if partial_observations is None
            else _read_partial_observations(partial_observations, assets_df, covars)
        )
