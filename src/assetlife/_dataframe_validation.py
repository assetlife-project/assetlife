"""Dataframe checks shared by the dataset validation of fittable models."""

from collections.abc import Hashable, Sequence
from typing import Any

import narwhals as nw
from narwhals.typing import IntoDataFrame


def to_eager_df(dataset: IntoDataFrame, name: str = "dataset") -> nw.DataFrame[Any]:
    """``dataset`` as an eager narwhals dataframe.

    Raises
    ------
    TypeError
        If ``dataset`` is not an eager dataframe.
    """
    try:
        return nw.from_native(dataset, eager_only=True)
    except TypeError:
        msg = f"Expected {name} to be a DataFrame, got {type(dataset).__name__}"
        raise TypeError(msg) from None


def check_columns_exist(
    df: nw.DataFrame[Any], columns: Sequence[str], name: str = "dataset"
) -> None:
    """Check that ``columns`` are columns of ``df``.

    Raises
    ------
    ValueError
        If a column is missing.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        msg = f"Columns not found in {name}: {missing}"
        raise ValueError(msg)


def check_not_empty(df: nw.DataFrame[Any], name: str = "dataset") -> None:
    """Check that ``df`` has rows.

    Raises
    ------
    ValueError
        If ``df`` has no rows.
    """
    if len(df) == 0:
        msg = f"{name} has no rows"
        raise ValueError(msg)


def check_numeric(
    df: nw.DataFrame[Any], columns: Sequence[str], name: str = "dataset"
) -> None:
    """Check that ``columns`` of ``df`` are numeric.

    Raises
    ------
    TypeError
        If a column is not numeric.
    """
    for col in columns:
        dtype = df.schema[col]
        if not dtype.is_numeric():
            msg = f"Column {col!r} of {name} must be numeric, got {dtype}"
            raise TypeError(msg)


def check_hashable(
    df: nw.DataFrame[Any], columns: Sequence[str], name: str = "dataset"
) -> None:
    """Check that values of ``columns`` of ``df`` are hashable.

    Nested dtypes (lists, arrays, structs) are not hashable. Object dtypes, e.g.
    pandas object columns, may hold any python object, so their values are
    checked one by one.

    Raises
    ------
    TypeError
        If a column is not hashable.
    """
    for col in columns:
        dtype = df.schema[col]
        if dtype.is_nested() or (
            dtype == nw.Object
            and not all(isinstance(v, Hashable) for v in df.get_column(col))
        ):
            msg = f"Column {col!r} of {name} must be hashable, got {dtype}"
            raise TypeError(msg)


def check_no_missing_values(
    df: nw.DataFrame[Any], columns: Sequence[str], name: str = "dataset"
) -> None:
    """Check that ``columns`` of ``df`` have no null values, nor NaN if float.

    Raises
    ------
    ValueError
        If a column has missing values.
    """
    for col in columns:
        series = df.get_column(col)
        if series.is_null().any() or (
            series.dtype.is_float() and series.is_nan().any()
        ):
            msg = f"Column {col!r} of {name} has missing values (null or NaN)"
            raise ValueError(msg)
