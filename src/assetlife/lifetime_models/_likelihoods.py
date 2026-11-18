from collections.abc import Mapping, Sequence
from typing import Any, Literal, final, override
from typing_extensions import TypeIs

import narwhals as nw
import numpy as np
import optype.numpy as onp
from narwhals.typing import IntoDataFrame

from ._base import FittableParametricLifetimeModel
from assetlife._optimization import FitConfig, MaximumLikelihoodOptimizer
from assetlife.typing import CoercibleFloat64_ND

type _ColumnArray[T: np.generic] = onp.Array[tuple[int, Literal[1]], T]

_LIFETIME_FIELDS = frozenset({"time", "time_inf", "time_sup", "event", "entry"})


def _is_column_array[T: np.generic](
    arr: onp.Array2D[T],
) -> TypeIs[_ColumnArray[T]]:
    """Whether a 2D array has a single column.

    Narrows the ``(int, int)`` shape returned by narwhals ``to_numpy`` to
    ``(int, 1)``, which an ``assert`` on ``arr.shape`` cannot do.
    """
    return arr.shape[1] == 1


def _to_column_array[T: np.generic](
    df: nw.DataFrame[Any], name: str, dtype: type[T]
) -> _ColumnArray[T]:
    """Column ``name`` as an ``(n, 1)`` array.

    Raises
    ------
    ValueError
        If the selection does not give a single column (not expected, as
        exactly one column is selected).
    """
    arr = df.select(nw.col(name)).to_numpy().astype(dtype)
    if not _is_column_array(arr):
        msg = f"Expected a single column for {name!r}, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


def _check_columns(
    names: Sequence[str],
    field_column_names: Mapping[str, str],
    covar_column_names: Sequence[str],
) -> None:
    """Check the column names given by the user against the dataset columns.

    Raises
    ------
    ValueError
        If a field is unknown, a column is missing, duplicated or would be
        overwritten by the renaming, or a covariate is a lifetime column.
    """
    fields, covars = field_column_names, covar_column_names
    unknown = set(fields) - _LIFETIME_FIELDS
    if unknown:
        msg = (
            f"Unknown fields in field_column_names: {sorted(unknown)}. "
            f"Valid fields are {sorted(_LIFETIME_FIELDS)}"
        )
        raise ValueError(msg)
    if len(set(fields.values())) != len(fields):
        msg = f"Several fields map to the same column in field_column_names: {fields}"
        raise ValueError(msg)
    if len(set(covars)) != len(covars):
        msg = f"Duplicated names in covar_column_names: {list(covars)}"
        raise ValueError(msg)
    missing = (set(fields.values()) | set(covars)) - set(names)
    if missing:
        msg = f"Columns not found in dataset: {sorted(missing)}"
        raise ValueError(msg)
    clash = (set(fields) & set(names)) - set(fields.values())
    if clash:
        msg = (
            f"Columns {sorted(clash)} already exist in dataset and would be "
            "overwritten by field_column_names. Drop or rename them first"
        )
        raise ValueError(msg)
    lifetime_columns = {fields.get(field, field) for field in _LIFETIME_FIELDS}
    overlap = set(covars) & lifetime_columns
    if overlap:
        msg = f"Lifetime columns cannot be used as covariates: {sorted(overlap)}"
        raise ValueError(msg)


def _check_schema(
    df: nw.DataFrame[Any],
    names: Sequence[str],
    display: Mapping[str, str],
) -> None:
    """Check dtypes and missing values of the used columns.

    Raises
    ------
    TypeError
        If ``event`` is neither boolean nor integer, or another column is not
        numeric.
    ValueError
        If a column has null or NaN values.
    """
    schema = df.schema
    for name in names:
        dtype = schema[name]
        valid = (
            dtype == nw.Boolean or dtype.is_integer()
            if name == "event"
            else dtype.is_numeric()
        )
        if not valid:
            expected = "boolean or 0/1 integer" if name == "event" else "numeric"
            msg = f"Column {display[name]!r} must be {expected}, got {dtype}"
            raise TypeError(msg)
    nb_missing = df.select(
        (
            nw.col(name).is_null() | nw.col(name).is_nan()
            if schema[name].is_float()
            else nw.col(name).is_null()
        ).sum()
        for name in names
    ).row(0)
    missing = {display[n]: k for n, k in zip(names, nb_missing, strict=True) if k}
    if missing:
        msg = f"Missing values (null or NaN) found, per column: {missing}"
        raise ValueError(msg)


def _raise_if_any(invalid: onp.ArrayND[np.bool_], msg: str) -> None:
    """Raise a ValueError naming the first invalid rows.

    Raises
    ------
    ValueError
        If any value of ``invalid`` is True.
    """
    rows = np.flatnonzero(invalid)
    if rows.size:
        msg = f"{msg}. Invalid rows: {rows[:5].tolist()}"
        if rows.size > 5:
            msg += f" and {rows.size - 5} others"
        raise ValueError(msg)


def _read_entry(
    df: nw.DataFrame[Any], display: Mapping[str, str]
) -> _ColumnArray[np.float64]:
    """Left truncation values, zeros if ``entry`` is not given.

    Values must be finite and non-negative.
    """
    if "entry" not in df.columns:
        return np.zeros((len(df), 1), dtype=np.float64)
    entry = _to_column_array(df, "entry", np.float64)
    _raise_if_any(
        ~np.isfinite(entry) | (entry < 0),
        f"{display['entry']!r} values must be finite and >= 0",
    )
    return entry


def _read_covar(df: nw.DataFrame[Any], name: str) -> _ColumnArray[np.float64]:
    """Covariate values.

    Values must be finite.
    """
    covar = _to_column_array(df, name, np.float64)
    _raise_if_any(~np.isfinite(covar), f"{name!r} values must be finite")
    return covar


def _read_time(
    df: nw.DataFrame[Any],
    entry: _ColumnArray[np.float64],
    display: Mapping[str, str],
) -> tuple[_ColumnArray[np.float64], _ColumnArray[np.float64], onp.Array1D[np.bool_]]:
    """Complete times, right-censored times and complete mask.

    ``time`` must be finite and greater than ``entry``, ``event`` must be 0
    or 1.
    """
    time = _to_column_array(df, "time", np.float64)
    name = display["time"]
    _raise_if_any(~np.isfinite(time), f"{name!r} values must be finite")
    _raise_if_any(time <= entry, f"{name!r} must be > {display['entry']!r}")
    if "event" in df.columns:
        event = df.get_column("event").to_numpy()
        _raise_if_any(
            ~np.isin(event, (0, 1)),
            f"{display['event']!r} values must be booleans or 0/1",
        )
        complete = event.astype(np.bool_)
    else:
        complete = np.ones(len(df), dtype=np.bool_)
    return time[complete], time[~complete], complete


def _read_interval_time(
    df: nw.DataFrame[Any],
    entry: _ColumnArray[np.float64],
    display: Mapping[str, str],
) -> tuple[
    _ColumnArray[np.float64],
    onp.Array[tuple[int, Literal[2]], np.float64],
    onp.Array1D[np.bool_],
]:
    """Complete times, censored intervals and complete mask.

    ``time_inf`` must be finite, and values must satisfy
    ``entry <= time_inf <= time_sup`` and ``entry < time_sup``.
    """
    lower = _to_column_array(df, "time_inf", np.float64)
    upper = _to_column_array(df, "time_sup", np.float64)
    inf, sup, entry_name = display["time_inf"], display["time_sup"], display["entry"]
    _raise_if_any(~np.isfinite(lower), f"{inf!r} values must be finite")
    _raise_if_any(lower > upper, f"{inf!r} must be <= {sup!r}")
    _raise_if_any(upper <= entry, f"{sup!r} must be > {entry_name!r}")
    _raise_if_any(lower < entry, f"{inf!r} must be >= {entry_name!r}")
    complete = (lower == upper).ravel()
    return upper[complete], np.column_stack((lower, upper))[~complete], complete


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
        If ``dataset`` is not an eager dataframe, if ``event`` is neither a
        boolean nor an integer column, or if another used column is not numeric.
    ValueError
        If ``field_column_names`` or ``covar_column_names`` are inconsistent
        with ``dataset`` (unknown field, missing or duplicated column, covariate
        that is also a lifetime column), if lifetimes are not given either as
        ``time`` or as both ``time_inf`` and ``time_sup``, if ``event`` is given with
        interval-censored lifetimes, if ``dataset`` is empty or has missing
        values, or if values are invalid: ``entry`` must be finite and
        non-negative, ``event`` must be 0 or 1, ``time`` must be finite and
        greater than ``entry``, ``time_inf`` must be finite and satisfy
        ``entry <= time_inf <= time_sup``, ``time_sup`` must be greater than
        ``entry`` and covariates must be finite.
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
        field_column_names: Mapping[str, str] | None = None,
        covar_column_names: Sequence[str] = (),
    ) -> None:
        try:
            df = nw.from_native(dataset, eager_only=True)
        except TypeError:
            msg = f"Expected a DataFrame, got {type(dataset).__name__}"
            raise TypeError(msg) from None
        fields = dict(field_column_names or {})
        covar_names = list(covar_column_names)
        _check_columns(df.columns, fields, covar_names)
        df = df.rename({column: field for field, column in fields.items()})
        names = set(df.columns)

        bounds = {"time_inf", "time_sup"} & names
        if ("time" in names) == bool(bounds) or len(bounds) == 1:
            msg = "Expected either a 'time' column, or 'time_inf' and 'time_sup'"
            raise ValueError(msg)
        if bounds and "event" in names:
            msg = "'event' must not be given with interval-censored lifetimes"
            raise ValueError(msg)

        n = len(df)
        if n == 0:
            msg = "dataset has no rows"
            raise ValueError(msg)
        used = [f for f in sorted(_LIFETIME_FIELDS) if f in names] + covar_names
        display = {name: fields.get(name, name) for name in used}
        display.setdefault("entry", "entry")
        _check_schema(df, used, display)

        entry = _read_entry(df, display)
        covars = tuple(_read_covar(df, name) for name in covar_names)
        if bounds:
            self.complete_time, self.censored_time, complete = _read_interval_time(
                df, entry, display
            )
        else:
            self.complete_time, self.censored_time, complete = _read_time(
                df, entry, display
            )

        truncated = (entry != 0).ravel()
        self.nb_observations = n
        self.nb_covars = len(covars)
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
