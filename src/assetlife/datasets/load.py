"""Example lifetime datasets, loaded as polars dataframes."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

_CSV_DIR = Path(__file__).parent / "csv"


def _read_lifetime_csv(name: str, covars: Sequence[str] = ()) -> pl.DataFrame:
    """Lifetime observations of ``csv/<name>.csv``, with ``event`` as booleans.

    The files store ``event`` as numbers (``1``/``0``, ``1.0``/``0.0``), which
    polars does not parse as booleans: it is read as floats, then cast.
    """
    schema = pl.Schema({
        "time": pl.Float64(),
        "event": pl.Float64(),
        "entry": pl.Float64(),
        **{covar: pl.Float64() for covar in covars},
    })
    return pl.read_csv(_CSV_DIR / f"{name}.csv", schema=schema).with_columns(
        pl.col("event").cast(pl.Boolean)
    )


def load_power_transformer() -> pl.DataFrame:
    r"""
    Load example dataset of power transformers lifetimes.

    Returns
    -------
    polars.DataFrame
        One row per power transformer, with columns:

        - ``time`` (``Float64``): observed lifetime values.
        - ``event`` (``Boolean``): whether the failure is observed. If
          ``False``, the lifetime is right-censored.
        - ``entry`` (``Float64``): left truncation values.

    Examples
    --------
    >>> from assetlife.datasets import load_power_transformer
    >>> from assetlife.lifetime_models import Weibull
    >>> data = load_power_transformer()
    >>> data.columns
    ['time', 'event', 'entry']
    >>> model = Weibull().fit(data)
    """

    return _read_lifetime_csv("power_transformer")


def load_insulator_string() -> pl.DataFrame:
    r"""
    Load example dataset of insulator string lifetimes with covariates.

    Returns
    -------
    polars.DataFrame
        One row per insulator string, with columns:

        - ``time`` (``Float64``): observed lifetime values.
        - ``event`` (``Boolean``): whether the failure is observed. If
          ``False``, the lifetime is right-censored.
        - ``entry`` (``Float64``): left truncation values.
        - ``pHCl``, ``pH2SO4`` and ``HNO3`` (``Float64``): covariates, the
          concentrations of pHCl, pH2SO4 and HNO3.

    Examples
    --------
    >>> from assetlife.datasets import load_insulator_string
    >>> from assetlife.lifetime_models import ParametricProportionalHazard, Weibull
    >>> data = load_insulator_string()
    >>> data.columns
    ['time', 'event', 'entry', 'pHCl', 'pH2SO4', 'HNO3']
    >>> model = ParametricProportionalHazard(Weibull()).fit(
    ...     data, ["pHCl", "pH2SO4", "HNO3"]
    ... )
    """

    return _read_lifetime_csv("insulator_string", ["pHCl", "pH2SO4", "HNO3"])


def load_circuit_breaker() -> pl.DataFrame:
    r"""
    Load example dataset of circuit breaker lifetimes.

    Returns
    -------
    polars.DataFrame
        One row per circuit breaker, with columns:

        - ``time`` (``Float64``): observed lifetime values.
        - ``event`` (``Boolean``): whether the failure is observed. If
          ``False``, the lifetime is right-censored.
        - ``entry`` (``Float64``): left truncation values.

    Examples
    --------
    >>> from assetlife.datasets import load_circuit_breaker
    >>> data = load_circuit_breaker()
    >>> data.columns
    ['time', 'event', 'entry']
    """

    return _read_lifetime_csv("circuit_breaker")
