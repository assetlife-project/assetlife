"""Example lifetime datasets, loaded as polars dataframes."""

from pathlib import Path

import polars as pl


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

    return pl.read_csv(
        Path(Path(__file__).parents[0], "csv/power_transformer.csv"),
        separator=",",
        schema=pl.Schema({
            "time": pl.Float64(),
            "event": pl.Boolean(),
            "entry": pl.Float64(),
        }),
    )


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

    return pl.read_csv(
        Path(Path(__file__).parents[0], "csv/insulator_string.csv"),
        separator=",",
        schema=pl.Schema({
            "time": pl.Float64(),
            "event": pl.Boolean(),
            "entry": pl.Float64(),
            "pHCl": pl.Float64(),
            "pH2SO4": pl.Float64(),
            "HNO3": pl.Float64(),
        }),
    )


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

    return pl.read_csv(
        Path(Path(__file__).parents[0], "csv/circuit_breaker.csv"),
        separator=",",
        schema=pl.Schema({
            "time": pl.Float64(),
            "event": pl.Boolean(),
            "entry": pl.Float64(),
        }),
    )
