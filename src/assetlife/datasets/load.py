from pathlib import Path

import polars as pl


def load_power_transformer() -> pl.DataFrame:
    r"""
    Load example dataset of power transformers lifetimes.

    Examples
    --------
    >>> from assetlife.datasets import load_power_transformer
    >>> data = load_power_transformer()
    >>> print(data["time"])
    [34.3 45.1 53.2 ... 30.  30.  30. ]

    Returns
    -------
    structured array
        A numpy structured array of 3 fields :

        - time (``np.float64``) : observed lifetime values
        - event (``np.bool_``) : boolean flag indicated if the event has been
          observed or not (if False, the observed lifetimes are right censored)
        - entry (``np.float64``) : left truncation values

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

    Examples
    --------
    >>> from assetlife.datasets import load_insulator_string
    >>> data = load_insulator_string()
    >>> print(data["time"])
    [70.  30.  45.  ...  8.8  7.6 53. ]
    >>> print(data["pHCl"])
    [0.49 0.76 0.43 ... 1.12 1.19 0.35]

    Returns
    -------
    structured array
        A numpy structured array of 6 fields :

        - time (``np.float64``) : observed lifetime values
        - event (``np.bool_``) : boolean flag indicated if the event has been
          observed or not (if False, the observed lifetimes are right censored)
        - entry (``np.float64``) : left truncation values
        - pHCl (``np.float64``) : quantitative covariate values (concentration of pHCl)
        - pH2SO4 (``np.float64``) : quantitative covariate values (concentration of
          pH2SO4)
        - HNO3 (``np.float64``) : quantitative covariate values (concentration of HNO3)

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

    Examples
    --------
    >>> from assetlife.datasets import load_circuit_breaker
    >>> data = load_circuit_breaker()
    >>> print(data["time"])
    [34. 28. 12. ... 42. 42. 37.]

    Returns
    -------
    structured array
        A numpy structured array of 3 fields :

        - time (``np.float64``) : observed lifetime values
        - event (``np.bool_``) : boolean flag indicated if the event has been
          observed or not (if False, the observed lifetimes are right censored)
        - entry (``np.float64``) : left truncation values
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
