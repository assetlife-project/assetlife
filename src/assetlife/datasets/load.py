from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_power_transformer() -> NDArray[np.void]:
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

    data = np.loadtxt(
        Path(Path(__file__).parents[0], "csv/power_transformer.csv"),
        delimiter=",",
        skiprows=1,
        dtype=np.dtype([
            ("time", np.float64),
            ("event", np.float64),
            ("entry", np.float64),
        ]),
    )
    # for some reason, numpy can't cast 1.0/0.0 to np.bool_
    new_dtype = np.dtype([
        ("time", np.float64),
        ("event", np.bool_),
        ("entry", np.float64),
    ])
    return data.astype(new_dtype)


def load_insulator_string() -> NDArray[np.void]:
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

    data = np.loadtxt(
        Path(Path(__file__).parents[0], "csv/insulator_string.csv"),
        delimiter=",",
        skiprows=1,
        dtype=np.dtype([
            ("time", np.float64),
            ("event", np.float64),
            ("entry", np.float64),
            ("pHCl", np.float64),
            ("pH2SO4", np.float64),
            ("HNO3", np.float64),
        ]),
    )
    # for some reason, numpy can't cast 1.0/0.0 to np.bool_
    new_dtype = np.dtype([
        ("time", np.float64),
        ("event", np.bool_),
        ("entry", np.float64),
        ("pHCl", np.float64),
        ("pH2SO4", np.float64),
        ("HNO3", np.float64),
    ])
    return data.astype(new_dtype)


def load_circuit_breaker() -> NDArray[np.void]:
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

    data = np.loadtxt(
        Path(Path(__file__).parents[0], "csv/circuit_breaker.csv"),
        delimiter=",",
        skiprows=1,
        dtype=np.dtype([
            ("time", np.float64),
            ("event", np.float64),
            ("entry", np.float64),
        ]),
    )
    # for some reason, numpy can't cast 1.0/0.0 to np.bool_
    new_dtype = np.dtype([
        ("time", np.float64),
        ("event", np.bool_),
        ("entry", np.float64),
    ])
    return data.astype(new_dtype)


def load_gas_refill_circuit_breaker() -> tuple[NDArray[np.void], NDArray[np.void]]:
    r"""
    Load example dataset of SF6 refills on circuit breakers.

    Examples
    --------
    >>> from assetlife.datasets import load_gas_refill_circuit_breaker
    >>> failures, assets = load_gas_refill_circuit_breaker()
    >>> print(failures[:3])
    [(4, 43.088) (4, 47.37 ) (4, 49.325)]
    >>> print(assets[3:5])
    [(3, 42., 51.) (4, 42., 51.)]

    Returns
    -------
    failures : structured array
        A numpy structured array of 2 fields, with one row per refill:

        - id (``np.int64``) : equipment id
        - time (``np.float64``) : age at the refill
    assets : structured array
        A numpy structured array of 3 fields, with one row per equipment,
        including those without any refill:

        - id (``np.int64``) : equipment id
        - start (``np.float64``) : age at the beginning of the observation
          window
        - end (``np.float64``) : age at the end of the observation window
    """

    # genfromtxt, unlike loadtxt, reads the empty age_at_repair field of
    # equipments without refill as nan
    data = np.genfromtxt(
        Path(Path(__file__).parents[0], "csv/gas_refill_circuit_breaker.csv"),
        delimiter=",",
        skip_header=1,
        dtype=np.dtype([
            ("id", np.int64),
            ("start", np.float64),
            ("end", np.float64),
            ("time", np.float64),
        ]),
    )
    # the observation window is repeated on every refill row of an equipment
    _, first_rows = np.unique(data["id"], return_index=True)
    assets = data[["id", "start", "end"]][first_rows]
    failures = data[["id", "time"]][~np.isnan(data["time"])]
    # multi-field indexing returns views with padded dtypes: repack them
    return (
        failures.astype(np.dtype([("id", np.int64), ("time", np.float64)])),
        assets.astype(
            np.dtype([("id", np.int64), ("start", np.float64), ("end", np.float64)])
        ),
    )
