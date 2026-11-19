"""Nonparametric lifetime models."""

from collections.abc import Mapping
from typing import Any, Literal, NamedTuple, TypeVar, final

import numpy as np
from matplotlib.axes import Axes
from narwhals.typing import IntoDataFrame
from optype.numpy import Array1D

from ._datasets import NonParametricData, read_complete_time
from assetlife.lifetime_models._base import _plot_probability_function


class NonParametricEstimation(NamedTuple):
    timeline: Array1D[np.float64]
    values: Array1D[np.float64]
    se: Array1D[np.float64]


KT = TypeVar("KT", bound=str)


class NonParametricLifetimeModel[KT: str]:
    _ci_bounds: tuple[float, float] = (0.0, 1.0)
    _estimations: dict[KT, NonParametricEstimation]

    def __init__(self) -> None:
        self._estimations = {}

    def plot(
        self,
        fname: KT,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes | None:
        """
        Plot an estimated function.

        Parameters
        ----------
        fname : str
            The name of the estimated function to plot, for instance ``"sf"``.
        ax : plt.Axes, optional
            An optional existing matplotlib.axes.
        **kwargs
            Extra arguments to configure the plot:
                - ci : bool, default is True
                - alpha_ci : float, default is 0.95
                - drawstyle : str, default is "steps-post"
                - any arguments allowed by matplotlib.plot

        Returns
        -------
        out : plt.Axes
            The axes the function is plotted on.

        Raises
        ------
        ValueError
            If ``fname`` is not a function estimated by this model.
        """

        estimations = self._estimations.get(fname, None)
        if estimations is None:
            available = ", ".join(repr(name) for name in self._estimations)
            msg = f"Unknown fname {fname!r}. Available: {available}"
            raise ValueError(msg)
        ci = kwargs.pop("ci", True)
        drawstyle = kwargs.pop("drawstyle", "steps-post")
        time, y, se = estimations
        if ci:
            return _plot_probability_function(
                time,
                y,
                se=se,
                ci_bounds=self._ci_bounds,
                ax=ax,
                drawstyle=drawstyle,
                **kwargs,
            )
        return _plot_probability_function(
            time,
            y,
            ax=ax,
            drawstyle=drawstyle,
            **kwargs,
        )


@final
class ECDF(NonParametricLifetimeModel[Literal["sf", "cdf"]]):
    """
    Empirical Cumulative Distribution Function.

    Parameters
    ----------
    dataset : DataFrame
        Lifetime observations, one row per observation, as any eager dataframe
        supported by narwhals (polars, pandas, pyarrow...), with a ``time``
        column of floats. All times are taken as complete lifetimes: ``event``,
        ``entry`` and other columns are ignored.
    field_column_names : mapping of str to str, optional
        Column name of the ``"time"`` field in ``dataset``, as
        ``{"time": column_name}``. If not given, ``time`` is looked up under
        its own name.
    """

    def __init__(
        self,
        dataset: IntoDataFrame,
        *,
        field_column_names: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        time = read_complete_time(dataset, field_column_names=field_column_names)
        timeline, counts = np.unique(time, return_counts=True)
        timeline = np.insert(timeline, 0, 0)
        cdf = np.insert(np.cumsum(counts), 0, 0) / np.sum(counts)
        se = np.sqrt(cdf * (1 - cdf) / len(time))
        self._estimations["sf"] = NonParametricEstimation(timeline, 1 - cdf, se)
        self._estimations["cdf"] = NonParametricEstimation(timeline, cdf, se)

    def sf(self) -> NonParametricEstimation:
        """
        The estimation of the survival function.

        Returns
        -------
        out : tuple of 3 1darray
            A tuple containing the timeline, the estimated values and the
            estimated standard errors.
        """
        return self._estimations["sf"]

    def cdf(self) -> NonParametricEstimation:
        """
        The estimation of the cumulative distribution function.

        Returns
        -------
        out : tuple of 3 1darray
            A tuple containing the timeline, the estimated values and the
            estimated standard errors.
        """
        return self._estimations["cdf"]


@final
class KaplanMeier(NonParametricLifetimeModel[Literal["sf"]]):
    r"""Kaplan-Meier estimator.

    Compute the non-parametric Kaplan-Meier estimator (also known as the product
    limit estimator) of the survival function from lifetime data.

    Parameters
    ----------
    dataset : DataFrame
        Lifetime observations, one row per observation, as any eager dataframe
        supported by narwhals (polars, pandas, pyarrow...). Lifetimes are
        complete or right-censored, given as a ``time`` column of floats. An
        optional ``event`` column of booleans tells whether each lifetime is a
        failure (``True``) or right-censored (``False``). If ``event`` is
        missing, all lifetimes are complete. An optional ``entry`` column of
        floats holds left truncation values (``0.`` if missing): an observation
        is at risk at time ``t`` if ``entry < t <= time``. Other columns are
        ignored.
    field_column_names : mapping of str to str, optional
        Column name of each lifetime field in ``dataset``, as
        ``{field: column_name}``, e.g. ``{"time": "duration", "event": "failed"}``.
        Fields are ``"time"``, ``"event"`` and ``"entry"``. Fields not given
        are looked up under their own name.

    Notes
    -----
    For a given time instant :math:`t` and :math:`n` total observations, this
    estimator is defined as:

    .. math::

        \hat{S}(t) = \prod_{i: t_i \leq t} \left( 1 - \frac{d_i}{n_i}\right)

    where :math:`d_i` is the number of failures until :math:`t_i` and
    :math:`n_i` is the number of assets at risk just prior to :math:`t_i`.

    The variance estimation is obtained by:

    .. math::

        \widehat{Var}[\hat{S}(t)] = \hat{S}(t)^2 \sum_{i: t_i \leq t}
        \frac{d_i}{n_i(n_i - d_i)}

    which is often referred to as Greenwood's formula.

    References
    ----------
    .. [1] Lawless, J. F. (2011). Statistical models and methods for lifetime
        data. John Wiley & Sons.

    .. [2] Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from
        incomplete observations. Journal of the American statistical
        association, 53(282), 457-481.

    """

    def __init__(
        self,
        dataset: IntoDataFrame,
        *,
        field_column_names: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        data = NonParametricData(dataset, field_column_names=field_column_names)
        timeline, n, d = data.timeline, data.nb_at_risk, data.nb_failures
        sf = (1 - d / n).cumprod()
        # Greenwood terms, 0 where everyone at risk fails (sf is then 0)
        terms = np.divide(
            d, n * (n - d), out=np.zeros(timeline.size, dtype=np.float64), where=n > d
        )
        var = sf**2 * terms.cumsum()

        self._estimations["sf"] = NonParametricEstimation(
            np.insert(timeline, 0, 0),
            np.insert(sf, 0, 1),
            np.insert(np.sqrt(var), 0, 0),
        )

    def sf(self) -> NonParametricEstimation:
        """
        The estimation of the survival function.

        Returns
        -------
        out : tuple of 3 1darray
            A tuple containing the timeline, the estimated values and the
            estimated standard errors.
        """
        return self._estimations["sf"]


@final
class NelsonAalen(NonParametricLifetimeModel[Literal["chf"]]):
    r"""Nelson-Aalen estimator.

    Compute the non-parametric Nelson-Aalen estimator of the cumulative hazard
    function from lifetime data.

    Parameters
    ----------
    dataset : DataFrame
        Lifetime observations, one row per observation, as any eager dataframe
        supported by narwhals (polars, pandas, pyarrow...). Lifetimes are
        complete or right-censored, given as a ``time`` column of floats. An
        optional ``event`` column of booleans tells whether each lifetime is a
        failure (``True``) or right-censored (``False``). If ``event`` is
        missing, all lifetimes are complete. An optional ``entry`` column of
        floats holds left truncation values (``0.`` if missing): an observation
        is at risk at time ``t`` if ``entry < t <= time``. Other columns are
        ignored.
    field_column_names : mapping of str to str, optional
        Column name of each lifetime field in ``dataset``, as
        ``{field: column_name}``, e.g. ``{"time": "duration", "event": "failed"}``.
        Fields are ``"time"``, ``"event"`` and ``"entry"``. Fields not given
        are looked up under their own name.

    Notes
    -----
    For a given time instant :math:`t` and :math:`n` total observations, this
    estimator is defined as:

    .. math::

        \hat{H}(t) = \sum_{i: t_i \leq t} \frac{d_i}{n_i}

    where :math:`d_i` is the number of failures until :math:`t_i` and
    :math:`n_i` is the number of assets at risk just prior to :math:`t_i`.

    The variance estimation is obtained by:

    .. math::

        \widehat{Var}[\hat{H}(t)] = \sum_{i: t_i \leq t} \frac{d_i}{n_i^2}

    Note that the alternative survivor function estimate:

    .. math::

        \tilde{S}(t) = \exp{(-\hat{H}(t))}

    is sometimes suggested for the continuous-time case.

    References
    ----------
    .. [1] Lawless, J. F. (2011). Statistical models and methods for lifetime
        data. John Wiley & Sons.
    """

    _ci_bounds: tuple[float, float] = (0.0, np.inf)

    def __init__(
        self,
        dataset: IntoDataFrame,
        *,
        field_column_names: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        data = NonParametricData(dataset, field_column_names=field_column_names)
        timeline, n, d = data.timeline, data.nb_at_risk, data.nb_failures
        chf = (d / n).cumsum()
        var = (d / n**2).cumsum()

        self._estimations["chf"] = NonParametricEstimation(
            np.insert(timeline, 0, 0),
            np.insert(chf, 0, 0),
            np.insert(np.sqrt(var), 0, 0),
        )

    def chf(self) -> NonParametricEstimation:
        """
        The estimation of the cumulative hazard function.

        Returns
        -------
        out : tuple of 3 1darray
            A tuple containing the timeline, the estimated values and the
            estimated standard errors.
        """
        return self._estimations["chf"]
