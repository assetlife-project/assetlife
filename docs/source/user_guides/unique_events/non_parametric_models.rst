Non-parametric lifetime models
================================

Before fitting a parametric distribution, it's often useful to estimate the survival or
hazard function directly from the data, without assuming any particular shape. AssetLife
provides two censoring-aware estimators for this, Kaplan-Meier and Nelson-Aalen. Both take the ``event`` and
``entry`` arguments and therefore account for right censoring and left truncation (see
:doc:`censoring_and_truncation`).

Both estimators share the same idea. Index the units of the dataset by :math:`j`, so that
unit :math:`j` is observed on :math:`[\textrm{entry}_j, \textrm{time}_j]`, and write
:math:`t_1 < t_2 < \dots < t_n` for the ordered distinct observed times, :math:`d_i` for the
number of failures recorded at :math:`t_i`, and :math:`n_i` for the number of units
**at risk** just prior to :math:`t_i`, that is the units already entered and not yet failed
or censored:

.. math::

    n_i = \#\{ j : \textrm{entry}_j \leq t_i \leq \textrm{time}_j \}

Censoring and truncation only enter through these two counts: a censored unit contributes to
:math:`n_i` up to the time it was last seen but never to :math:`d_i`, and a left-truncated
unit only starts contributing to :math:`n_i` once it enters the observation window.

Kaplan-Meier: survival function
---------------------------------------------------------

The Kaplan-Meier (product-limit) estimator reads each observed failure time as a conditional
probability of surviving that instant, :math:`1 - d_i/n_i`, and multiplies them along the
timeline:

.. math::

    \hat{S}(t) = \prod_{i: t_i \leq t} \left( 1 - \frac{d_i}{n_i}\right)

The estimate is a step function that drops only where a failure was actually observed and
stays flat everywhere else, so censored units never push the curve down:

.. plot::
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> from assetlife.datasets import load_circuit_breaker
    >>> from assetlife.lifetime_models import KaplanMeier
    >>> dataset = load_circuit_breaker()
    >>> km = KaplanMeier(dataset["time"], event=dataset["event"], entry=dataset["entry"])
    >>> _ = km.plot("sf")
    >>> _ = plt.xlabel("Time")
    >>> _ = plt.ylabel("Estimated survival function")
    >>> _ = plt.title("Kaplan-Meier")
    >>> plt.show()

The shaded band is a 95% confidence interval. It widens along the timeline because fewer and
fewer assets remain at risk, so the tail of the curve rests on very few observations. The curve
drops to zero only if the longest observed time is an event, i.e. every asset still at risk at
that age had the event. If the longest observed time is censored, the curve stops above zero: the
oldest assets were last seen at that age without the event, and the estimator says nothing about
survival beyond it.

Nelson-Aalen: cumulative hazard
----------------------------------------------------------

The Nelson-Aalen estimator targets the cumulative hazard function :math:`H(t)` instead of
the survival function. Where Kaplan-Meier multiplies survival probabilities, Nelson-Aalen 
accumulates failure rates:

.. math::

    \hat{H}(t) = \sum_{i: t_i \leq t} \frac{d_i}{n_i}

Being a sum of non-negative increments, the estimate increases with time. Its slope is the
hazard rate: a steepening curve means the units are wearing out.

.. plot::
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> from assetlife.datasets import load_circuit_breaker
    >>> from assetlife.lifetime_models import NelsonAalen
    >>> dataset = load_circuit_breaker()
    >>> na = NelsonAalen(dataset["time"], event=dataset["event"], entry=dataset["entry"])
    >>> _ = na.plot("chf")
    >>> _ = plt.xlabel("Time")
    >>> _ = plt.ylabel("Estimated cumulative hazard")
    >>> _ = plt.title("Nelson-Aalen")
    >>> plt.show()

These estimators are a good **sanity check** before committing to a parametric shape (see
:doc:`distributions`): if a fitted distribution's survival curve strays far
from the Kaplan-Meier estimate, that shape is probably a poor fit for the data.
