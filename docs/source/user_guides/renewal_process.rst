Renewal process
===============

A unique event model describes **one** asset until **one** event. In practice the asset is 
replaced when that event occurs, its replacement is replaced in turn, and so on. If every 
replacement is *as good as new*, the successive lifetimes are independent draws from the same 
lifetime model: this sequence of replacements is a **renewal process**.

:py:class:`~assetlife.stochastic_processes.RenewalProcess` wraps any lifetime model this way:

>>> from assetlife.datasets import load_circuit_breaker
>>> from assetlife.lifetime_models import Weibull
>>> from assetlife.stochastic_processes import RenewalProcess
>>> dataset = load_circuit_breaker()
>>> weibull = Weibull().fit(time=dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> renewal_process = RenewalProcess(lifetime_model=weibull)

The renewal function
--------------------

The number of renewals :math:`N(t)` observed up to time :math:`t` is random. What is useful for
planning is its average over a fleet, the **renewal function** :math:`m(t) = \mathbb{E}[N(t)]`.
It is the solution of the **renewal equation**:

.. math::

    m(t) = F_1(t) + \int_0^t m(t-x) \, \mathrm{d}F(x)

:math:`F` is the cumulative distribution function of the lifetime model and :math:`F_1` the one
of the *first* lifetime (the two only differ for assets that are not new at time 0, see
:ref:`below <rp_delayed>`). The first renewal happens at some date :math:`x`, after which
everything starts over: hence the integral. AssetLife solves this equation numerically on a
regular timeline, given a final time ``tf`` and a number of points ``nb_steps``:

>>> timeline, m = renewal_process.renewal_function(tf=200., nb_steps=5)
>>> timeline
array([  0.,  50., 100., 150., 200.])
>>> m
array([0.        , 0.15360633, 0.9776673 , 1.59320522, 2.30440966])

After 200 years, each position of the fleet has been renewed about 2.3 times on average. Its
derivative, the **renewal density** :math:`\omega(t) = m'(t)`, is the rate of renewals at a
given date. Over a long horizon it tends to one renewal every mean lifetime (elementary renewal
theorem):

.. math::

    \lim_{t \to \infty} \omega(t) = \lim_{t \to \infty} \frac{m(t)}{t} = \frac{1}{\mathbb{E}[X]}

>>> timeline, density = renewal_process.renewal_density(tf=200., nb_steps=5)
>>> density
array([0.        , 0.01053298, 0.01478043, 0.01268122, 0.01366078])
>>> round(float(1 / weibull.mean()), 5)
0.01365

.. _rp_delayed:

Assets already in service
-------------------------

If an asset is already ``a0`` years old at time 0, its first lifetime is conditioned on 
having survived up to ``a0``, while the following ones start from zero. Such a process is 
called **delayed**. Passing an array gives one column per asset:

>>> import numpy as np
>>> timeline, m = renewal_process.renewal_function(tf=200., nb_steps=5, a0=np.array([0., 30., 60.]))
>>> m.shape
(5, 3)
>>> np.round(m, 3)
array([[0.   , 0.   , 0.   ],
       [0.154, 0.611, 0.95 ],
       [0.978, 1.327, 1.506],
       [1.593, 1.991, 2.245],
       [2.304, 2.679, 2.899]])

The older the asset, the sooner its first replacement, and the gap persists over the whole
horizon. More generally, a different law for the first lifetime can be given at construction with
``RenewalProcess(lifetime_model, first_lifetime_model=...)``. This is useful if the technology of
the underlying asset changed between the old one and the replaced one.

Preventive replacements
-----------------------

With an age of replacement ``ar``, each asset is replaced either at failure or when it reaches
``ar``, whichever comes first: the lifetime model is replaced by :math:`\min(X, a_r)`. The
renewals then split into two kinds, each with its own counter:

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - Method
      - Expected number of ...
    * - ``renewal_function``
      - renewals of any kind
    * - ``expected_number_of_events``
      - failures only (preventive replacements are not events)
    * - ``expected_number_of_preventive_renewals``
      - preventive replacements at ``ar``

Every renewal is one or the other, so the two counters add up to the renewal function. With the
optimal age of the :doc:`getting_started`, about 59 years:

>>> timeline, m = renewal_process.renewal_function(tf=200., nb_steps=5, ar=58.8)
>>> timeline, failures = renewal_process.expected_number_of_events(tf=200., nb_steps=5, ar=58.8)
>>> timeline, preventive = renewal_process.expected_number_of_preventive_renewals(
...     tf=200., nb_steps=5, ar=58.8
... )
>>> np.round(failures, 3)
array([0.   , 0.154, 0.417, 0.68 , 0.943])
>>> np.round(preventive, 3)
array([0.   , 0.   , 0.749, 1.499, 2.248])
>>> bool(np.allclose(m, failures + preventive))
True

Replacing at 59 years increases the total number of renewals (3.2 instead of 2.3 over 200
years) but divides the number of failures by more than two.

.. plot::
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> from assetlife.datasets import load_circuit_breaker
    >>> from assetlife.lifetime_models import Weibull
    >>> from assetlife.stochastic_processes import RenewalProcess
    >>> dataset = load_circuit_breaker()
    >>> weibull = Weibull().fit(
    ...     time=dataset["time"], event=dataset["event"], entry=dataset["entry"]
    ... )
    >>> renewal_process = RenewalProcess(lifetime_model=weibull)
    >>> timeline, m = renewal_process.renewal_function(tf=200., nb_steps=201)
    >>> _, m_ar = renewal_process.renewal_function(tf=200., nb_steps=201, ar=58.8)
    >>> _, failures_ar = renewal_process.expected_number_of_events(tf=200., nb_steps=201, ar=58.8)
    >>> _ = plt.plot(timeline, m, label="run to failure: all renewals")
    >>> _ = plt.plot(timeline, m_ar, label="ar = 58.8: all renewals")
    >>> _ = plt.plot(timeline, failures_ar, label="ar = 58.8: failures only")
    >>> _ = plt.plot(
    ...     timeline, timeline / weibull.mean(), "k--", label="t / E[X]"
    ... )
    >>> _ = plt.xlabel("Time")
    >>> _ = plt.ylabel("Expected number of renewals")
    >>> _ = plt.legend()
    >>> plt.show()

The renewal equation assumes each replacement leaves the asset *as good as new*. A
:py:class:`~assetlife.stochastic_processes.NonHomogeneousPoissonProcess` leaves it *as bad as
old*: repair does not equal renew, and the expected number of events is the cumulative intensity
:math:`\Lambda(t)`.

This section shows that a renewal process only counts. To attach a cost to each renewal, see the 
details in :doc:`renewal_reward_process`.
