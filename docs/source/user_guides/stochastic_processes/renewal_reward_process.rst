Renewal process with rewards
============================

A :doc:`renewal process <renewal_process>` says *how many* renewals to expect. To decide
between maintenance strategies you also need to know *how much* they cost.  A **renewal reward
process** attaches an amount, the **reward**, to each renewal, and accumulates these amounts over
time. Note that the vocabulary is generic: a reward is an amount of any sign and unit.
:py:class:`~assetlife.stochastic_processes.RenewalRewardProcess` is built like a ``RenewalProcess``, 
and inherits all its counters:

>>> from assetlife.datasets import load_circuit_breaker
>>> from assetlife.lifetime_models import Weibull
>>> from assetlife.stochastic_processes import RenewalRewardProcess
>>> dataset = load_circuit_breaker()
>>> weibull = Weibull().fit(time=dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> process = RenewalRewardProcess(lifetime_model=weibull)

What is a reward?
-----------------

A reward gives the amount paid at the end of a cycle, as a function of how long that cycle 
lasted. You describe it with the cost keyword arguments of each method:

.. list-table::
    :header-rows: 1
    :widths: 25 40 35

    * - Arguments
      - Reward :math:`Y` of a cycle of duration :math:`X`
      - Corresponds to
    * - ``cf``
      - :math:`Y = c_f`, whatever :math:`X`
      - run-to-failure
    * - ``cf``, ``cp``, ``ar``
      - :math:`Y = c_f` if :math:`X < a_r`, :math:`c_p` otherwise
      - preventive age replacement
    * - ``cf1``, ``cp1`` (optional)
      - for the first cycle only
      - first replacement of an asset already in service

The first case is the simplest possible reward: every renewal is a failure and costs ``cf``. In
the long run, it costs ``cf`` once every mean lifetime:

>>> round(float(process.asymptotic_expected_equivalent_annual_worth(cf=11.)), 5)
0.15015

The second case makes the amount depend on *why* the asset was renewed: a failure before ``ar``
costs ``cf``, reaching ``ar`` costs ``cp``. ``cp`` and ``ar`` describe a single decision, so 
they go together: one without the other raises an error.With the costs and the optimal age of 
the :doc:`../getting_started`:

>>> round(float(process.asymptotic_expected_equivalent_annual_worth(cf=11., cp=3., ar=58.8)), 5)
0.09177

Like ``a0`` and ``ar``, costs accept at most 1d arrays, which gives one result per value:

>>> import numpy as np
>>> process.asymptotic_expected_equivalent_annual_worth(
...     cf=np.array([5., 11., 20.]), cp=3., ar=[58.8, 45., 27.]
... )
array([0.06359451, 0.08735154, 0.12187421])

Discounting
-----------

An amount paid later is worth less than the same amount paid now. Every method takes a
``discounting_rate`` :math:`\delta` (0 by default) that applies exponential discounting. Two
factors derive from it:

.. math::

    D(x) = e^{-\delta x}
    \qquad\qquad
    AF(t) = \int_0^t D(x)\,\mathrm{d}x = \frac{1 - e^{-\delta t}}{\delta}

The **discount factor** :math:`D(x)` gives today's value of an amount paid at date :math:`x`. The
**annuity factor** :math:`AF(t)` is today's value of paying an amount per year from 0 to :math:`t`:

>>> round(float(process.asymptotic_expected_equivalent_annual_worth(
...     cf=11., cp=3., ar=58.8, discounting_rate=0.04
... )), 5)
0.03261

Total reward and equivalent annual worth
----------------------------------------

The **expected total reward** :math:`z(t)` is the discounted sum of the rewards paid up to
:math:`t`. It satisfies a renewal equation of the same shape as the renewal function, where
each renewal contributes its discounted reward instead:

.. math::

    z(t) = \int_0^t \mathbb{E}[Y \mid X = x] \, D(x) \, \mathrm{d}F(x)
         + \int_0^t z(t-x) \, D(x) \, \mathrm{d}F(x)

Without discounting, it is simply the two counters of the :doc:`renewal_process` weighted by
their costs:

>>> timeline, z = process.expected_total_reward(tf=100., nb_steps=10, cf=11., cp=3., ar=58.8)
>>> np.round(z, 3)
array([0.   , 0.007, 0.088, 0.393, 1.11 , 2.392, 5.134, 5.289, 5.712,
       6.559])
>>> _, failures = process.expected_number_of_events(tf=100., nb_steps=10, ar=58.8)
>>> _, preventive = process.expected_number_of_preventive_renewals(tf=100., nb_steps=10, ar=58.8)
>>> bool(np.allclose(z, 11. * failures + 3. * preventive))
True

The jump between 55.56 and 66.67 years is the wave of preventive replacements at 58.8 years.
The total grows with the horizon, so it does not compare well across strategies. The
**equivalent annual worth** :math:`z(t) / AF(t)` turns it into a constant annual amount, which
stabilizes over time. Each quantity also has an ``asymptotic_`` variant giving its limit for an
infinite horizon:

.. list-table::
    :header-rows: 1
    :widths: 45 15 40

    * - Method
      - Horizon
      - Returns
    * - ``expected_total_reward``
      - ``tf``
      - :math:`z(t)` along the timeline
    * - ``expected_equivalent_annual_worth``
      - ``tf``
      - :math:`z(t) / AF(t)` along the timeline
    * - ``asymptotic_expected_total_reward``
      - infinite
      - :math:`\lim z(t)`, infinite if :math:`\delta = 0`
    * - ``asymptotic_expected_equivalent_annual_worth``
      - infinite
      - long-run annual amount

>>> float(process.asymptotic_expected_total_reward(cf=11., cp=3., ar=58.8))
inf

Without discounting, renewing forever costs an infinite amount, which is why the annual worth
is the figure used to compare strategies.

.. plot::
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> from assetlife.datasets import load_circuit_breaker
    >>> from assetlife.lifetime_models import Weibull
    >>> from assetlife.stochastic_processes import RenewalRewardProcess
    >>> dataset = load_circuit_breaker()
    >>> weibull = Weibull().fit(
    ...     time=dataset["time"], event=dataset["event"], entry=dataset["entry"]
    ... )
    >>> process = RenewalRewardProcess(lifetime_model=weibull)
    >>> for label, costs in (
    ...     ("run to failure", {"cf": 11.}),
    ...     ("ar = 50", {"cf": 11., "cp": 3., "ar": 50}),
    ... ):
    ...     timeline, q = process.expected_equivalent_annual_worth(
    ...         tf=300., nb_steps=301, discounting_rate=0.04, **costs
    ...     )
    ...     q_inf = process.asymptotic_expected_equivalent_annual_worth(
    ...         discounting_rate=0.04, **costs
    ...     )
    ...     line, = plt.plot(timeline, q, label=label)
    ...     _ = plt.axhline(q_inf, color=line.get_color(), linestyle="--")
    >>> _ = plt.xlabel("Time")
    >>> _ = plt.ylabel("Expected equivalent annual worth")
    >>> _ = plt.legend()
    >>> plt.show()

The dashed lines are the asymptotic values. The age of replacement is set to 50 years for
illustration only: it is not the optimum, 58.8 years, but it shows more replacement cycles, with
a jump at 50, 100 and 150 years. Each jump is a wave of preventive replacements, smaller than the
previous one, since failures shift the replacement dates and discounting makes later costs
count less.

First-cycle rewards
-------------------

For assets already in service (``a0``), the first replacement may cost something different, for
instance because it includes a removal or a technology change. ``cf1`` and ``cp1`` set the
rewards of that first cycle only:

>>> timeline, z = process.expected_total_reward(tf=100., nb_steps=5, cf=11., a0=30.)
>>> np.round(z, 3)
array([ 0.   ,  2.094,  6.782, 11.136, 14.237])
>>> timeline, z1 = process.expected_total_reward(tf=100., nb_steps=5, cf=11., cf1=20., a0=30.)
>>> np.round(z1, 3)
array([ 0.   ,  3.8  , 12.209, 19.46 , 23.209])

They only apply to a delayed process (``a0`` given, or a ``first_lifetime_model``): otherwise
nothing distinguishes the first cycle from the others.

See the maintenance policies of the :doc:`/api/index` for the full list of methods.
