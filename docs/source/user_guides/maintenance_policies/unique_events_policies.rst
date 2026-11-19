Unique events policies
======================

A policy answers an asset manager's question: **when should the assets be replaced, and what will
it cost?** These policies apply to assets described by a :doc:`unique event
<../unique_events/index>`: each asset is replaced as good as new when its unique event occurs.
Two strategies exist:

* **run-to-failure**: the asset is in service until event happens, then replaced at a cost ``cf``;
* **age replacement**: the asset is replaced preventively when it reaches an age ``ar``, at a
  cost ``cp``, or at the cost ``cf`` if event happened before.

Run-to-failure is a specific case of age replacement with an infinite ``ar``. ``cf`` is not only
the price of a new asset since it can include everything related to the event: emergency work, 
collateral damage, or costs related to energy not supplied.

Each strategy is computed over two horizons: assets replaced **indefinitely**, to plan a fleet
over the long run, or **one cycle only**, to decide about the assets currently in service.

>>> from assetlife.datasets import load_circuit_breaker
>>> from assetlife.lifetime_models import Weibull
>>> from assetlife.policies import AgeReplacementPolicy, RunToFailurePolicy
>>> dataset = load_circuit_breaker()
>>> weibull = Weibull().fit(time=dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> run_to_failure = RunToFailurePolicy(baseline=weibull)
>>> age_replacement = AgeReplacementPolicy(baseline=weibull)

Replacing indefinitely
----------------------

``RunToFailurePolicy`` and ``AgeReplacementPolicy`` replace each asset again and again: they are
:doc:`renewal reward processes <../stochastic_processes/renewal_reward_process>` with the associated 
costs. A **cycle** is the life of one asset, from its installation to its replacement. With
a lifetime :math:`X`, it lasts :math:`\min(X, a_r)` and costs :math:`c_f` if :math:`X < a_r`,
:math:`c_p` otherwise.

When repeating the policy over many cycles, the total cost is about the number of cycles times the
expected cost of a cycle. The total duration is about the number of cycles times its expected
duration. Their ratio no longer depends on the number of cycles: it is the **cost per unit of
time** in the long run:

.. math::

    q^\infty(a_r) = \frac{c_f \, F(a_r) + c_p \, S(a_r)}{\int_0^{a_r} S(x) \, \mathrm{d}x}

The numerator weights each cost by its probability: the asset event happens before :math:`a_r` with
probability :math:`F(a_r)`, and survives up to it with probability :math:`S(a_r)`. The
denominator sums, over time, the probability that the asset did not experience the event: each instant
it survives adds to the duration of the cycle. For run-to-failure, it reduces to one event
cost every mean lifetime:

>>> round(float(run_to_failure.asymptotic_expected_equivalent_annual_cost(cf=11.)), 5)
0.15015
>>> round(float(11. / weibull.mean()), 5)
0.15015

For age replacement, it depends on ``ar``. ``cf``, ``cp`` and ``ar`` describe a single decision,
so they always go together:

>>> import numpy as np
>>> np.round(age_replacement.asymptotic_expected_equivalent_annual_cost(
...     ar=np.array([30., 45., 60., 80.]), cf=11., cp=3.
... ), 4)
array([0.107 , 0.0874, 0.0928, 0.1172])

Every method also takes a ``discounting_rate`` and the ages ``a0`` of the assets already in 
service (see :doc:`../stochastic_processes/renewal_reward_process`).
``AgeReplacementPolicy`` also projects the workload, year by year: ``annual_number_of_replacements`` 
counts all replacements, ``annual_number_of_failures`` only those caused by an event (see
:doc:`../stochastic_processes/renewal_process`):

>>> a0 = np.array([0., 20., 40.])
>>> timeline, replacements = age_replacement.annual_number_of_replacements(
...     nb_years=100, ar=48., a0=a0
... )
>>> timeline, failures = age_replacement.annual_number_of_failures(nb_years=100, ar=48., a0=a0)
>>> np.round(replacements.sum(axis=0), 2)
array([2.  , 2.04, 2.22])
>>> np.round(failures.sum(axis=0), 2)
array([0.27, 0.28, 0.3 ])

Each asset is replaced about twice over 100 years, and approximately 13.5% are an
event-related replacement.

Optimal age of replacement
--------------------------

Both terms of :math:`q^\infty` grow with :math:`a_r`. A small :math:`a_r` gives short cycles,
so ``cp`` is paid too often while a large :math:`a_r` lets more failures happen, and the cost drifts
back towards run-to-failure. In between lies a minimum:

.. plot::
    :context: close-figs

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from assetlife.datasets import load_circuit_breaker
    >>> from assetlife.lifetime_models import Weibull
    >>> from assetlife.policies import AgeReplacementPolicy, RunToFailurePolicy
    >>> dataset = load_circuit_breaker()
    >>> weibull = Weibull().fit(
    ...     time=dataset["time"], event=dataset["event"], entry=dataset["entry"]
    ... )
    >>> policy = AgeReplacementPolicy(baseline=weibull)
    >>> ar_range = np.linspace(15., 150., 100)
    >>> costs = policy.asymptotic_expected_equivalent_annual_cost(ar=ar_range, cf=11., cp=3.)
    >>> ar_star = policy.compute_optimal_ar(cf=11., cp=3.)
    >>> rtf_cost = RunToFailurePolicy(baseline=weibull).asymptotic_expected_equivalent_annual_cost(
    ...     cf=11.
    ... )
    >>> _ = plt.plot(ar_range, costs, label="age replacement")
    >>> _ = plt.axhline(rtf_cost, color="k", linestyle="--", label="run to failure")
    >>> _ = plt.axvline(ar_star, color="grey", linestyle=":", label="optimal ar")
    >>> _ = plt.xlabel("Age of replacement ar")
    >>> _ = plt.ylabel("Asymptotic expected equivalent annual cost")
    >>> _ = plt.legend()
    >>> plt.show()

``compute_optimal_ar`` finds this minimum by solving
:math:`\mathrm{d}q^\infty / \mathrm{d}a_r = 0`. Without discounting, this reads:

.. math::

    h(a_r) \int_0^{a_r} S(x) \, \mathrm{d}x - F(a_r) = \frac{c_p}{c_f - c_p}

The left-hand side grows with :math:`a_r` when the hazard rate :math:`h` increases. The
right-hand side only depends on the cost ratio: the cheaper the preventive replacement, the
earlier the optimum.

Without discounting, the cost per unit of time at the optimum equals
:math:`(c_f - c_p) \, h(a_r^*)`: the extra cost of an event, times the risk of failing right now. 
Below that age, keeping the asset one more year is cheaper than the average cost of the policy.

>>> ar_star = age_replacement.compute_optimal_ar(cf=11., cp=3.)
>>> round(float(ar_star), 2)
47.84
>>> round(float(age_replacement.asymptotic_expected_equivalent_annual_cost(
...     ar=ar_star, cf=11., cp=3.
... )), 5)
0.08696
>>> round(float((11. - 3.) * weibull.hf(time=ar_star)), 5)
0.08696

With a discounting rate :math:`\delta`, the cost at the optimum becomes
:math:`(c_f - c_p) \, h(a_r^*) - \delta \, c_p`:

>>> ar_disc = age_replacement.compute_optimal_ar(cf=11., cp=3., discounting_rate=0.04)
>>> round(float(ar_disc), 2)
58.8
>>> round(float(age_replacement.asymptotic_expected_equivalent_annual_cost(
...     ar=ar_disc, cf=11., cp=3., discounting_rate=0.04
... )), 5)
0.03261
>>> round(float((11. - 3.) * weibull.hf(time=ar_disc) - 0.04 * 3.), 5)
0.03261

.. note::

    The minimum exists because the hazard rate of the fitted Weibull increases: an old asset is
    more likely to have an event than a new one. More precisely, a unique optimum exists if 
    :math:`h` is strictly increasing. With a constant hazard rate (exponential lifetime), 
    replacing a working asset gains nothing, and run-to-failure is the best policy.

One cycle only
--------------

To decide about the assets currently in service only, ``OneCycleRunToFailurePolicy`` and
``OneCycleAgeReplacementPolicy`` stop after the first replacement. Without
renewal, the expected cost is the cost of one cycle, :math:`c_f \, F(a_r) + c_p \, S(a_r)`.
For one-cycle policies, ``asymptotic_`` means that the whole cycle is counted, however long it
lasts: it is not an average over many cycles.

>>> from assetlife.policies import OneCycleAgeReplacementPolicy, OneCycleRunToFailurePolicy
>>> one_cycle = OneCycleAgeReplacementPolicy(lifetime_model=weibull)
>>> round(float(one_cycle.asymptotic_expected_net_present_value(ar=ar_star, cf=11., cp=3.)), 3)
4.042
>>> round(float(11. * weibull.cdf(time=ar_star) + 3. * weibull.sf(time=ar_star)), 3)
4.042

The annual cost differs from the renewal one. The renewal policy divides the expected cost by
the expected duration. The one-cycle policy spreads the cost of *this* cycle over *its*
duration, then averages. Short cycles weigh more, so it is higher:

>>> one_cycle_rtf = OneCycleRunToFailurePolicy(lifetime_model=weibull)
>>> round(float(one_cycle_rtf.asymptotic_expected_equivalent_annual_cost(cf=11.)), 5)
0.16953

The criterion is not the same, so neither is the optimal age, even though it lies in the same
region:

>>> ar_one_cycle = one_cycle.compute_optimal_ar(cf=11., cp=3.)
>>> round(float(ar_one_cycle), 2)
43.82

.. note::

    These policies assume each asset is replaced as good as new after a unique event. When the 
    underlying event is defined differently, other policies should be used.
