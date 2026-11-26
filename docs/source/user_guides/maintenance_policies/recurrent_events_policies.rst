Recurrent events policies
=========================

When failures are repairable, an asset is not replaced at each event: it is repaired, and the
:doc:`non-homogeneous Poisson process <../stochastic_processes/nhpp>` (NHPP) describes the pace
of these repairs. The question becomes: **when does replacing the asset become cheaper than 
repairing it?**

:py:class:`~assetlife.policies.RecurrentEventAgeReplacementPolicy` answers it with an age of
replacement ``ar`` and two costs:

* ``cr``, the cost of a **repair**. The asset is repaired minimally and put back
  into service. ``cr`` therefore covers everything related to a repair, the repair itself but
  also emergency work, collateral damage or energy not supplied;
* ``cp``, the cost of the **preventive replacement** of the asset when it reaches ``ar``.

.. warning::

    This policy has no ``cf`` argument: a repair never ends the life of the asset, so its whole
    cost goes into ``cr``.

We take the Gompertz process fitted on the SF6 gas refills of circuit breakers dataset:

>>> import numpy as np
>>> import pandas as pd
>>> from assetlife.datasets import load_gas_refill_circuit_breaker
>>> from assetlife.lifetime_models import Gompertz
>>> from assetlife.policies import RecurrentEventAgeReplacementPolicy
>>> from assetlife.stochastic_processes import NonHomogeneousPoissonProcess
>>> failures, assets = load_gas_refill_circuit_breaker()
>>> failures, assets = pd.DataFrame(failures), pd.DataFrame(assets)
>>> nhpp = NonHomogeneousPoissonProcess(Gompertz()).fit(failures, assets)
>>> policy = RecurrentEventAgeReplacementPolicy(nhpp)

Cost per unit of time
---------------------

A **cycle** is the life of one asset, from its installation to its replacement. Unlike with
unique events, it always lasts exactly :math:`a_r`: the asset is never replaced before, even
when it fails. During the cycle, the asset goes through :math:`\Lambda(a_r)` repairs on
average, each costing :math:`c_r`, where :math:`\Lambda` is the cumulative intensity. Over 
many cycles, the **cost per unit of time** is the expected cost of a cycle divided by its 
duration:

.. math::

    q^\infty(a_r) = \frac{c_p + c_r \, \Lambda(a_r)}{a_r}

>>> np.round(policy.asymptotic_expected_equivalent_annual_cost(
...     ar=np.array([30., 40., 50., 60.]), cr=1., cp=10.
... ), 4)
array([0.3345, 0.256 , 0.2337, 0.363 ])

A small :math:`a_r` pays ``cp`` too often while a large one allows assets to experience more 
repairs. In between lies a minimum that can be computed.

Each replacement resets the intensity of the position to that of a new asset. With the 
optimal age computed in the next section:

.. plot::
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import pandas as pd
    >>> from assetlife.datasets import load_gas_refill_circuit_breaker
    >>> from assetlife.lifetime_models import Gompertz
    >>> from assetlife.policies import RecurrentEventAgeReplacementPolicy
    >>> from assetlife.stochastic_processes import NonHomogeneousPoissonProcess
    >>> failures, assets = load_gas_refill_circuit_breaker()
    >>> failures, assets = pd.DataFrame(failures), pd.DataFrame(assets)
    >>> nhpp = NonHomogeneousPoissonProcess(Gompertz()).fit(failures, assets)
    >>> policy = RecurrentEventAgeReplacementPolicy(nhpp)
    >>> ar_star = policy.compute_optimal_ar(cr=1., cp=10.)
    >>> timeline = np.linspace(0., 150., 1001)
    >>> _ = plt.plot(timeline, nhpp.intensity(timeline), "k--", label="never replaced")
    >>> _ = plt.plot(
    ...     timeline, nhpp.intensity(timeline % ar_star), label=f"replaced at {ar_star:.1f} years"
    ... )
    >>> _ = plt.ylim(0., 0.6)
    >>> _ = plt.xlabel("Time")
    >>> _ = plt.ylabel("Intensity of the position")
    >>> _ = plt.legend()
    >>> plt.show()

Optimal age of replacement
--------------------------

``compute_optimal_ar`` finds this minimum. Without discounting, it solves:

.. math::

    a_r \, \lambda(a_r) - \Lambda(a_r) = \frac{c_p}{c_r}

At the optimum, the cost per unit of time equals :math:`c_r \, \lambda(a_r^*)`. Below that age, 
keeping the asset is cheaper than the average cost of the policy:

>>> ar_star = policy.compute_optimal_ar(cr=1., cp=10.)
>>> round(float(ar_star), 2)
48.22
>>> round(float(policy.asymptotic_expected_equivalent_annual_cost(ar=ar_star, cr=1., cp=10.)), 4)
0.2321
>>> round(float(1. * nhpp.intensity(ar_star)), 4)
0.2321

On a plot, the cost of the repairs :math:`c_r \, \lambda(a)` crosses the cost per unit of time
of the policy exactly at its minimum:

.. plot::
    :context: close-figs

    >>> ar_range = np.linspace(20., 70., 100)
    >>> costs = policy.asymptotic_expected_equivalent_annual_cost(ar=ar_range, cr=1., cp=10.)
    >>> ar_star = policy.compute_optimal_ar(cr=1., cp=10.)
    >>> _ = plt.plot(
    ...     ar_range, costs, label=r"$q^\infty(a_r)$: cost per unit of time, replacing at $a_r$"
    ... )
    >>> _ = plt.plot(
    ...     ar_range, 1. * nhpp.intensity(ar_range),
    ...     label=r"$c_r \, \lambda(a)$: cost per unit of time of repairs at age $a$",
    ... )
    >>> _ = plt.axvline(ar_star, color="grey", linestyle=":", label=r"optimal $a_r^*$")
    >>> _ = plt.ylim(0., 1.)
    >>> _ = plt.xlabel(r"Age of replacement $a_r$, or age $a$")
    >>> _ = plt.ylabel("Cost per unit of time")
    >>> _ = plt.legend()
    >>> plt.show()

The method takes a ``discounting_rate`` as well (see
:doc:`../stochastic_processes/renewal_reward_process`). Discounting makes the replacement, paid
at the end of the cycle, count less, so the optimum is postponed:

>>> ar_disc = policy.compute_optimal_ar(cr=1., cp=10., discounting_rate=0.04)
>>> round(float(ar_disc), 2)
51.91
>>> np.round(policy.asymptotic_expected_equivalent_annual_cost(
...     ar=ar_disc, cr=1., cp=10., discounting_rate=0.04
... ), 4)
array([0.075])

The optimal age goes from 48.22 to 51.91 years. The cost, 0.075, is an equivalent annual cost in
today's money: costs paid decades from now are discounted, so it cannot be directly compared with 
the 0.2321 obtained without discounting.

.. note::

    As for unique events, a minimum only exists if the intensity increases. With a constant
    intensity (homogeneous Poisson process), an old asset is repaired as often as a new one,
    and replacing it gains nothing.

Assets already in service
-------------------------

For assets already ``a0`` years old, only the repairs between :math:`a_0` and :math:`a_r` 
remain to pay. With discounting, the expected net present value accounts for it:

>>> np.round(policy.asymptotic_expected_net_present_value(
...     ar=ar_disc, a0=np.array([0., 20., 40.]), cr=1., cp=10., discounting_rate=0.04
... ), 3)
array([1.875, 4.166, 8.982])

The older the asset, the sooner its replacement, and the higher the cost in today's money.

Planning the fleet
------------------

To estimate the maintenance workload, apply the optimal replacement age to the
entire fleet of 35,695 breakers. Each breaker's current age is its age at the end
of the observation period. Breakers already older than :math:`a_r^*` are overdue
and replaced immediately, so their age is reset to zero:

>>> ages = np.minimum(assets["end"].to_numpy(), ar_star)
>>> int((assets["end"] >= ar_star).sum())
2712

The policy does not provide annual counts directly, but they can be derived
from the cumulative intensity. Consider a breaker with current age :math:`a_0`,
after applying the initial replacements. Its scheduled replacement dates are

.. math::

    a_r - a_0,\quad 2a_r - a_0,\quad 3a_r - a_0,\quad \dots

where :math:`a_r = a_r^*`. Between replacements, the breaker follows the same
failure process as a new asset, with its age reset to zero at each replacement.

To calculate the expected number of repairs over the next :math:`t` years,
count the completed cycles and the elapsed time in the current cycle:

.. math::

    k \Lambda(a_r)
    + \Lambda\big((a_0 + t) \bmod a_r\big)
    - \Lambda(a_0)

where

.. math::

    k = \left\lfloor \frac{a_0 + t}{a_r} \right\rfloor

is the number of completed cycles. Each completed cycle contributes
:math:`\Lambda(a_r)` expected repairs. The last two terms account for repairs
during the current cycle, subtracting the cumulative intensity accumulated
before the observation period. In particular, the expected number of future
repairs is zero at :math:`t=0`.

Summing these quantities over the entire fleet gives the expected cumulative
numbers of replacements and repairs at each year:

>>> years = np.arange(0., 101.)[:, None]
>>> replacements = np.clip(np.ceil((years - (ar_star - ages)) / ar_star), 0., None).sum(axis=1)
>>> cycles = np.floor((ages + years) / ar_star)
>>> repairs = (
...     cycles * nhpp.cumulative_intensity(ar_star)
...     + nhpp.cumulative_intensity((ages + years) % ar_star)
...     - nhpp.cumulative_intensity(ages)
... ).sum(axis=1)
>>> annual_replacements, annual_repairs = np.diff(replacements), np.diff(repairs)
>>> annual_replacements[:5]
array([3267.,  729.,  884., 1206., 1628.])
>>> np.round(annual_repairs[:5])
array([1369., 1487., 1589., 1641., 1602.])

The cumulative counts are evaluated at integer years, and their differences
give the annual workload. Replacements occurring immediately at the start of
the planning period are included in the first year.

The first year concentrates the 2,712 overdue breakers, bringing the total to
3,267 replacements. Subsequent replacements follow the current age
distribution of the fleet, while the expected number of repairs stabilizes
around 1,500 to 1,600 per year:

.. plot::
    :context: close-figs

    >>> ages = np.minimum(assets["end"].to_numpy(), ar_star)
    >>> years = np.arange(0., 101.)[:, None]
    >>> replacements = np.clip(np.ceil((years - (ar_star - ages)) / ar_star), 0., None).sum(axis=1)
    >>> cycles = np.floor((ages + years) / ar_star)
    >>> repairs = (
    ...     cycles * nhpp.cumulative_intensity(ar_star)
    ...     + nhpp.cumulative_intensity((ages + years) % ar_star)
    ...     - nhpp.cumulative_intensity(ages)
    ... ).sum(axis=1)
    >>> calendar = 2025 + years[:-1, 0]
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    >>> _ = ax1.bar(
    ...     calendar, np.diff(replacements), width=1., align="edge", color="C1",
    ...     edgecolor="black", linewidth=0.3, label="preventive replacements",
    ... )
    >>> _ = ax1.set_ylabel("Replacements per year")
    >>> _ = ax1.legend(loc="upper right")
    >>> _ = ax2.bar(
    ...     calendar, np.diff(repairs), width=1., align="edge", color="C0",
    ...     edgecolor="black", linewidth=0.3, label=f"repairs, replaced at {ar_star:.1f} years",
    ... )
    >>> _ = ax2.set_xlabel("Year")
    >>> _ = ax2.set_ylabel("Expected repairs per year")
    >>> _ = ax2.legend(loc="upper right")
    >>> plt.show()

The replacement schedule is periodic: each breaker is replaced every
:math:`a_r^*` years after its initial replacement. Consequently, the annual
replacement pattern repeats every 48 years. The initial replacement wave
therefore reappears after 48 years. Spreading overdue replacements over
several years would smooth this initial peak and the subsequent replacement
waves as well.
