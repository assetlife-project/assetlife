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
