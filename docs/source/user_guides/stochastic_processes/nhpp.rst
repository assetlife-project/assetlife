Non-homogeneous Poisson process
===============================

A :doc:`renewal process <renewal_process>` assumes that each event is followed by a replacement:
the asset restarts *as good as new*. Many events are **repairs** though, and the repair
does not rejuvenate the asset. Take an SF6 circuit breaker whose seals slowly degrade: when gas
leaks, a technician refills it. Service is restored, but the leaking seal is still there, and the
asset restarts *as bad as old*. Such a repair is called **minimal**. Refills come back more and more 
often as the asset ages, without replacement.

The number of events :math:`N(t)` an asset goes through up to age :math:`t` is then a
**non-homogeneous Poisson process** (NHPP). Its pace is given by the **intensity**
:math:`\lambda(t)`. A constant intensity gives the (homogeneous) Poisson process. An 
increasing one describes an asset that is aging, the most common case in maintenance.

Repair histories
----------------

An NHPP is not fitted on one duration per asset but on the **sequence** of events observed on
each asset. This information comes as two tables:

* ``failures``: one row per event, with the asset ``id`` and its age ``time`` at the event;
* ``assets``: one row per asset, with its ``id`` and the ages ``start`` and ``end`` at which its
  observation began and ended. An asset without any event appears in this table only.

The *gas refill circuit breaker* dataset, about SF6 refills, is loaded directly in this format,
as two NumPy structured arrays:

>>> import numpy as np
>>> import pandas as pd
>>> from assetlife.datasets import load_gas_refill_circuit_breaker
>>> failures, assets = load_gas_refill_circuit_breaker()
>>> print(failures[:3])
[(4, 43.088) (4, 47.37 ) (4, 49.325)]
>>> print(assets[3:5])
[(3, 42., 51.) (4, 42., 51.)]

Equipment 3 was observed from 42 to 51 years without any refill. Equipment 4, observed over the
same ages, was refilled three times. ``fit`` expects pandas DataFrames, so the two arrays are
converted first:

>>> failures, assets = pd.DataFrame(failures), pd.DataFrame(assets)
>>> len(assets), len(failures), failures["id"].nunique()
(35695, 14154, 6329)

Out of 35,695 circuit breakers, 14,1546 happened in total and 6,329 had at least one refill over 
their observation window.

Fitting
-------

The parameters are estimated by maximum likelihood. Each event contributes its intensity, and
each observation window the expected number of events in it:

.. math::

    \ln L = \sum_{\text{events}} \ln \lambda(t_{ij})
          - \sum_{\text{assets}} \left( \Lambda(e_i) - \Lambda(s_i) \right)

where :math:`t_{ij}` is the age at the :math:`j`-th event of asset :math:`i`, observed between
:math:`s_i` and :math:`e_i`, and :math:`\Lambda(t) = \int_0^t \lambda(x) \, \mathrm{d}x` is the
**cumulative intensity**.

The intensity is given by a model (Weibull for instance). The time to the first event follows that 
model.
After a minimal repair, the asset is as old as it was just before, so the time to the next
event follows the same model, conditioned on having survived up to the current age. The
intensity of
:py:class:`~assetlife.stochastic_processes.NonHomogeneousPoissonProcess` is thus the hazard
function of the model, and its cumulative intensity the cumulative hazard function. The 
model passed to the process gives the shape of the intensity only: its parameters are
estimated by ``fit`` and you can compare candidate shapes:

>>> from assetlife.lifetime_models import Gompertz, Weibull
>>> from assetlife.stochastic_processes import NonHomogeneousPoissonProcess
>>> weibull_nhpp = NonHomogeneousPoissonProcess(Weibull()).fit(failures, assets)
>>> print(weibull_nhpp.fitting_results)
fitted params : [11.5486, 0.0211381]
AIC           : 51741.8
AICc          : 51741.8
BIC           : 51756.9
>>> nhpp = NonHomogeneousPoissonProcess(Gompertz()).fit(failures, assets)
>>> print(nhpp.fitting_results)
fitted params : [0.000101132, 0.194428]
AIC           : 50397.7
AICc          : 50397.7
BIC           : 50412.8

The Gompertz intensity, :math:`c \lambda e^{\lambda t}`, fits better.

The intensity
-------------

``intensity`` and ``cumulative_intensity`` evaluate :math:`\lambda` and :math:`\Lambda`:

>>> np.round(nhpp.intensity(np.array([30., 40., 50.])), 3)
array([0.007, 0.047, 0.328])

A 30-year-old breaker needs about one refill every 150 years, a 50-year-old one about one every
3 years.

The cumulative intensity is the expected number of events up to :math:`t`,
:math:`\mathbb{E}[N(t)] = \Lambda(t)`. For a breaker aged 40, the expected number of refills 
over the next 10 years are:

>>> expected = nhpp.cumulative_intensity(50.) - nhpp.cumulative_intensity(40.)
>>> round(float(expected), 3)
1.445

This expected number is the area under the intensity between the two ages:

.. plot::
    :context: close-figs

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from assetlife.datasets import load_gas_refill_circuit_breaker
    >>> from assetlife.lifetime_models import Gompertz, Weibull
    >>> from assetlife.stochastic_processes import NonHomogeneousPoissonProcess
    >>> failures, assets = load_gas_refill_circuit_breaker()
    >>> failures, assets = pd.DataFrame(failures), pd.DataFrame(assets)
    >>> nhpp = NonHomogeneousPoissonProcess(Gompertz()).fit(failures, assets)
    >>> timeline = np.linspace(0., 55., 200)
    >>> _ = plt.plot(timeline, nhpp.intensity(timeline), label="Gompertz intensity")
    >>> window = np.linspace(40., 50., 50)
    >>> _ = plt.fill_between(
    ...     window, nhpp.intensity(window), alpha=0.3,
    ...     label="Λ(50) - Λ(40): expected refills between 40 and 50 years",
    ... )
    >>> _ = plt.xlabel("Age")
    >>> _ = plt.ylabel("Intensity (refills per year)")
    >>> _ = plt.legend(loc="upper left")
    >>> plt.show()

Counts per period
-----------------

Records sometimes give the number of events per period instead of their dates, for instance
from a yearly report. ``partial_observations`` takes these counts, one row per period. The
asset still appears in ``assets`` with its whole observation window, which the periods must
cover. With one more breaker, observed from 40 to 50 years, with one refill in the first 5 years
and two in the next 5:

>>> partial = pd.DataFrame({
...     "id": [35698, 35698], "start": [40., 45.], "end": [45., 50.], "count": [1, 2],
... })
>>> new_asset = pd.DataFrame({"id": [35698], "start": [40.], "end": [50.]})
>>> nhpp_partial = NonHomogeneousPoissonProcess(Gompertz()).fit(
...     failures, pd.concat([assets, new_asset]), partial_observations=partial
... )
>>> print(nhpp_partial.fitting_results)
fitted params : [0.000101162, 0.194425]
AIC           : 50402.2
AICc          : 50402.2
BIC           : 50417.3

To sample trajectories of a fitted process, see :doc:`../sampling`. An NHPP counts repairs but
does not replace the asset. Deciding when replacing becomes cheaper than repairing is the subject
of :doc:`../maintenance_policies/recurrent_events_policies`.
