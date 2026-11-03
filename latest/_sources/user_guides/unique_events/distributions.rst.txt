Parametric distributions
=============================================

A parametric lifetime distribution assumes a fixed functional shape (Weibull, Gamma,
Gompertz, ...) for the survival function, and estimates its parameters from data. Fitting
one is a single call, accounting for censoring and truncation as described in
:doc:`censoring_and_truncation`:

>>> from assetlife.datasets import load_circuit_breaker
>>> from assetlife.lifetime_models import Weibull
>>> dataset = load_circuit_breaker()
>>> weibull = Weibull().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> weibull.get_params()
array([3.7267452 , 0.01232326])

The two fitted values are the Weibull ``shape`` and ``rate``.

Comparing candidate shapes
--------------------------------------------

Nothing in the data tells you which shape to assume, so fit several and look at them. The
Kaplan-Meier estimator (see :doc:`non_parametric_models`) is the natural reference here: it
makes no shape assumption, so a parametric curve that strays far from it is a poor
description of the data.

.. plot::
    :context: close-figs

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from assetlife.datasets import load_circuit_breaker
    >>> from assetlife.lifetime_models import KaplanMeier, Weibull, Gamma, Gompertz, LogLogistic
    >>> dataset = load_circuit_breaker()
    >>> km = KaplanMeier(dataset["time"], event=dataset["event"], entry=dataset["entry"])
    >>> _ = km.plot("sf", ci=False, color="black", linestyle="--", label="Kaplan-Meier")
    >>> timeline = np.arange(0, 145)
    >>> for distribution in (Weibull(), Gamma(), Gompertz(), LogLogistic()):
    ...     fitted = distribution.fit(
    ...         dataset["time"], event=dataset["event"], entry=dataset["entry"]
    ...     )
    ...     _ = fitted.plot("sf", timeline, ci=False, label=type(fitted).__name__)
    >>> _ = plt.xlabel("Time")
    >>> _ = plt.ylabel("Estimated survival function")
    >>> _ = plt.legend()
    >>> plt.show()

All four shapes agree while the fleet is young and only separate past about 55 years: Gompertz
and Weibull bend down faster, Gamma and LogLogistic keep a heavier tail. Gompertz stays 
closest to the Kaplan-Meier function over the observed range.

Curves can look equally plausible, and a distribution with more parameters can always be made to 
hug the data more closely. The comparison has to be made on the likelihood and the number of 
parameters.

.. _maximum_likelihood:

Maximum likelihood
------------------------------------------

Fitting maximizes the likelihood of the observed data under the chosen shape. Each row
contributes a factor that depends on what was actually observed.

:math:`\theta` denotes the parameters, :math:`f` the probability density, :math:`S` the survival 
function, :math:`F` the cumulative distribution function and :math:`H` the cumulative hazard.

With :math:`\mathcal{D}` the set of observed failures, :math:`\mathcal{C}` the right-censored
ones, :math:`\mathcal{I}` the interval-censored ones and :math:`a_i` the entry age of asset
:math:`i`, ``fit`` minimizes:

.. math::

    -\log L(\theta) = - \sum_{i \in \mathcal{D}} \log f(t_i; \theta)
                      + \sum_{i \in \mathcal{C}} H(t_i; \theta)
                      - \sum_{i \in \mathcal{I}} \log \left[ F(b_i; \theta) - F(a_i; \theta) \right]
                      - \sum_i H(a_i; \theta)

The four contributions to the likelihood can be interpreted as follows:

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - What is known about the asset
      - Contribution to :math:`-\log L(\theta)`
    * - it failed at :math:`t` (complete observation)
      - :math:`-\log f(t;\theta)`
    * - it was still alive at :math:`t` (right censoring)
      - :math:`H(t;\theta)`
    * - it failed somewhere in :math:`[a, b]` (interval censoring)
      - :math:`-\log \left[ F(b;\theta) - F(a;\theta) \right]`
    * - it was already :math:`a` units old when observation began (left truncation)
      - :math:`-H(a;\theta)`

The first three are mutually exclusive: a row is either complete, right-censored or
interval-censored. The fourth one adds to whichever of them applies, on every row that
entered observation after birth. Right censoring is the interval case with
:math:`b = +\infty` and left censoring the interval case with :math:`a = 0`, which is exactly
how they are encoded in AssetLife (see :doc:`censoring_and_truncation`).

Information criteria
------------------------------------------

The log-likelihood alone can't arbitrate between shapes with different numbers of
parameters, since adding parameters can only improve the fit. Information criteria penalize
that: with :math:`k` parameters, :math:`n` observations and :math:`\hat{\ell}` the maximized
log-likelihood,

.. math::

    \textrm{AIC} = 2k - 2\hat{\ell} \qquad
    \textrm{AICc} = \textrm{AIC} + \frac{2k(k+1)}{n - k - 1} \qquad
    \textrm{BIC} = k \log n - 2\hat{\ell}

AICc is the small-sample correction to AIC, and BIC penalizes parameters more heavily as the
sample grows. All three are computed on ``fitting_results``, and in every case the lower the
better:

>>> print(weibull.fitting_results)
fitted params : [3.72675, 0.0123233]
AIC           : 2493.72
AICc          : 2493.72
BIC           : 2506.41

>>> from assetlife.lifetime_models import Gamma, Gompertz
>>> gamma = Gamma().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> print(gamma.fitting_results)
fitted params : [5.57387, 0.0644293]
AIC           : 2503.5
AICc          : 2503.5
BIC           : 2516.19

>>> gompertz = Gompertz().fit(dataset["time"], event=dataset["event"], entry=dataset["entry"])
>>> print(gompertz.fitting_results)
fitted params : [0.00390781, 0.0757955]
AIC           : 2485.57
AICc          : 2485.57
BIC           : 2498.25

All three shapes have two parameters here, so the penalty term is identical and the ranking
is the ranking of the likelihoods: Gompertz first, then Weibull, then Gamma. This confirms
what the plot suggested.

Covariates are the next step: see :doc:`parametric_regressions` for models where lifetime depends 
on the conditions each asset operates under.
