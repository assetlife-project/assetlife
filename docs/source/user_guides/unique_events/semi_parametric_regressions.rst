.. _cox_model:

Semi-parametric regressions
===========================

``ParametricProportionalHazard`` (see :doc:`parametric_regressions`) requires a baseline shape, 
and the coefficients absorb whatever the shape gets wrong. The Cox proportional hazard
model removes that assumption: it estimates the covariate effect while leaving the baseline
hazard entirely unspecified. Hence "semi-parametric": only the covariate part is parametric.

Fitting the Cox model
---------------------

>>> import numpy as np
>>> import polars as pl
>>> from assetlife.datasets import load_insulator_string
>>> from assetlife.lifetime_models import SemiParametricProportionalHazard
>>> insulator_data = load_insulator_string()
>>> names = ["pHCl", "pH2SO4", "HNO3"]
>>> standardized_data = insulator_data.with_columns(
...     (pl.col(name) - pl.col(name).mean()) / pl.col(name).std() for name in names
... )
>>> cox = SemiParametricProportionalHazard(standardized_data, names)
>>> np.round(cox.get_params(), 3)
array([ 0.851, -0.255,  0.233])

The covariates are standardized exactly as in :doc:`parametric_regressions`, so each
coefficient is expressed per standard deviation of its covariate. These three coefficients are
directly comparable to the first three of the ``ParametricProportionalHazard`` fit
(``[0.794, -0.229, 0.195]``): both are fitted on the same rows, with the same ``event`` and
``entry``, and give the same signs, the same ordering and close magnitudes, obtained here
without needing to get the baseline shape right. The remaining gap is what the Gompertz shape
absorbs in the parametric fit. Like every coefficient in these pages, they are returned on the
log-hazard scale, so they must be exponentiated to be read as hazard ratios, which is how Cox
regression is usually reported:

>>> np.round(np.exp(cox.get_params()), 3)
array([2.342, 0.775, 1.262])

These hazard ratios are read exactly as in the parametric case (see :ref:`hazard_ratio`): the
baseline cancels out of the ratio, so leaving it unspecified changes nothing to their meaning.
Per standard deviation, all others held fixed:

- one standard deviation more of ``pHCl`` (+0.193) multiplies the failure rate by about 2.3;
- one standard deviation more of ``pH2SO4`` (+0.086) lowers it by about 22%;
- one standard deviation more of ``HNO3`` (+0.06) raises it by about 26%.

The same story as the parametric fit: ``pHCl`` dominates, ``HNO3`` raises the failure rate and
``pH2SO4`` lowers it.

The fitted model gives the survival function of an individual asset from its covariate values,
here for the first two assets of the dataset:

>>> first_two = standardized_data.head(2)
>>> estimation = cox.sf(*(first_two[name].to_numpy() for name in names), se=False)
>>> timeline, sf_values = estimation.timeline, estimation.values
>>> timeline[:3]
array([1.1, 2.6, 3. ])
>>> np.round(sf_values[0][:3], 4)
array([0.9999, 0.9999, 0.9998])
>>> np.round(sf_values[1][:3], 4)
array([0.9997, 0.9994, 0.9991])

.. plot::
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> import polars as pl
    >>> from assetlife.datasets import load_insulator_string
    >>> from assetlife.lifetime_models import SemiParametricProportionalHazard
    >>> names = ["pHCl", "pH2SO4", "HNO3"]
    >>> dataset = load_insulator_string().with_columns(
    ...     (pl.col(name) - pl.col(name).mean()) / pl.col(name).std() for name in names
    ... )
    >>> cox = SemiParametricProportionalHazard(dataset, names)
    >>> first_two = dataset.head(2)
    >>> estimation = cox.sf(*(first_two[name].to_numpy() for name in names), se=False)
    >>> timeline, sf_values = estimation.timeline, estimation.values
    >>> _ = plt.plot(timeline, sf_values[0], label="asset 0")
    >>> _ = plt.plot(timeline, sf_values[1], label="asset 1")
    >>> _ = plt.xlabel("time")
    >>> _ = plt.ylabel("survival probability")
    >>> _ = plt.legend()
    >>> plt.show()

The two assets shown here have different covariate values (different acid exposure), hence
the different estimated survival curves.

The partial likelihood
----------------------

The Cox model is not fitted on the full likelihood of the lifetimes, but on a *partial*
likelihood that never involves the baseline hazard. The idea is to look at each observed
failure in turn, and at the assets still in service at that moment. The question asked is:
given that one of them failed, how likely is it that it was precisely this one, rather than
another? Under the proportional hazard form, all these assets share the same baseline hazard,
so it cancels out of that question, and the answer only depends on their covariates: the higher
an asset's hazard ratio, the more likely it is to be the one that failed.

Multiplying these probabilities over all observed failures gives the partial likelihood, and
the coefficients are the ones that make the observed order of failures most likely. Only the
order of the event times matters, not their exact values, which is why the baseline never
needs to be specified. When several failures share the same date, their order is unknown and
AssetLife uses an approximation (Breslow's, or Efron's when ties are frequent).
