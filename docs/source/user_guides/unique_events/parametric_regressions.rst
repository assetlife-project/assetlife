Parametric regressions
=============================================

A distribution alone can't account for the fact that different assets operate under
different conditions. Regressions extend a baseline distribution with covariates: for
instance, an insulator string's lifetime plausibly depends on the acid concentrations it's
exposed to (see :doc:`datasets`).

Two parametric models
------------------------------------------

AssetLife offers two parametric ways of letting a covariate vector :math:`x` act on a baseline
cumulative hazard :math:`H_0`. Proportional hazard scales the hazard,

.. math::

    H(t, x) = e^{\beta \cdot x} \, H_0(t)

while accelerated failure time rescales the timeline, so that a covariate makes the asset age
faster or slower:

.. math::

    H(t, x) = H_0 \left( t \, e^{-\beta \cdot x} \right)

In both cases :math:`\beta` is the vector of covariate coefficients, but their signs read in
opposite directions. Under the proportional hazard form, a positive coefficient raises the
hazard, so it **shortens** lifetimes. Under the accelerated failure time form, a positive
coefficient slows the asset's aging, so it **lengthens** lifetimes.

Standardizing the covariates
------------------------------------------

AssetLife does not rescale covariates internally: each coefficient is expressed per unit of
the covariate exactly as it is passed. Covariates measured on different scales therefore give
coefficients that cannot be compared with each other. Standardizing them first (subtract the
mean, divide by the standard deviation, as scikit-learn's ``StandardScaler`` does) puts every
covariate on the same scale:

>>> import numpy as np
>>> from assetlife.datasets import load_insulator_string
>>> insulator_data = load_insulator_string()
>>> names = ("pHCl", "pH2SO4", "HNO3")
>>> np.round([insulator_data[name].std() for name in names], 3)
array([0.193, 0.086, 0.06 ])
>>> covar = [
...     (insulator_data[name] - insulator_data[name].mean()) / insulator_data[name].std()
...     for name in names
... ]

From now on, a one-unit change of a covariate means a change of one **standard deviation**:
0.193 for ``pHCl``, 0.086 for ``pH2SO4`` and 0.06 for ``HNO3``.

Fitting a regression
------------------------------------------

Both forms are fitted the same way, with the covariates passed alongside the lifetimes:

>>> from assetlife.lifetime_models import Gompertz, ParametricProportionalHazard
>>> regression = ParametricProportionalHazard(Gompertz()).fit(
...     insulator_data["time"], covar,
...     event=insulator_data["event"], entry=insulator_data["entry"],
... )
>>> np.round(regression.get_params(), 3)
array([ 0.794, -0.229,  0.195,  0.063,  0.029])

The first three parameters are the covariate coefficients (for ``pHCl``, ``pH2SO4`` and
``HNO3`` respectively), and the last two are the baseline Gompertz distribution's own
parameters. ``ParametricAcceleratedFailureTime`` takes the same arguments.

.. _hazard_ratio:

The hazard ratio
------------------------------------------

Coefficients on the hazard scale are hard to read directly. Under the proportional hazard
form, what is interpretable is the ratio of the hazard rates of two assets, because the
baseline cancels:

.. math::

    \textrm{HR} = \frac{h(t, x_2)}{h(t, x_1)}
                = \frac{e^{\beta \cdot x_2} \, h_0(t)}{e^{\beta \cdot x_1} \, h_0(t)}
                = e^{\beta \cdot (x_2 - x_1)}

First, the ratio does not depend on :math:`t`: that is exactly what "proportional hazard" means,
and it is an assumption about the data. If a covariate's effect fades with age, this family of
models cannot represent it.
Second, taking :math:`x_{1, k}` and :math:`x_{2, k} = x_{1, k} + 1` on covariate :math:`k`
(raised by one standard deviation, since it is standardized) leaves:

.. math::

    \textrm{HR} = e^{\beta_k \cdot (x_{2, k} - x_{1, k})}
                = e^{\beta_k}

so :math:`e^{\beta_k}` is the multiplicative effect on the instantaneous failure rate of a
one standard deviation increase in the :math:`k`-th covariate, all others held fixed. Above 1
the asset fails faster, below 1 it lasts longer, and 1 means no effect:

>>> np.round(np.exp(regression.get_params()[:3]), 3)
array([2.212, 0.795, 1.215])

Per standard deviation, all others held fixed:

- one standard deviation more of ``pHCl`` (+0.193) multiplies the failure rate by about 2.2;
- one standard deviation more of ``pH2SO4`` (+0.086) lowers it by about 20%;
- one standard deviation more of ``HNO3`` (+0.06) raises it by about 22%.

Because every covariate is on the same scale, these ratios can be compared directly:
``pHCl`` is by far the most influential of the three.

The accelerated failure time form has no constant hazard ratio at all: there,
:math:`e^{\beta_k}` stretches the time axis, so the effect of a one standard deviation
increase is read as an aging speed rather than as a rate multiplier.
