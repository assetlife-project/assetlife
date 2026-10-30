Unique events
=============

Lifetime models describe the time until a **single event** occurs: the duration from a time
origin, typically the installation of the asset, to an event of interest. That event is often
a failure, but it can be any event you choose, as long as it happens at most once per asset.
In this guide, "lifetime" therefore means "time until that event", not necessarily the time
until the asset dies.

This section starts with how duration data is observed (censoring and truncation), then shows
how to build lifetime models from it, from non-parametric estimators to regressions with
covariates.

.. toctree::
    :maxdepth: 1

    datasets
    censoring_and_truncation
    non_parametric_models
    distributions
    parametric_regressions
    semi_parametric_regressions
