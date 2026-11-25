API
===

This section provides comprehensive details about the exposed AssetLife API.
AssetLife is structured into different modules, each with a clear and specific role.
We divided the API documentation close to the same logic.

Lifetime models
---------------

Parametric lifetime models
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: assetlife.lifetime_models

.. rubric:: Parametric lifetime distributions

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :caption: Parametric lifetime models
    :nosignatures:

    Exponential
    Weibull
    Gompertz
    Gamma
    LogLogistic
    EquilibriumDistribution
    MinimumDistribution

.. rubric:: Parametric lifetime regressions

.. autosummary::
    :toctree: parametric_lifetime_models
    :template: class_template.rst
    :nosignatures:

    ParametricProportionalHazard
    ParametricAcceleratedFailureTime

Semiparametric lifetime regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: semiparametric_lifetime_models
    :template: class_template.rst
    :caption: Semiparametric lifetime models
    :nosignatures:

    SemiParametricProportionalHazard

Non parametric lifetime models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: nonparametric_lifetime_models
    :template: class_template.rst
    :caption: Nonparametric lifetime models
    :nosignatures:

    KaplanMeier
    NelsonAalen
    ECDF

Stochastic processes
--------------------

.. currentmodule:: assetlife.stochastic_processes

.. autosummary::
    :toctree: stochastic_process
    :template: class_template.rst
    :caption: Stochastic processes
    :nosignatures:

    RenewalProcess
    RenewalRewardProcess
    NonHomogeneousPoissonProcess

Maintenance policies
--------------------

.. currentmodule:: assetlife.policies

.. autosummary::
    :toctree: maintenance_policies
    :caption: Maintenance policies
    :template: class_template.rst
    :nosignatures:

    AgeReplacementPolicy
    OneCycleAgeReplacementPolicy
    RunToFailurePolicy
    OneCycleRunToFailurePolicy
    RecurrentEventAgeReplacementPolicy

Built-in datasets
-----------------

.. currentmodule:: assetlife.datasets

.. autosummary::
    :toctree: datasets
    :template: function_template.rst
    :caption: Built-in datasets
    :nosignatures:

    ~load_circuit_breaker
    ~load_insulator_string
    ~load_power_transformer

Sampling
--------

AssetLife offers few functions to sample data.

.. currentmodule:: assetlife.sampling

.. autosummary::
    :toctree: sampling
    :template: function_template.rst
    :caption: Sampling
    :nosignatures:

    ~sample_failures_from_nhpp
    ~sample_lifetimes_from_renewal_process
    ~sample_process

