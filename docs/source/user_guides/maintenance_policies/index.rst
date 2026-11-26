Maintenance policies
====================

A **maintenance policy** turns rewards into a decision. Every policy is built from a model 
and exposes the same four cost methods. For unique events, they are the methods of the
underlying renewal reward process but from two vocabularies. A renewal reward process uses the
terms of probability theory while a policy uses the financial terms. Unique event policies also
come in two variants: a *renewal* one, where assets are replaced and a *one-cycle* one, which
stops at the first replacement. Recurrent event policies are built on a non-homogeneous Poisson
process instead.

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Policy
      - ``RenewalRewardProcess``
    * - ``expected_net_present_value``
      - ``expected_total_reward``
    * - ``expected_equivalent_annual_cost``
      - ``expected_equivalent_annual_worth``
    * - ``asymptotic_expected_net_present_value``
      - ``asymptotic_expected_total_reward``
    * - ``asymptotic_expected_equivalent_annual_cost``
      - ``asymptotic_expected_equivalent_annual_worth``

.. toctree::
    :maxdepth: 1

    unique_events_policies
    recurrent_events_policies
