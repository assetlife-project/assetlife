Maintenance policies
====================

A **maintenance policy** turns rewards into a decision. Every policy is built from a lifetime 
model and exposes the same four cost methods. They are the methods of the underlying renewal 
reward process, renamed:

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

Each policy also comes in two variants: a *renewal* one, where assets are replaced over a timeline or
indefinitely, to plan a fleet over the long run, and a *one-cycle* one, which stops at the first
replacement, to decide about the assets currently in service.

.. toctree::
    :maxdepth: 1

    run_to_failure
    age_replacement
