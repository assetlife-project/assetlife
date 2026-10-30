Datasets
---------

AssetLife contains three example datasets under ``assetlife.datasets``, all about
power-grid equipment lifetime data. Each one is loaded as a NumPy `structured array
<https://numpy.org/doc/stable/user/basics.rec.html>`_ with three fields (see
:doc:`censoring_and_truncation` for what they mean):

- ``time``: the observed lifetime values.
- ``event``: whether the event actually happened inside the observation window
  (``False`` means the lifetime is *right-censored*, i.e. the asset had not had the event by
  the end of the observation window).
- ``entry``: the age of the asset at the beginning of the observation window (*left-truncation*).

Circuit breakers
^^^^^^^^^^^^^^^^^^

A circuit breaker is a switching device that opens and closes an electrical circuit, and
automatically interrupts the current when a fault such as a short circuit is detected.

>>> from assetlife.datasets import load_circuit_breaker
>>> dataset = load_circuit_breaker()
>>> dataset.dtype.names
('time', 'event', 'entry')
>>> len(dataset)
4204

Power transformers
^^^^^^^^^^^^^^^^^^^^^

A power transformer changes the voltage level of the electricity flowing through the grid,
for instance stepping it up for long-distance transmission or down for distribution.

>>> from assetlife.datasets import load_power_transformer
>>> dataset = load_power_transformer()
>>> dataset.dtype.names
('time', 'event', 'entry')
>>> len(dataset)
1650

Insulator strings
^^^^^^^^^^^^^^^^^^^^

An insulator string is a chain of insulating discs that holds an overhead line conductor on
its tower while keeping it electrically isolated from it.

>>> from assetlife.datasets import load_insulator_string
>>> dataset = load_insulator_string()
>>> dataset.dtype.names
('time', 'event', 'entry', 'pHCl', 'pH2SO4', 'HNO3')
>>> len(dataset)
12000

This is the only dataset with covariates: ``pHCl``, ``pH2SO4`` and ``HNO3``, the acid concentrations 
the insulators are exposed to, which may accelerate their degradation. Used for both the parametric 
regression (:doc:`parametric_regressions`) and the semi-parametric Cox 
model (:doc:`semi_parametric_regressions`).
