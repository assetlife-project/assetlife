"""Likelihoods for stochastic processes."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Self, final, override

import numpy as np

from ._datasets_validation import NHPPData
from assetlife._optimization import FitConfig, MaximumLikelihoodOptimizer
from assetlife.lifetime_models import (
    FittableParametricLifetimeModel,
)
from assetlife.typing import (
    CoercibleFloat64_ND,
)

if TYPE_CHECKING:
    import optype.numpy as onp


@final
class NHPPLikelihood(
    MaximumLikelihoodOptimizer[
        FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]], NHPPData
    ]
):
    r"""
    Likelihood of a non-homogeneous Poisson process given failure observations.

    With :math:`\lambda` the intensity and :math:`\Lambda` the cumulative
    intensity, the negative log-likelihood sums:

    - :math:`-\sum_i \log \lambda(t_i)` over the failure times :math:`t_i`,
    - :math:`\sum_k \Lambda(e_k) - \Lambda(s_k)` over the observation windows
      :math:`]s_k, e_k]` of the assets,
    - :math:`-\sum_j c_j \log(\Lambda(b_j) - \Lambda(a_j))` over the partial
      observations, :math:`c_j` failures counted in :math:`]a_j, b_j]`.

    Use ``from_model`` to build it from a model.

    Parameters
    ----------
    model : FittableParametricLifetimeModel
        Lifetime model whose hazard function is the intensity of the process.
    data : NHPPData
        Validated failure observations.
    config : FitConfig
        Configuration used by the optimizer.
    """

    def __init__(
        self,
        model: FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]],
        data: NHPPData,
        config: FitConfig,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log

    @classmethod
    def from_model(
        cls,
        model: FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]],
        data: NHPPData,
        **kwargs: Any,
    ) -> Self:
        """Likelihood of a NHPP of intensity ``model.hf`` given ``data``.

        ``model`` is adapted in place to the number of covariates of ``data``.
        The likelihood works on a copy of it, so ``model`` is not changed by
        the optimization: use ``.fit(process)`` to set the optimal parameters
        on the process holding ``model``.

        Parameters
        ----------
        model : FittableParametricLifetimeModel
            Lifetime model whose hazard function is the intensity of the process.
        data : NHPPData
            Validated failure observations.
        **kwargs
            Extra keyword arguments passed to `scipy.optimize.minimize
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
            (e.g. ``x0``, ``bounds``, ``method``) and ``covariance_method``.
            Defaults are the parameters initialized by ``model``, its
            bounds, ``method="L-BFGS-B"`` and its default covariance method.

        Returns
        -------
        NHPPLikelihood
        """
        # resize first: x0 and bounds depend on the number of parameters
        model._set_nb_covars(data.nb_covars)  # ruff: ignore[private-member-access]
        x0 = (
            kwargs["x0"] if "x0" in kwargs else model._init_params(data.failures_time)  # ruff: ignore[private-member-access]
        )
        config = FitConfig(x0)
        config.scipy_minimize_options["bounds"] = kwargs.get(
            "bounds", model.get_params_bounds()
        )
        config.scipy_minimize_options["method"] = kwargs.get("method", "L-BFGS-B")
        config.covariance_method = kwargs.get(
            "covariance_method",
            model._default_covariance_method(),  # ruff: ignore[private-member-access]
        )
        return cls(copy.deepcopy(model), data, config)

    @property
    @override
    def nb_observations(self) -> int:
        n = self.data.failures_time.size
        if self.data.partial_observations is not None:
            n += self.data.partial_observations.failures_count.size
        return n

    @override
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        self.model.set_params(params)
        return (
            self._exact_events_contrib()
            + self._observation_period_contrib()
            + self._partial_observation_contrib()
        )

    def jac_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array1D[np.float64]:
        self.model.set_params(params)
        return (
            self._jac_exact_events_contrib()
            + self._jac_observation_period_contrib()
            + self._jac_partial_observation_contrib()
        )

    def _exact_events_contrib(self) -> float:
        return -np.sum(
            np.log(self.model.hf(self.data.failures_time, *self.data.failures_covars))
        )

    def _jac_exact_events_contrib(self) -> onp.ArrayND[np.float64]:
        jac = -self.model.jac_hf(
            self.data.failures_time, *self.data.failures_covars
        ) / self.model.hf(self.data.failures_time, *self.data.failures_covars)
        return np.sum(jac, axis=1)

    def _observation_period_contrib(self) -> float:
        return np.sum(
            self.model.chf(self.data.observations_end, *self.data.observations_covars)
            - self.model.chf(
                self.data.observations_start, *self.data.observations_covars
            )
        )

    def _jac_chf(
        self,
        time: onp.Array1D[np.float64],
        covars: tuple[onp.Array1D[np.float64], ...],
    ) -> onp.Array2D[np.float64]:
        # chf(0) = 0 whatever the parameters, so its jacobian is 0. It is not
        # evaluated at 0, where some models give nan (e.g. log(0) * 0)
        positive = time > 0
        if positive.all():
            return self.model.jac_chf(time, *covars)
        nb_params = self.model.get_params().size
        if not positive.any():
            return np.zeros((nb_params, time.size), dtype=np.float64)
        jac_positive = self.model.jac_chf(
            time[positive], *(covar[positive] for covar in covars)
        )
        # dtype of jac_positive may be complex (complex-step covariance)
        jac = np.zeros((nb_params, time.size), dtype=jac_positive.dtype)
        jac[:, positive] = jac_positive
        return jac

    def _jac_observation_period_contrib(self) -> onp.ArrayND[np.float64]:
        jac = self._jac_chf(
            self.data.observations_end, self.data.observations_covars
        ) - self._jac_chf(self.data.observations_start, self.data.observations_covars)
        return np.sum(jac, axis=1)

    def _partial_observation_contrib(self) -> float:
        partial = self.data.partial_observations
        if partial is None:
            return 0.0
        return np.sum(
            -partial.failures_count
            * np.log(
                self.model.chf(partial.end, *partial.covars)
                - self.model.chf(partial.start, *partial.covars)
            )
        )

    def _jac_partial_observation_contrib(self) -> onp.ArrayND[np.float64]:
        partial = self.data.partial_observations
        if partial is None:
            return np.zeros_like(self.model.get_params(), dtype=np.float64)
        a = self._jac_chf(partial.end, partial.covars) - self._jac_chf(
            partial.start, partial.covars
        )
        b = self.model.chf(partial.end, *partial.covars) - self.model.chf(
            partial.start, *partial.covars
        )
        jac = -partial.failures_count * (a / b)
        return np.sum(jac, axis=1)
