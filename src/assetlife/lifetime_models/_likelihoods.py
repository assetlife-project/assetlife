from typing import final, override

import numpy as np
import optype.numpy as onp

from ._base import FittableParametricLifetimeModel
from ._datasets import LifetimeData
from assetlife._optimization import FitConfig, MaximumLikelihoodOptimizer
from assetlife.typing import CoercibleFloat64_ND


@final
class LifetimeLikelihood(
    MaximumLikelihoodOptimizer[
        FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]], LifetimeData
    ]
):
    """
    Maximum likelihood estimator from lifetime data.

    Parameters
    ----------
    model : FittableParametricLifetimeModel
        Model with initialized parameters.
    data : LifetimeData
        Preprocessed lifetime observations.
    config : FitConfig
        Configuration used by the optimizer.

    Attributes
    ----------
    model : FittableParametricLifetimeModel
        Model used by the likelihood.
    data : LifetimeData
        Preprocessed lifetime observations.
    config : FitConfig
        Configuration used by the optimizer.
    """

    data: LifetimeData

    def __init__(
        self,
        model: FittableParametricLifetimeModel[*tuple[CoercibleFloat64_ND, ...]],
        data: LifetimeData,
        config: FitConfig,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log

    @property
    @override
    def nb_observations(self) -> int:
        return self.data.nb_observations

    @override
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        self.model.set_params(params)
        return (
            self._complete_time_contrib()
            + self._censored_time_contrib()
            + self._left_truncations_contrib()
        )

    def jac_negative_log(
        self, params: onp.Array1D[np.float64]
    ) -> onp.Array1D[np.float64]:
        """
        Jacobian of the negative log likelihood.

        The Jacobian is computed with respect to parameters.

        Parameters
        ----------
        params : 1d array of floats
            Parameter values.

        Returns
        -------
        out : 1d array of floats
        """
        self.model.set_params(params)
        return (
            self._jac_complete_time_contrib()
            + self._jac_censored_time_contrib()
            + self._jac_left_truncations_contrib()
        )

    def _complete_time_contrib(self) -> float:
        if self.data.complete_time.size == 0:
            return 0.0
        return -np.sum(
            np.log(
                self.model.pdf(self.data.complete_time, *self.data.complete_time_covars)
            )
        )

    def _jac_complete_time_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.complete_time.size == 0:
            return np.zeros_like(self.model.get_params())
        jac = -self.model.jac_pdf(
            self.data.complete_time, *self.data.complete_time_covars
        ) / self.model.pdf(self.data.complete_time, *self.data.complete_time_covars)

        return np.sum(jac, axis=(1, 2))

    def _censored_time_contrib(self) -> float:
        if self.data.censored_time.size == 0:
            return 0.0
        if self.data.censored_time.shape[-1] > 1:
            # interval censored time
            return np.sum(
                -np.log(
                    10**-10
                    + self.model.cdf(
                        self.data.censored_time[:, 1], *self.data.censored_time_covars
                    )
                    - self.model.cdf(
                        self.data.censored_time[:, 0], *self.data.censored_time_covars
                    )
                ),
            )
        # right censored time
        return np.sum(
            self.model.chf(self.data.censored_time, *self.data.censored_time_covars)
        )

    def _jac_censored_time_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.censored_time.size == 0:
            return np.zeros_like(self.model.get_params())
        if self.data.censored_time.shape[-1] > 1:
            # interval censored time
            jac_interval_censored = (
                self.model.jac_sf(
                    self.data.censored_time[:, 1], *self.data.censored_time_covars
                )
                - self.model.jac_sf(
                    self.data.censored_time[:, 0], *self.data.censored_time_covars
                )
            ) / (
                10**-10
                + self.model.cdf(
                    self.data.censored_time[:, 1], *self.data.censored_time_covars
                )
                - self.model.cdf(
                    self.data.censored_time[:, 0], *self.data.censored_time_covars
                )
            )

            return np.sum(jac_interval_censored, axis=(1, 2))
        # right censored time
        return np.sum(
            self.model.jac_chf(
                self.data.censored_time, *self.data.censored_time_covars
            ),
            axis=(1, 2),
        )

    def _left_truncations_contrib(self) -> float:
        if self.data.left_truncations.size == 0:
            return 0.0
        return -np.sum(
            self.model.chf(
                self.data.left_truncations, *self.data.left_truncations_covars
            )
        )

    def _jac_left_truncations_contrib(self) -> onp.ArrayND[np.float64]:
        if self.data.left_truncations.size == 0:
            return np.zeros_like(self.model.get_params())
        jac = -self.model.jac_chf(
            self.data.left_truncations, *self.data.left_truncations_covars
        )
        return np.sum(jac, axis=(1, 2))
