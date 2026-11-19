"""Covariates effect of lifetime regressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from assetlife.base import ParametricModel

if TYPE_CHECKING:
    import optype.numpy as onp

    from assetlife.typing import CoercibleFloat64_ND, Float64_ND


class LinearCovarEffect(ParametricModel):
    """
    Covariates effect.

    Parameters
    ----------
    coefficients : tuple of float, default is (None,)
        Coefficients of the covariates effect.
    """

    def __init__(self, *coefficients: float | None) -> None:
        super().__init__(*coefficients)

    def g(self, *covar: CoercibleFloat64_ND) -> Float64_ND:
        """
        Returns the covariates effect.

        Parameters
        ----------
        covar : float or np.ndarray
            The covariate values

        Returns
        -------
        out : np.float64 or np.ndarray

        Raises
        ------
        ValueError
            If the number of covariates differs from the number of coefficients.
        """
        nb_coef = self.get_params().size
        if len(covar) != nb_coef:
            msg = (
                f"Invalid number of covar. Got {nb_coef} coefficients but "
                f"{len(covar)} covariates are given."
            )
            raise ValueError(msg)
        broadcasted_covar = np.broadcast_arrays(*covar)
        stack_covar = np.stack(broadcasted_covar, axis=-1)
        return np.exp(np.sum(stack_covar * self.get_params(), axis=-1))

    def jac_g(self, *covar: CoercibleFloat64_ND) -> onp.ArrayND[np.float64]:
        """
        Returns the jacobian of the covariates effect.

        Parameters
        ----------
        covar : float or np.ndarray
            The covariate values

        Returns
        -------
        out : np.ndarray
        """
        g = self.g(*covar)
        broadcasted_covar = np.broadcast_arrays(*covar)
        stack_covar = np.stack(broadcasted_covar, axis=0)
        return stack_covar * g

    @override
    def __repr__(self) -> str:
        return f"LinearCovarEffect({self.get_params().item()!r})"
