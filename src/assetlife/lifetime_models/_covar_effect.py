"""Covariates effect of lifetime regressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from assetlife._base import ParametricModel

if TYPE_CHECKING:
    import optype.numpy as onp

    from assetlife._typing import CoercibleFloat64_ND, Float64_ND


class LinearCovarEffect(ParametricModel):
    r"""
    Linear covariates effect of lifetime regressions.

    .. math::

        g(\beta, x) = e^{\beta \cdot x}

    where :math:`x` is a vector of covariates and :math:`\beta` the vector of
    coefficients.

    Parameters
    ----------
    *coefficients : float or None
        Coefficients :math:`\beta`, one per covariate. None if unknown, e.g.
        before fitting.
    """

    def __init__(self, *coefficients: float | None) -> None:
        super().__init__(*coefficients)

    def g(self, *covar: CoercibleFloat64_ND) -> Float64_ND:
        """
        The covariates effect.

        Parameters
        ----------
        *covar : float or np.ndarray
            Covariate values, one per coefficient. They are broadcast together.

        Returns
        -------
        out : np.float64 or np.ndarray
            Values of the covariates effect, of the broadcast shape of
            ``covar``.

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
        The jacobian of the covariates effect with respect to the coefficients.

        Parameters
        ----------
        *covar : float or np.ndarray
            Covariate values, one per coefficient. They are broadcast together.

        Returns
        -------
        out : np.ndarray
            Derivatives with respect to each coefficient. The first dimension
            holds the coefficients, the others are the broadcast shape of
            ``covar``.
        """
        g = self.g(*covar)
        broadcasted_covar = np.broadcast_arrays(*covar)
        stack_covar = np.stack(broadcasted_covar, axis=0)
        return stack_covar * g

    @override
    def __repr__(self) -> str:
        return f"LinearCovarEffect({self.get_params().item()!r})"
