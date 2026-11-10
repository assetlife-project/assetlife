import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, TypeVar

import numpy as np
import optype.numpy as onp
from scipy.optimize import Bounds, approx_fprime, minimize

from assetlife.base import FittingResults, ParametricModel

M = TypeVar("M", bound=ParametricModel)
D = TypeVar("D")


type MinimizeMethod = Literal[
    "Nelder-Mead", "nelder-mead",
    "Powell", "powell",
    "CG", "cg",
    "BFGS", "bfgs",
    "Newton-CG", "newton-cg",
    "L-BFGS-B", "l-bfgs-b",
    "TNC", "tnc",
    "COBYLA", "cobyla",
    "COBYQA", "cobyqa",
    "SLSQP", "slsqp",
    "Trust-Constr", "trust-constr",
    "Dogleg", "dogleg",
    "Trust-NCG", "trust-ncg",
    "Trust-Exact", "trust-exact",
    "Trust-Krylov", "trust-krylov",
]  # fmt: skip


class ScipyMinimizeOptions(TypedDict, total=False):
    """
    Keyword arguments forwarded to ``scipy.optimize.minimize``.

    All keys are optional.
    """

    method: MinimizeMethod
    # float64 arrays during the minimization, but complex arrays when the
    # covariance is approximated by complex step ("cs"): the dtype is left open
    jac: Callable[[onp.Array1D[Any]], onp.Array1D[Any]]
    hess: Callable[[onp.Array1D[np.float64]], onp.Array2D[np.float64]]
    bounds: Bounds
    tol: float


@dataclass
class FitConfig:
    """Configuration for maximum-likelihood fitting.

    Parameters
    ----------
    x0 : float or 1d array-like of float
        Initial parameter value or vector passed to SciPy's optimizer.
    scipy_minimize_options : dict, optional
        Keyword arguments forwarded to ``scipy.optimize.minimize``.
    covariance_method : {"cs", "2point", "exact", False}, default=False
        Method used to estimate the covariance matrix, or ``False`` to skip it.
    """

    x0: onp.ToFloat | onp.ToFloat1D
    scipy_minimize_options: ScipyMinimizeOptions = field(
        default_factory=ScipyMinimizeOptions
    )
    covariance_method: Literal["cs", "2point", "exact", False] = False


class MaximumLikelihoodOptimizer[M: ParametricModel, D](ABC):
    """
    Abstract generic class for maximum-likelihood estimation.

    Subclasses provide the data-specific negative log-likelihood. This class
    runs SciPy minimization and builds the associated ``FittingResults``.

    Notes
    -----
    Jacobian and hessian are not required but they can be implemented in
    concrete likelihoods. To use the jacobian or hessian implementations in the
    likelihood, pass them into ``self.config["scipy_minimize_options"]``.

    Attributes
    ----------
    nb_observations : int
        The number of observations.
    """

    model: M
    data: D
    config: FitConfig

    @property
    @abstractmethod
    def nb_observations(self) -> int: ...

    @abstractmethod
    def negative_log(self, params: onp.Array1D[np.float64]) -> float:
        """
        Negative log likelihood.

        Parameters
        ----------
        params : 1d array of floats
            Parameter values.

        Returns
        -------
        out : float
            Negative log likelihood value.
        """

    def optimize(self) -> FittingResults:
        """
        Search parameter values that maximize the likelihood given data.

        Returns
        -------
        out : FittingResults
            An object that encapsulates optimal parameters and fitting
            information (AIC, variance, etc.).
        """

        optimizer = minimize(
            self.negative_log,
            self.config.x0,
            **self.config.scipy_minimize_options,
        )

        fitting_results = FittingResults(
            self.nb_observations,
            np.copy(optimizer.x),
            optimizer.success,
            optimizer.fun,
        )

        if not fitting_results.success:
            warnings.warn(
                "The negative log-likelihood minimization did not exited successfully.",
                stacklevel=2,
            )

        if self.config.covariance_method is False:
            return fitting_results

        jac = self.config.scipy_minimize_options.get("jac", None)
        hess = self.config.scipy_minimize_options.get("hess", None)
        if jac is not None and self.config.covariance_method != "exact":
            fitting_results.covariance_matrix = _approx_parameters_covariance(
                fitting_results.optimal_params,
                jac,
                method=self.config.covariance_method,
            )
        if hess is not None and self.config.covariance_method == "exact":
            fitting_results.covariance_matrix = np.linalg.pinv(
                hess(fitting_results.optimal_params),
            )
        return fitting_results


def _approx_parameters_covariance(
    params: onp.Array1D[np.float64],
    jac_negative_log: Callable[[onp.Array1D[np.number]], onp.Array1D[np.number]],
    method: Literal["2point", "cs"] = "cs",
) -> onp.Array2D[np.float64] | None:
    """
    Approximate the covariance matrix of fitted parameters.

    Parameters
    ----------
    params : 1darray of floats
        The parameter values.
    jac_negative_log : Callable
        A function taking 1d array of floats and returning 1d array of floats.
    method : "2point" or "cs", default to "cs"
        The approximation method to use.
    """

    size = params.size
    eps = 1e-6
    hess = np.empty((size, size), dtype=np.float64)

    # hessian 2 point
    if method == "2point":

        def jac_param_i(
            i: int,
        ) -> Callable[[onp.Array1D[np.float64]], np.float64]:
            def f(params: onp.Array1D[np.float64]) -> np.float64:
                return jac_negative_log(params)[i]

            return f

        for i in range(size):
            hess[i] = approx_fprime(
                params,
                jac_param_i(i),
                eps,
            )
        return hess
    # hessian cs
    u = eps * 1j * np.eye(size)
    complex_params = params.astype(np.complex64)  # change params to complex
    for i in range(size):
        for j in range(i, size):
            hess[i, j] = np.imag(jac_negative_log(complex_params + u[i])[j]) / eps
            if i != j:
                hess[j, i] = hess[i, j]
    covariance_matrix = None
    try:
        covariance_matrix = np.linalg.pinv(hess).astype(np.float64)
    except np.linalg.LinAlgError as err:
        warnings.warn(
            f"""
            Failed to compute parameters covariance due to non-invertible
            hessian matrix. Numpy pseudo-inversion algorithm returned : {err}

            You can skip parameters covariance computation by setting
            covariance_method to False.
            """,
            stacklevel=2,
        )

    return covariance_matrix
