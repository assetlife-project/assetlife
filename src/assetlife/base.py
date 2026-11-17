"""Parametric models base class and fitting utilities."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    TypedDict,
    TypeVar,
    final,
    override,
)

import numpy as np
from scipy import stats
from scipy.optimize import Bounds, approx_fprime, minimize

if TYPE_CHECKING:
    import optype.numpy as onp


__all__ = ["FitConfig", "MaximumLikelihoodOptimizer", "ParametricModel"]


@final
class _Parameters:
    """
    Tree-structured dictionary-like container for model parameters.

    A ``ParametricModel`` owns one ``_Parameters`` instance. Nested parametric
    components are stored as leaves so their values can be flattened, updated,
    and restored as a single parameter vector.
    """

    _leaves: dict[str, _Parameters]
    _values: list[float | None]

    def __init__(self, *values: float | None) -> None:
        self._values = list(values)
        self._leaves = {}

    def _iter_values(self) -> Iterator[float | None]:
        yield from self._values
        for leaf in self._leaves.values():
            yield from leaf.all_values

    @property
    def all_values(self) -> tuple[float | None, ...]:
        """Flatten all parameter values in tree order."""
        return tuple(self._iter_values())

    @property
    def size(self) -> int:
        """Number of parameters in the tree."""
        return len(self._values) + sum(leaf.size for leaf in self._leaves.values())

    def set_leaf(self, leaf_name: str, leaf: Self) -> None:
        """Set or replace a parameter subtree."""
        self._leaves[leaf_name] = leaf

    def set_all_values(self, values: Sequence[float | None]) -> None:
        """Set values of the whole parameter tree.

        Parameters
        ----------
        values : sequence of float, optional.
            The values to set in the parameters tree.

        Raises
        ------
        ValueError
            If the parameter tree has leaves and the number of values does not
            match its size.
        """
        if len(values) != self.size and self._leaves:
            msg = f"Expected {self.size} values but got {len(values)}"
            raise ValueError(msg)
        if not self._leaves:
            self._values = list(values)
        start = len(self._values)
        self._values = list(values[:start])
        for leaf in self._leaves.values():
            stop = start + leaf.size
            leaf.set_all_values(values[start:stop])
            start = stop


class ParametricModel:
    """
    Base class for models with parameters.

    The class stores parameters in a tree structure, exposes them as a flat
    vector with ``get_params`` and ``set_params``, and tracks fitting results.

    Examples
    --------
    >>> class ModelA(ParametricModel):
    ...     def __init__(self, a, b):
    ...         super().__init__(a, b)
    >>> class ModelB(ParametricModel):
    ...     def __init__(self, baseline: ModelA):
    ...         super().__init__()
    ...         self.baseline = baseline
    >>> model_a = ModelA(1, 2)
    >>> model_b = ModelB(model_a)
    >>> model_b.get_params()
    array([1, 2])
    """

    _params: _Parameters
    fitting_results: FittingResults | None

    def __init__(self, *params: float | None) -> None:
        self._params = _Parameters(*params)
        self.fitting_results = None

    def is_parametrized(self) -> bool:
        """Whether at least one parameter value is set."""
        return bool(~np.all(np.isnan(self.get_params())))

    def is_fitted(self) -> bool:
        """Whether fitting results are set."""
        return self.fitting_results is not None

    def get_params(self) -> onp.Array1D[np.float64]:
        """
        Get the parameters of this model.

        Returns
        -------
        out : 1darray of floats
            Model parameters.

        Notes
        -----
        If parameter values are not set, they default to ``np.nan`` values.
        """
        return np.array(self._params.all_values)

    def set_params(self, new_params: onp.ToFloat1D) -> None:
        """
        Set the parameters of this model.

        Parameters
        ----------
        new_params : 1d array-like of floats
            Model parameters.

        Notes
        -----
        ``set_params`` definition expects an array-like of floats. At runtime,
        complex parameters might be setted temporarily to approximate fitted
        parameters covariance. This is contradictory to the given typing. At
        the moment, we don't see a better solution and we believe that this is
        actually a limitation of what can be expressed in the static typesystem.
        """
        # not @params.setter to allow a different type for the values to set
        new_params = np.asarray(new_params)
        assert new_params.ndim == 1
        self._params.set_all_values(list(new_params))

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        # automatically add params of new component_model
        if isinstance(value, ParametricModel):
            # a reference of component._params is kept in the _Parameters tree
            # thus changing model params will affect each component params
            component: ParametricModel = value
            self._params.set_leaf(f"{name}.params", component._params)
        super().__setattr__(name, value)

    def save(self, path: str | Path) -> None:
        """
        Save model to pickle file.

        Parameters
        ----------
        path: str or Path
            Pickle filepath.

        """
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> ParametricModel:
        """
        Load model from pickle file.

        Parameters
        ----------
        path: str or Path
            Pickle filepath.

        Raises
        ------
        FileNotFoundError
            If the path doesn't exist.

        Warnings
        --------
        Loading pickled data can execute arbitrary code. Only load files from
        trusted sources.
        """
        if not Path(path).exists():
            raise FileNotFoundError(path)

        with Path(path).open("rb") as f:
            return pickle.load(f)  # ruff: ignore[suspicious-pickle-usage]


@dataclass
class FittingResults:
    """Results returned by parametric model fitting.

    The object stores optimizer status, optimal parameter values, information
    criteria, and optional covariance estimates.
    """

    nb_observations: int  #: Number of observations (samples)
    optimal_params: onp.Array1D[np.float64]  #: Optimal parameter values
    success: bool  #: Whether or not the optimizer exited successfully.
    neg_log_likelihood: float = field(
        repr=False,
    )  #: Negative log likelihood value at optimal parameter values

    covariance_matrix: onp.Array2D[np.float64] | None = field(
        repr=False,
        default=None,
    )  #: Covariance matrix (computed as the inverse of the Hessian matrix).

    nb_params: int = field(init=False, repr=False)  #: Number of parameters.
    aic: float = field(init=False)  #: Akaike Information Criterion.
    aicc: float = field(
        init=False,
    )  #: Akaike Information Criterion with a correction for small sample sizes.
    bic: float = field(init=False)  #: Bayesian Information Criterion.
    se: onp.Array1D[np.float64] | None = field(
        init=False,
        repr=False,
    )  #: Standard error, square root of the diagonal of the covariance matrix
    ci: onp.Array[tuple[int, Literal[2]], np.float64] | None = field(
        init=False,
        repr=False,
    )  #: 95% CI

    def __post_init__(self) -> None:
        self.nb_params = self.optimal_params.size
        self.aic = 2 * self.nb_params + 2 * self.neg_log_likelihood
        self.aicc = self.aic + 2 * self.nb_params * (self.nb_params + 1) / (
            self.nb_observations - self.nb_params - 1
        )
        self.bic = (
            np.log(self.nb_observations) * self.nb_params + 2 * self.neg_log_likelihood
        )
        self.se = None
        self.ci = None
        if self.covariance_matrix is not None:
            self.se = np.sqrt(np.diag(self.covariance_matrix))
            self.ci = self.optimal_params.reshape(-1, 1) + stats.norm.ppf((
                0.05,
                0.95,
            )) * self.se.reshape(-1, 1) / np.sqrt(self.nb_observations)  # (p, 2)

    @override
    def __str__(self) -> str:
        fields = {
            "fitted params": self.optimal_params,
            "AIC": self.aic,
            "AICc": self.aicc,
            "BIC": self.bic,
        }
        # Find the maximum field name length for alignment
        max_name_length = max(len(name) for name in fields)
        lines: list[str] = []
        for name, value in fields.items():
            # Format arrays to be more compact
            if isinstance(value, np.ndarray):
                value_str = f"[{', '.join(f'{x:.6g}' for x in value)}]"
            else:
                value_str = f"{value:.6g}" if isinstance(value, float) else str(value)
            lines.append(f"{name:<{max_name_length}} : {value_str}")
        return "\n".join(lines)


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

        match self.config.covariance_method:
            case False:
                return fitting_results
            case "exact":
                hess_fn = self.config.scipy_minimize_options.get("hess", None)
                if hess_fn is None:
                    return fitting_results
                hess = hess_fn(fitting_results.optimal_params)
            case "2point" | "cs":
                jac = self.config.scipy_minimize_options.get("jac", None)
                if jac is None:
                    return fitting_results
                hess = _approx_parameters_hess(
                    fitting_results.optimal_params,
                    jac,
                    method=self.config.covariance_method,
                )

        fitting_results.covariance_matrix = _get_covariance_matrix(hess)

        return fitting_results


def _approx_parameters_hess(
    params: onp.Array1D[np.float64],
    jac_negative_log: Callable[[onp.Array1D[np.number]], onp.Array1D[np.number]],
    method: Literal["2point", "cs"] = "cs",
) -> onp.Array2D[np.float64]:
    """
    Approximate the hessian matrix of fitted parameters.

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

    u = eps * 1j * np.eye(size)
    complex_params = params.astype(np.complex64)  # change params to complex
    for i in range(size):
        for j in range(i, size):
            hess[i, j] = np.imag(jac_negative_log(complex_params + u[i])[j]) / eps
            if i != j:
                hess[j, i] = hess[i, j]
    return hess


def _get_covariance_matrix(
    hess: onp.Array2D[np.float64],
) -> onp.Array2D[np.float64] | None:
    """
    Get the covariance matrix from the hessian.

    Parameters
    ----------
    hess : onp.Array2D[np.float64]
        The hessian matrix
    """

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
