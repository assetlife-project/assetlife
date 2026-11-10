"""Parametric models base class and fitting utilities."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    final,
    override,
)

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    import optype.numpy as onp


__all__ = ["FittingResults", "ParametricModel"]


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
