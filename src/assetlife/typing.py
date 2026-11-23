"""Type variables and aliases."""

from collections.abc import Mapping
from typing import Any, TypeVarTuple

import numpy as np
import optype.numpy as onp
from narwhals.typing import IntoDataFrame

__all__ = [
    "CoercibleFloat64_1D",
    "CoercibleFloat64_ND",
    "CovarTs",
    "Float64_1D",
    "Float64_ND",
    "Seed",
    "TabularDataset",
    "Timeline",
]

#: Generic variadic type variable tuple used for additional model arguments,
#: such as covariates.
CovarTs = TypeVarTuple("CovarTs")

# Numpy types coercible to np.float64
type _Float64_Co = np.float64 | np.float32 | np.float16 | np.integer | np.bool

#: Scalar or NumPy array coercible to ``np.float64``.
#:
#: This accepts scalar values coercible to ``np.float64`` and NumPy arrays with
#: dtype safely coercible to ``np.float64``. It does not accept arbitrary
#: array-like sequences.
type CoercibleFloat64_ND = onp.ToFloat64 | onp.ArrayND[_Float64_Co]

#: Scalar or one-dimensional NumPy array coercible to ``np.float64``.
type CoercibleFloat64_1D = onp.ToFloat64 | onp.Array1D[_Float64_Co]

#: Scalar ``np.float64`` or NumPy array with dtype ``np.float64``.
type Float64_ND = np.float64 | onp.ArrayND[np.float64]

#: Scalar ``np.float64`` or one-dimensional NumPy array with dtype ``np.float64``.
type Float64_1D = np.float64 | onp.Array1D[np.float64]

#: One-dimensional timeline array with dtype ``np.float64``.
type Timeline = onp.Array1D[np.float64]

#: Accepted random seed inputs.
type Seed = (
    int | np.random.Generator | np.random.BitGenerator | np.random.RandomState | None
)

#: Any tabular like dataset
type TabularDataset = Mapping[str, Any] | IntoDataFrame
