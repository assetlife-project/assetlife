"""Type variables and aliases used throughout ReLife."""

from typing import TypeAlias, TypeVarTuple

import numpy as np
import optype.numpy as onp

__all__ = [
    "CoercibleFloat64_1D",
    "CoercibleFloat64_ND",
    "CovarTs",
    "Float64_1D",
    "Float64_ND",
    "Seed",
    "Timeline",
]

#: Generic variadic type variable tuple used for additional model arguments,
#: such as covariates.
CovarTs = TypeVarTuple("CovarTs")

# Numpy types coercible to np.float64
_f64_co: TypeAlias = np.float64 | np.float32 | np.float16 | np.integer | np.bool

#: Scalar or NumPy array coercible to ``np.float64``.
#:
#: This accepts scalar values coercible to ``np.float64`` and NumPy arrays with
#: dtype safely coercible to ``np.float64``. It does not accept arbitrary
#: array-like sequences.
CoercibleFloat64_ND: TypeAlias = onp.ToFloat64 | onp.ArrayND[_f64_co]

#: Scalar or one-dimensional NumPy array coercible to ``np.float64``.
CoercibleFloat64_1D: TypeAlias = onp.ToFloat64 | onp.Array1D[_f64_co]

#: Scalar ``np.float64`` or NumPy array with dtype ``np.float64``.
Float64_ND: TypeAlias = np.float64 | onp.ArrayND[np.float64]

#: Scalar ``np.float64`` or one-dimensional NumPy array with dtype ``np.float64``.
Float64_1D: TypeAlias = np.float64 | onp.Array1D[np.float64]

#: One-dimensional timeline array with dtype ``np.float64``.
Timeline: TypeAlias = onp.Array1D[np.float64]

#: Accepted random seed inputs.
Seed: TypeAlias = (
    int | np.random.Generator | np.random.BitGenerator | np.random.RandomState | None
)
