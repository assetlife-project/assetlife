import itertools
from collections.abc import Generator
from typing import TypeAlias

import numpy as np

Shape: TypeAlias = tuple[int, ...]


def _generate_without_duplicates(
    *inputs: list[Shape], repeat: int = 1
) -> Generator[tuple[Shape, ...]]:
    seen: set[frozenset[tuple[int, ...]]] = set()
    for prod in itertools.product(*inputs, repeat=repeat):
        prod_set: frozenset[tuple[int, ...]] = frozenset(prod)
        if len(prod_set) == 1:  # all items are the same
            continue
        if prod_set not in seen:
            seen.add(prod_set)
            yield prod


def generate_shapes(
    max_ndim: int, nb_args: int
) -> list[tuple[Shape, ...]] | list[Shape]:
    """
    Generate sets of mutually broadcastable shapes without repetitions and scalars.

    Parameters
    ----------
    n : int
        The number of shapes to generate in each set.
    num_axes : int
        The number of axes

    Examples
    --------
    >>> generate_shapes(2, 1)
    [(4,), (2, 1), (2, 4)]
    >>> generate_shapes(2, 2)
    [((4,), (2, 1)), ((4,), (2, 4)), ((2, 1), (2, 4))]
    """
    assert max_ndim >= 2
    shape_patterns = itertools.product([0, 1], repeat=max_ndim)
    shape_ref = np.arange(2, 2 * max_ndim + 1, 2)
    ones_ref = np.ones_like(shape_ref)
    shapes: list[tuple[int, ...]] = []
    for pattern in shape_patterns:
        shape = np.where(pattern, shape_ref, ones_ref)
        mask = np.cumsum(pattern) > 0  # remove first dims if 1
        shape = shape[mask]
        if np.all(shape == 1):  # skip scalar-like
            continue
        shapes.append(tuple(shape.tolist()))
    if nb_args > 1:
        return list(_generate_without_duplicates(shapes, repeat=nb_args))
    else:
        return shapes


def shape_id(shape: Shape) -> str:
    if tuple(shape) == ():
        return "scalar"
    return "x".join(map(str, shape))
