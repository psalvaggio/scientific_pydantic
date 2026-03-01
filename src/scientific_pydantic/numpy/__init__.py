"""Pydantic adapters for numpy types

The current supported types are:

- `dtype` - [DTypeAdapter][scientific_pydantic.numpy.DTypeAdapter]
- `ndarray` - [NDArrayAdapter][scientific_pydantic.numpy.NDArrayAdapter]
"""

from .dtype_adapter import DTypeAdapter
from .ndarray_adapter import NDArrayAdapter

__all__ = [
    "DTypeAdapter",
    "NDArrayAdapter",
]
