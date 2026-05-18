"""Pydantic adapters for numpy types

The current supported types are:

- `dtype` - [DTypeAdapter][scientific_pydantic.numpy.DTypeAdapter]
- `ndarray` - [NDArrayAdapter][scientific_pydantic.numpy.NDArrayAdapter]
- `bool_`, `integer`, `inexact` -
      [ScalarAdapter][scientific_pydantic.numpy.ScalarAdapter]
"""

from .dtype_adapter import DTypeAdapter
from .ndarray_adapter import NDArrayAdapter
from .scalar_adapter import ScalarAdapter

__all__ = [
    "DTypeAdapter",
    "NDArrayAdapter",
    "ScalarAdapter",
]
