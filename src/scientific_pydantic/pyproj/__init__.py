"""Pydantic adapters for pyproj types

The current supported types are:

- `CRS` - [CRS][scientific_pydantic.pyproj.CRSAdapter]
"""

from .crs import CRSAdapter

__all__ = [
    "CRSAdapter",
]
