"""Pydantic adapters for scipy.spatial.transform

The current supported types are:

- `Rotation` -
    [RotationAdapter][scientific_pydantic.scipy.spatial.transform.RotationAdapter]
"""

from .rotation import RotationAdapter

__all__ = [
    "RotationAdapter",
]
