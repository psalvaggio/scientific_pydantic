"""Type adaptors for shapely

The current supported types are:

- Geometry types - [GeometryAdapter][scientific_pydantic.shapely.GeometryAdapter]
"""

from .adapters import CoordinateBounds, GeometryAdapter

__all__ = [
    "CoordinateBounds",
    "GeometryAdapter",
]
