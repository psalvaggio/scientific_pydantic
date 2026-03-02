"""Pydantic adapters for astropy.time

The current supported types are:

- `Time` - [`TimeAdapter`][scientific_pydantic.astropy.time.TimeAdapter]
"""

from .time_adapter import TimeAdapter

__all__ = [
    "TimeAdapter",
]
