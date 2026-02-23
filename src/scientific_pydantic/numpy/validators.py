"""Validation logic"""

import typing as ty
from collections.abc import Sequence

import numpy as np
import pydantic
from numpy.typing import NDArray
from pydantic_core import PydanticCustomError

from ..ellipsis import EllipsisLiteral
from ..range import RangeAdapter
from ..slice import IntSliceAdapter
from .dtype_adapter import DTypeAdapter


class DTypeValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for the array's data type"""

    dtype: ty.Annotated[np.dtype, DTypeAdapter()]

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply data type validation"""
        try:
            return arr.astype(self.dtype)
        except (ValueError, TypeError) as e:
            err_t = "dtype_error"
            msg = "the array could not be converted to {dtype}"
            raise PydanticCustomError(err_t, msg, {"dtype": self.dtype}) from e


class NDimValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for the number of dimensions"""

    ndim: int = pydantic.Field(ge=0)

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply ndim validation"""
        if arr.ndim != self.ndim:
            err_t = "ndim_error"
            msg = "the array had {arr_ndim} dimension(s), expected {ndim}"
            raise PydanticCustomError(
                err_t, msg, {"arr_ndim": arr.ndim, "ndim": self.ndim}
            )
        return arr


class ShapeValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Data type for the shape of the array"""

    shape: Sequence[
        EllipsisLiteral
        | int
        | ty.Annotated[range, RangeAdapter()]
        | ty.Annotated[slice, IntSliceAdapter]
        | None,
    ]

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply shape validation"""
        shape = list(self.shape)
        arr_shape = arr.shape

        def match(shape_idx: int, arr_idx: int) -> bool:
            """Return True if shape[shape_idx:] can match arr_shape[arr_idx:]"""
            if shape_idx == len(shape) and arr_idx == len(arr_shape):
                return True  # Base case: both exhausted
            if shape_idx == len(shape):
                return False  # array dims left but no specs to match them
            if arr_idx == len(arr_shape):
                # Remaining specs must all be ellipses (each can match 0 dims)
                return all(s is ... for s in shape[shape_idx:])

            spec = shape[shape_idx]
            if spec is ...:
                # Try matching 0, 1, 2, ... array dims to this ellipsis
                for n in range(arr_idx, len(arr_shape) + 1):
                    if match(shape_idx + 1, n):
                        return True
                return False

            return _matches_spec(arr_shape[arr_idx], spec) and match(  # type: ignore[bad-argument-type]
                shape_idx + 1, arr_idx + 1
            )

        if not match(0, 0):
            msg = f"Array shape {arr_shape} does not match spec {shape}"
            raise ValueError(msg)

        return arr


def _matches_spec(dim_size: int, spec: int | range | slice | None) -> bool:
    if spec is None:
        return True
    if isinstance(spec, int):
        return dim_size == spec
    if isinstance(spec, range):
        return dim_size in spec
    if isinstance(spec, slice):
        start = spec.start if spec.start is not None else 0
        stop = spec.stop
        step = spec.step if spec.step is not None else 1
        if stop is None:
            return dim_size >= start and (dim_size - start) % step == 0
        return start <= dim_size < stop and (dim_size - start) % step == 0
    msg = f"Unknown shape spec type: {type(spec)}"
    raise ValueError(msg)


class GtValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for a bounds check"""

    gt: float

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply gt validation"""
        if not np.all(arr > self.gt):
            err = PydanticCustomError(
                "bounds_error",
                f"Not all elements were greater than {self.gt}",
            )
            raise err
        return arr


class GeValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for a bounds check"""

    ge: float

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply ge validation"""
        if not np.all(arr >= self.ge):
            err = PydanticCustomError(
                "bounds_error",
                f"Not all elements were greater than or equal to {self.ge}",
            )
            raise err
        return arr


class LtValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for a bounds check"""

    lt: float

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply lt validation"""
        if not np.all(arr < self.lt):
            err = PydanticCustomError(
                "bounds_error",
                f"Not all elements were less than {self.lt}",
            )
            raise err
        return arr


class LeValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for a bounds check"""

    le: float

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply le validation"""
        if not np.all(arr <= self.le):
            err = PydanticCustomError(
                "bounds_error",
                f"Not all elements were less than or equal to {self.le}",
            )
            raise err
        return arr


class ClipValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for a clip check"""

    clip: tuple[float | None, float | None]

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply clip validation"""
        if self.clip[0] is not None or self.clip[1] is not None:
            return np.clip(arr, self.clip[0], self.clip[1])
        return arr


class NDArrayValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for an ndarray"""

    dtype: DTypeValidator | None = None
    ndim: NDimValidator | None = None
    shape: ShapeValidator | None = None
    gt: GtValidator | None = None
    ge: GeValidator | None = None
    lt: LtValidator | None = None
    le: LeValidator | None = None
    clip: ClipValidator | None = None

    @classmethod
    def from_kwargs(cls, **kwargs) -> "NDArrayValidator":
        """Create from field/value pairs"""
        try:
            return cls(
                **{key: {key: val} for key, val in kwargs.items() if val is not None},
            )
        except pydantic.ValidationError as e:
            msg = "Invalid constraint value(s):\n" + (
                "\n".join(
                    f"{i + 1}. {err['loc'][0]} ({err['input']!r})- {err['msg']}"
                    for i, err in enumerate(e.errors())
                )
            )
            raise ValueError(msg) from None

    def __call__(self, arr: np.typing.ArrayLike) -> np.typing.NDArray:
        """Validate and optionally transform a numpy ndarray.

        Parameters
        ----------
        arr : ndarray
            The array to validate

        Returns
        -------
        ndarray
            The validated (and possibly transformed) array

        Raises
        ------
        ValueError
            If validation fails
        TypeError
            If dtype conversion fails
        """
        # Convert to ndarray if needed
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        for field in type(self).model_fields:
            val = getattr(self, field)
            if val is None:
                continue

            arr = val(arr)

        return arr
