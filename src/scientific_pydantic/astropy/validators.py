"""Common validators for astropy"""

import types
import typing as ty
from collections.abc import Sequence

from pydantic import BaseModel, PydanticSchemaGenerationError
from pydantic_core import PydanticCustomError

from scientific_pydantic.numpy.validators import (
    HasNdim,
    HasShape,
    NDimValidator,
    ShapeValidator,
)


class HasIsScalar(ty.Protocol):
    """Protocol for an astropy array-like object (Quantity/Time)"""

    @property
    def isscalar(self) -> bool: ...  # noqa: D102


HasIsScalarT = ty.TypeVar("HasIsScalarT", bound=HasIsScalar)


class ScalarValidator(BaseModel, frozen=True, extra="forbid"):
    """Validator for the scalar property on a Quantity"""

    scalar: bool

    def __call__(self, val: HasIsScalarT) -> HasIsScalarT:
        """Apply scalar validation"""
        if val.isscalar != self.scalar:
            err_t = "scalar_error"
            msg = "Expected isscalar to be {exp}, was {actual}"
            raise PydanticCustomError(
                err_t, msg, {"exp": self.scalar, "actual": val.isscalar}
            )
        return val


class AstropyArrayLike(HasShape, HasNdim, HasIsScalar):
    """Protocol for an astropy array-like object (Quantity/Time)"""


AstropyArrayLikeT = ty.TypeVar("AstropyArrayLikeT", bound=AstropyArrayLike)


class ArrayShapeValidator:
    """Validator for the scalar property on a Quantity

    Parameters
    ----------
    scalar : bool
        If True, only scalar quantities will be accepted. If False, only vector
        quantities will be accepted. If None, no scalar constraints are enforced,
        unless `ndim` or `shape` are provided.
    ndim : int | None
        If given, the dimensionality of the quantity must match this value. Must
        be >= 0.
    shape : Sequence[Ellipsis | int | range | slice | None] | None
        Shape specifier for the given array. See `NDArrayValidator` for a
        description of how this works.
    """

    def __init__(
        self,
        *,
        scalar: bool | None = None,
        ndim: int | None = None,
        shape: Sequence[types.EllipsisType | int | range | slice | None] | None = None,
    ) -> None:
        # Handle contradictions in the shape arguments
        if scalar is not None:
            if scalar:
                if ndim is not None and ndim != 0:
                    msg = f"scalar=True and ndim={ndim} contradict"
                    raise PydanticSchemaGenerationError(msg)
                ndim = None  # ndim = 0 is redundant
                if shape is not None and shape != ():
                    msg = f"scalar=True and shape={shape} contradict"
                    raise PydanticSchemaGenerationError(msg)
                shape = None  # shape = () is redundant
            else:
                if ndim == 0:
                    msg = "scalar=False and ndim=0 contradict"
                    raise PydanticSchemaGenerationError(msg)
                if shape == ():
                    msg = "scalar=False and shape=() contradict"
                    raise PydanticSchemaGenerationError(msg)

        self._validators: list[ty.Callable] = []
        if scalar is not None:
            self._validators.append(ScalarValidator(scalar=scalar))
        if ndim is not None:
            self._validators.append(NDimValidator(ndim=ndim))
        if shape is not None:
            self._validators.append(ShapeValidator(shape=shape))

    def __call__(self, arr: AstropyArrayLikeT) -> AstropyArrayLikeT:
        """Perform validation"""
        for val in self._validators:
            arr = val(arr)
        return arr
