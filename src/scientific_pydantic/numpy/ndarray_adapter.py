"""Pydantic adapters for numpy."""

import types
import typing as ty
from collections.abc import Sequence

import pydantic
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

if ty.TYPE_CHECKING:
    import numpy as np


class NDArrayAdapter:
    """Pydantic type adapter for numpy `ndarray`s with validation constraints.

    Inputs can be coerced from:

    1. `ndarray` - Identity
    2. `ArrayLike` - Any object that can be converted to an `ndarray` via
       [`np.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).

    Shape specifiers for arrays are a sequence of entries, which support the
    options:

    - `Ellipsis`/`...` - A wildcard match that matches any number of dimensions
      with any size. Multiple can be used in a shape specifier.
    - `int` - The corresponding dimension of the array must have exactly this
      size.
    - `range` - The corresponding dimension of the array must have a size that
      is in this `range`.
    - `slice` - The corresponding dimension of the array must have a size that
      is in this `slice`. A `None` in the start or stop of the `slice` indicates
      that there is no lower or upper bound, respectively, for the dimension
      size.
    - `None` - The corresponding dimension must exist, but no constraint is
      applied to the size.

    For instance, a shape specifier of:
    ```python
    (..., 3, None, range(1, 3), slice(3, None))
    ```
    would indicate the array must have at least 4 dimensions, where the last 4
    dimensions must be of size 4, anything, 1 or 2, and at least 3.

    Parameters
    ----------
    dtype
        If given, the array will be coerced into this data type via `.astype()`.
    ndim
        If given, the array must have this dimensionality.
    shape
        If given a shape specifier for the array.
    gt
        If given, all elements in the array must be `>` this value.
    ge
        If given, all elements in the array must be `>=` this value.
    lt
        If given, all elements in the array must be `<` this value.
    le
        If given, all elements in the array must be `<=` this value.
    clip
        If not `(None, None)`, the array will be passed through
        `numpy.clip(array, clip[0], clip[1])` to bound the values.

    Examples
    --------
    >>> import pydantic
    >>> import numpy as np
    >>> from scientific_pydantic.numpy import (
    ...     NDArrayAdapter,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> class Model(pydantic.BaseModel):
    ...     a: ty.Annotated[
    ...         np.ndarray, NDArrayAdapter()
    ...     ]  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> Model(a=[[1, 2], [3, 4]])
    Model(a=array([[1, 2],
           [3, 4]]))
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dtype: "type | np.dtype | str | None" = None,
        ndim: int | None = None,
        shape: Sequence[types.EllipsisType | int | range | slice | None] | None = None,
        gt: float | None = None,
        ge: float | None = None,
        lt: float | None = None,
        le: float | None = None,
        clip: tuple[float | None, float | None] = (None, None),
    ) -> None:
        from .validators import NDArrayValidator

        self._validator = NDArrayValidator.from_kwargs(
            dtype=dtype,
            ndim=ndim,
            shape=shape,
            gt=gt,
            ge=ge,
            lt=lt,
            le=le,
            clip=clip,
        )

    def __get_pydantic_core_schema__(
        self,
        _source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for an NDArray"""
        import numpy as np

        def validate(value: ty.Any) -> np.typing.NDArray:
            return self._validator(value)

        def serialize(value: np.typing.NDArray) -> list:
            """Serialize ndarray to nested lists for JSON"""
            return value.tolist()

        python_schema = core_schema.no_info_after_validator_function(
            validate,
            core_schema.any_schema(),
        )

        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema(
                [
                    core_schema.any_schema(),
                    core_schema.no_info_after_validator_function(
                        validate,
                        core_schema.any_schema(),
                    ),
                ],
            ),
            python_schema=python_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize,
                info_arg=False,
                return_schema=core_schema.any_schema(),
            ),
        )

    def __get_pydantic_json_schema__(  # noqa: C901
        self,
        _core_schema: core_schema.CoreSchema,
        _handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Generate JSON schema for the ndarray field"""
        json_schema: dict[str, ty.Any] = {"type": "array"}

        # Add description of constraints
        constraints = []
        if self._validator.dtype is not None:
            constraints.append(f"dtype: {self._validator.dtype.dtype}")

        ndim = self._validator.ndim.ndim if self._validator.ndim is not None else None
        if ndim is not None:
            constraints.append(f"{ndim}D array")
        if self._validator.shape is not None:
            shape_desc = "x".join(
                str(s) if s is not None else "?" for s in self._validator.shape.shape
            )
            constraints.append(f"shape: ({shape_desc})")
        if self._validator.ge is not None:
            constraints.append(f"values >= {self._validator.ge.ge}")
        if self._validator.le is not None:
            constraints.append(f"values <= {self._validator.le.le}")
        if self._validator.gt is not None:
            constraints.append(f"values > {self._validator.gt.gt}")
        if self._validator.lt is not None:
            constraints.append(f"values < {self._validator.lt.lt}")
        if self._validator.clip is not None:
            constraints.append(
                f"clipped to [{self._validator.clip.clip[0]}, "
                f"{self._validator.clip.clip[1]}]",
            )

        if constraints:
            json_schema["description"] = "NumPy array: " + ", ".join(constraints)

        # Add items constraint based on ndim
        if ndim is not None and ndim <= 1:
            json_schema["items"] = {"type": "number"}
        elif ndim is not None and ndim > 1:
            # Nested arrays
            items_schema = {"type": "number"}
            for _ in range(ndim - 1):
                items_schema = {"type": "array", "items": items_schema}
            json_schema["items"] = items_schema

        return json_schema
