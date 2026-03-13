"""Pydantic adapters for numpy data types."""

from __future__ import annotations

import typing as ty

from pydantic_core import PydanticCustomError, core_schema

from scientific_pydantic.schema import make_core_schema

if ty.TYPE_CHECKING:
    import numpy as np
    import pydantic
    from pydantic.json_schema import JsonSchemaValue


class DTypeAdapter:
    """Pydantic adapter for numpy.dtype

    Validation Options
    ------------------
    1. `dtype`: Identity.
    2. `dtype`-like object: Anything that can be converted to a numpy `dtype`
       object via `dtype.__init__`. This notably includes the string
       representations (e.g. '<f8', '|i4') or Python/numpy numeric types.

    JSON Serialization
    ------------------
    `dtype`'s are serialized to JSON via the `.str` property.

    Examples
    --------
    >>> import pydantic
    >>> import numpy as np
    >>> from scientific_pydantic.numpy import (
    ...     DTypeAdapter,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> class Model(pydantic.BaseModel):
    ...     dt: ty.Annotated[np.dtype, DTypeAdapter()]  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> Model(dt="|i4")
    Model(dt=dtype('int32'))
    >>> Model(dt=float)
    Model(dt=dtype('float64'))
    >>> Model(dt=np.float64)
    Model(dt=dtype('float64'))
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: type[ty.Any],
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        import numpy as np

        return make_core_schema(
            np.dtype,
            serializer=lambda dt: dt.str,
            before_validator=_validate,
            json_schema=core_schema.str_schema(),
        )

    def __get_pydantic_json_schema__(
        self,
        core_schema: core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Generate JSON schema for the ndarray field"""
        json_schema = handler(core_schema)
        json_schema["description"] = "NumPy dtype"
        return json_schema


def _validate(value: ty.Any) -> np.dtype:
    import numpy as np

    try:
        return np.dtype(value)
    except Exception as exc:
        err_t = "invalid_dtype"
        msg = "invalid numpy dtype: {e}"
        raise PydanticCustomError(err_t, msg, {"value": value, "e": str(exc)}) from exc
