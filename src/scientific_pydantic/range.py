"""Adaptor for range"""

import typing as ty

import pydantic
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, core_schema

from .schema import Encoding, make_core_schema
from .slice_syntax import (
    SliceSyntaxError,
    format_slice_syntax,
    parse_slice_syntax,
)


class RangeAdapter:
    """Pydantic adapter for Python `range` using slice syntax.

    Validation Options
    ------------------
    1. `range` - Identity
    2. `str` - A slice-like syntax (`[start:]stop[:step]`) is used. This
        representation is also used for the JSON encoding of range.

    Examples
    --------
    >>> import typing as ty
    >>> import pydantic
    >>> from scientific_pydantic import RangeAdapter  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> class Model(pydantic.BaseModel):
    ...     field: ty.Annotated[range, RangeAdapter()]  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> Model(field="12:25:2")
    Model(field=range(12, 25, 2))
    >>> Model(field=range(12, 25, 2))
    Model(field=range(12, 25, 2))
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        return make_core_schema(
            range,
            encoding=Encoding(
                serializer=_serialize,
                before_validator=_validate,
                json_schema=core_schema.str_schema(
                    pattern=r"^\s*-?\d+\s*:\s*-?\d+\s*(?::\s*-?\d+\s*)?$"
                ),
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Get the JSON schema for this type"""
        schema = handler(core_schema)
        schema["description"] = "Python range syntax: start:stop[:step]"
        return schema


def _validate(value: ty.Any) -> range:
    if isinstance(value, range):
        return value

    if isinstance(value, str):
        try:
            start, stop, step = parse_slice_syntax(
                value,
                converter=int,
                require_start=False,
                require_stop=True,
            )
        except SliceSyntaxError as exc:
            raise ValueError(str(exc)) from exc

        return range(
            start if start is not None else 0,
            stop,
            step if step is not None else 1,
        )

    err_t = "invalid_range"
    msg = "expected range or slice-syntax string, got {t}"
    raise PydanticCustomError(err_t, msg, {"t": type(value).__name__})


def _serialize(value: range) -> str:
    step = None if value.step == 1 else value.step
    return format_slice_syntax(value.start, value.stop, step)
