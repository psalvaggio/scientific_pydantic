"""Adaptor for range"""

import typing as ty
from collections.abc import Hashable

import pydantic
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, core_schema

from .common_range_slice import UNSET, StartStopStepAdapters
from .schema import Encoding, make_core_schema
from .slice_syntax import (
    SliceSyntaxError,
    format_slice_syntax,
    parse_slice_syntax,
)


class RangeAdapter:
    """Pydantic adapter for Python `range` using slice syntax.

    Currently, only `int` slices are supported.

    Validation Options
    ------------------
    1. `range` - Identity
    2. `str` - A slice-like syntax (`[start:]stop[:step]`) is used. This
        representation is also used for the JSON encoding of range.

    Parameters
    ----------
    default_type
        The default type annotation for all 3 elements of the range. This must
        either be int or an annotated int.
    start_type
        If given, overrides `default_type` as the type annotation for the start
        of the range.
    stop_type
        If given, overrides `default_type` as the type annotation for the stop
        of the range.
    step_type
        If given, overrides `default_type` as the type annotation for the step
        of the range.
    encoding
        A custom encoding for this type

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

    def __init__(
        self,
        default_type: Hashable = int,
        *,
        start_type: Hashable = UNSET,
        stop_type: Hashable = UNSET,
        step_type: Hashable = UNSET,
        encoding: Encoding | None = None,
    ) -> None:
        for t, label in (
            (default_type, "default_type"),
            (start_type, "start_type"),
            (stop_type, "stop_type"),
            (step_type, "step_type"),
        ):
            if (
                t is UNSET
                or t is int  # type: ignore[unnecessary-comparison]
                or (ty.get_origin(t) is ty.Annotated and ty.get_args(t)[0] is int)
            ):
                continue

            msg = (
                f"RangeAdapter: {label} was {t}, but only int's or annotated "
                "int's are currently supported"
            )
            raise ValueError(msg)

        self._adapters = StartStopStepAdapters(
            default_type,
            start_type=start_type,
            stop_type=stop_type,
            step_type=step_type,
        )
        self._encoding = encoding if encoding is not None else self._default_encoding()

    def __get_pydantic_core_schema__(
        self,
        _source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        return make_core_schema(
            range,
            encoding=self._encoding,
            after_validators=[self._adapters.after_validator],
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

    def _default_encoding(self) -> Encoding[range]:
        return Encoding(
            serializer=_serialize,
            before_validator=_validate,
            json_schema=core_schema.str_schema(
                pattern=r"^\s*-?\d+\s*:\s*-?\d+\s*(?::\s*-?\d+\s*)?$"
            ),
        )


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
                dest_type=range,
            )
        except SliceSyntaxError as e:
            err_t = "range_syntax_error"
            msg = "{what}"
            raise PydanticCustomError(err_t, msg, {"what": str(e)}) from e

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
