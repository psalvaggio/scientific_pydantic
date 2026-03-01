"""Pydantic adapter for slice's"""

import numbers
import typing as ty
from collections.abc import Hashable, Mapping, Sequence

import pydantic
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from scientific_pydantic.slice_syntax import (
    SliceSyntaxError,
    format_slice_syntax,
    parse_slice_syntax,
)

UNSET = object()


class SliceAdapter:
    """Pydantic adapter for Python's built-in `slice`.

    `slice` is more complex than `range` in that it is essentially a generic
    3-tuple (`tuple[Any, Any, Any]`). Integer values are only required if/when
    the user calls `.indices()` on the `slice`. There are a number of other
    valid uses of `slice` that do not use integers as the elements and thus
    this adapter supports non-integer elements.

    Inputs can be coerced from:

    1. `slice` - Identity.
    2. `str` - A string of the format `"[start]:[stop][:step]"`. This is also
       used as the JSON representation when all elements are either numeric
       or `None`.
    3. `Mapping` - A mapping with `"start"`, `"stop"` and `"step"` keys
       (all optional). This is used as the JSON representation when the
       conditions for an `str` encoding are not met.
    4. `Sequence` - A sequence of length 1, 2 or 3 with generic elements.

    A public alias [IntSliceAdapter][scientific_pydantic.IntSliceAdapter] is
    exposed for `SliceAdapter(int | None)`.

    Parameters
    ----------
    default_type : Hashable
        The default type annotation for all 3 elements of the slice. This should
        normally include `None` unless all 3 elements are always required..
    start_type : Hashable
        If given, overrides `default_type` as the type annotation for the start
        of the slice.
    stop_type : Hashable
        If given, overrides `default_type` as the type annotation for the stop
        of the slice.
    step_type : Hashable
        If given, overrides `default_type` as the type annotation for the step
        of the slice.

    Examples
    --------
    >>> import pydantic
    >>> from scientific_pydantic import SliceAdapter  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> class Model(pydantic.BaseModel):
    ...     s: ty.Annotated[
    ...         slice, SliceAdapter(int | None)
    ...     ]  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> Model(s=slice(1, 3))
    Model(s=slice(1, 3, None))
    >>> Model(s="1:10:2")
    Model(s=slice(1, 10, 2))
    >>> Model(s={"start": 1})
    Model(s=slice(1, None, None))
    """

    def __init__(
        self,
        default_type: Hashable = ty.Any,
        *,
        start_type: Hashable = UNSET,
        stop_type: Hashable = UNSET,
        step_type: Hashable = UNSET,
    ) -> None:
        adapters = {
            t: pydantic.TypeAdapter(t)
            for t in {default_type, start_type, stop_type, step_type}
            if t is not UNSET
        }
        self._default_adapter = adapters[default_type]
        self._start_adapter = (
            adapters[start_type] if start_type is not UNSET else self._default_adapter
        )
        self._stop_adapter = (
            adapters[stop_type] if stop_type is not UNSET else self._default_adapter
        )
        self._step_adapter = (
            adapters[step_type] if step_type is not UNSET else self._default_adapter
        )

    def __get_pydantic_core_schema__(
        self,
        _source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""

        def _validate(value: ty.Any) -> slice:
            match value:
                case slice():
                    return value
                case Mapping():
                    start, stop, step = _from_mapping(value)
                case str():
                    start, stop, step = _from_str(value)
                case Sequence():
                    start, stop, step = _from_sequence(value)
                case _:
                    msg = "Expected a slice, sequence, mapping or str"
                    raise ValueError(msg)

            start = self._start_adapter.validate_python(start)
            stop = self._stop_adapter.validate_python(stop)
            step = self._step_adapter.validate_python(step)
            return slice(start, stop, step)

        def _serialize(value: slice) -> str | dict[str, ty.Any]:
            if all(
                x is None or isinstance(x, numbers.Number)
                for x in (value.start, value.stop, value.step)
            ):
                return format_slice_syntax(value.start, value.stop, value.step)

            return {
                "start": value.start,
                "stop": value.stop,
                "step": value.step,
            }

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize,
                when_used="json",
            ),
        )

    def __get_pydantic_json_schema__(
        self,
        _core_schema: core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Get the JSON schema for this object"""
        return handler(
            core_schema.union_schema(
                [
                    core_schema.str_schema(),
                    core_schema.list_schema(min_length=1, max_length=3),
                    core_schema.typed_dict_schema(
                        {
                            "start": core_schema.typed_dict_field(
                                self._start_adapter.core_schema
                                if self._start_adapter is not None
                                else core_schema.any_schema(),
                            ),
                            "stop": core_schema.typed_dict_field(
                                self._stop_adapter.core_schema
                                if self._stop_adapter is not None
                                else core_schema.any_schema(),
                            ),
                            "step": core_schema.typed_dict_field(
                                self._step_adapter.core_schema
                                if self._step_adapter is not None
                                else core_schema.any_schema(),
                            ),
                        },
                        total=False,
                    ),
                ],
            ),
        )


IntSliceAdapter = SliceAdapter(int | None)


def _from_mapping(value: Mapping[str, ty.Any]) -> tuple[ty.Any, ty.Any, ty.Any]:
    if any(x not in ("start", "stop", "step") for x in value):
        msg = 'Invalid key for slice, can only accept "start"/"stop"/"step"'
        raise ValueError(msg)
    return (value.get("start"), value.get("stop"), value.get("step"))


def _from_str(value: str) -> tuple[ty.Any, ty.Any, ty.Any]:
    try:
        start, stop, step = parse_slice_syntax(
            value,
            converter=str,
            require_start=False,
            require_stop=True,
        )
    except SliceSyntaxError as exc:
        raise ValueError(str(exc)) from exc

    return (start, stop, step)


def _from_sequence(value: Sequence[ty.Any]) -> tuple[ty.Any, ty.Any, ty.Any]:
    if 1 <= len(value) <= 3:  # noqa: PLR2004
        return tuple(value)
    msg = "A sequence input to slice must have 1-3 elements"
    raise ValueError(msg)
