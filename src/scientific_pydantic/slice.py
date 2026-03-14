"""Pydantic adapter for slice's"""

import numbers
import typing as ty
from collections.abc import Hashable, Mapping, Sequence

import pydantic
from pydantic_core import InitErrorDetails, PydanticCustomError, core_schema

from .schema import Encoding, make_core_schema
from .slice_syntax import (
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

    A public alias [IntSliceAdapter][scientific_pydantic.IntSliceAdapter] is
    exposed for `SliceAdapter(int | None)`.

    Validation Options
    ------------------
    1. `slice` - Identity.
    2. `str` - A string of the format `"[start]:[stop][:step]"`. This is also
       used as the JSON representation when all elements are either numeric
       or `None`.
    3. `Mapping` - A mapping with `"start"`, `"stop"` and `"step"` keys
       (all optional). This is used as the JSON representation when the
       conditions for an `str` encoding are not met.
    4. `Sequence` - A sequence of length 1, 2 or 3 with generic elements.

    Parameters
    ----------
    default_type
        The default type annotation for all 3 elements of the slice. This should
        normally include `None` unless all 3 elements are always required..
    start_type
        If given, overrides `default_type` as the type annotation for the start
        of the slice.
    stop_type
        If given, overrides `default_type` as the type annotation for the stop
        of the slice.
    step_type
        If given, overrides `default_type` as the type annotation for the step
        of the slice.
    encoding
        A custom encoding for this type


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
        encoding: Encoding | None = None,
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
        self._encoding = encoding if encoding is not None else self._default_encoding()

    def __get_pydantic_core_schema__(
        self,
        _source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""

        def _validate(value: slice) -> slice:
            try:
                start = self._start_adapter.validate_python(value.start)
            except pydantic.ValidationError as e:
                raise _prefix_validation_error(e, "start") from None
            try:
                stop = self._stop_adapter.validate_python(value.stop)
            except pydantic.ValidationError as e:
                raise _prefix_validation_error(e, "stop") from None
            try:
                step = self._step_adapter.validate_python(value.step)
            except pydantic.ValidationError as e:
                raise _prefix_validation_error(e, "step") from None
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

        return make_core_schema(
            slice,
            encoding=self._encoding,
            after_validators=[_validate],
        )

    def _default_encoding(self) -> Encoding[slice]:
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

        return Encoding(
            serializer=_serialize,
            before_validator=_validate_slice,
            json_schema=core_schema.union_schema(
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


def _from_mapping(value: Mapping[str, ty.Any]) -> tuple[ty.Any, ty.Any, ty.Any]:
    for key in value:
        if key not in ("start", "stop", "step"):
            err_t = "slice_invalid_key"
            msg = 'invalid key "{key}" for slice, can only accept "start"/"stop"/"step"'
            raise PydanticCustomError(err_t, msg, {"key": key})
    return (value.get("start"), value.get("stop"), value.get("step"))


def _from_str(value: str) -> tuple[ty.Any, ty.Any, ty.Any]:
    try:
        start, stop, step = parse_slice_syntax(
            value,
            converter=str,
            require_start=False,
            require_stop=True,
        )
    except SliceSyntaxError as e:
        err_t = "slice_syntax_error"
        msg = "{what}"
        raise PydanticCustomError(err_t, msg, {"what": str(e)}) from e

    return (start, stop, step)


def _from_sequence(value: Sequence[ty.Any]) -> tuple[ty.Any, ty.Any, ty.Any]:
    size = len(value)
    if 1 <= size <= 3:  # noqa: PLR2004
        return tuple(value)
    err_t = "slice_length_error"
    msg = "a sequence input to slice must have 1-3 elements, was {size}"
    raise PydanticCustomError(err_t, msg, {"size": size})


def _validate_slice(value: ty.Any) -> slice:
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
            err_t = "slice_type_error"
            msg = "expected a slice, sequence, mapping or str, got {t}"
            raise PydanticCustomError(err_t, msg, {"t": type(value).__name__})

    return slice(start, stop, step)


def _prefix_validation_error(
    exc: pydantic.ValidationError,
    field: str,
) -> pydantic.ValidationError:
    """Rebuild a ValidationError with `field` prepended to every error path."""
    details: list[InitErrorDetails] = []
    for error in exc.errors(include_url=False):
        err_t = error["type"]
        msg = error["msg"]
        details.append(
            InitErrorDetails(
                # The message is rendered at this point, so these aren't literal
                # string anymore. It works at runtime.
                type=PydanticCustomError(err_t, msg),  # type: ignore[bad-argument-type]
                loc=(field, *error["loc"]),
                input=error["input"],
            )
        )
    return pydantic.ValidationError.from_exception_data(
        title=exc.title,
        line_errors=details,
    )


IntSliceAdapter = SliceAdapter(int | None)
