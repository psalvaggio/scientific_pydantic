"""Common functionality between slice and range"""

import typing as ty
from collections.abc import Hashable

import pydantic
from pydantic_core import InitErrorDetails, PydanticCustomError

UNSET = object()

T = ty.TypeVar("T", range, slice)


class StartStopStepAdapters:
    """Manages type adapters for the start/stop/step fields in range/slice"""

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
        default_adapter = adapters[default_type]
        self.start_adapter = (
            adapters[start_type] if start_type is not UNSET else default_adapter
        )
        self.stop_adapter = (
            adapters[stop_type] if stop_type is not UNSET else default_adapter
        )
        self.step_adapter = (
            adapters[step_type] if step_type is not UNSET else default_adapter
        )

    def after_validator(self, val: T) -> T:
        """Validate the fields w.r.t. to type adapters"""
        try:
            start = self.start_adapter.validate_python(val.start)
        except pydantic.ValidationError as e:
            raise _prefix_validation_error(e, "start") from None
        try:
            stop = self.stop_adapter.validate_python(val.stop)
        except pydantic.ValidationError as e:
            raise _prefix_validation_error(e, "stop") from None
        try:
            step = self.step_adapter.validate_python(val.step)
        except pydantic.ValidationError as e:
            raise _prefix_validation_error(e, "step") from None
        return type(val)(start, stop, step)


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
