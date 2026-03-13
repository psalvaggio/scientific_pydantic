"""Schema generation utilities"""

import inspect
import typing as ty

from pydantic_core import core_schema

NativeT = ty.TypeVar("NativeT")


def make_core_schema(  # noqa: PLR0913
    dtype: type[NativeT],
    *,
    serializer: ty.Callable[[NativeT], ty.Any],
    before_validator: ty.Callable[[ty.Any], NativeT]
    | ty.Callable[[object, core_schema.ValidationInfo], NativeT],
    after_validators: ty.Sequence[
        ty.Callable[[NativeT], NativeT]
        | ty.Callable[[NativeT, core_schema.ValidationInfo], NativeT]
    ]
    | None = None,
    json_schema: core_schema.CoreSchema | None = None,
    serializer_when_used: core_schema.WhenUsed = "json-unless-none",
) -> core_schema.CoreSchema:
    """Make a pydantic core schema for our custom adapters

    Parameters
    ----------
    dtype
        The data type being adapted
    serializer
        Custom serialization function. By default, this is only used for JSON
        serialization. `model_dump()` in Python mode will just pass the field
        value through as-is.
    before_validator
        A before validator to used for type transformations. This is used in
        both Python and JSON validation, so be sure to check for the type
        itself.
    after_validators
        After validators to apply to the value once validated into the type.
    json_schema
        Core schema for the value in JSON. This is the
        `json_schema_input_schema` for before_validator.
    serializer_when_used
        The `when_used` argument for the `serializer`. By default, the
        serializer is only called in JSON mode when the value is non-None.
    """
    is_dtype = core_schema.is_instance_schema(dtype)

    after_validator_schemas = [
        (
            core_schema.with_info_after_validator_function
            if _takes_info_param(val)
            else core_schema.no_info_after_validator_function
            # We can use any_schema here because is_dtype() is applied to the
            # output of before_validator. The types of the arguments say that we
            # have to pass through NativeT, that is a possible user error, but
            # we're not going to waste the overhead of additional checks here.
        )(val, core_schema.any_schema())  # type: ignore[bad-argument-type]
        for val in after_validators or []
    ]

    return core_schema.chain_schema(
        [
            (
                core_schema.with_info_before_validator_function
                if _takes_info_param(before_validator)
                else core_schema.no_info_before_validator_function
            )(
                before_validator,  # type: ignore[bad-argument-type]
                is_dtype,
                json_schema_input_schema=json_schema,
            ),
            *after_validator_schemas,
        ],
        serialization=core_schema.plain_serializer_function_ser_schema(
            serializer,
            when_used=serializer_when_used,
            info_arg=_takes_info_param(serializer),
        ),
    )


def _takes_info_param(callable_func: ty.Callable) -> bool:
    """Test whether the given callable is a "with info" callable

    The current implementation just tests if it is a 2-argument function.
    """
    try:
        sig = inspect.signature(callable_func)
    except (ValueError, TypeError):
        return False

    # Filter out parameters with default values and *args, **kwargs
    required_params = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    ]
    num_required = len(required_params)

    # The function should be callable with exactly 1 or 2 arguments,
    # considering *args and **kwargs make the number of arguments variable
    has_var_args = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
    has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    if has_var_args or has_var_kwargs:
        return False
    return num_required == 2  # noqa: PLR2004
