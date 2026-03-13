"""Schema generation utilities"""

import dataclasses
import inspect
import typing as ty

from pydantic_core import core_schema

NativeT = ty.TypeVar("NativeT")


@dataclasses.dataclass(frozen=True)
class Encoding(ty.Generic[NativeT]):
    """A matched serializer / deserializer pair for type encoding

    Use this to specify how type conversions are performed in a
    scientific-pydantic type adapter.

    Parameters
    ----------
    serializer
        Converts a validated value to a JSON-serializable object. May take one
        argument ``(value,)`` or two ``(value, info)`` where ``info`` is a
        ``pydantic_core.SerializationInfo`` instance.
    before_validator
        Converts the raw input (after ``json_schema`` coercion on the JSON path)
        into the destination type.  The library's default ``before_validator``
        is replaced entirely, so this function must also handle the
        Python-mode path (i.e. accept an already-constructed instance and
        return it unchanged).
    json_schema
        A ``pydantic_core`` schema describing the expected JSON input data. This
        becomes the ``json_schema_input_schema`` for ``before_validator``
        and is used for JSON schema generation.
    serializer_when_used
        The ``when_used`` argument for the ``serializer``. By default, the
        serializer is only called in JSON mode when the value is non-None.

    Examples
    --------
    Encode an ``astropy.units.Quantity`` as a plain float in meters instead of
    the default ``{"value": ..., "unit": "..."}`` dict:

    >>> import astropy.units as u
    >>> import pydantic
    >>> from pydantic_core import core_schema
    >>> from scientific_pydantic import Encoding
    >>> from scientific_pydantic.astropy.units import QuantityAdapter
    >>>
    >>> def _to_meters(value: object) -> u.Quantity:
    ...     if isinstance(value, u.Quantity):
    ...         return value.to(u.m)
    ...     return u.Quantity(float(value), u.m)  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> meters_encoding = Encoding[u.Quantity](
    ...     serializer=lambda q, _info: q.to(u.m).value,
    ...     before_validator=_to_meters,
    ...     json_schema=core_schema.float_schema(),
    ... )  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> class Model(pydantic.BaseModel):
    ...     q: ty.Annotated[
    ...         u.Quantity,
    ...         QuantityAdapter(encoding=meters_encoding),
    ...     ]  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> Model(q=5 << u.m).model_dump(mode="json")
    {'q': 5.0}
    >>> Model(q=5 << u.mm).model_dump(mode="json")
    {'q': 0.005}
    """

    serializer: (
        ty.Callable[[NativeT], ty.Any]
        | ty.Callable[[NativeT, core_schema.SerializationInfo], ty.Any]
    )
    before_validator: (
        ty.Callable[[ty.Any], NativeT]
        | ty.Callable[[ty.Any, core_schema.ValidationInfo], NativeT]
    )
    json_schema: core_schema.CoreSchema | None = None
    serializer_when_used: core_schema.WhenUsed = "json-unless-none"


def make_core_schema(
    dtype: type[NativeT],
    *,
    encoding: Encoding[NativeT],
    after_validators: ty.Sequence[
        ty.Callable[[NativeT], NativeT]
        | ty.Callable[[NativeT, core_schema.ValidationInfo], NativeT]
    ]
    | None = None,
) -> core_schema.CoreSchema:
    """Make a pydantic core schema for our custom adapters

    Parameters
    ----------
    dtype
        The data type being adapted
    encoding
        Encoding object that describes how to serialize/validate the object
        along both the JSON and Python modes.
    after_validators
        After validators to apply to the value once validated into the type.
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
                if _takes_info_param(encoding.before_validator)
                else core_schema.no_info_before_validator_function
            )(
                encoding.before_validator,  # type: ignore[bad-argument-type]
                is_dtype,
                json_schema_input_schema=encoding.json_schema,
            ),
            *after_validator_schemas,
        ],
        serialization=core_schema.plain_serializer_function_ser_schema(
            encoding.serializer,
            when_used=encoding.serializer_when_used,
            info_arg=_takes_info_param(encoding.serializer),
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
