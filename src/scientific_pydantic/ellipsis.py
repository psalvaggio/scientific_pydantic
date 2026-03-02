"""Pydantic adapter for Ellipsis"""

import types
import typing as ty

import pydantic
from pydantic_core import core_schema


class EllipsisAdapter:
    """A Pydantic annotation for the `Ellipsis` singleton (`...`).

    In general, you should use the publicly-defined alias
    [EllipsisLiteral][scientific_pydantic.EllipsisLiteral] to express when you
    want `...` stored in your model.

    Validation Options
    ------------------
    1. `Ellipsis`/`...`: Identity.
    2. `ty.Literal["..."]` - The string "...". This is used in JSON encoding.

    Examples
    --------
    >>> import pydantic
    >>> from scientific_pydantic import (
    ...     EllipsisLiteral,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> class Model(pydantic.BaseModel):
    ...     shape: list[EllipsisLiteral | int]  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> Model(shape=[1, ..., 2])
    Model(shape=[1, Ellipsis, 2])
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: ty.Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        del handler

        if source_type is not types.EllipsisType and not (
            ty.get_origin(source_type) is ty.Literal
            and len(args := ty.get_args(source_type)) > 0
            and args[0] is Ellipsis
        ):
            msg = (
                "EllipsisAdapter is only usable with EllipsisType or "
                f"Literal[Ellipsis], not {source_type}"
            )
            raise pydantic.PydanticSchemaGenerationError(msg)

        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda _: "...",
                when_used="json",
            ),
        )

    @classmethod
    def _validate(cls, value: ty.Any) -> types.EllipsisType:
        if value is ... or value == "...":
            return ...
        msg = f"Expected Ellipsis (...), got {value!r}"
        raise ValueError(msg)


EllipsisLiteral = ty.Annotated[types.EllipsisType, EllipsisAdapter()]
