"""Pydantic adapter for astropy.units.PhysicalType."""

from __future__ import annotations

import typing as ty

import pydantic
from pydantic_core import core_schema

from scientific_pydantic.schema import make_core_schema


class PhysicalTypeAdapter:
    """A pydantic adapter for astropy.units.PhysicalType

    Validation Options
    ------------------
    1. `PhysicalType` - Identity.
    2. `str` - The name of the physical type (e.g. `"length"`, `"mass"`).
        This is the form used for JSON encoding.

    Examples
    --------
    >>> import typing as ty
    >>> import pydantic
    >>> import astropy.units as u
    >>> from scientific_pydantic.astropy.units import (
    ...     PhysicalTypeAdapter,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> class Model(pydantic.BaseModel):
    ...     pt: ty.Annotated[
    ...         u.PhysicalType, PhysicalTypeAdapter()
    ...     ]  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> Model(pt="length")
    Model(pt=PhysicalType('length'))
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        import astropy.units as u

        from .validators import validate_physical_type

        if source_type is not u.PhysicalType:
            msg = (
                "PhysicalTypeAdapter is only usable with "
                f"astropy.units.PhysicalType, not {source_type}."
            )
            raise pydantic.PydanticSchemaGenerationError(msg)

        return make_core_schema(
            u.PhysicalType,
            serializer=str,
            before_validator=validate_physical_type,
            json_schema=core_schema.str_schema(),
        )

    def __get_pydantic_json_schema__(
        self,
        core_schema: core_schema.CoreSchema,
        handler: pydantic.json_schema.GetJsonSchemaHandler,
    ) -> pydantic.json_schema.JsonSchemaValue:
        """Get the JSON schema for this type"""
        desc = "An astropy PhysicalType expressed as a string."
        return handler(core_schema) | {
            "description": desc,
            "examples": ["length", "area"],
        }
