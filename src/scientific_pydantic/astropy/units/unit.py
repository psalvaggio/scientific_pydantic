"""Pydantic adapter for astropy.units.Unit."""

from __future__ import annotations

import dataclasses
import typing as ty

from scientific_pydantic.schema import make_core_schema

if ty.TYPE_CHECKING:
    import astropy.units as u

import pydantic
from pydantic_core import core_schema


class UnitAdapter:
    """A `pydantic` adapter for `astropy.units.UnitBase`

    Validation Options
    ------------------
    1. `UnitBase` - Identity.
    2. `str` - A string encoding of units that can be passed to the constructor
           of `Unit` (e.g. `"kg m / s2"`). This is the form used for JSON
           encoding.

    Parameters
    ----------
    equivalent_unit
        If given, validated values must be equivalent to this unit.
    equivalencies
        Optional list of astropy equivalency pairs (as returned by e.g.
        ``astropy.units.spectral()``).  Passed verbatim to
        ``UnitBase.is_equivalent``.
    physical_type
        If given, the unit by have this physical type

    Examples
    --------
    >>> import typing as ty
    >>> import pydantic
    >>> import astropy.units as u
    >>> from scientific_pydantic.astropy.units import (
    ...     UnitAdapter,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> class Model(pydantic.BaseModel):
    ...     u: ty.Annotated[u.UnitBase, UnitAdapter()]  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> Model(u="kg m / s2")
    Model(u=Unit("kg m / s2"))
    """

    def __init__(
        self,
        equivalent_unit: u.UnitBase | str | None = None,
        *,
        equivalencies: list[tuple] | None = None,
        physical_type: u.PhysicalType | str | u.Quantity | u.UnitBase | None = None,
    ) -> None:
        from .validators import (
            EquivalencyValidator,
            PhysicalTypeValidator,
            validate_physical_type,
        )

        @dataclasses.dataclass
        class Validators:
            equivalency: EquivalencyValidator | None = None
            physical_type: PhysicalTypeValidator | None = None

        validators: dict[str, ty.Any] = {}

        if equivalent_unit is not None:
            validators["equivalency"] = EquivalencyValidator(
                equivalent_unit, equivalencies=equivalencies
            )

        if physical_type is not None:
            validators["physical_type"] = PhysicalTypeValidator(
                validate_physical_type(physical_type)
            )
        self._validators = Validators(**validators)

    @property
    def equivalent_unit(self) -> u.UnitBase | None:
        """If non-None, validated units must be equivalent to this unit"""
        val = self._validators.equivalency
        return val.equivalent_unit if val is not None else None

    @property
    def equivalencies(self) -> list[tuple] | None:
        """Custom equivalencies for the equivalency check"""
        val = self._validators.equivalency
        return val.equivalencies if val is not None else None

    @property
    def physical_type(self) -> u.PhysicalType | None:
        """If non-None, validated unit must be of this physical type"""
        val = self._validators.physical_type
        return val.physical_type if val is not None else None

    def __get_pydantic_core_schema__(
        self,
        source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        import astropy.units as u

        from .validators import validate_unit

        if source_type is not u.UnitBase:
            msg = (
                "UnitAdapter is only usable with "
                f"astropy.units.UnitBase, not {source_type}."
            )
            raise pydantic.PydanticSchemaGenerationError(msg)

        validators = [
            f
            for f in (self._validators.equivalency, self._validators.physical_type)
            if f is not None
        ]

        return make_core_schema(
            u.UnitBase,
            serializer=str,
            before_validator=validate_unit,
            after_validators=validators,
            json_schema=core_schema.str_schema(),
        )

    def __get_pydantic_json_schema__(
        self,
        core_schema: core_schema.CoreSchema,
        handler: pydantic.json_schema.GetJsonSchemaHandler,
    ) -> pydantic.json_schema.JsonSchemaValue:
        """Get the JSON schema for this type"""
        desc = "An astropy unit expressed as a string."
        if self.equivalent_unit is not None:
            equiv_hint = ""
            if self.equivalencies is not None:
                equiv_hint = (
                    " (with custom equivalencies: "
                    + ", ".join(f"{x[0]} <-> {x[1]}" for x in self.equivalencies)
                    + ")"
                )
            desc += f' Must be equivalent to "{self.equivalent_unit}"{equiv_hint}.'
        if self.physical_type is not None:
            desc += f' Must be of type "{self.physical_type!s}".'

        return handler(core_schema) | {
            "description": desc,
            "examples": ["m / s", "km / h", "kg", "deg", "J / (kg K)"]
            if self.equivalent_unit is None
            else [str(self.equivalent_unit)],
        }
