"""Pydantic adapter for astropy.units.Unit."""

from __future__ import annotations

import typing as ty

if ty.TYPE_CHECKING:
    import astropy.units as u

import pydantic_core


class EquivalencyValidator:
    """Validator for unit equivalency

    Parameters
    ----------
    equivalent_unit : astropy.units.UnitBase | str
        Validated values must be equivalent to this unit.
    equivalencies:
        Optional list of astropy equivalency pairs (as returned by e.g.
        ``astropy.units.spectral()``).  Passed verbatim to
        ``UnitBase.is_equivalent``.
    """

    def __init__(
        self,
        equivalent_unit: u.UnitBase | str,
        *,
        equivalencies: list[tuple] | None = None,
    ) -> None:
        if isinstance(equivalent_unit, str):
            import astropy.units as u

            equivalent_unit = u.Unit(equivalent_unit)

        self._equivalent_unit = equivalent_unit
        self._equivalencies = equivalencies

    @property
    def equivalent_unit(self) -> u.UnitBase:
        """If non-None, validated units must be equivalent to this unit"""
        return self._equivalent_unit

    @property
    def equivalencies(self) -> list[tuple] | None:
        """Custom equivalencies for the equivalency check"""
        return self._equivalencies

    def __call__(self, unit: u.UnitBase) -> u.UnitBase:
        """Validate the given unit for equivalency

        Parameters
        ----------
        unit : astropy.units.UnitBase
            The unit to validate

        Returns
        -------
        astropy.units.UnitBase
            The input ``unit`` (for validator chaining)
        """
        if self.equivalent_unit is None or unit.is_equivalent(
            self.equivalent_unit, equivalencies=self.equivalencies
        ):
            return unit

        equiv_hint = (
            f" (with equivalencies: {self.equivalencies})"
            if self.equivalencies is not None
            else ""
        )
        err_t = "astropy_unit_not_equivalent"
        msg = "Unit {unit} is not equivalent to {target}{hint}"
        raise pydantic_core.PydanticCustomError(
            err_t,
            msg,
            {
                "unit": unit.to_string(),
                "target": self.equivalent_unit.to_string(),
                "hint": equiv_hint,
            },
        )
