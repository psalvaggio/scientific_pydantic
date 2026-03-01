"""Pydantic adapter for astropy.time.Time"""

from __future__ import annotations

import dataclasses
import functools
import typing as ty
from collections.abc import Mapping, Sequence

import pydantic
from pydantic_core import PydanticCustomError, core_schema

if ty.TYPE_CHECKING:
    import types

    from astropy.time import Time
    from pydantic import GetCoreSchemaHandler


class TimeAdapter:
    """Pydantic adapter for astropy.time.Time

    Parameters
    ----------
    scalar : bool
        If True, only scalar times will be accepted. If False, only vector times
        will be accepted. If None, no scalar constraints are enforced, unless
        `ndim` or `shape` are provided.
    ndim : int | None
        If given, the dimensionality of the time must match this value. Must
        be >= 0.
    shape : Sequence[Ellipsis | int | range | slice | None] | None
        Shape specifier for the given time(s). See `NDArrayValidator` for a
        description of how this works.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        scalar: bool | None = None,
        ndim: int | None = None,
        shape: Sequence[types.EllipsisType | int | range | slice | None] | None = None,
        ge: ty.Any = None,
        gt: ty.Any = None,
        le: ty.Any = None,
        lt: ty.Any = None,
    ) -> None:

        from scientific_pydantic.numpy.validators import (
            validate_all_ge,
            validate_all_gt,
            validate_all_le,
            validate_all_lt,
        )

        from ..validators import ArrayShapeValidator

        @dataclasses.dataclass
        class CtorValidators:
            shape: ArrayShapeValidator
            ge: ty.Callable[[Time], Time] | None = None
            gt: ty.Callable[[Time], Time] | None = None
            le: ty.Callable[[Time], Time] | None = None
            lt: ty.Callable[[Time], Time] | None = None

        validators: dict[str, ty.Callable] = {}
        for bound, name, val in (
            (gt, "gt", validate_all_gt),
            (ge, "ge", validate_all_ge),
            (lt, "lt", validate_all_lt),
            (le, "le", validate_all_le),
        ):
            if bound is None:
                continue
            try:
                bound_t = _validate_time(bound)
            except ValueError as e:
                msg = f"while validating the {name} constraint:\n{e}"
                raise pydantic.PydanticSchemaGenerationError(msg) from e

            validators[name] = functools.partial(val, bound=bound_t)

        self._validators = CtorValidators(
            shape=ArrayShapeValidator(scalar=scalar, ndim=ndim, shape=shape),
            **validators,
        )

    def __get_pydantic_core_schema__(
        self,
        _source_type: ty.Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        to_time = core_schema.no_info_plain_validator_function(_validate_time)
        validators = core_schema.chain_schema(
            [to_time]
            + [
                core_schema.no_info_plain_validator_function(func)
                for field in dataclasses.fields(self._validators)
                if (func := getattr(self._validators, field.name)) is not None
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=validators,
            python_schema=validators,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_json,
                when_used="json-unless-none",
            ),
        )


def _unambiguous_parse(val: ty.Any) -> Time:
    """Create a Time directly from an input object, must work unambiguously"""
    from astropy.time import Time

    try:
        return Time(val)
    except (ValueError, TypeError) as e:
        err_t = "invalid_time"
        msg = "Unable to construct an astropy Time:\n{e}"
        raise PydanticCustomError(err_t, msg, {"e": str(e)}) from e


def _from_mapping(data: Mapping[str, ty.Any]) -> Time:
    """Create a Time from a mapping representing the constructor arguments"""
    from astropy.time import Time

    ctor_kwargs: dict[str, ty.Any] = {
        "format": data.get("format"),
        "scale": data.get("scale"),
        "precision": data.get("precision"),
        "in_subfmt": data.get("in_subfmt"),
        "out_subfmt": data.get("out_subfmt"),
        "location": data.get("location"),
    }
    if (val := data.get("value", data.get("val"))) is None:
        err_t = "missing_value"
        msg = 'Mapping is missing a "value" or "val" key'
        raise PydanticCustomError(err_t, msg)
    val2 = data.get("value2", data.get("val2"))

    try:
        return Time(val, val2=val2, **ctor_kwargs)
    except (ValueError, TypeError) as e:
        err_t = "invalid_time"
        msg = "Unable to construct an astropy Time:\n{e}"
        raise PydanticCustomError(err_t, msg, {"e": str(e)}) from e


def _validate_time(data: ty.Any) -> Time:
    """Create a Time from user input"""
    from astropy.time import Time

    if isinstance(data, Time):
        return data
    if isinstance(data, Mapping):
        return _from_mapping(data)

    return _unambiguous_parse(data)


def _serialize_json(t: Time) -> dict[str, ty.Any]:
    """Serialize for use in JSON"""
    if t.masked:
        msg = "Serialization of masked times is not supported yet"
        raise NotImplementedError(msg)

    if t.format == "datetime":  # not JSON serializeable
        t = t.copy("isot")

    res: dict[str, ty.Any] = {}
    if t.precision > 6:  # noqa: PLR2004
        res |= {
            "value": t.jd1 if t.isscalar else t.jd1.tolist(),  # type: ignore[missing-attribute]
            "value2": t.jd2 if t.isscalar else t.jd2.tolist(),  # type: ignore[missing-attribute]
            "format": "jd",
        }
    else:
        res |= {
            "value": t.value if t.isscalar else t.value.tolist(),
            "format": t.format,
        }

    res["scale"] = t.scale
    if t.precision != 3:  # noqa: PLR2004
        res["precision"] = t.precision
    if t.in_subfmt != "*":
        res["in_subfmt"] = t.in_subfmt
    if t.out_subfmt != "*":
        res["out_subfmt"] = t.out_subfmt
    if t.location is not None:
        import astropy.units as u

        res["location"] = [
            float(t.location.geodetic.lon.to_value(u.deg)),
            float(t.location.geodetic.lat.to_value(u.deg)),
            float(t.location.geodetic.height.to_value(u.m)),
        ]

    return res
