"""Tests for scientific_pydantic.numpy.scalar_adapter."""

from __future__ import annotations

import json
import typing as ty

import numpy as np
import pydantic
import pytest

from scientific_pydantic.numpy import ScalarAdapter

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


_INT_TYPES: list[type] = [np.int8, np.int16, np.int32, np.int64]
_UINT_TYPES: list[type] = [np.uint8, np.uint16, np.uint32, np.uint64]
_FLOAT_TYPES: list[type] = [np.float16, np.float32, np.float64]
_COMPLEX_TYPES: list[type] = [np.complex64, np.complex128]


def _model(scalar_type: type, **adapter_kwargs: ty.Any) -> type[pydantic.BaseModel]:
    """Return a single-field pydantic model annotated with *scalar_type*."""
    adapter = ScalarAdapter(**adapter_kwargs)

    class _M(pydantic.BaseModel):
        v: ty.Annotated[scalar_type, adapter]  # type: ignore[valid-type]

    return _M


# ---------------------------------------------------------------------------
# Identity / type preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scalar_type", "value"),
    [
        pytest.param(np.int8, np.int8(42), id="int8-identity"),
        pytest.param(np.int32, np.int32(-7), id="int32-identity"),
        pytest.param(np.uint16, np.uint16(255), id="uint16-identity"),
        pytest.param(np.float32, np.float32(3.14), id="float32-identity"),
        pytest.param(np.float64, np.float64(-1.0), id="float64-identity"),
        pytest.param(np.complex128, np.complex128(1 + 2j), id="complex128-identity"),
        pytest.param(np.bool_, np.True_, id="bool-identity"),
    ],
)
def test_identity_passthrough(scalar_type: type, value: ty.Any) -> None:
    """Already-correct scalars are returned unchanged."""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[scalar_type, ScalarAdapter()]

    m = Model(v=value)
    assert isinstance(m.v, scalar_type)
    assert m.v == value


# ---------------------------------------------------------------------------
# Python scalar coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scalar_type", "python_value"),
    [
        pytest.param(np.int16, 100, id="int16-from-int"),
        pytest.param(np.uint8, 200, id="uint8-from-int"),
        pytest.param(np.float32, 2.718, id="float32-from-float"),
        pytest.param(np.float64, 0, id="float64-from-int"),
        pytest.param(np.bool_, True, id="bool-from-bool"),
        pytest.param(np.bool_, 0, id="bool-from-zero"),
    ],
)
def test_python_scalar_coercion(scalar_type: type, python_value: ty.Any) -> None:
    """Python scalars are coerced to the target numpy type."""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[scalar_type, ScalarAdapter()]

    m = Model(v=python_value)
    assert isinstance(m.v, scalar_type)


# ---------------------------------------------------------------------------
# String coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scalar_type", "str_value", "expected"),
    [
        pytest.param(np.float32, "3.14", np.float32(3.14), id="float32-from-str"),
        pytest.param(np.int64, "-99", np.int64(-99), id="int64-from-str"),
        pytest.param(np.uint32, "42", np.uint32(42), id="uint32-from-str"),
    ],
)
def test_string_coercion(scalar_type: type, str_value: str, expected: ty.Any) -> None:
    """String inputs are parsed and cast to the target numpy scalar."""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[scalar_type, ScalarAdapter()]

    m = Model(v=str_value)
    assert isinstance(m.v, scalar_type)
    np.testing.assert_allclose(float(m.v), float(expected))


# ---------------------------------------------------------------------------
# Cross-kind numpy casting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target_type", "input_scalar"),
    [
        pytest.param(np.float64, np.int32(5), id="float64-from-int32"),
        pytest.param(np.int32, np.float32(7.0), id="int32-from-float32"),
        pytest.param(np.uint8, np.int64(200), id="uint8-from-int64"),
    ],
)
def test_cross_kind_cast(target_type: type, input_scalar: np.generic) -> None:
    """Numpy scalars of different kinds are cast to the target type."""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[target_type, ScalarAdapter()]

    m = Model(v=input_scalar)
    assert isinstance(m.v, target_type)
    assert m.v == target_type(input_scalar)


# ---------------------------------------------------------------------------
# Bounds validation - accepted values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scalar_type", "kwargs", "value"),
    [
        pytest.param(np.float64, {"ge": 0.0}, 0.0, id="float64-ge=0-boundary"),
        pytest.param(np.float64, {"ge": 0.0}, 1.5, id="float64-ge=0-above"),
        pytest.param(np.float64, {"gt": 0.0}, 0.001, id="float64-gt=0"),
        pytest.param(np.float32, {"le": 1.0}, 1.0, id="float32-le=1-boundary"),
        pytest.param(np.float32, {"lt": 1.0}, 0.999, id="float32-lt=1"),
        pytest.param(np.int32, {"ge": -10, "le": 10}, 0, id="int32-bounded-middle"),
        pytest.param(np.uint8, {"ge": 0, "le": 255}, 255, id="uint8-max"),
    ],
)
def test_bounds_accepted(
    scalar_type: type, kwargs: dict[str, float], value: ty.Any
) -> None:
    """Values within the specified bounds are accepted."""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[scalar_type, ScalarAdapter(**kwargs)]

    m = Model(v=value)
    assert isinstance(m.v, scalar_type)


# ---------------------------------------------------------------------------
# Bounds validation - rejected values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scalar_type", "kwargs", "bad_value"),
    [
        pytest.param(np.float64, {"ge": 0.0}, -0.001, id="float64-ge=0-below"),
        pytest.param(np.float64, {"gt": 0.0}, 0.0, id="float64-gt=0-at-zero"),
        pytest.param(np.float32, {"le": 1.0}, 1.001, id="float32-le=1-above"),
        pytest.param(np.float32, {"lt": 1.0}, 1.0, id="float32-lt=1-at-bound"),
        pytest.param(np.int32, {"ge": -10, "le": 10}, 11, id="int32-exceeds-upper"),
        pytest.param(np.int32, {"ge": -10, "le": 10}, -11, id="int32-below-lower"),
    ],
)
def test_bounds_rejected(
    scalar_type: type, kwargs: dict[str, float], bad_value: ty.Any
) -> None:
    """Values outside the specified bounds raise ValidationError."""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[scalar_type, ScalarAdapter(**kwargs)]

    with pytest.raises(pydantic.ValidationError):
        Model(v=bad_value)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scalar_type", "bad_value"),
    [
        pytest.param(np.float32, "not-a-number", id="float32-bad-str"),
        pytest.param(np.int16, [1, 2], id="int16-from-list"),
        pytest.param(np.uint8, -1, id="uint8-overflow-neg"),
        pytest.param(np.int8, 200, id="int8-overflow-pos"),
    ],
)
def test_invalid_inputs_rejected(scalar_type: type, bad_value: ty.Any) -> None:
    """Clearly invalid inputs raise ValidationError."""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[scalar_type, ScalarAdapter()]

    with pytest.raises(pydantic.ValidationError):
        Model(v=bad_value)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scalar_type", "value"),
    [
        pytest.param(np.int32, np.int32(7), id="int32-roundtrip"),
        pytest.param(np.uint8, np.uint8(255), id="uint8-roundtrip"),
        pytest.param(np.float32, np.float32(1.5), id="float32-roundtrip"),
        pytest.param(np.float64, np.float64(-2.71828), id="float64-roundtrip"),
        pytest.param(np.bool_, np.False_, id="bool-roundtrip"),
    ],
)
def test_json_roundtrip(scalar_type: type, value: ty.Any) -> None:
    """model_dump(mode='json') and model_validate_json produce equivalent models."""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[scalar_type, ScalarAdapter()]

    m = Model(v=value)
    json_str = m.model_dump_json()
    # Ensure the serialised payload is valid JSON
    payload = json.loads(json_str)
    assert "v" in payload

    # Re-validate from JSON
    m2 = Model.model_validate_json(json_str)
    assert isinstance(m2.v, scalar_type)
    np.testing.assert_allclose(float(m2.v), float(value), rtol=1e-4)


def test_complex_json_roundtrip() -> None:
    """Complex scalars serialise to a string and round-trip cleanly."""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[np.complex128, ScalarAdapter()]

    val = np.complex128(3.0 + 4.0j)
    m = Model(v=val)
    json_str = m.model_dump_json()
    payload = json.loads(json_str)
    # The serialised form must be a string representing the complex number
    assert isinstance(payload["v"], str)
    m2 = Model.model_validate_json(json_str)
    assert isinstance(m2.v, np.complex128)
    assert np.isclose(m2.v, val)


# ---------------------------------------------------------------------------
# JSON schema generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scalar_type", "kwargs", "expected_fragments"),
    [
        pytest.param(np.float32, {}, ["float32"], id="float32-no-bounds"),
        pytest.param(
            np.float64,
            {"ge": 0.0, "le": 1.0},
            ["float64", ">= 0.0", "<= 1.0"],
            id="float64-with-bounds",
        ),
        pytest.param(
            np.int32,
            {"gt": -5, "lt": 5},
            ["int32", "> -5", "< 5"],
            id="int32-strict-bounds",
        ),
    ],
)
def test_json_schema_description(
    scalar_type: type,
    kwargs: dict[str, float],
    expected_fragments: list[str],
) -> None:
    """The generated JSON schema description includes dtype and constraint info."""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[scalar_type, ScalarAdapter(**kwargs)]

    schema = Model.model_json_schema()
    desc: str = schema.get("properties", {}).get("v", {}).get("description", "")
    for fragment in expected_fragments:
        assert fragment in desc, f"Expected {fragment} in description: {desc}"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_bad_base_type() -> None:
    """Test using the adapter on an invalid base type"""
    with pytest.raises(
        pydantic.PydanticSchemaGenerationError,
        match="was not a supported NumPy scalar type",
    ):

        class Model(pydantic.BaseModel):
            f: ty.Annotated[bool, ScalarAdapter()]


@pytest.mark.parametrize(
    ("source_type", "value"),
    [
        pytest.param(np.uint16, np.bytes_(b"abc"), id="bytes-to-u16"),
        pytest.param(np.int16, np.str_("hi"), id="str-to-i16"),
    ],
)
def test_cast_overflow(
    source_type: type,
    value: np.generic,
) -> None:
    """Test overflow from numpy type casting"""

    class Model(pydantic.BaseModel):
        v: ty.Annotated[source_type, ScalarAdapter()]

    with pytest.raises(pydantic.ValidationError, match="numpy_scalar_cast"):
        Model(v=value)
