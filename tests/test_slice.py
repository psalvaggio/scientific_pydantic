"""Unit test for range.py"""

import typing as ty

import pydantic
import pytest
from pydantic_core import core_schema

from scientific_pydantic import Encoding, IntSliceAdapter, SliceAdapter


class IntModel(pydantic.BaseModel):
    """Model with an int slice"""

    s: ty.Annotated[slice, IntSliceAdapter]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(slice(5), slice(5), id="slice(5)"),
        pytest.param(":5", slice(5), id=":5"),
        pytest.param(" : 5", slice(5), id=" : 5"),
        pytest.param("1:5", slice(1, 5), id="1:5"),
        pytest.param(" 1\t:\n5 : ", slice(1, 5), id=" 1 : 5 : "),
        pytest.param("1:10:2", slice(1, 10, 2), id="1:10:2"),
        pytest.param(
            {"start": 1, "stop": 10, "step": 2},
            slice(1, 10, 2),
            id="1:10:2-dict",
        ),
    ],
)
def test_int_slice_validation(value: ty.Any, expected: range) -> None:
    """Valid inputs are converted to a range."""
    assert IntModel(s=value).s == expected


@pytest.mark.parametrize(
    ("value", "match"),
    [
        pytest.param(123, "slice_type_error", id="int"),
        pytest.param("5", "slice_syntax_error", id="5"),
        pytest.param("random text", "slice_syntax_error", id="random text"),
        pytest.param(
            "random:text:with colons", r"(?s)\.start.*int_parsing", id="non-ints"
        ),
        pytest.param("1:2:3:4", "(?s)got 4.*slice_syntax_error", id="1:2:3:4"),
        pytest.param(slice("a", "b"), "int_parsing", id="str-slice"),
        pytest.param(slice(1, 2, "c"), r"(?s)\.step.*int_parsing", id="str-slice-step"),
    ],
)
def test_int_slice_validation_errors(value: ty.Any, match: str) -> None:
    """Invalid inputs raise ValidationError."""
    with pytest.raises(pydantic.ValidationError, match=match):
        IntModel(s=value)


@pytest.mark.parametrize(
    ("value", "truth"),
    [
        pytest.param(slice(5), ":5", id=":5"),
        pytest.param(slice(1, 5), "1:5", id="1:5"),
        pytest.param(slice(1, 4, 2), "1:4:2", id="1:4:2"),
    ],
)
def test_int_slice_serialization(value: slice, truth: str) -> None:
    """Range serializes to JSON-compatible mapping."""
    model = IntModel(s=value)
    assert model.model_dump() == {"s": value}
    assert model.model_dump(mode="json") == {"s": truth}


def test_json_schema() -> None:
    """JSON schema is stable and well-defined."""
    schema = IntModel.model_json_schema()
    assert schema["properties"]["s"] == {
        "title": "S",
        "anyOf": [
            {"type": "string"},
            {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {},  # this should be populated better
            },
            {
                "type": "object",
                "properties": {
                    "start": {
                        "title": "Start",
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                    },
                    "stop": {
                        "title": "Stop",
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                    },
                    "step": {
                        "title": "Step",
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                    },
                },
            },
        ],
    }


def test_datetime_timedelta() -> None:
    """Make a heterogenuous slice"""
    import datetime

    class Model(pydantic.BaseModel):
        s: ty.Annotated[
            slice, SliceAdapter(datetime.datetime, step_type=datetime.timedelta)
        ]

    x = Model(
        s=("2026-01-02T01:00:00", "2026-01-03T02:00:00", "PT2H1M"),  # type: ignore[bad-argument-type]
    )

    assert x.s.start == datetime.datetime.fromisoformat("2026-01-02T01:00:00")
    assert x.s.stop == datetime.datetime.fromisoformat("2026-01-03T02:00:00")
    assert x.s.step == datetime.timedelta(seconds=7260)


def test_union_members() -> None:
    """Test whether unions can be used for the types"""

    class Model(pydantic.BaseModel):
        s: ty.Annotated[slice, SliceAdapter(int, stop_type=int | str)]

    assert Model(s=(1, "s", 2)).s == slice(1, "s", 2)  # type: ignore[bad-argument-type]
    assert Model(s=(1, 0, 2)).s == slice(1, 0, 2)  # type: ignore[bad-argument-type]


def test_custom_encoding() -> None:
    """Test a custom encoding of a slice"""

    def serialize(s: slice) -> str:
        return f"{s.start}::{s.stop}::{s.step}"

    def validate(val: ty.Any) -> slice:
        if isinstance(val, slice):
            return val
        return slice(*val.split("::"))

    encoding = Encoding(
        serializer=serialize,
        before_validator=validate,
        json_schema=core_schema.str_schema(),
    )

    class Model(pydantic.BaseModel):
        field: ty.Annotated[slice, SliceAdapter(int, encoding=encoding)]

    m = Model(field="1::2::3")  # type: ignore[bad-argument-type]
    assert m.field == slice(1, 2, 3)
    assert m.model_dump(mode="json") == {"field": "1::2::3"}
