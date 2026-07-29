"""Unit test for range.py"""

import typing as ty

import pydantic
import pytest

from scientific_pydantic import RangeAdapter


@pytest.mark.parametrize(
    ("kwargs", "value", "expected"),
    [
        pytest.param({}, range(5), range(5), id="range(5)"),
        pytest.param({}, ":5", range(5), id=":5"),
        pytest.param({}, " : 5", range(5), id=" : 5"),
        pytest.param({}, "1:5", range(1, 5), id="1:5"),
        pytest.param({}, " 1\t:\n5 : ", range(1, 5), id=" 1 : 5 : "),
        pytest.param({}, "1:10:2", range(1, 10, 2), id="1:10:2"),
        pytest.param(
            {"default_type": pydantic.PositiveInt},
            "1:10:2",
            range(1, 10, 2),
            id="pos-1:10:2",
        ),
        pytest.param(
            {"step_type": pydantic.NegativeInt},
            "1:10:-2",
            range(1, 10, -12),
            id="neg-step-1:10:-12",
        ),
    ],
)
def test_range_validation(
    kwargs: dict[str, ty.Any], value: ty.Any, expected: range
) -> None:
    """Valid inputs are converted to a range."""

    class Model(pydantic.BaseModel):
        r: ty.Annotated[range, RangeAdapter(**kwargs)]

    model = Model(r=value)
    assert model.r == expected


@pytest.mark.parametrize(
    ("kwargs", "value", "match"),
    [
        pytest.param(
            {},
            123,
            "expected range or slice-syntax string, got int",
            id="invalid_range",
        ),
        pytest.param(
            {},
            "5",
            "invalid range syntax, expected 2-3 parts separated by :'s, got 1",
            id="5",
        ),
        pytest.param(
            {},
            "random text",
            "invalid range syntax, expected 2-3 parts separated by :'s, got 1",
            id="random text",
        ),
        pytest.param(
            {},
            "random:text:with colons",
            "invalid integer in range string",
            id="non-ints",
        ),
        pytest.param(
            {},
            "1:2:3:4",
            "invalid range syntax, expected 2-3 parts separated by :'s, got 4",
            id="1:2:3:4",
        ),
        pytest.param(
            {"default_type": pydantic.PositiveInt},
            "-1:3",
            r"(?s)r\.start.*Input should be greater than 0",
            id="pos--1:3:1",
        ),
        pytest.param(
            {"stop_type": pydantic.PositiveInt},
            "-10:-2:3",
            r"(?s)r\.stop.*Input should be greater than 0",
            id="pos-stop--10:-2:3",
        ),
        pytest.param(
            {"start_type": pydantic.PositiveInt},
            "-1:3",
            r"(?s)r\.start.*Input should be greater than 0",
            id="pos-start--1:3",
        ),
    ],
)
def test_range_validation_errors(
    kwargs: dict[str, ty.Any], value: ty.Any, match: str
) -> None:
    """Invalid inputs raise ValidationError."""

    class Model(pydantic.BaseModel):
        r: ty.Annotated[range, RangeAdapter(**kwargs)]

    with pytest.raises(pydantic.ValidationError, match=match):
        Model(r=value)


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "match"),
    [
        pytest.param(
            {"default_type": float},
            ValueError,
            "RangeAdapter: default_type was <class 'float'>, but only int's or "
            "annotated int's are currently supported",
            id="default-float",
        ),
        pytest.param(
            {"start_type": ty.Annotated[float, 16], "stop_type": pydantic.PositiveInt},
            ValueError,
            r"RangeAdapter: start_type was typing.Annotated\[float, 16\], but "
            "only int's or annotated int's are currently supported",
            id="start-annotated-float",
        ),
        pytest.param(
            {"stop_type": ty.Annotated[float, 16], "start_type": pydantic.PositiveInt},
            ValueError,
            r"RangeAdapter: stop_type was typing.Annotated\[float, 16\], but "
            "only int's or annotated int's are currently supported",
            id="stop-annotated-float",
        ),
        pytest.param(
            {"step_type": ty.Annotated[float, 16]},
            ValueError,
            r"RangeAdapter: step_type was typing.Annotated\[float, 16\], but "
            "only int's or annotated int's are currently supported",
            id="step-annotated-float",
        ),
    ],
)
def test_invalid_kwargs(
    kwargs: dict[str, ty.Any], exc_type: type[BaseException], match: str
) -> None:
    """Test an invalid configuration of RangeAdapter"""
    with pytest.raises(exc_type, match=match):
        RangeAdapter(**kwargs)


@pytest.mark.parametrize(
    ("value", "truth"),
    [
        pytest.param(range(5), ":5", id=":5"),
        pytest.param(range(1, 5), "1:5", id="1:5"),
        pytest.param(range(1, 4, 2), "1:4:2", id="1:4:2"),
    ],
)
def test_range_serialization(value: range, truth: str) -> None:
    """Range serializes to JSON-compatible mapping."""

    class Model(pydantic.BaseModel):
        r: ty.Annotated[range, RangeAdapter()]

    model = Model(r=value)
    assert model.model_dump() == {"r": value}
    assert model.model_dump(mode="json") == {"r": truth}


def test_json_schema() -> None:
    """JSON schema is stable and well-defined."""

    class Model(pydantic.BaseModel):
        r: ty.Annotated[range, RangeAdapter()]

    schema = Model.model_json_schema()
    r = schema["properties"]["r"]
    assert r["type"] == "string"
    assert r["description"] == "Python range syntax: start:stop[:step]"
    assert "pattern" in r
