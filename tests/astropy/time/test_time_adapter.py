"""Unit tests for time_adapter.py"""

import datetime
import typing as ty

import astropy.units as u
import numpy as np
import numpy.testing as npt
import pydantic
import pydantic_core
import pytest
from astropy.time import Time

from scientific_pydantic.astropy.time import TimeAdapter


class Unconstrained(pydantic.BaseModel):
    """An unconstrained model for testing"""

    field: ty.Annotated[Time, TimeAdapter()]


def assert_times_close(test: Time, truth: Time) -> None:
    """Assert that two Time objects are close to each other"""
    diff = np.abs((test - truth).to_value(u.s))
    atol = 10 ** (-truth.precision)
    npt.assert_allclose(diff, 0, atol=atol)


@pytest.mark.parametrize(
    ("value", "truth"),
    [
        pytest.param(
            Time("2024-06-01T00:00:00", format="isot", scale="tai"),
            Time("2024-06-01T00:00:00", format="isot", scale="tai"),
            id="isot-tai-ident",
        ),
        pytest.param(
            Time(["2024-01-01", "2024-06-01", "2024-12-31"], format="iso", scale="utc"),
            Time(["2024-01-01", "2024-06-01", "2024-12-31"], format="iso", scale="utc"),
            id="iso-utc-ident-vector",
        ),
        pytest.param(
            Time("2024-06-01T00:00:00.123456789", format="isot", scale="tai"),
            Time(
                "2024-06-01T00:00:00.123456789", format="isot", scale="tai", precision=9
            ),
            id="isot-tai-high-precision-ident",
        ),
        pytest.param(
            "2024-06-01T00:00:00.123456789",
            Time(
                "2024-06-01T00:00:00.123456789", format="isot", scale="utc", precision=9
            ),
            id="isot-tai-high-precision-str",
        ),
        pytest.param(
            [["2024-06-01T00:00:00.123456789"], ["2025-06-01T00:00:00.123456789"]],
            Time(
                [["2024-06-01T00:00:00.123456789"], ["2025-06-01T00:00:00.123456789"]],
                format="isot",
                scale="utc",
                precision=9,
            ),
            id="isot-tai-high-precision-list[str]",
        ),
    ],
)
def test_valid(value: ty.Any, truth: Time) -> None:
    """Test for a valid field validation"""
    m = Unconstrained(field=value)
    assert_times_close(m.field, truth)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        pytest.param(None, "invalid_time", id="None"),
        pytest.param("20250301T12:13:00", "invalid_time", id="almost-isot-str"),
        pytest.param(
            {"value": "20250301T12:13:00"}, "invalid_time", id="almost-isot-dict"
        ),
        pytest.param({"vlue": "2025-03-01T12:13:00"}, "missing_value", id="value-type"),
    ],
)
def test_invalid(value: ty.Any, match: str) -> None:
    """Test for a valid field validation"""
    with pytest.raises(pydantic.ValidationError, match=match):
        Unconstrained(field=value)


def test_mask_error() -> None:
    """Test that masked times are not supported"""
    m = Unconstrained(field=["2025-03-01T12:13:00", "2025-03-01T12:14:00"])
    m.field[0] = np.ma.masked
    with pytest.raises(
        pydantic_core.PydanticSerializationError,
        match="Serialization of masked times is not supported yet",
    ):
        m.model_dump_json()


@pytest.mark.parametrize(
    "value",
    [
        # --- ISO 8601 strings ---
        Time("2024-01-15T12:30:00", format="isot", scale="utc"),
        Time("2000-01-01T00:00:00", format="isot", scale="utc"),  # J2000 epoch
        Time("1999-12-31T23:59:59", format="isot", scale="utc"),  # pre-J2000
        Time("2024-02-29T00:00:00", format="isot", scale="utc"),  # leap day
        Time("2016-12-31T23:59:60", format="isot", scale="utc"),  # leap second
        # --- Different scales ---
        Time("2024-06-01T00:00:00", format="isot", scale="tai"),
        Time("2024-06-01T00:00:00", format="isot", scale="tt"),
        Time("2024-06-01T00:00:00", format="isot", scale="tcg"),
        Time("2024-06-01T00:00:00", format="isot", scale="tcb"),
        Time("2024-06-01T00:00:00", format="isot", scale="tdb"),
        Time("2024-06-01T00:00:00", format="isot", scale="local"),
        # --- Julian Date formats ---
        Time(2451545.0, format="jd", scale="utc"),  # J2000.0
        Time(2451545.5, format="jd", scale="utc"),
        Time(0.0, format="jd", scale="tai"),  # JD zero point
        Time(2460000.5, format="jd", scale="utc"),  # recent epoch
        # --- Modified Julian Date ---
        Time(51544.5, format="mjd", scale="utc"),  # J2000 in MJD
        Time(0.0, format="mjd", scale="tai"),
        Time(60310.0, format="mjd", scale="utc"),  # recent date
        # --- Unix timestamps ---
        Time(0.0, format="unix", scale="utc"),  # Unix epoch
        Time(1_000_000_000.0, format="unix", scale="utc"),
        Time(1_700_000_000.0, format="unix", scale="utc"),
        # --- GPS time ---
        Time(0.0, format="gps"),  # GPS epoch (1980-01-06)
        Time(1_000_000_000.0, format="gps"),
        # --- FITS format ---
        Time("2024-01-15T12:30:00.000", format="fits", scale="utc"),
        # --- Byear / Jyear (Besselian / Julian year) ---
        Time(2000.0, format="jyear", scale="utc"),
        Time(1970.0, format="byear", scale="utc"),  # B1950
        Time(2024.5, format="jyear", scale="utc"),
        # --- Datetime object ---
        Time(
            datetime.datetime.fromisoformat("2024-03-20T09:06:00"),
            format="datetime",
            scale="utc",
        ),  # spring equinox ~2024
        # --- Far past / far future ---
        Time("1000-06-15T00:00:00", format="isot", scale="tai"),
        Time("3000-01-01T00:00:00", format="isot", scale="tai"),
        # --- Array of times (scalar equivalent) ---
        Time(["2024-01-01", "2024-06-01", "2024-12-31"], format="iso", scale="utc"),
    ],
    ids=lambda x: x.isot,
)
def test_roundtrip(value: Time) -> None:
    """Test round-tripping through Python/JSON"""
    m = Unconstrained(field=value)

    py = Unconstrained.model_validate(m.model_dump(mode="python"))
    assert_times_close(py.field, value)

    json = Unconstrained.model_validate_json(m.model_dump_json())
    assert_times_close(json.field, value)


def test_serialization_py() -> None:
    """Test Python serialization behavior"""
    m = Unconstrained(
        field=Time(
            "2026-03-01T12:32:01.123456789",
            precision=9,
            scale="utc",
            location=(-77.0, 43, 30.0),
        )
    )

    py = m.model_dump()
    assert isinstance(py["field"], Time)
    assert_times_close(py["field"], m.field)


@pytest.mark.parametrize(
    ("value", "truth"),
    [
        pytest.param(
            Time(
                "2026-03-01T12:32:01.123456789",
                precision=9,
                scale="utc",
                location=(-77.0, 43, 30.0),
            ),
            {
                "format": "jd",
                "precision": 9,
                "value": 2461101.0,
                "value2": 0.022235225194317088,
                "scale": "utc",
                "location": [
                    pytest.approx(-77.0, abs=1e-10),
                    pytest.approx(43.0, abs=1e-10),
                    pytest.approx(30.0, abs=1e-6),
                ],
            },
            id="high-prec-isot",
        ),
        pytest.param(
            Time(
                "2026-03-01T12:32:01.123456789",
                precision=6,
                scale="utc",
                location=(-77.0, 43, 30.0),
            ),
            {
                "format": "isot",
                "value": "2026-03-01T12:32:01.123457",
                "scale": "utc",
                "precision": 6,
                "location": [
                    pytest.approx(-77.0, abs=1e-10),
                    pytest.approx(43.0, abs=1e-10),
                    pytest.approx(30.0, abs=1e-6),
                ],
            },
            id="us-precision-isot",
        ),
        pytest.param(
            Time(
                "2026-03-01 12:32:01.123456789",
                in_subfmt="date_hms",
                out_subfmt="date_hms",
                scale="utc",
            ),
            {
                "format": "iso",
                "value": "2026-03-01 12:32:01.123",
                "in_subfmt": "date_hms",
                "out_subfmt": "date_hms",
                "scale": "utc",
            },
            id="custom-subfmt",
        ),
    ],
)
def test_serialization_json(value: Time, truth: dict[str, ty.Any]) -> None:
    """Test serialization behavior"""
    m = Unconstrained(field=value)
    json = m.model_dump(mode="json")
    assert isinstance(json["field"], dict)
    assert json["field"] == truth


@pytest.mark.parametrize(
    ("kwargs", "value", "result"),
    [
        pytest.param(
            {"scalar": True},
            "2026-03-01T13:27:00",
            Time("2026-03-01T13:27:00"),
            id="scalar-True-pass",
        ),
        pytest.param(
            {"scalar": True},
            ["2026-03-01T13:27:00"],
            "scalar_error",
            id="scalar-True-fail",
        ),
        pytest.param(
            {"scalar": False},
            ["2026-03-01T13:27:00"],
            Time(["2026-03-01T13:27:00"]),
            id="scalar-False-pass",
        ),
        pytest.param(
            {"scalar": False},
            "2026-03-01T13:27:00",
            "scalar_error",
            id="scalar-False-fail",
        ),
        pytest.param(
            {"ndim": 2},
            [["2026-03-01T13:27:00"]],
            Time([["2026-03-01T13:27:00"]]),
            id="ndim-2-pass",
        ),
        pytest.param(
            {"ndim": 2},
            ["2026-03-01T13:27:00"],
            "ndim_error",
            id="ndim-2-fail",
        ),
        pytest.param(
            {"shape": [1, 2]},
            [["2026-03-01T13:27:00", "2026-03-01T13:27:00"]],
            Time([["2026-03-01T13:27:00", "2026-03-01T13:27:00"]]),
            id="shape-[1,2]-pass",
        ),
        pytest.param(
            {"shape": [1, 2]},
            [["2026-03-01T13:27:00"]],
            "shape_error",
            id="shape-[1,2]-fail",
        ),
        pytest.param(
            {
                "gt": "2026-03-01T13:26:00",
                "ge": "2026-03-01T13:27:00",
                "lt": "2026-03-02T13:27:01",
                "le": "2026-03-02T13:27:00",
            },
            [["2026-03-01T13:27:00", "2026-03-02T13:27:00"]],
            Time([["2026-03-01T13:27:00", "2026-03-02T13:27:00"]]),
            id="all-bounds-pass",
        ),
        pytest.param(
            {"gt": "2026-03-01T13:27:00"},
            [["2026-03-01T13:27:00", "2026-03-02T13:27:00"]],
            "bounds_error",
            id="gt-fail",
        ),
        pytest.param(
            {"ge": "2026-03-01T13:27:01"},
            [["2026-03-01T13:27:00", "2026-03-02T13:27:00"]],
            "bounds_error",
            id="ge-fail",
        ),
        pytest.param(
            {"lt": "2026-03-02T13:27:00"},
            [["2026-03-01T13:27:00", "2026-03-02T13:27:00"]],
            "bounds_error",
            id="lt-fail",
        ),
        pytest.param(
            {"le": "2026-03-02T12:27:00"},
            [["2026-03-01T13:27:00", "2026-03-02T13:27:00"]],
            "bounds_error",
            id="le-fail",
        ),
    ],
)
def test_contraints(
    kwargs: dict[str, ty.Any], value: ty.Any, result: Time | str
) -> None:
    """Test theshape constraints on Time's"""

    class Model(pydantic.BaseModel):
        field: ty.Annotated[Time, TimeAdapter(**kwargs)]

    if isinstance(result, Time):
        m = Model(field=value)
        assert_times_close(m.field, result)
    else:
        with pytest.raises(pydantic.ValidationError, match=result):
            Model(field=value)
