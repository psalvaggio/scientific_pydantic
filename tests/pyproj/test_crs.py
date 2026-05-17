"""Unit tests for CRS adapters"""

import typing as ty

import pydantic
import pytest
from pyproj import CRS

from scientific_pydantic.pyproj import CRSAdapter


class Model(pydantic.BaseModel):
    """A simple test model"""

    f: ty.Annotated[CRS, CRSAdapter()]


@pytest.mark.parametrize(
    ("data", "crs_input"),
    [
        pytest.param(
            "+proj=longlat +datum=WGS84 +no_defs",
            "+proj=longlat +datum=WGS84 +no_defs",
            id="proj-string",
        ),
        pytest.param(
            {"proj": "longlat", "datum": "WGS84", "no_defs": None},
            "+proj=longlat +datum=WGS84 +no_defs",
            id="proj-dict",
        ),
        pytest.param(
            """
    GEODCRS["WGS 84",
      ENSEMBLE["World Geodetic System 1984 ensemble",
          MEMBER["World Geodetic System 1984 (Transit)"],
          MEMBER["World Geodetic System 1984 (G730)"],
          MEMBER["World Geodetic System 1984 (G873)"],
          MEMBER["World Geodetic System 1984 (G1150)"],
          MEMBER["World Geodetic System 1984 (G1674)"],
          MEMBER["World Geodetic System 1984 (G1762)"],
          MEMBER["World Geodetic System 1984 (G2139)"],
          MEMBER["World Geodetic System 1984 (G2296)"],
          ELLIPSOID["WGS 84",6378137,298.257223563,
              LENGTHUNIT["metre",1]],
          ENSEMBLEACCURACY[2.0]],
      PRIMEM["Greenwich",0,
        ANGLEUNIT["degree",0.0174532925199433]],
      CS[Cartesian,3],
        AXIS["(X)",geocentricX,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(Y)",geocentricY,
            ORDER[2],
            LENGTHUNIT["metre",1]],
        AXIS["(Z)",geocentricZ,
            ORDER[3],
            LENGTHUNIT["metre",1]],
      USAGE[
        SCOPE["Geodesy. Navigation and positioning using GPS satellite system."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
      ID["EPSG",4978]]
            """,
            4978,
            id="wkt-str",
        ),
        pytest.param("EPSG:4979", 4979, id="authority-str"),
        pytest.param(3857, "EPSG:3857", id="epsg-int"),
        pytest.param(("epsg", 32601), 32601, id="epsg-tuple"),
        pytest.param(CRS(4326), 4326, id="identity"),
    ],
)
def test_valid_input(data: ty.Any, crs_input: ty.Any) -> None:
    """Tests valid input for a CRS field"""
    m = Model(f=data)
    truth = CRS(crs_input)
    assert m.f.to_wkt() == truth.to_wkt()


@pytest.mark.parametrize(
    ("data", "match"),
    [
        pytest.param(1, "Invalid projection: EPSG:1", id="invalid-int"),
        pytest.param("abc", "Invalid projection: abc", id="invalid-str"),
        pytest.param({"a": "b"}, 'Invalid projection: {"a": "b"}', id="invalid-dict"),
        pytest.param(("a", "b"), "Invalid projection: a:b", id="invalid-tuple"),
    ],
)
def test_invalid_input(data: ty.Any, match: str) -> None:
    """Tests invalid input for a CRS field"""
    with pytest.raises(pydantic.ValidationError, match=match):
        Model(f=data)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            4919,
            (
                'GEODCRS["ITRF2000",DYNAMIC[FRAMEEPOCH[1997]],'
                'DATUM["International Terrestrial Reference Frame 2000",'
                'ELLIPSOID["GRS 1980",6378137,298.257222101,'
                'LENGTHUNIT["metre",1]]],'
                'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
                'CS[Cartesian,3],AXIS["(X)",geocentricX,ORDER[1],'
                'LENGTHUNIT["metre",1]],AXIS["(Y)",geocentricY,ORDER[2],'
                'LENGTHUNIT["metre",1]],AXIS["(Z)",geocentricZ,ORDER[3],'
                'LENGTHUNIT["metre",1]],USAGE[SCOPE["Geodesy."],AREA["World."],'
                "BBOX[-90,-180,90,180]],"
                'ID["EPSG",4919]]'
            ),
            id="4919",
        ),
    ],
)
def test_serialization(data: ty.Any, expected: str) -> None:
    """Test serialization (and round-tripping)"""
    m = Model(f=data)
    py_dict = m.model_dump()
    assert m.f.is_exact_same(py_dict["f"])
    assert Model.model_validate(py_dict).f.to_wkt() == m.f.to_wkt()

    json_dict = m.model_dump(mode="json")
    assert json_dict["f"] == expected
    assert Model.model_validate(json_dict).f.to_wkt() == m.f.to_wkt()
