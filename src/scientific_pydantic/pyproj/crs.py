"""Pydantic adapters for pyproj CRS data types."""

from __future__ import annotations

import typing as ty

from pydantic_core import PydanticCustomError, core_schema

from scientific_pydantic.schema import Encoding, make_core_schema

if ty.TYPE_CHECKING:
    import pydantic
    from pyproj import CRS


class CRSAdapter:
    """Pydantic adapter for pyproj.CRS

    Validation Options
    ------------------
    1. An existing `CRS` object. Identity.
    2. Valid constructor options for CRS. See
        https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.__init__
       for the supported options for your version of `pyproj`.

    JSON Serialization
    ------------------
    An EPSG authority string if pyproj is 100% confident. Otherwise, the WKT
    representation of the CRS is used via `.to_wkt()`.

    Parameters
    ----------
    encoding
        A custom encoding for this type.


    Examples
    --------
    >>> import pydantic
    >>> from pyproj import CRS
    >>> from scientific_pydantic.pyproj import (
    ...     CRSAdapter,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> class Model(pydantic.BaseModel):
    ...     crs: ty.Annotated[CRS, CRSAdapter()]  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> Model(crs=4326)
    Model(crs=<Geographic 2D CRS: EPSG:4326>
    Name: WGS 84
    Axis Info [ellipsoidal]:
    - Lat[north]: Geodetic latitude (degree)
    - Lon[east]: Geodetic longitude (degree)
    Area of Use:
    - name: World.
    - bounds: (-180.0, -90.0, 180.0, 90.0)
    Datum: World Geodetic System 1984 ensemble
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich
    )
    >>> Model(crs="EPSG:4978")
    Model(crs=<Geocentric CRS: EPSG:4978>
    Name: WGS 84
    Axis Info [cartesian]:
    - X[geocentricX]: Geocentric X (metre)
    - Y[geocentricY]: Geocentric Y (metre)
    - Z[geocentricZ]: Geocentric Z (metre)
    Area of Use:
    - name: World.
    - bounds: (-180.0, -90.0, 180.0, 90.0)
    Datum: World Geodetic System 1984 ensemble
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich
    )
    """

    def __init__(
        self,
        *,
        encoding: Encoding | None = None,
    ) -> None:
        self._encoding = encoding

    def __get_pydantic_core_schema__(
        self,
        _source_type: type[ty.Any],
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        from pyproj import CRS

        return make_core_schema(
            CRS,
            encoding=self._encoding
            or Encoding(
                serializer=_serialize,
                before_validator=_validate,
                json_schema=core_schema.any_schema(
                    metadata={
                        "details": "https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.__init__",
                    }
                ),
            ),
        )


def _validate(val: ty.Any) -> CRS:
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    if isinstance(val, CRS):
        return val

    try:
        return CRS(val)
    except CRSError as exc:
        err_t = "crs_error"
        msg = "{e}"
        raise PydanticCustomError(err_t, msg, {"e": str(exc)}) from exc


def _serialize(val: CRS) -> str:
    epsg = val.to_epsg(100)
    if epsg is not None:
        return f"EPSG:{epsg}"
    return val.to_wkt()
