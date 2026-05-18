"""Pydantic adapters for numpy scalar types."""

from __future__ import annotations

import typing as ty

from pydantic import PydanticSchemaGenerationError
from pydantic_core import PydanticCustomError, core_schema

from scientific_pydantic.schema import Encoding, make_core_schema

if ty.TYPE_CHECKING:
    import numpy as np
    from pydantic import GetCoreSchemaHandler
    from pydantic_core.core_schema import CoreSchema


class ScalarAdapter:
    """Pydantic adapter for NumPy scalar types.

    Supports all numeric NumPy scalar kinds: signed/unsigned integers,
    floating-point, complex, and boolean.

    Validation Options
    ------------------
    1. Matching numpy scalar type - identity pass-through.
    2. Any other numpy scalar - cast via the target type's constructor.
    3. Python ``int``, ``float``, ``bool``, ``str`` - converted via the
       target type's constructor.

    JSON Serialization
    ------------------
    Scalars are serialized to JSON-safe Python primitives:

    - ``np.bool_``: ``bool``
    - ``np.integer`` subtypes: ``int``
    - ``np.floating`` subtypes: ``float``
    - ``np.complexfloating`` subtypes: ``str`` (e.g. ``"1+2j"``)

    Parameters
    ----------
    gt
        If given, validated values must be strictly greater than this.
    ge
        If given, validated values must be greater than or equal to this.
    lt
        If given, validated values must be strictly less than this.
    le
        If given, validated values must be less than or equal to this.
    encoding
        A custom [`Encoding`][scientific_pydantic.Encoding] to override the
        default serialization/deserialization behaviour entirely.

    Examples
    --------
    >>> from typing import Annotated
    >>> import numpy as np
    >>> import pydantic
    >>> from scientific_pydantic.numpy import (
    ...     ScalarAdapter,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> class Model(pydantic.BaseModel):
    ...     prob: Annotated[np.float64, ScalarAdapter(ge=0.0, le=1.0)]
    ...     n: Annotated[np.uint16, ScalarAdapter()]  # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    >>> Model(prob=0.5, n=42)
    Model(prob=np.float64(0.5), n=np.uint16(42))
    >>> Model(prob="0.25", n="7")
    Model(prob=np.float64(0.25), n=np.uint16(7))
    """

    def __init__(
        self,
        *,
        gt: float | np.generic | None = None,
        ge: float | np.generic | None = None,
        lt: float | np.generic | None = None,
        le: float | np.generic | None = None,
        encoding: Encoding | None = None,
    ) -> None:
        self._gt = gt
        self._ge = ge
        self._lt = lt
        self._le = le
        self._encoding = encoding

    def __get_pydantic_core_schema__(
        self,
        source_type: ty.Any,
        _handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Build the pydantic-core schema for this numpy scalar type."""
        import numpy as np

        if not isinstance(source_type, type) or not issubclass(
            source_type, (np.bool_, np.integer, np.inexact)
        ):
            msg = f"Source type {source_type} was not a supported NumPy scalar type"
            raise PydanticSchemaGenerationError(msg)

        kind: str = np.dtype(source_type).kind

        before_validator = _build_before_validator(source_type)
        bounds_validator = _build_bounds_validator(
            source_type, gt=self._gt, ge=self._ge, lt=self._lt, le=self._le
        )
        after_validators: list[ty.Callable[[ty.Any], ty.Any]] = (
            [bounds_validator] if bounds_validator is not None else []
        )

        # Insert the custom description
        desc = self._get_json_description(source_type)
        json_schema = _np_json_schema(kind)
        json_schema["metadata"] = {
            "pydantic_js_functions": [lambda c, h: h(c) | {"description": desc}]
        }

        encoding = (
            self._encoding
            if self._encoding is not None
            else Encoding(
                serializer=_serializer,
                before_validator=before_validator,
                json_schema=json_schema,
            )
        )

        return make_core_schema(
            source_type,
            encoding=encoding,
            after_validators=after_validators,
        )

    def _get_json_description(self, source_type: type[np.generic]) -> str:
        """Get the description for the JSON schema"""
        import numpy as np

        dtype = np.dtype(source_type)
        parts = [f"NumPy scalar: {dtype}"]
        if self._gt is not None:
            parts.append(f"value > {self._gt}")
        if self._ge is not None:
            parts.append(f"value >= {self._ge}")
        if self._lt is not None:
            parts.append(f"value < {self._lt}")
        if self._le is not None:
            parts.append(f"value <= {self._le}")
        return ", ".join(parts)


# Numpy scalar kind characters that map to a JSON number schema.
_NUMERIC_KINDS: frozenset[str] = frozenset("uifc")  # uint, int, float, complex


def _np_json_schema(kind: str) -> CoreSchema:
    """Return the pydantic-core JSON input schema for a numpy scalar kind.

    Parameters
    ----------
    kind
        The single-character numpy dtype kind string (e.g. ``'f'``, ``'i'``).

    Returns
    -------
    CoreSchema
        A ``pydantic_core`` schema for the expected JSON payload.
    """
    if kind == "b":
        # np.bool_ - accept JSON bool or 0/1
        return core_schema.union_schema(
            [core_schema.bool_schema(), core_schema.int_schema(ge=0, le=1)]
        )
    if kind in _NUMERIC_KINDS:
        return core_schema.union_schema(
            [
                core_schema.float_schema(),
                core_schema.int_schema(),
                core_schema.str_schema(),
            ]
        )
    # Fallback - accept any JSON value and let numpy decide
    return core_schema.any_schema()


def _build_before_validator(
    scalar_type: type[np.generic],
) -> ty.Callable[[ty.Any], ty.Any]:
    """Return a before-validator that coerces *value* to *scalar_type*.

    Parameters
    ----------
    scalar_type
        A concrete numpy scalar type, e.g. ``np.float32``.

    Returns
    -------
    Callable[[Any], Any]
        A function that returns a validated numpy scalar.
    """

    def _validate(value: ty.Any) -> ty.Any:
        import numpy as np

        if isinstance(value, scalar_type):
            return value
        if isinstance(value, np.generic):
            # Allow casting from other numpy scalar kinds
            try:
                return scalar_type(value)
            except (ValueError, TypeError, OverflowError) as e:
                err_t = "numpy_scalar_cast"
                msg = "Cannot cast {src} to {dst}: {e}"
                raise PydanticCustomError(
                    err_t,
                    msg,
                    {"src": type(value).__name__, "dst": scalar_type.__name__, "e": e},
                ) from e
        # Accept Python scalars and strings
        try:
            return scalar_type(value)
        except (TypeError, ValueError, OverflowError) as e:
            err_t = "numpy_scalar_invalid"
            msg = "Cannot convert {v!r} to {dst}: {e}"
            raise PydanticCustomError(
                err_t, msg, {"v": value, "dst": scalar_type.__name__, "e": e}
            ) from e

    return _validate


def _build_bounds_validator(
    scalar_type: type[np.generic],
    *,
    gt: float | np.generic | None,
    ge: float | np.generic | None,
    lt: float | np.generic | None,
    le: float | np.generic | None,
) -> ty.Callable[[ty.Any], ty.Any] | None:
    """Return an after-validator enforcing numeric bounds, or *None* if unconstrained.

    Parameters
    ----------
    scalar_type : type
        The target numpy scalar type (used in error messages).
    gt : float or None
        Exclusive lower bound.
    ge : float or None
        Inclusive lower bound.
    lt : float or None
        Exclusive upper bound.
    le : float or None
        Inclusive upper bound.

    Returns
    -------
    Callable or None
        Validator function, or ``None`` when no bounds are specified.
    """
    if gt is None and ge is None and lt is None and le is None:
        return None

    def _check(value: ty.Any) -> ty.Any:
        if gt is not None and not value > gt:
            err_t = "numpy_scalar_gt"
            msg = "{name} value {v} must be > {bound}"
            raise PydanticCustomError(
                err_t, msg, {"name": scalar_type.__name__, "v": value, "bound": gt}
            )
        if ge is not None and not value >= ge:
            err_t = "numpy_scalar_ge"
            msg = "{name} value {v} must be >= {bound}"
            raise PydanticCustomError(
                err_t, msg, {"name": scalar_type.__name__, "v": value, "bound": ge}
            )
        if lt is not None and not value < lt:
            err_t = "numpy_scalar_lt"
            msg = "{name} value {v} must be < {bound}"
            raise PydanticCustomError(
                err_t, msg, {"name": scalar_type.__name__, "v": value, "bound": lt}
            )
        if le is not None and not value <= le:
            err_t = "numpy_scalar_le"
            msg = "{name} value {v} must be <= {bound}"
            raise PydanticCustomError(
                err_t, msg, {"name": scalar_type.__name__, "v": value, "bound": le}
            )
        return value

    return _check


def _serializer(value: ty.Any) -> float | int | bool | str:
    """Serialize a numpy scalar to a JSON-safe Python primitive.

    Parameters
    ----------
    value : numpy scalar
        The validated numpy scalar to serialize.

    Returns
    -------
    float or int or bool or str
        The JSON-safe representation.
    """
    import numpy as np

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.complexfloating):
        # JSON has no native complex; serialize as "a+bj" string
        return str(complex(value))
    if isinstance(value, np.integer):
        return int(value)
    return float(value)
