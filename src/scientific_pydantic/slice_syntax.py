"""Utilities for slice string syntax"""

import typing as ty


class SliceSyntaxError(ValueError):
    """Raised when a slice syntax string is invalid."""


T = ty.TypeVar("T")


@ty.overload
def parse_slice_syntax(
    value: str,
    *,
    converter: ty.Callable[[str], T],
    require_start: ty.Literal[True],
    require_stop: ty.Literal[True],
    dest_type: type,
) -> tuple[T, T, T | None]: ...


@ty.overload
def parse_slice_syntax(
    value: str,
    *,
    converter: ty.Callable[[str], T],
    require_start: ty.Literal[False],
    require_stop: ty.Literal[True],
    dest_type: type,
) -> tuple[T | None, T, T | None]: ...


@ty.overload
def parse_slice_syntax(
    value: str,
    *,
    converter: ty.Callable[[str], T],
    require_start: ty.Literal[False],
    require_stop: ty.Literal[False],
    dest_type: type,
) -> tuple[T | None, T | None, T | None]: ...


@ty.overload
def parse_slice_syntax(
    value: str,
    *,
    converter: ty.Callable[[str], T],
    require_start: ty.Literal[True],
    require_stop: ty.Literal[False],
    dest_type: type,
) -> tuple[T, T | None, T | None]: ...


def parse_slice_syntax(
    value: str,
    *,
    converter: ty.Callable[[str], T],
    require_start: bool,
    require_stop: bool,
    dest_type: type,
) -> tuple[T | None, T | None, T | None]:
    """Parse a Python-style slice string: [start]:[stop][:step].

    Parameters
    ----------
    value
        Slice syntax string.
    converter
        Conversion function to use on each element if present
    require_start
        Whether start must be present and non-empty.
    require_stop
        Whether stop must be present and non-empty.
    dest_type
        Destination type for the resulting tuple (only used in errors)

    Returns
    -------
    (start, stop, step)
    """
    parts = value.split(":")
    n_parts = len(parts)
    if not 2 <= n_parts <= 3:  # noqa: PLR2004
        msg = (
            f"invalid {dest_type.__name__} syntax, expected 2-3 parts "
            f"separated by :'s, got {n_parts}"
        )
        raise SliceSyntaxError(msg)

    def _parse(part: str) -> T | None:
        part = part.strip()
        if part == "":
            return None
        return converter(part)

    try:
        start = _parse(parts[0])
        stop = _parse(parts[1])
        step = _parse(parts[2]) if len(parts) == 3 else None  # noqa: PLR2004
    except (ValueError, TypeError) as exc:
        msg = f"invalid integer in {dest_type.__name__} string"
        raise SliceSyntaxError(msg) from exc

    if require_start and start is None:
        msg = "start is required"
        raise SliceSyntaxError(msg)
    if require_stop and stop is None:
        msg = "stop is required"
        raise SliceSyntaxError(msg)
    if step == 0:
        msg = "step must not be zero"
        raise SliceSyntaxError(msg)

    return start, stop, step


def format_slice_syntax(start: ty.Any, stop: ty.Any, step: ty.Any) -> str:
    """Format slice components into canonical slice syntax."""
    if step is None:
        return f"{start or ''}:{stop or ''}"
    return f"{start or ''}:{stop or ''}:{step}"
