"""Unit test for version_check.py"""

import types
import typing as ty
from collections.abc import Callable, Sequence
from unittest import mock

import pytest

from scientific_pydantic.version_check import (
    version_eq,
    version_ge,
    version_gt,
    version_le,
    version_lt,
    version_ne,
)


def make_module(version: str) -> types.ModuleType:
    """Make a test module"""
    m = mock.MagicMock()
    m.__version__ = version
    return ty.cast("types.ModuleType", m)


@pytest.mark.parametrize(
    ("version", "cmp", "pred", "truth"),
    [
        # --- version_ge ---
        pytest.param("2.0.0", "1.9.9", version_ge, True, id="ge-major-greater"),
        pytest.param("1.1.0", "1.0.9", version_ge, True, id="ge-minor-greater"),
        pytest.param("1.0.1", "1.0.0", version_ge, True, id="ge-patch-greater"),
        pytest.param("1.0.0", "1.0.0", version_ge, True, id="ge-equal"),
        pytest.param("1.0.0", "1.0.1", version_ge, False, id="ge-patch-less"),
        pytest.param("1.0.0", "1.0", version_ge, True, id="ge-eq-major-minor"),
        pytest.param("1", "1.2.3", version_ge, False, id="ge-lt-major-minor"),
        pytest.param("1.0.0", "1", version_ge, True, id="ge-eq-major"),
        pytest.param("1.0.0", (1, 0, 0), version_ge, True, id="ge-equal-seq"),
        pytest.param("1.0.0", (1, 0, 1), version_ge, False, id="ge-less-seq"),
        # --- version_gt ---
        pytest.param("2.0.0", "1.9.9", version_gt, True, id="gt-major-greater"),
        pytest.param(
            "2.0.0", "1.9", version_gt, True, id="gt-major-greater-major-minor"
        ),
        pytest.param("2.0.0", "1", version_gt, True, id="gt-major-greater-major"),
        pytest.param("1", "2.0.0", version_gt, False, id="gt-major-less-major"),
        pytest.param("1.0.0", "1.0.0", version_gt, False, id="gt-equal"),
        pytest.param("1.0.0", "1.0.1", version_gt, False, id="gt-patch-less"),
        pytest.param("1.0.0", (0, 9, 9), version_gt, True, id="gt-greater-seq"),
        pytest.param("1.0.0", (1, 0, 0), version_gt, False, id="gt-equal-seq"),
        pytest.param("1.0.0", (1, 0), version_gt, False, id="gt-equal-seq-major-minor"),
        pytest.param("1.0", (1,), version_gt, False, id="gt-equal-seq-major"),
        # --- version_le ---
        pytest.param("1.0.0", "1.0.1", version_le, True, id="le-patch-less"),
        pytest.param("1.0.0", "1.0.0", version_le, True, id="le-equal"),
        pytest.param("1.0.1", "1.0.0", version_le, False, id="le-patch-greater"),
        pytest.param("1.0.0", (1, 0, 0), version_le, True, id="le-equal-seq"),
        pytest.param("2.0.0", (1, 9, 9), version_le, False, id="le-major-greater-seq"),
        pytest.param(
            "2.0.0", (1, 9), version_le, False, id="le-major-greater-seq-major-minor"
        ),
        pytest.param("1.9", (2,), version_le, True, id="le-major-lt-major-minor"),
        # --- version_lt ---
        pytest.param("1.0.0", "1.0.1", version_lt, True, id="lt-patch-less"),
        pytest.param("1.0.0", "1.0.0", version_lt, False, id="lt-equal"),
        pytest.param("2.0.0", "1.9.9", version_lt, False, id="lt-major-greater"),
        pytest.param("1.0.0", (1, 0, 1), version_lt, True, id="lt-less-seq"),
        pytest.param("1.0.0", (1, 0, 0), version_lt, False, id="lt-equal-seq"),
        # --- version_eq ---
        pytest.param("1.2.3", "1.2.3", version_eq, True, id="eq-identical"),
        pytest.param("1.2.3", "1.2.4", version_eq, False, id="eq-patch-differs"),
        pytest.param("1.2.3", "2.2.3", version_eq, False, id="eq-major-differs"),
        pytest.param("1.2.3", (1, 2, 3), version_eq, True, id="eq-identical-seq"),
        pytest.param("1.2.3", (1, 2, 4), version_eq, False, id="eq-patch-differs-seq"),
        pytest.param("1.2.0", (1, 2), version_eq, True, id="eq-patch-ident-no-patch"),
        pytest.param("1.0", (1,), version_eq, True, id="eq-patch-ident-no-minor"),
        pytest.param("1", (1,), version_eq, True, id="eq-patch-ident-seq-major"),
        pytest.param("1", (1, 1), version_eq, False, id="eq-patch-no-seq-major"),
        pytest.param("1.1", (1,), version_eq, False, id="eq-minor-no-seq"),
        # --- version_ne ---
        pytest.param("1.2.3", "1.2.4", version_ne, True, id="ne-patch-differs"),
        pytest.param("1.2.3", "1.2.3", version_ne, False, id="ne-identical"),
        pytest.param("2.0.0", (1, 0, 0), version_ne, True, id="ne-major-differs-seq"),
        pytest.param("1.0.0", (1, 0, 0), version_ne, False, id="ne-identical-seq"),
    ],
)
def test_version_compare(
    *,
    version: str,
    cmp: Sequence[int] | str,
    pred: Callable[[types.ModuleType, Sequence[int] | str], bool],
    truth: bool,
) -> None:
    """Test a version comparison function"""
    assert pred(make_module(version), cmp) == truth


def test_normalize_missing_version() -> None:
    """Raise ValueError when the module lacks __version__."""
    m = mock.MagicMock(spec=[])  # no __version__ attribute
    with pytest.raises(ValueError, match="does not have a __version__"):
        version_ge(m, "1.0.0")
