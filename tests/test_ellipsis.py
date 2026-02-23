"""Unit test for ellipsis.py"""

import types
import typing as ty

import pydantic
import pytest

from scientific_pydantic import EllipsisAdapter, EllipsisLiteral


class Model(pydantic.BaseModel):
    """Test model"""

    e: EllipsisLiteral


def test_ellipsis_literal_valid() -> None:
    """Test usage of the EllipsisLiteral"""
    assert Model(e=Ellipsis).e is Ellipsis
    assert Model(e=...).e is Ellipsis
    assert Model(e="...").e is Ellipsis

    class Foo:  # noqa: PLW1641
        def __eq__(self, x: object) -> bool:
            return x == "..."

    assert Model(e=Foo()).e is Ellipsis


def test_ellipsis_literal_invalid() -> None:
    """Test invalid usage of the EllipsisLiteral"""
    with pytest.raises(
        pydantic.ValidationError,
        match=r"Expected Ellipsis \(\.\.\.\), got 5",
    ):
        Model(e=5)


def test_can_only_use_with_valid_types() -> None:
    """Test the valid type check"""

    class Valid(pydantic.BaseModel):
        f0: ty.Annotated[ty.Literal[Ellipsis], EllipsisAdapter()]  # type: ignore[not-a-type]
        f1: ty.Annotated[ty.Literal[...], EllipsisAdapter()]  # type: ignore[invalid-literal]
        f2: ty.Annotated[types.EllipsisType, EllipsisAdapter()]

    with pytest.raises(pydantic.PydanticSchemaGenerationError):

        class M1(pydantic.BaseModel):
            f: ty.Annotated[int, EllipsisAdapter()]
