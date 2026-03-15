# Copilot Instructions — `scientific_pydantic`

`scientific_pydantic` is a pure-Python library (src layout) that integrates
scientific types (NumPy arrays, shapely geometries, etc) into Pydantic models.
All tooling is managed with `uv` linting/formatting with `ruff`, docs with
`mkdocs`, type-checking with `pyrefly`, CI via GitHub Actions.

## Language & runtime

- Python >=3.10
- Code must pass `ruff check` and `ruff format` with no diagnostics. Use
  `# noqa: <CODE>` only when genuinely unavoidable, always include the rule
  code (e.g. `# noqa: PLR2004`).
- All code must satisfy `pyrefly`. Use `# type: ignore[<code>]` sparingly
  and always with an explicit error code (e.g. `# type: ignore[missing-attribute]`).

## Imports

```python
# 1. stdlib
import typing as ty  # always alias typing as ty
from collections.abc import Callable, Sequence   # ABCs from collections.abc, not typing

# 2. third-party
import pydantic

# 3. local
from scientific_pydantic.foo import Bar
```

- Group imports: stdlib, third-party, local. Blank line between each group.
- Heavy imports used only in type annotations go behind `TYPE_CHECKING`
- The only imports allowed at global scope are stdlib, pydantic, and local, any
  other third party library must be a nested import.

## Type annotations

- Annotate every function parameter and return type.
- Prefer `X | Y` union syntax over `ty.Union[X, Y]`

## Pydantic models

- Subclass `pydantic.BaseModel` with `frozen=True, extra="forbid"` unless there
  is an explicit reason not to, document that reason in the class docstring.

```python
class Model(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validated wrapper around a NumPy ndarray."""
    ...
```

- Custom validation errors use `PydanticCustomError` with a `snake_case`
  error-type string variable, a separate message variable with `{placeholder}`
  substitution:

```python
from pydantic_core import PydanticCustomError

err_t = "invalid_dtype"
msg = "Expected dtype {expected}, got {actual}"
raise PydanticCustomError(err_t, msg, {"expected": expected, "actual": actual})
```

## Docstrings

- One-line summary for simple private functions
- NumPy-style (Parameters / Returns / Raises sections) for public API and
  complex private functions
- Class docstrings describe what the class *is*
- Use doctest-style examples.
- No docstrings on trivial private helpers.

## Testing (`tests/`)

### Parametrize

```python
@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        pytest.param(np.float32, (3,), id="float32-1d"),
        pytest.param(np.int64, (2, 4), id="int64-2d"),
        pytest.param(np.float64, (), id="float64-scalar"),
    ],
)
def test_array_dtype(dtype: type, shape: tuple[int, ...]) -> None:
    """Round-trips an array through the model and checks preserved dtype."""
```

- `@pytest.mark.parametrize` always receives a tuple of param names
- Every case uses `pytest.param(..., id="...")`.  IDs are short, descriptive,
  kebab-case or use operators (`">0"`, `"any_point-wkt"`, `"polygon-for-pt_or_multipt"`).
  The id conveys the *scenario*, not just a number.

### Assertions

- Use `numpy.testing` and `shapely.testing` assertions where applicable.

### Test functions

- One-line docstring on every test function.
- Type annotations on all parameters, including fixtures.
- Return type `-> None` on every test function.
- Use fixtures when they genuinely improve clarity; module-level constants are
  fine and often preferable for simple shared data.

## Tooling cheatsheet

| Task | Command |
|---|---|
| Install deps | `uv sync` |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type-check | `uv run pyrefly check` |
| Pre-commit (all hooks) | `uv run prek run --all-files` |
| Build docs (local) | `uv run mkdocs serve` |
