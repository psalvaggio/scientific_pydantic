# `scientific_pydantic`

[![CI](https://github.com/psalvaggio/scientific_pydantic/actions/workflows/ci.yml/badge.svg)](https://github.com/psalvaggio/scientific_pydantic/actions/workflows/ci.yml)
[![Pre-commit](https://github.com/psalvaggio/scientific_pydantic/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/psalvaggio/scientific_pydantic/actions/workflows/pre-commit.yml)
[![Docs](https://img.shields.io/badge/docs-Docs-blue?style=flat-square&logo=github&logoColor=white&link=https://psalvaggio.github.io/scientific_pydantic/)](https://psalvaggio.github.io/scientific_pydantic/)
[![Coverage Status](https://coveralls.io/repos/github/psalvaggio/scientific_pydantic/badge.svg?branch=main)](https://coveralls.io/github/psalvaggio/scientific_pydantic?branch=main)

`scientific_pydantic` is an extension module to
[`pydantic`](https://docs.pydantic.dev/latest/) that adds support for a number
of common data types in scientific computing.

## Motivation

Let's say with only `pydantic`, you wanted to put a `numpy.ndarray` object into
one of your models:
```python
import numpy as np
import pydantic

class MyModel(pydantic.BaseModel):
    arr: np.ndarray
```
this will produce the following error:
```
pydantic.errors.PydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'numpy.ndarray'>. Set `arbitrary_types_allowed=True` in the model_config to ignore this error or implement `__get_pydantic_core_schema__` on your type to fully support it.
```
Specifying `arbitrary_types_allowed=True` can work, but disables JSON
serialization and validation and does not support lax parsing and input
conversions, which are a powerful feature of pydantic. This library takes the
approach of implementing `__get_pydantic_core_schema__` in adapter objects and
using `Annotated`, as described
[here](https://docs.pydantic.dev/latest/concepts/types/#as-an-annotation).

With `scientific_pydantic`, this example looks like:
```python
import typing as ty

import numpy as np
import pydantic
from scientific_pydantic.numpy import NDArrayAdapter

class MyModel(pydantic.BaseModel):
    arr: ty.Annotated[np.ndarray, NDArrayAdapter()]
```
Using this pattern, you can embed scientific data types into your models and get
the full `pydantic` experience with serialization and input conversions.

## Usage

In general, it is recommended to use `from`-style imports with this library. The
import path for an adapter is normally akin to the import path of the type it is
adapting. For instance, the adapter for `scipy.spatial.transform.Rotation` would
be:
```python
from scientific_pydantic.scipy.spatial.transform import RotationAdapter
```
Adapters are used via the `Annotated` pattern, as was shown in the Motivation
section. This allows for typecheckers that support `pydantic`, such as
[`pyrefly`](https://pyrefly.org) to understand the type of the fields.

A number of adapters provided with this library also take parameters that define
common validation operations. This takes inspiration from `pydantic.Field`. For
instance,
```python
import pydantic

class MyModel(pydantic.BaseModel):
    a: int = pydantic.Field(ge=0)
```
defines a non-negative integer. In `scientific_pydantic`, this style of
validation logic can be accomplished via:
```python
import typing as ty

import numpy as np
import pydantic
from scientific_pydantic.numpy import NDArrayAdapter

class MyModel(pydantic.BaseModel):
    a: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(None, 3), ge=0)]
```
which constrains `a` to be an `N x 3` `ndarray` where all elements are
non-negative. See the individual adapters in the API documentation for a
description of the parameters each one takes.

## Design Philosophy

This library has an interesting conundrum from a dependency standpoint. Since
the goal is to provide adapters for common types that come from many different
libraries, there are a few options for how dependency management can work:

1. Depend on all packages being supported and enforce version constraints. This
   would violate the "pay for what you use" principle and is thus not a good
   option.
2. Split the package into `N` different, but related, packages (e.g.
   `scientific_pydantic_shapely`) that enforce version constraints individually.
   This is tractable, but leads to a large number of packages to maintain and
   version together.
3. Have 1 package and only depend on `pydantic` (and `pydantic_core`). Users
   will bring their own versions of the packages they want to use.

This library takes approach #3. This puts the burden of version compatibility
onto this library. For instance, `scipy.spatial.transform.Rotation` objects
gained the ability to support N-D arrays of rotation transforms in version
1.17.0. Thus, validation features related to this must be disabled if the user
brings their own `scipy` version that is `< 1.17.0`. This adds complexity to
this library, but prevents either the dependency bloat from option 1 or the
package bloat from option 2.

By only depending on `pydantic`, the library must not import anything from a
third-party library at global scope. This is accomplished via liberal use of
delayed and nested import statments and enforced via a unit test.
