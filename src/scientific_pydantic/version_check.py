"""Version check functions"""

import types
from collections.abc import Sequence


def version_ge(module: types.ModuleType, version: Sequence[int] | str) -> bool:
    """Test whether the given module's version is >= `version`"""
    module_ver, version = _normalize(module, version)
    return module_ver >= version


def version_gt(module: types.ModuleType, version: Sequence[int] | str) -> bool:
    """Test whether the given module's version is > `version`"""
    module_ver, version = _normalize(module, version)
    return module_ver > version


def version_le(module: types.ModuleType, version: Sequence[int] | str) -> bool:
    """Test whether the given module's version is <= `version`"""
    module_ver, version = _normalize(module, version)
    return module_ver <= version


def version_lt(module: types.ModuleType, version: Sequence[int] | str) -> bool:
    """Test whether the given module's version is < `version`"""
    module_ver, version = _normalize(module, version)
    return module_ver < version


def version_eq(module: types.ModuleType, version: Sequence[int] | str) -> bool:
    """Test whether the given module's version is == `version`"""
    module_ver, version = _normalize(module, version)
    return module_ver == version


def version_ne(module: types.ModuleType, version: Sequence[int] | str) -> bool:
    """Test whether the given module's version is != `version`"""
    module_ver, version = _normalize(module, version)
    return module_ver != version


def _normalize(
    module: types.ModuleType, version: Sequence[int] | str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    v = getattr(module, "__version__", None)
    if v is None:
        msg = f"{module} does not have a __version__"
        raise ValueError(msg)

    module_ver = tuple(int(x) for x in v.split("."))
    version = (
        tuple(int(x) for x in version.split("."))
        if isinstance(version, str)
        else tuple(version)
    )
    return module_ver, version
