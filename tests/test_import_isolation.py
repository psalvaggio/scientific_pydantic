"""Test for import isolation"""

import pathlib
import subprocess


def test_import_without_optional_deps() -> None:
    """Test isolated import

    Ensure scientific_pydantic can be imported with only
    pydantic/pydantic_core available.
    """
    result = subprocess.run(
        [  # noqa: S607 (it's a unit test)
            "uv",
            "run",
            "--isolated",
            "--with",
            "pydantic",
            "--no-project",
            "--",
            "python",
            "-c",
            "import scientific_pydantic",
        ],
        cwd=pathlib.Path(__file__).parent.parent / "src",
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Import failed with only pydantic/pydantic_core available.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
