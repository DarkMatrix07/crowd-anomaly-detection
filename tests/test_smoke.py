import subprocess
import sys


def test_package_import_smoke() -> None:
    __import__("src")


def test_module_cli_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    help_output = result.stdout.lower()
    assert "usage" in help_output
    assert "train" in help_output
    assert "infer" in help_output
    assert "test" in help_output
