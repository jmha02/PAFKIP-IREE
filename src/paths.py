from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def repo_relative(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return str(path.relative_to(PACKAGE_ROOT))
    except ValueError:
        return str(path)
