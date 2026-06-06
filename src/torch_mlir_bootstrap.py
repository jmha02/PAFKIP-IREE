from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TORCH_MLIR_PACKAGE = ROOT / "build-torch-mlir" / "tools" / "torch-mlir" / "python_packages" / "torch_mlir"


def add_torch_mlir_build_path() -> None:
    if TORCH_MLIR_PACKAGE.exists():
        path = str(TORCH_MLIR_PACKAGE)
        if path not in sys.path:
            sys.path.insert(0, path)
