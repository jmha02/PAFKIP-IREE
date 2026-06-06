#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCH_MLIR_PACKAGE = ROOT / "build-torch-mlir/tools/torch-mlir/python_packages/torch_mlir"

if TORCH_MLIR_PACKAGE.exists():
    sys.path.insert(0, str(TORCH_MLIR_PACKAGE))


def ok(label: str, value: str) -> None:
    print(f"[ok] {label}: {value}")


def missing(label: str, hint: str) -> bool:
    print(f"[missing] {label}: {hint}")
    return False


def main() -> int:
    checks = []

    for tool in ["git", "cmake", "ninja"]:
        path = shutil.which(tool)
        checks.append(bool(path))
        ok(tool, path) if path else missing(tool, f"install {tool} and add it to PATH")

    for tool in ["clang", "clang++"]:
        path = shutil.which(tool)
        checks.append(bool(path))
        ok(tool, path) if path else missing(tool, f"install {tool} and add it to PATH")

    iree_dir = ROOT / "third_party" / "iree"
    checks.append((iree_dir / "CMakeLists.txt").exists())
    if (iree_dir / "CMakeLists.txt").exists():
        ok("third_party/iree", str(iree_dir))
    else:
        missing("third_party/iree", "run `git submodule update --init --recursive`")

    runtime_build = ROOT / "third_party/iree/build-riscv-baremetal/runtime/src/iree/base/libiree_base_base.a"
    legacy_runtime_build = ROOT / "third_party/iree/build-riscv-pk/runtime/src/iree/base/libiree_base_base.a"
    for rel in [
        "third_party/iree/build/tools/iree-compile",
        "third_party/iree/build/tools/iree-run-module",
        "third_party/iree/runtime/tools/iree-bundle-baremetal",
    ]:
        path = ROOT / rel
        checks.append(path.exists())
        ok(rel, str(path)) if path.exists() else missing(rel, "run `tools/setup_from_scratch.sh`")
    checks.append(runtime_build.exists() or legacy_runtime_build.exists())
    if runtime_build.exists():
        ok("third_party/iree/build-riscv-baremetal", str(runtime_build))
    elif legacy_runtime_build.exists():
        ok("third_party/iree/build-riscv-pk", f"{legacy_runtime_build} (legacy local build dir)")
    else:
        missing(
            "third_party/iree/build-riscv-baremetal",
            "set RISCV and run `tools/setup_from_scratch.sh`",
        )

    riscv = os.environ.get("RISCV")
    if riscv:
        ok("RISCV", riscv)
        for tool in ["riscv64-unknown-elf-gcc", "riscv64-unknown-elf-g++", "llvm-nm", "spike"]:
            path = shutil.which(tool) or str(Path(riscv) / "bin" / tool)
            exists = Path(path).exists()
            checks.append(exists)
            ok(tool, path) if exists else missing(tool, f"expected in $RISCV/bin or PATH")
        objcopy_path = (
            shutil.which("llvm-objcopy")
            or shutil.which("riscv64-unknown-elf-objcopy")
            or str(Path(riscv) / "bin" / "llvm-objcopy")
        )
        objcopy_exists = Path(objcopy_path).exists()
        checks.append(objcopy_exists)
        ok("objcopy", objcopy_path) if objcopy_exists else missing(
            "objcopy",
            "expected llvm-objcopy or riscv64-unknown-elf-objcopy in $RISCV/bin or PATH",
        )
    else:
        checks.append(False)
        missing("RISCV", "set RISCV to a riscv64-unknown-elf toolchain root")

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import torch_mlir.fx  # noqa: F401
        checks.append(True)
        ok("python deps", "torch, torchvision, torch_mlir.fx")
    except Exception as exc:
        checks.append(False)
        missing(
            "python deps",
            f"{exc}; install requirements and run `tools/setup_from_scratch.sh` to build torch-mlir",
        )

    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
