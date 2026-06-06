#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IREE_DIR="${ROOT}/third_party/iree"
BAREMETAL_BUILD_DIR="${IREE_DIR}/build-riscv-baremetal"
TORCH_MLIR_DIR="${IREE_DIR}/third_party/torch-mlir"
TORCH_MLIR_BUILD_DIR="${ROOT}/build-torch-mlir"
TORCH_MLIR_PACKAGE="${TORCH_MLIR_BUILD_DIR}/tools/torch-mlir/python_packages/torch_mlir"
JOBS="${JOBS:-$(nproc)}"

git -C "${ROOT}" submodule update --init --recursive

for tool in cmake ninja clang clang++; do
  if ! command -v "${tool}" >/dev/null; then
    echo "${tool} is required" >&2
    exit 1
  fi
done

if ! python3 - <<'PY'
import numpy  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
PY
then
  cat >&2 <<'EOF'

Python packages needed for artifact export are missing.
Install the Python requirements in your environment and rerun this script:

  python3 -m pip install -r requirements.txt
  tools/setup_from_scratch.sh
EOF
  exit 1
fi

cmake -S "${IREE_DIR}" -B "${IREE_DIR}/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_BENCHMARKS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=OFF \
  -DPython3_EXECUTABLE="$(command -v python3)"

cmake --build "${IREE_DIR}/build" \
  --target iree-compile iree-run-module \
  -j "${JOBS}"

if ! PYTHONPATH="${TORCH_MLIR_PACKAGE}:${PYTHONPATH:-}" python3 - <<'PY'
import torch_mlir.fx  # noqa: F401
PY
then
  if [[ ! -f "${TORCH_MLIR_DIR}/externals/llvm-project/llvm/CMakeLists.txt" ]]; then
    echo "torch-mlir LLVM submodule is missing; run git submodule update --init --recursive" >&2
    exit 1
  fi

  cmake -S "${TORCH_MLIR_DIR}/externals/llvm-project/llvm" -B "${TORCH_MLIR_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=torch-mlir \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="${TORCH_MLIR_DIR}" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DTORCH_MLIR_BUILD_PYTHON_PACKAGE=ON \
    -DTORCH_MLIR_ENABLE_REFBACKEND=OFF \
    -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
    -DTORCH_MLIR_ENABLE_TOSA=OFF \
    -DTORCH_MLIR_ENABLE_ONNX_C_IMPORTER=OFF \
    -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=OFF \
    -DTORCH_MLIR_USE_INSTALLED_PYTORCH=ON \
    -DPython3_EXECUTABLE="$(command -v python3)"

  cmake --build "${TORCH_MLIR_BUILD_DIR}" \
    --target TorchMLIRPythonModules torch-mlir-opt \
    -j "${JOBS}"
fi

if [[ -z "${RISCV:-}" ]]; then
  cat >&2 <<'EOF'

Host IREE tools were built, but the RISC-V runtime was not configured.
Set RISCV to a riscv64-unknown-elf toolchain root and rerun this script:

  export RISCV=/path/to/riscv-tools
  export PATH="$RISCV/bin:$PATH"
  tools/setup_from_scratch.sh

The toolchain must provide riscv64-unknown-elf-gcc/g++ and spike must be in PATH
or $RISCV/bin for Spike simulation.
EOF
  exit 0
fi

for tool in riscv64-unknown-elf-gcc riscv64-unknown-elf-g++; do
  if [[ ! -x "${RISCV}/bin/${tool}" ]]; then
    echo "${tool} is required under \$RISCV/bin" >&2
    exit 1
  fi
done

if [[ -f "${BAREMETAL_BUILD_DIR}/CMakeCache.txt" ]] &&
   ! grep -q "CMAKE_TOOLCHAIN_FILE.*riscv.toolchain.cmake" "${BAREMETAL_BUILD_DIR}/CMakeCache.txt"; then
  echo "Recreating ${BAREMETAL_BUILD_DIR}; previous cache was not configured with the RISC-V toolchain file."
  rm -rf "${BAREMETAL_BUILD_DIR}"
fi

cmake -S "${IREE_DIR}" -B "${BAREMETAL_BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${IREE_DIR}/build_tools/cmake/riscv.toolchain.cmake" \
  -DRISCV_CPU=generic-riscv_64 \
  -DRISCV_TOOLCHAIN_PREFIX=riscv64-unknown-elf- \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_BENCHMARKS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=OFF \
  -DIREE_BUILD_STATIC_RUN_MODULE=OFF \
  -DIREE_ENABLE_LIBBACKTRACE=OFF \
  -DIREE_ENABLE_POSIX=OFF \
  -DIREE_ENABLE_THREADING=OFF \
  -DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=ON \
  -DIREE_HOST_BIN_DIR="${IREE_DIR}/build/tools" \
  -DCMAKE_C_COMPILER="${RISCV}/bin/riscv64-unknown-elf-gcc" \
  -DCMAKE_CXX_COMPILER="${RISCV}/bin/riscv64-unknown-elf-g++"

cmake --build "${BAREMETAL_BUILD_DIR}" \
  --target \
    iree_base_base \
    iree_hal_hal \
    iree_hal_drivers_local_sync_sync_driver \
    iree_hal_local_elf_arch \
    iree_hal_local_loaders_embedded_elf_loader \
    iree_modules_hal_hal \
    iree_modules_hal_inline_inline \
    iree_modules_vmvx_vmvx \
    iree_vm_bytecode_module \
    iree_builtins_ukernel_ukernel \
    iree_builtins_ukernel_arch_riscv_64_riscv_64 \
    iree_builtins_ukernel_arch_riscv_64_riscv_64_v \
    iree_tooling_run_module \
  -j "${JOBS}"

python3 "${ROOT}/tools/check_env.py"
