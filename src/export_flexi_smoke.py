#!/usr/bin/env python3
"""Small FlexiNPU lowering smoke stages.

These stages are intentionally handwritten MLIR so that the pass under test is
exactly IREE's linalg-to-FlexiNPU path, without relying on PyTorch pattern
choices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from paths import repo_relative


DTYPE_TO_NP = {
    "f32": np.float32,
    "f16": np.float16,
}
DTYPE_TO_TORCH = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = values.astype(np.float32, copy=False)
    bits = f32.view(np.uint32)
    rounded = bits + (((bits >> 16) & 1) + 0x7FFF)
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_f32(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


def write_tensor(path: Path, values: np.ndarray, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dtype == "bf16":
        f32_to_bf16_bits(values).tofile(path)
    else:
        values.astype(DTYPE_TO_NP[dtype]).tofile(path)


def write_golden(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values.astype(np.float32).tofile(path)


def tensor_type(shape: tuple[int, ...], dtype: str) -> str:
    return f"{'x'.join(str(dim) for dim in shape)}x{dtype}"


def write_matmul_mlir(
    path: Path, dtype: str, out_dtype: str, m: int, n: int, k: int
) -> None:
    path.write_text(
        f"""module {{
  func.func @main(%arg0: tensor<{m}x{k}x{dtype}>, %arg1: tensor<{k}x{n}x{dtype}>, %arg2: tensor<{m}x{n}x{out_dtype}>) -> tensor<{m}x{n}x{out_dtype}> {{
    %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<{m}x{k}x{dtype}>, tensor<{k}x{n}x{dtype}>)
      outs(%arg2 : tensor<{m}x{n}x{out_dtype}>) -> tensor<{m}x{n}x{out_dtype}>
    return %0 : tensor<{m}x{n}x{out_dtype}>
  }}
}}
"""
    )


def export_matmul(
    out_dir: Path, dtype: str, out_dtype: str, seed: int, m: int, n: int, k: int
) -> None:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k), dtype=np.float32) * 0.25
    b = rng.standard_normal((k, n), dtype=np.float32) * 0.25
    c = np.zeros((m, n), dtype=np.float32)
    golden = a @ b + c

    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = out_dir / "inputs"
    golden_dir = out_dir / "golden"
    mlir_path = out_dir / "matmul.mlir"
    write_matmul_mlir(mlir_path, dtype, out_dtype, m, n, k)

    inputs = []
    for name, values, item_dtype in [
        ("mat_a", a, dtype),
        ("mat_b", b, dtype),
        ("mat_c", c, out_dtype),
    ]:
        path = input_dir / f"{name}.bin"
        write_tensor(path, values, item_dtype)
        shape = list(values.shape)
        inputs.append(
            {
                "name": name,
                "shape": shape,
                "dtype": item_dtype,
                "iree": tensor_type(tuple(shape), item_dtype),
                "file": repo_relative(path),
            }
        )

    golden_path = golden_dir / "mat_d.bin"
    write_golden(golden_path, golden)
    manifest = {
        "name": f"flexi_matmul_{dtype}x{dtype}x{out_dtype}",
        "function": "main",
        "mlir": repo_relative(mlir_path),
        "input_type": "none",
        "inputs": inputs,
        "outputs": [
            {
                "name": "mat_d",
                "shape": [m, n],
                "dtype": out_dtype,
                "iree": tensor_type((m, n), out_dtype),
                "golden": repo_relative(golden_path),
                "compare_as": "f32_reference",
            }
        ],
        "llvmcpu_vector_pproc_strategy": "none",
        "llvmcpu_stack_allocation_limit": 1048576,
        "spike_extension": "flexi",
        "compare_atol": 2.0e-2 if dtype != "f32" else 5.0e-4,
        "compare_rtol": 2.0e-2 if dtype != "f32" else 5.0e-4,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(mlir_path)


def write_conv_mlir(path: Path, dtype: str, out_dtype: str) -> None:
    path.write_text(
        f"""module {{
  func.func @main(%arg0: tensor<1x32x6x6x{dtype}>, %arg1: tensor<32x32x3x3x{dtype}>, %arg2: tensor<1x32x4x4x{out_dtype}>) -> tensor<1x32x4x4x{out_dtype}> {{
    %0 = linalg.conv_2d_nchw_fchw
      {{dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}}
      ins(%arg0, %arg1 : tensor<1x32x6x6x{dtype}>, tensor<32x32x3x3x{dtype}>)
      outs(%arg2 : tensor<1x32x4x4x{out_dtype}>) -> tensor<1x32x4x4x{out_dtype}>
    return %0 : tensor<1x32x4x4x{out_dtype}>
  }}
}}
"""
    )


def export_conv(out_dir: Path, dtype: str, out_dtype: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.standard_normal((1, 32, 6, 6), dtype=np.float32) * 0.25
    weight = rng.standard_normal((32, 32, 3, 3), dtype=np.float32) * 0.25
    init = np.zeros((1, 32, 4, 4), dtype=np.float32)
    with torch.no_grad():
        output = torch.nn.functional.conv2d(
            torch.from_numpy(image),
            torch.from_numpy(weight),
        ).numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = out_dir / "inputs"
    golden_dir = out_dir / "golden"
    mlir_path = out_dir / "conv.mlir"
    write_conv_mlir(mlir_path, dtype, out_dtype)

    inputs = []
    for name, values, item_dtype in [
        ("image", image, dtype),
        ("weight", weight, dtype),
        ("init", init, out_dtype),
    ]:
        path = input_dir / f"{name}.bin"
        write_tensor(path, values, item_dtype)
        inputs.append(
            {
                "name": name,
                "shape": list(values.shape),
                "dtype": item_dtype,
                "iree": tensor_type(tuple(values.shape), item_dtype),
                "file": repo_relative(path),
            }
        )
    golden_path = golden_dir / "output.bin"
    write_golden(golden_path, output)

    manifest = {
        "name": f"flexi_conv_{dtype}x{dtype}x{out_dtype}",
        "function": "main",
        "mlir": repo_relative(mlir_path),
        "input_type": "none",
        "inputs": inputs,
        "outputs": [
            {
                "name": "output",
                "shape": [1, 32, 4, 4],
                "dtype": out_dtype,
                "iree": tensor_type((1, 32, 4, 4), out_dtype),
                "golden": repo_relative(golden_path),
                "compare_as": "f32_reference",
            }
        ],
        "llvmcpu_vector_pproc_strategy": "none",
        "llvmcpu_stack_allocation_limit": 1048576,
        "spike_extension": "flexi",
        "compare_atol": 2.0e-2 if dtype != "f32" else 5.0e-4,
        "compare_rtol": 2.0e-2 if dtype != "f32" else 5.0e-4,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(mlir_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "artifact_aux" / "flexi_matmul")
    parser.add_argument("--kind", choices=["matmul", "conv"], default="matmul")
    parser.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="bf16")
    parser.add_argument("--out-dtype", choices=["f32", "f16", "bf16"])
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    args = parser.parse_args()
    if args.kind == "matmul":
        export_matmul(
            args.out_dir,
            args.dtype,
            args.out_dtype or args.dtype,
            args.seed,
            args.m,
            args.n,
            args.k,
        )
    else:
        export_conv(args.out_dir, args.dtype, args.out_dtype or args.dtype, args.seed)


if __name__ == "__main__":
    main()
