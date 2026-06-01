#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch_mlir.fx
from torch._decomp import get_decompositions
from torch.export import export
from torch.export.experimental import _export_forward_backward
from torch_mlir.extras.fx_decomp_util import DEFAULT_DECOMPOSITIONS

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from model import (
    FlatBNResNet50PAFLoss,
    collect_bn_affine_params,
    flatten_bn_params,
    make_resnet50_tta_models_with_weights,
)


DTYPE_TO_MLIR = {torch.float32: "f32"}


def shape_dtype_to_iree(tensor: torch.Tensor) -> str:
    shape = "x".join(str(d) for d in tensor.shape)
    return f"{shape}x{DTYPE_TO_MLIR[tensor.dtype]}" if shape else DTYPE_TO_MLIR[tensor.dtype]


def write_bin(path: Path, tensor: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tensor.detach().cpu().numpy()).astype(np.float32).tofile(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "artifact_train",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--weights", choices=["none", "default"], default="none")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(91)

    model = FlatBNResNet50PAFLoss(args.classes, args.weights)
    ref_model, _, _ = make_resnet50_tta_models_with_weights(args.classes, args.weights)
    ref_params, ref_names = collect_bn_affine_params(ref_model)
    flat_bn_params = flatten_bn_params(ref_params).requires_grad_(True)
    images = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    ema_logits = torch.randn(1, args.classes, dtype=torch.float32)
    lr = torch.tensor(args.lr, dtype=torch.float32)

    ep = export(model, (images, ema_logits, flat_bn_params), strict=False)
    fb = _export_forward_backward(ep)
    decomposition_table = get_decompositions([*DEFAULT_DECOMPOSITIONS, torch.ops.aten.as_strided])
    mlir = torch_mlir.fx.export_and_import(
        fb,
        output_type="torch",
        decomposition_table=decomposition_table,
    )
    mlir_path = args.out_dir / "train.mlir"
    mlir_path.write_text(str(mlir))

    loss = model(images, ema_logits, flat_bn_params)
    loss.backward()
    flat_grads = flat_bn_params.grad.detach()
    new_bn_params = flat_bn_params.detach() - lr * flat_grads

    input_dir = args.out_dir / "inputs"
    golden_dir = args.out_dir / "golden"
    input_tensors = {
        "images": images,
        "ema_logits": ema_logits,
        "flat_bn_params": flat_bn_params.detach(),
    }
    inputs = []
    for name, tensor in input_tensors.items():
        path = input_dir / f"{name}.bin"
        write_bin(path, tensor)
        inputs.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "f32",
                "iree": shape_dtype_to_iree(tensor),
                "file": str(path),
            }
        )

    outputs = []
    for name, tensor in {
        "loss": loss.detach(),
        "flat_bn_grads": flat_grads,
    }.items():
        path = golden_dir / f"{name}.bin"
        write_bin(path, tensor)
        outputs.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "f32",
                "iree": shape_dtype_to_iree(tensor),
                "golden": str(path),
            }
        )

    update_path = golden_dir / "new_bn_params.bin"
    write_bin(update_path, new_bn_params)

    manifest = {
        "name": "train",
        "function": "main",
        "mlir": str(mlir_path),
        "inputs": inputs,
        "outputs": outputs,
        "bn_param_count": len(ref_names),
        "bn_scalar_count": int(flat_bn_params.numel()),
        "bn_param_names": ref_names,
        "lr": float(args.lr),
        "golden_new_bn_params": str(update_path),
        "llvmcpu_vector_pproc_strategy": "none",
        "llvmcpu_stack_allocation_limit": 1048576,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(mlir_path)
    print(f"bn_param_count={len(ref_names)}")
    print(f"bn_scalar_count={flat_bn_params.numel()}")


if __name__ == "__main__":
    main()
