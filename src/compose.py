#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent


def split_module(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[0].strip() != "module {":
        raise ValueError("expected MLIR to start with `module {`")
    module_end = None
    for index, line in enumerate(lines[1:], start=1):
        if line == "}":
            module_end = index
            break
    if module_end is None:
        raise ValueError("expected top-level module-closing `}`")
    return "\n".join(lines[1:module_end]), "\n".join(lines[module_end + 1 :])


def rename_main(text: str, new_name: str) -> str:
    return text.replace("func.func @main(", f"func.func @{new_name}(", 1)


def load_body(manifest_path: Path, new_name: str) -> tuple[str, str, dict]:
    manifest = json.loads(manifest_path.read_text())
    body, resources = split_module(Path(manifest["mlir"]).read_text())
    return rename_main(body, new_name), resources, manifest


def write_scalar(path: Path, value: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(np.array(value, dtype=np.float32)).tofile(path)


def copy_bin(src: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(Path(src).read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-artifacts",
        type=Path,
        default=ROOT_DIR / "artifact_train",
    )
    parser.add_argument(
        "--update-artifacts",
        type=Path,
        default=ROOT_DIR / "artifact_sgd",
    )
    parser.add_argument(
        "--aux-artifacts",
        type=Path,
        default=ROOT_DIR / "artifact_aux",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "artifact_full",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--momentum", type=float, default=0.999)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    train_body, train_resources, train_manifest = load_body(
        args.train_artifacts / "manifest.json", "paf_train_step"
    )
    update_body, update_resources, _ = load_body(
        args.update_artifacts / "manifest.json", "sgd"
    )
    logits_body, logits_resources, _ = load_body(
        args.aux_artifacts / "logits" / "manifest.json", "logits"
    )
    ema_body, ema_resources, _ = load_body(
        args.aux_artifacts / "ema" / "manifest.json", "ema"
    )
    kip_body, kip_resources, _ = load_body(
        args.aux_artifacts / "kip" / "manifest.json", "kip"
    )
    for name, resources in {
        "update": update_resources,
        "ema": ema_resources,
        "kip": kip_resources,
    }.items():
        if resources.strip():
            raise ValueError(f"{name} MLIR unexpectedly has dialect resources")
    # The logits function was exported from the same seeded model as the
    # train-step. Reuse train-step resources to avoid duplicating 100MB+ of
    # constants in the composed module.
    if not train_resources.strip() or not logits_resources.strip():
        raise ValueError("expected train/logits MLIR to carry weight resources")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.out_dir / "inputs"
    input_by_name = {item["name"]: item for item in train_manifest["inputs"]}
    images_item = dict(input_by_name["images"])
    main_item = dict(input_by_name["flat_bn_params"])
    ema_path = input_dir / "ema_bn_params.bin"
    anchor_path = input_dir / "anchor_bn_params.bin"
    lr_path = input_dir / "lr.bin"
    momentum_path = input_dir / "momentum.bin"
    copy_bin(main_item["file"], ema_path)
    copy_bin(main_item["file"], anchor_path)
    write_scalar(lr_path, args.lr)
    write_scalar(momentum_path, args.momentum)

    loop_body = f"""
  func.func @main(%arg0: !torch.vtensor<[1,3,224,224],f32>, %arg1: !torch.vtensor<[53120],f32>, %arg2: !torch.vtensor<[53120],f32>, %arg3: !torch.vtensor<[53120],f32>, %arg4: !torch.vtensor<[],f32>, %arg5: !torch.vtensor<[],f32>) -> (!torch.vtensor<[1,1000],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[53120],f32>, !torch.vtensor<[53120],f32>, !torch.vtensor<[53120],f32>) {{
    %c1 = arith.constant 1 : index
    %c_steps = arith.constant {args.steps} : index
    %main_logits0 = func.call @logits(%arg0, %arg1) : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[53120],f32>) -> !torch.vtensor<[1,1000],f32>
    %ema_logits0 = func.call @logits(%arg0, %arg2) : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[53120],f32>) -> !torch.vtensor<[1,1000],f32>
    %anchor_logits0 = func.call @logits(%arg0, %arg3) : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[53120],f32>) -> !torch.vtensor<[1,1000],f32>
    %train0:2 = func.call @paf_train_step(%arg0, %ema_logits0, %arg1) : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[1,1000],f32>, !torch.vtensor<[53120],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[53120],f32>)
    %final0:2 = func.call @kip(%main_logits0, %ema_logits0, %anchor_logits0) : (!torch.vtensor<[1,1000],f32>, !torch.vtensor<[1,1000],f32>, !torch.vtensor<[1,1000],f32>) -> (!torch.vtensor<[1,1000],f32>, !torch.vtensor<[1],f32>)
    %main1 = func.call @sgd(%arg1, %train0#1, %arg4) : (!torch.vtensor<[53120],f32>, !torch.vtensor<[53120],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[53120],f32>
    %ema1 = func.call @ema(%arg2, %main1, %arg5) : (!torch.vtensor<[53120],f32>, !torch.vtensor<[53120],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[53120],f32>
    %loop:6 = scf.for %i = %c1 to %c_steps step %c1 iter_args(%main_state = %main1, %ema_state = %ema1, %last_final = %final0#0, %last_energy = %final0#1, %last_loss = %train0#0, %last_grad = %train0#1) -> (!torch.vtensor<[53120],f32>, !torch.vtensor<[53120],f32>, !torch.vtensor<[1,1000],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[53120],f32>) {{
      %main_logits = func.call @logits(%arg0, %main_state) : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[53120],f32>) -> !torch.vtensor<[1,1000],f32>
      %ema_logits = func.call @logits(%arg0, %ema_state) : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[53120],f32>) -> !torch.vtensor<[1,1000],f32>
      %anchor_logits = func.call @logits(%arg0, %arg3) : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[53120],f32>) -> !torch.vtensor<[1,1000],f32>
      %train:2 = func.call @paf_train_step(%arg0, %ema_logits, %main_state) : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[1,1000],f32>, !torch.vtensor<[53120],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[53120],f32>)
      %final:2 = func.call @kip(%main_logits, %ema_logits, %anchor_logits) : (!torch.vtensor<[1,1000],f32>, !torch.vtensor<[1,1000],f32>, !torch.vtensor<[1,1000],f32>) -> (!torch.vtensor<[1,1000],f32>, !torch.vtensor<[1],f32>)
      %next_main = func.call @sgd(%main_state, %train#1, %arg4) : (!torch.vtensor<[53120],f32>, !torch.vtensor<[53120],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[53120],f32>
      %next_ema = func.call @ema(%ema_state, %next_main, %arg5) : (!torch.vtensor<[53120],f32>, !torch.vtensor<[53120],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[53120],f32>
      scf.yield %next_main, %next_ema, %final#0, %final#1, %train#0, %train#1 : !torch.vtensor<[53120],f32>, !torch.vtensor<[53120],f32>, !torch.vtensor<[1,1000],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[53120],f32>
    }}
    return %loop#2, %loop#3, %loop#4, %loop#0, %loop#1, %loop#5 : !torch.vtensor<[1,1000],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[53120],f32>, !torch.vtensor<[53120],f32>, !torch.vtensor<[53120],f32>
  }}
"""
    mlir_path = args.out_dir / "full.mlir"
    resource_suffix = "\n" + train_resources + "\n"
    mlir_path.write_text(
        "module {\n"
        + train_body
        + "\n"
        + logits_body
        + "\n"
        + update_body
        + "\n"
        + ema_body
        + "\n"
        + kip_body
        + loop_body
        + "}\n"
        + resource_suffix
    )

    images_item["file"] = input_by_name["images"]["file"]
    main_item["name"] = "main_bn_params"
    manifest = {
        "name": "full",
        "function": "main",
        "mlir": str(mlir_path),
        "inputs": [
            images_item,
            main_item,
            {
                "name": "ema_bn_params",
                "shape": [53120],
                "dtype": "f32",
                "iree": "53120xf32",
                "file": str(ema_path),
            },
            {
                "name": "anchor_bn_params",
                "shape": [53120],
                "dtype": "f32",
                "iree": "53120xf32",
                "file": str(anchor_path),
            },
            {"name": "lr", "shape": [], "dtype": "f32", "iree": "f32", "file": str(lr_path)},
            {
                "name": "momentum",
                "shape": [],
                "dtype": "f32",
                "iree": "f32",
                "file": str(momentum_path),
            },
        ],
        "outputs": [
            {"name": "final_logits", "shape": [1, 1000], "dtype": "f32", "iree": "1x1000xf32"},
            {"name": "energy", "shape": [1], "dtype": "f32", "iree": "1xf32"},
            {"name": "loss", "shape": [], "dtype": "f32", "iree": "f32"},
            {
                "name": "new_main_bn_params",
                "shape": [53120],
                "dtype": "f32",
                "iree": "53120xf32",
            },
            {
                "name": "new_ema_bn_params",
                "shape": [53120],
                "dtype": "f32",
                "iree": "53120xf32",
            },
            {"name": "flat_bn_grads", "shape": [53120], "dtype": "f32", "iree": "53120xf32"},
        ],
        "steps": args.steps,
        "lr": args.lr,
        "momentum": args.momentum,
        "bn_param_count": train_manifest["bn_param_count"],
        "bn_scalar_count": train_manifest["bn_scalar_count"],
        "llvmcpu_vector_pproc_strategy": "none",
        "llvmcpu_stack_allocation_limit": 1048576,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(mlir_path)


if __name__ == "__main__":
    main()
