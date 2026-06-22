#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from export_lib import export_one
from model import (
    AllBNSGDMomentumUpdate,
    collect_bn_affine_params,
    flatten_bn_grads,
    flatten_bn_params,
    make_resnet50_tta_models_with_weights,
    paf_loss,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "artifact_sgd")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--weights", choices=["none", "default"], default="none")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    args = parser.parse_args()

    torch.manual_seed(117)
    dtype = torch.float16 if args.dtype == "f16" else torch.float32
    main_model, ema_model, _ = make_resnet50_tta_models_with_weights(args.classes, args.weights)
    main_model = main_model.to(dtype)
    ema_model = ema_model.to(dtype)
    params, names = collect_bn_affine_params(main_model)
    train_images = torch.randn(1, 3, args.image_size, args.image_size, dtype=dtype)
    main_filter_images = torch.randn(1, 3, args.image_size, args.image_size, dtype=dtype)
    with torch.no_grad():
        ema_filter_logits = ema_model(main_filter_images)
        main_filter_logits = main_model(main_filter_images)
    train_logits = main_model(train_images)
    loss = paf_loss(train_logits, main_filter_logits, ema_filter_logits)
    loss.backward()
    bn_params = flatten_bn_params(params).to(dtype)
    bn_grads = flatten_bn_grads(params).to(dtype)
    velocity = torch.zeros_like(bn_params)
    lr = torch.tensor(args.lr, dtype=dtype)
    sgd_momentum = torch.tensor(args.sgd_momentum, dtype=dtype)

    outputs = export_one(
        args.out_dir,
        "sgd",
        AllBNSGDMomentumUpdate(),
        (bn_params, bn_grads, velocity, lr, sgd_momentum),
        ["bn_params", "bn_grads", "velocity", "lr", "sgd_momentum"],
        ["new_bn_params", "new_velocity"],
    )
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bn_param_count"] = len(names)
    manifest["bn_scalar_count"] = int(bn_params.numel())
    manifest["bn_param_names"] = names
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "bn_param_count": len(names),
                "bn_scalar_count": int(bn_params.numel()),
                "max_param_delta": float(torch.max(torch.abs(outputs[0] - bn_params)).item()),
                "max_velocity_delta": float(torch.max(torch.abs(outputs[1] - velocity)).item()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
