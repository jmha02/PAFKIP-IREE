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
    AllBNSGDUpdate,
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
    args = parser.parse_args()

    torch.manual_seed(117)
    main_model, ema_model, _ = make_resnet50_tta_models_with_weights(args.classes, args.weights)
    params, names = collect_bn_affine_params(main_model)
    images = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    with torch.no_grad():
        ema_logits = ema_model(images)
    logits = main_model(images)
    loss = paf_loss(logits, ema_logits)
    loss.backward()
    bn_params = flatten_bn_params(params)
    bn_grads = flatten_bn_grads(params)
    lr = torch.tensor(args.lr, dtype=torch.float32)

    outputs = export_one(
        args.out_dir,
        "sgd",
        AllBNSGDUpdate(),
        (bn_params, bn_grads, lr),
        ["bn_params", "bn_grads", "lr"],
        ["new_bn_params"],
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
