#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from export_lib import export_one
from model import (
    FlatBNResNet50PAFLoss,
    collect_bn_affine_params,
    flatten_bn_params,
    kip,
    make_resnet50_tta_models_with_weights,
)


class FlatBNResNet50Logits(torch.nn.Module):
    def __init__(self, classes=1000, weights_name="none"):
        super().__init__()
        self.model = FlatBNResNet50PAFLoss(classes, weights_name)

    def forward(self, images, flat_bn_params):
        return self.model.forward_logits(images, flat_bn_params)


class FlatBNEMAUpdate(torch.nn.Module):
    def forward(self, ema_bn_params, main_bn_params, momentum):
        return momentum * ema_bn_params + (1.0 - momentum) * main_bn_params


class PAFKIPKIPFinal(torch.nn.Module):
    def forward(self, main_logits, ema_logits, anchor_logits):
        final_logits = kip(main_logits, ema_logits, anchor_logits)
        energy = torch.log(torch.exp(main_logits).sum(dim=1))
        return final_logits, energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "artifact_aux",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--weights", choices=["none", "default"], default="none")
    parser.add_argument("--momentum", type=float, default=0.999)
    args = parser.parse_args()

    torch.manual_seed(91)
    ref_model, _, _ = make_resnet50_tta_models_with_weights(args.classes, args.weights)
    ref_params, _ = collect_bn_affine_params(ref_model)
    flat_bn_params = flatten_bn_params(ref_params)
    shifted_bn_params = flat_bn_params + torch.linspace(
        0.0, 1.0e-3, flat_bn_params.numel(), dtype=torch.float32
    )
    images = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    logits = torch.randn(1, args.classes, dtype=torch.float32)
    ema_logits = torch.randn(1, args.classes, dtype=torch.float32)
    anchor_logits = torch.randn(1, args.classes, dtype=torch.float32)
    momentum = torch.tensor(args.momentum, dtype=torch.float32)

    export_one(
        args.out_dir / "logits",
        "logits",
        FlatBNResNet50Logits(args.classes, args.weights),
        (images, flat_bn_params),
        ["images", "flat_bn_params"],
        ["logits"],
        {"llvmcpu_vector_pproc_strategy": "none", "llvmcpu_stack_allocation_limit": 1048576},
    )
    export_one(
        args.out_dir / "ema",
        "ema",
        FlatBNEMAUpdate(),
        (flat_bn_params, shifted_bn_params, momentum),
        ["ema_bn_params", "main_bn_params", "momentum"],
        ["new_ema_bn_params"],
    )
    export_one(
        args.out_dir / "kip",
        "kip",
        PAFKIPKIPFinal(),
        (logits, ema_logits, anchor_logits),
        ["main_logits", "ema_logits", "anchor_logits"],
        ["final_logits", "energy"],
    )


if __name__ == "__main__":
    main()
