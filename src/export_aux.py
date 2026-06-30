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
    SeededPAFKIPTTAViews,
    collect_bn_affine_params,
    flatten_bn_params,
    kip,
    make_resnet50_tta_models_with_weights,
    sanitize_finite,
    seeded_pafkip_transform_specs,
    stable_energy,
)


class FlatBNResNet50Logits(torch.nn.Module):
    def __init__(self, classes=1000, weights_name="none", npu_dtype="f32"):
        super().__init__()
        self.model = FlatBNResNet50PAFLoss(classes, weights_name, npu_dtype)

    def forward(self, images, flat_bn_params):
        return self.model.forward_logits(images, flat_bn_params)


class FlatBNEMAUpdate(torch.nn.Module):
    def forward(self, ema_bn_params, main_bn_params, ema_decay):
        safe_ema = sanitize_finite(ema_bn_params)
        safe_main = sanitize_finite(main_bn_params)
        return sanitize_finite(ema_decay * safe_ema + (1.0 - ema_decay) * safe_main)


class PAFKIPKIPFinal(torch.nn.Module):
    def forward(self, main_logits, ema_logits, anchor_logits):
        final_logits = kip(main_logits, ema_logits, anchor_logits)
        energy = stable_energy(main_logits)
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
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    parser.add_argument("--npu-dtype", choices=["f32", "f16", "bf16"], default="f32")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--transform-seed", type=int, default=91)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "f16" else torch.float32
    torch.manual_seed(91)
    ref_model, _, _ = make_resnet50_tta_models_with_weights(args.classes, args.weights)
    ref_params, _ = collect_bn_affine_params(ref_model)
    flat_bn_params = flatten_bn_params(ref_params).to(dtype)
    shifted_bn_params = flat_bn_params + torch.linspace(
        0.0, 1.0e-3, flat_bn_params.numel(), dtype=dtype
    )
    images = torch.randn(1, 3, args.image_size, args.image_size, dtype=dtype)
    logits = torch.randn(1, args.classes, dtype=dtype)
    ema_logits = torch.randn(1, args.classes, dtype=dtype)
    anchor_logits = torch.randn(1, args.classes, dtype=dtype)
    ema_decay = torch.tensor(args.ema_decay, dtype=dtype)

    transform_outputs = export_one(
        args.out_dir / "tta_views",
        "tta_views",
        SeededPAFKIPTTAViews(args.transform_seed),
        (images,),
        ["raw_images"],
        ["train_images", "main_filter_images", "ema_filter_images", "anchor_images"],
        {
            "transform": "RandomCrop(224,padding=4)+RandomHorizontalFlip",
            "transform_seed": int(args.transform_seed),
            "transform_specs": [
                {"top": top, "left": left, "flip": flip}
                for top, left, flip in seeded_pafkip_transform_specs(args.transform_seed)
            ],
        },
    )

    export_one(
        args.out_dir / "logits",
        "logits",
        FlatBNResNet50Logits(args.classes, args.weights, args.npu_dtype).to(dtype),
        (images, flat_bn_params),
        ["images", "flat_bn_params"],
        ["logits"],
        {
            "llvmcpu_vector_pproc_strategy": "none",
            "llvmcpu_stack_allocation_limit": 1048576,
            "npu_dtype": args.npu_dtype,
        },
    )
    export_one(
        args.out_dir / "ema",
        "ema",
        FlatBNEMAUpdate(),
        (flat_bn_params, shifted_bn_params, ema_decay),
        ["ema_bn_params", "main_bn_params", "ema_decay"],
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
