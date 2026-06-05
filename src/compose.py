#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

from paths import repo_path, repo_relative


def by_name(items: list[dict]) -> dict[str, dict]:
    result = {}
    for item in items:
        name = item["name"]
        if name in result:
            raise ValueError(f"duplicate manifest item name: {name}")
        result[name] = item
    return result


def torch_vtensor(item: dict) -> str:
    if item["dtype"] != "f32":
        raise ValueError(f"{item['name']} must be f32, got {item['dtype']}")
    shape = item["shape"]
    if shape:
        return "!torch.vtensor<[" + ",".join(str(dim) for dim in shape) + "],f32>"
    return "!torch.vtensor<[],f32>"


def check_shape(item: dict, expected: list[int], label: str):
    actual = item["shape"]
    if actual != expected:
        raise ValueError(f"{label} shape mismatch: expected {expected}, got {actual}")


def check_same_shape(lhs: dict, rhs: dict, label: str):
    if lhs["shape"] != rhs["shape"]:
        raise ValueError(
            f"{label} shape mismatch: {lhs['name']}={lhs['shape']} "
            f"{rhs['name']}={rhs['shape']}"
        )


def check_names(stage: str, manifest: dict, inputs: list[str], outputs: list[str]):
    actual_inputs = [item["name"] for item in manifest["inputs"]]
    actual_outputs = [item["name"] for item in manifest["outputs"]]
    if actual_inputs != inputs:
        raise ValueError(f"{stage} inputs mismatch: expected {inputs}, got {actual_inputs}")
    if actual_outputs != outputs:
        raise ValueError(f"{stage} outputs mismatch: expected {outputs}, got {actual_outputs}")


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
    body, resources = split_module(repo_path(manifest["mlir"]).read_text())
    return rename_main(body, new_name), resources, manifest


def write_scalar(path: Path, value: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(np.array(value, dtype=np.float32)).tofile(path)


def copy_bin(src: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(repo_path(src).read_bytes())


def write_zeros(path: Path, shape: list[int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    np.zeros(count, dtype=np.float32).tofile(path)


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
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--per-step-inputs",
        action="store_true",
        help="Expose one raw image input per unrolled TTA step.",
    )
    args = parser.parse_args()
    args.train_artifacts = args.train_artifacts.resolve()
    args.update_artifacts = args.update_artifacts.resolve()
    args.aux_artifacts = args.aux_artifacts.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    train_body, train_resources, train_manifest = load_body(
        args.train_artifacts / "manifest.json", "paf_train_step"
    )
    update_body, update_resources, update_manifest = load_body(
        args.update_artifacts / "manifest.json", "sgd"
    )
    logits_body, logits_resources, logits_manifest = load_body(
        args.aux_artifacts / "logits" / "manifest.json", "logits"
    )
    tta_body, tta_resources, tta_manifest = load_body(
        args.aux_artifacts / "tta_views" / "manifest.json", "tta_views"
    )
    ema_body, ema_resources, ema_manifest = load_body(
        args.aux_artifacts / "ema" / "manifest.json", "ema"
    )
    kip_body, kip_resources, kip_manifest = load_body(
        args.aux_artifacts / "kip" / "manifest.json", "kip"
    )
    check_names(
        "train",
        train_manifest,
        ["train_images", "main_filter_images", "ema_filter_logits", "flat_bn_params"],
        ["loss", "flat_bn_grads"],
    )
    check_names(
        "tta_views",
        tta_manifest,
        ["raw_images"],
        ["train_images", "main_filter_images", "ema_filter_images", "anchor_images"],
    )
    check_names(
        "sgd",
        update_manifest,
        ["bn_params", "bn_grads", "velocity", "lr", "sgd_momentum"],
        ["new_bn_params", "new_velocity"],
    )
    check_names("logits", logits_manifest, ["images", "flat_bn_params"], ["logits"])
    check_names(
        "ema",
        ema_manifest,
        ["ema_bn_params", "main_bn_params", "ema_decay"],
        ["new_ema_bn_params"],
    )
    check_names(
        "kip",
        kip_manifest,
        ["main_logits", "ema_logits", "anchor_logits"],
        ["final_logits", "energy"],
    )
    for name, resources in {
        "tta_views": tta_resources,
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
    input_by_name = by_name(train_manifest["inputs"])
    train_outputs = by_name(train_manifest["outputs"])
    update_inputs = by_name(update_manifest["inputs"])
    update_outputs = by_name(update_manifest["outputs"])
    logits_inputs = by_name(logits_manifest["inputs"])
    logits_outputs = by_name(logits_manifest["outputs"])
    tta_inputs = by_name(tta_manifest["inputs"])
    tta_outputs = by_name(tta_manifest["outputs"])
    ema_inputs = by_name(ema_manifest["inputs"])
    ema_outputs = by_name(ema_manifest["outputs"])
    kip_inputs = by_name(kip_manifest["inputs"])
    kip_outputs = by_name(kip_manifest["outputs"])
    train_images_item = dict(input_by_name["train_images"])
    main_filter_images_item = dict(input_by_name["main_filter_images"])
    raw_images_item = dict(tta_inputs["raw_images"])
    main_item = dict(input_by_name["flat_bn_params"])
    loss_item = train_outputs["loss"]
    grad_item = train_outputs["flat_bn_grads"]
    logits_item = logits_outputs["logits"]
    energy_item = kip_outputs["energy"]
    final_item = kip_outputs["final_logits"]
    new_main_item = update_outputs["new_bn_params"]
    new_velocity_item = update_outputs["new_velocity"]
    new_ema_item = ema_outputs["new_ema_bn_params"]
    bn_shape = main_item["shape"]
    logits_shape = logits_item["shape"]
    check_same_shape(train_images_item, logits_inputs["images"], "train/logits image ABI")
    check_same_shape(main_filter_images_item, logits_inputs["images"], "filter/logits image ABI")
    check_same_shape(raw_images_item, train_images_item, "tta raw/train image ABI")
    check_same_shape(tta_outputs["train_images"], train_images_item, "tta train image ABI")
    check_same_shape(tta_outputs["main_filter_images"], main_filter_images_item, "tta main filter image ABI")
    check_same_shape(tta_outputs["ema_filter_images"], logits_inputs["images"], "tta ema image ABI")
    check_same_shape(tta_outputs["anchor_images"], logits_inputs["images"], "tta anchor image ABI")
    check_same_shape(main_item, logits_inputs["flat_bn_params"], "train/logits BN ABI")
    check_same_shape(main_item, update_inputs["bn_params"], "train/sgd BN ABI")
    check_same_shape(grad_item, update_inputs["bn_grads"], "train/sgd grad ABI")
    check_same_shape(main_item, update_inputs["velocity"], "train/sgd velocity ABI")
    check_same_shape(main_item, update_outputs["new_bn_params"], "sgd output ABI")
    check_same_shape(main_item, update_outputs["new_velocity"], "sgd velocity output ABI")
    check_same_shape(main_item, ema_inputs["ema_bn_params"], "ema input ABI")
    check_same_shape(main_item, ema_inputs["main_bn_params"], "ema input ABI")
    check_same_shape(main_item, ema_outputs["new_ema_bn_params"], "ema output ABI")
    check_same_shape(input_by_name["ema_filter_logits"], logits_item, "train/logits output ABI")
    check_same_shape(logits_item, kip_inputs["main_logits"], "kip main logits ABI")
    check_same_shape(logits_item, kip_inputs["ema_logits"], "kip ema logits ABI")
    check_same_shape(logits_item, kip_inputs["anchor_logits"], "kip anchor logits ABI")
    check_same_shape(logits_item, kip_outputs["final_logits"], "kip final logits ABI")
    check_shape(loss_item, [], "train loss")
    check_shape(update_inputs["lr"], [], "sgd lr")
    check_shape(update_inputs["sgd_momentum"], [], "sgd momentum")
    check_shape(ema_inputs["ema_decay"], [], "ema decay")
    if train_manifest["bn_scalar_count"] != bn_shape[0]:
        raise ValueError(
            "bn_scalar_count mismatch: "
            f"{train_manifest['bn_scalar_count']} != flat BN shape {bn_shape}"
        )

    image_type = torch_vtensor(train_images_item)
    bn_type = torch_vtensor(main_item)
    logits_type = torch_vtensor(logits_item)
    scalar_type = torch_vtensor(loss_item)
    energy_type = torch_vtensor(energy_item)
    ema_path = input_dir / "ema_bn_params.bin"
    anchor_path = input_dir / "anchor_bn_params.bin"
    velocity_path = input_dir / "sgd_velocity.bin"
    lr_path = input_dir / "lr.bin"
    sgd_momentum_path = input_dir / "sgd_momentum.bin"
    ema_decay_path = input_dir / "ema_decay.bin"
    raw_image_paths = []
    if args.per_step_inputs:
        raw = np.fromfile(repo_path(raw_images_item["file"]), dtype=np.float32).reshape(raw_images_item["shape"])
        for step in range(args.steps):
            path = input_dir / f"raw_images_{step}.bin"
            # Deterministic synthetic stream for stage validation. Real ImageNet-C
            # loaders can replace these files without changing the ABI.
            shifted = np.roll(raw, shift=step, axis=3) + np.float32(step * 1.0e-3)
            shifted.astype(np.float32).tofile(path)
            raw_image_paths.append(path)
    else:
        raw_images_path = input_dir / "raw_images.bin"
        copy_bin(raw_images_item["file"], raw_images_path)
        raw_image_paths.append(raw_images_path)
    copy_bin(main_item["file"], ema_path)
    copy_bin(main_item["file"], anchor_path)
    write_zeros(velocity_path, bn_shape)
    write_scalar(lr_path, args.lr)
    write_scalar(sgd_momentum_path, args.sgd_momentum)
    write_scalar(ema_decay_path, args.ema_decay)

    raw_arg_count = args.steps if args.per_step_inputs else 1
    main_arg = f"%arg{raw_arg_count}"
    ema_arg = f"%arg{raw_arg_count + 1}"
    anchor_arg = f"%arg{raw_arg_count + 2}"
    velocity_arg = f"%arg{raw_arg_count + 3}"
    lr_arg = f"%arg{raw_arg_count + 4}"
    sgd_momentum_arg = f"%arg{raw_arg_count + 5}"
    ema_decay_arg = f"%arg{raw_arg_count + 6}"
    function_args = [
        f"%arg{i}: {image_type}" for i in range(raw_arg_count)
    ] + [
        f"{main_arg}: {bn_type}",
        f"{ema_arg}: {bn_type}",
        f"{anchor_arg}: {bn_type}",
        f"{velocity_arg}: {bn_type}",
        f"{lr_arg}: {scalar_type}",
        f"{sgd_momentum_arg}: {scalar_type}",
        f"{ema_decay_arg}: {scalar_type}",
    ]

    raw_for_step0 = "%arg0"
    step_lines = [
        f"    %views0:4 = func.call @tta_views({raw_for_step0}) : ({image_type}) -> ({image_type}, {image_type}, {image_type}, {image_type})",
        f"    %main_logits0 = func.call @logits(%views0#0, {main_arg}) : ({image_type}, {bn_type}) -> {logits_type}",
        f"    %ema_filter_logits0 = func.call @logits(%views0#2, {ema_arg}) : ({image_type}, {bn_type}) -> {logits_type}",
        f"    %anchor_logits0 = func.call @logits(%views0#3, {anchor_arg}) : ({image_type}, {bn_type}) -> {logits_type}",
        f"    %train0:2 = func.call @paf_train_step(%views0#0, %views0#1, %ema_filter_logits0, {main_arg}) : ({image_type}, {image_type}, {logits_type}, {bn_type}) -> ({scalar_type}, {bn_type})",
        f"    %final0:2 = func.call @kip(%main_logits0, %ema_filter_logits0, %anchor_logits0) : ({logits_type}, {logits_type}, {logits_type}) -> ({logits_type}, {energy_type})",
        f"    %sgd0:2 = func.call @sgd({main_arg}, %train0#1, {velocity_arg}, {lr_arg}, {sgd_momentum_arg}) : ({bn_type}, {bn_type}, {bn_type}, {scalar_type}, {scalar_type}) -> ({bn_type}, {bn_type})",
        f"    %ema1 = func.call @ema({ema_arg}, %sgd0#0, {ema_decay_arg}) : ({bn_type}, {bn_type}, {scalar_type}) -> {bn_type}",
    ]
    last_main = "%sgd0#0"
    last_ema = "%ema1"
    last_velocity = "%sgd0#1"
    last_final = "%final0#0"
    last_energy = "%final0#1"
    last_loss = "%train0#0"
    last_grad = "%train0#1"
    for step in range(1, args.steps):
        raw_for_step = f"%arg{step}" if args.per_step_inputs else "%arg0"
        step_lines.append(
            f"    %views{step}:4 = func.call @tta_views({raw_for_step}) : ({image_type}) -> ({image_type}, {image_type}, {image_type}, {image_type})"
        )
        step_lines.extend(
            [
                f"    %main_logits{step} = func.call @logits(%views{step}#0, {last_main}) : ({image_type}, {bn_type}) -> {logits_type}",
                f"    %ema_filter_logits{step} = func.call @logits(%views{step}#2, {last_ema}) : ({image_type}, {bn_type}) -> {logits_type}",
                f"    %anchor_logits{step} = func.call @logits(%views{step}#3, {anchor_arg}) : ({image_type}, {bn_type}) -> {logits_type}",
                f"    %train{step}:2 = func.call @paf_train_step(%views{step}#0, %views{step}#1, %ema_filter_logits{step}, {last_main}) : ({image_type}, {image_type}, {logits_type}, {bn_type}) -> ({scalar_type}, {bn_type})",
                f"    %final{step}:2 = func.call @kip(%main_logits{step}, %ema_filter_logits{step}, %anchor_logits{step}) : ({logits_type}, {logits_type}, {logits_type}) -> ({logits_type}, {energy_type})",
                f"    %sgd{step}:2 = func.call @sgd({last_main}, %train{step}#1, {last_velocity}, {lr_arg}, {sgd_momentum_arg}) : ({bn_type}, {bn_type}, {bn_type}, {scalar_type}, {scalar_type}) -> ({bn_type}, {bn_type})",
                f"    %ema{step + 1} = func.call @ema({last_ema}, %sgd{step}#0, {ema_decay_arg}) : ({bn_type}, {bn_type}, {scalar_type}) -> {bn_type}",
            ]
        )
        last_main = f"%sgd{step}#0"
        last_ema = f"%ema{step + 1}"
        last_velocity = f"%sgd{step}#1"
        last_final = f"%final{step}#0"
        last_energy = f"%final{step}#1"
        last_loss = f"%train{step}#0"
        last_grad = f"%train{step}#1"

    loop_body = f"""
  func.func @main({", ".join(function_args)}) -> ({logits_type}, {energy_type}, {scalar_type}, {bn_type}, {bn_type}, {bn_type}, {bn_type}) {{
{chr(10).join(step_lines)}
    return {last_final}, {last_energy}, {last_loss}, {last_main}, {last_ema}, {last_velocity}, {last_grad} : {logits_type}, {energy_type}, {scalar_type}, {bn_type}, {bn_type}, {bn_type}, {bn_type}
  }}
"""
    mlir_path = args.out_dir / "full.mlir"
    resource_suffix = "\n" + train_resources + "\n"
    mlir_path.write_text(
        "module {\n"
        + train_body
        + "\n"
        + tta_body
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

    raw_inputs = []
    if args.per_step_inputs:
        for step, path in enumerate(raw_image_paths):
            item = dict(raw_images_item)
            item["name"] = f"raw_images_{step}"
            item["file"] = repo_relative(path)
            raw_inputs.append(item)
    else:
        raw_images_item["name"] = "raw_images"
        raw_images_item["file"] = repo_relative(raw_image_paths[0])
        raw_inputs.append(raw_images_item)
    main_item["name"] = "main_bn_params"
    main_item["file"] = repo_relative(repo_path(main_item["file"]))
    manifest = {
        "name": "full",
        "function": "main",
        "mlir": repo_relative(mlir_path),
        "inputs": [
            *raw_inputs,
            main_item,
            {
                "name": "ema_bn_params",
                "shape": bn_shape,
                "dtype": "f32",
                "iree": main_item["iree"],
                "file": repo_relative(ema_path),
            },
            {
                "name": "anchor_bn_params",
                "shape": bn_shape,
                "dtype": "f32",
                "iree": main_item["iree"],
                "file": repo_relative(anchor_path),
            },
            {
                "name": "sgd_velocity",
                "shape": bn_shape,
                "dtype": "f32",
                "iree": main_item["iree"],
                "file": repo_relative(velocity_path),
            },
            {"name": "lr", "shape": [], "dtype": "f32", "iree": "f32", "file": repo_relative(lr_path)},
            {
                "name": "sgd_momentum",
                "shape": [],
                "dtype": "f32",
                "iree": "f32",
                "file": repo_relative(sgd_momentum_path),
            },
            {
                "name": "ema_decay",
                "shape": [],
                "dtype": "f32",
                "iree": "f32",
                "file": repo_relative(ema_decay_path),
            },
        ],
        "outputs": [
            {"name": "final_logits", "shape": logits_shape, "dtype": "f32", "iree": final_item["iree"]},
            {"name": "energy", "shape": energy_item["shape"], "dtype": "f32", "iree": energy_item["iree"]},
            {"name": "loss", "shape": [], "dtype": "f32", "iree": "f32"},
            {
                "name": "new_main_bn_params",
                "shape": new_main_item["shape"],
                "dtype": "f32",
                "iree": new_main_item["iree"],
            },
            {
                "name": "new_ema_bn_params",
                "shape": new_ema_item["shape"],
                "dtype": "f32",
                "iree": new_ema_item["iree"],
            },
            {
                "name": "new_sgd_velocity",
                "shape": new_velocity_item["shape"],
                "dtype": "f32",
                "iree": new_velocity_item["iree"],
            },
            {"name": "flat_bn_grads", "shape": grad_item["shape"], "dtype": "f32", "iree": grad_item["iree"]},
        ],
        "steps": args.steps,
        "lr": args.lr,
        "sgd_momentum": args.sgd_momentum,
        "ema_decay": args.ema_decay,
        "per_step_inputs": args.per_step_inputs,
        "transform": tta_manifest.get("transform"),
        "transform_seed": tta_manifest.get("transform_seed"),
        "transform_specs": tta_manifest.get("transform_specs"),
        "bn_param_count": train_manifest["bn_param_count"],
        "bn_scalar_count": train_manifest["bn_scalar_count"],
        "llvmcpu_vector_pproc_strategy": "none",
        "llvmcpu_stack_allocation_limit": 1048576,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(mlir_path)


if __name__ == "__main__":
    main()
