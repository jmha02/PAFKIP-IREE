#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from paths import repo_path


FULL_LOOP_ARTIFACTS = THIS_DIR / "artifact_full"
TRAIN_STEP_ARTIFACTS = THIS_DIR / "artifact_train"
UPDATE_ARTIFACTS = THIS_DIR / "artifact_sgd"
AUX_ARTIFACTS = THIS_DIR / "artifact_aux"
FORWARD_ARTIFACTS = AUX_ARTIFACTS / "logits"
FLEXI_SMOKE_ARTIFACTS = AUX_ARTIFACTS / "flexi_matmul"
STAGE_ARTIFACTS = {
    "tta_views": AUX_ARTIFACTS / "tta_views",
    "logits": FORWARD_ARTIFACTS,
    "ema": AUX_ARTIFACTS / "ema",
    "kip": AUX_ARTIFACTS / "kip",
    "sgd": UPDATE_ARTIFACTS,
    "train": TRAIN_STEP_ARTIFACTS,
    "flexi_matmul": FLEXI_SMOKE_ARTIFACTS,
    "full": FULL_LOOP_ARTIFACTS,
}


def torch_mlir_package_path() -> Path:
    return THIS_DIR / "build-torch-mlir" / "tools" / "torch-mlir" / "python_packages" / "torch_mlir"


def command_env(extra_env=None):
    env = os.environ.copy()
    package_path = torch_mlir_package_path()
    if package_path.exists():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(package_path) if not existing else f"{package_path}:{existing}"
    if extra_env:
        env.update(extra_env)
    return env


def run_cmd(cmd: list[str]):
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=THIS_DIR, env=command_env(), check=True)


def run_capture(cmd: list[str], *, env=None, allow_timeout: bool = False):
    print("+ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd,
        cwd=THIS_DIR,
        env=command_env(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if allow_timeout and result.returncode == 124:
        print(result.stdout)
        print("[timeout] command reached the requested time limit")
        return result
    if result.returncode != 0:
        print(result.stdout)
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
    return result


def script(name: str) -> str:
    return str(SRC_DIR / name)


def has_manifest(path: Path) -> bool:
    return (path / "manifest.json").exists()


def prepare(args):
    if args.force or not has_manifest(TRAIN_STEP_ARTIFACTS):
        run_cmd(
            [
                sys.executable,
                script("export_train.py"),
                "--out-dir",
                str(TRAIN_STEP_ARTIFACTS),
                "--image-size",
                str(args.image_size),
                "--classes",
                str(args.classes),
                "--weights",
                args.weights,
                "--lr",
                str(args.lr),
                "--dtype",
                args.dtype,
                "--npu-dtype",
                args.npu_dtype,
            ]
        )
    if args.force or not has_manifest(UPDATE_ARTIFACTS):
        run_cmd(
            [
                sys.executable,
                script("export_sgd.py"),
                "--out-dir",
                str(UPDATE_ARTIFACTS),
                "--image-size",
                str(args.image_size),
                "--classes",
                str(args.classes),
                "--weights",
                args.weights,
                "--lr",
                str(args.lr),
                "--dtype",
                args.dtype,
            ]
        )
    if args.force or not has_manifest(AUX_ARTIFACTS / "logits"):
        run_cmd(
            [
                sys.executable,
                script("export_aux.py"),
                "--out-dir",
                str(AUX_ARTIFACTS),
                "--image-size",
                str(args.image_size),
                "--classes",
                str(args.classes),
                "--weights",
                args.weights,
                "--ema-decay",
                str(args.ema_decay),
                "--transform-seed",
                str(args.transform_seed),
                "--dtype",
                args.dtype,
                "--npu-dtype",
                args.npu_dtype,
            ]
        )
    if args.force or not has_manifest(FULL_LOOP_ARTIFACTS):
        run_cmd(
            [
                sys.executable,
                script("compose.py"),
                "--train-artifacts",
                str(TRAIN_STEP_ARTIFACTS),
                "--update-artifacts",
                str(UPDATE_ARTIFACTS),
                "--aux-artifacts",
                str(AUX_ARTIFACTS),
                "--out-dir",
                str(FULL_LOOP_ARTIFACTS),
                "--steps",
                str(args.steps),
                "--lr",
                str(args.lr),
                "--sgd-momentum",
                str(args.sgd_momentum),
                "--ema-decay",
                str(args.ema_decay),
            ]
        )
    print(FULL_LOOP_ARTIFACTS)


def prepare_flexi_smoke(args):
    out_dir = args.out_dir or (AUX_ARTIFACTS / f"flexi_{args.kind}_{args.dtype}")
    run_cmd(
        [
            sys.executable,
            script("export_flexi_smoke.py"),
            "--out-dir",
            str(out_dir),
            "--kind",
            args.kind,
            "--dtype",
            args.dtype,
            *(["--out-dtype", args.out_dtype] if args.out_dtype else []),
            "--seed",
            str(args.seed),
            "--m",
            str(args.m),
            "--n",
            str(args.n),
            "--k",
            str(args.k),
            "--init-value",
            str(args.init_value),
        ]
    )
    print(out_dir)


def prepare_aux(args):
    run_cmd(
        [
            sys.executable,
            script("export_aux.py"),
            "--out-dir",
            str(args.out_dir),
            "--image-size",
            str(args.image_size),
            "--classes",
            str(args.classes),
            "--weights",
            args.weights,
            "--dtype",
            args.dtype,
            "--npu-dtype",
            args.npu_dtype,
            "--ema-decay",
            str(args.ema_decay),
            "--transform-seed",
            str(args.transform_seed),
        ]
    )
    print(args.out_dir)


def load_manifest(artifacts: Path = FULL_LOOP_ARTIFACTS) -> dict:
    manifest_path = artifacts / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"artifacts are missing: {manifest_path}")
    return json.loads(manifest_path.read_text())


def default_runtime_root() -> Path:
    baremetal = THIS_DIR / "third_party" / "iree" / "build-riscv-baremetal"
    if baremetal.exists():
        return baremetal
    return THIS_DIR / "third_party" / "iree" / "build-riscv-pk"


def compile_target(args):
    from iree_run import compile_module

    manifest = load_manifest()
    targets = ["host", "saturn"] if args.target == "both" else [args.target]
    for target in targets:
        vmfb = compile_module(manifest, FULL_LOOP_ARTIFACTS, target)
        print(vmfb)


def summarize_outputs(output_dir: Path):
    import numpy as np

    arrays = {
        "final_logits": np.fromfile(output_dir / "final_logits.bin", dtype=np.float32),
        "energy": np.fromfile(output_dir / "energy.bin", dtype=np.float32),
        "loss": np.fromfile(output_dir / "loss.bin", dtype=np.float32),
        "new_main_bn_params": np.fromfile(output_dir / "new_main_bn_params.bin", dtype=np.float32),
        "new_ema_bn_params": np.fromfile(output_dir / "new_ema_bn_params.bin", dtype=np.float32),
        "new_sgd_velocity": np.fromfile(output_dir / "new_sgd_velocity.bin", dtype=np.float32),
        "flat_bn_grads": np.fromfile(output_dir / "flat_bn_grads.bin", dtype=np.float32),
    }
    summarize_full_outputs(arrays, load_manifest())


OUTPUT_RE = re.compile(
    r"^\[IREE\]\[output\s+(\d+)(?:\s+[^\]]+)?\]\s+shape=.*\s+dtype=([A-Za-z0-9]+)\s+elements=(\d+)"
)
VALUE_RE = re.compile(r"^\s*\[\s*(\d+)\]\s+([-+a-zA-Z0-9.]+)\s*$")


def parse_baremetal_outputs(text: str, manifest: dict) -> dict[str, "np.ndarray"]:
    import numpy as np

    values_by_index: dict[int, dict[int, float]] = {}
    expected_counts: dict[int, int] = {}
    current = None
    for line in text.splitlines():
        header = OUTPUT_RE.match(line)
        if header:
            current = int(header.group(1))
            expected_counts[current] = int(header.group(3))
            values_by_index[current] = {}
            continue
        match = VALUE_RE.match(line)
        if match and current is not None:
            raw_value = match.group(2).lower()
            if raw_value in ("nan", "+nan", "-nan"):
                value = float("nan")
            elif raw_value in ("inf", "+inf"):
                value = float("inf")
            elif raw_value == "-inf":
                value = float("-inf")
            else:
                value = float(raw_value)
            values_by_index[current][int(match.group(1))] = value

    parsed = {}
    for index, item in enumerate(manifest["outputs"]):
        if index not in values_by_index:
            raise RuntimeError(f"missing baremetal output {index} ({item['name']})")
        expected = expected_counts[index]
        values = values_by_index[index]
        if len(values) != expected:
            raise RuntimeError(
                f"output {index} ({item['name']}) printed {len(values)} values, "
                f"expected {expected}. Rebuild the ELF with `--output-print-limit 0` "
                "before using `--summary`."
            )
        parsed[item["name"]] = np.array([values[i] for i in range(expected)], dtype=np.float32)
    return parsed


def _stats(values):
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "l2": None}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "l2": float(np.linalg.norm(finite)),
    }


def summarize_full_outputs(arrays: dict, manifest: dict):
    import numpy as np

    required = [
        "final_logits",
        "energy",
        "loss",
        "new_main_bn_params",
        "new_ema_bn_params",
        "new_sgd_velocity",
        "flat_bn_grads",
    ]
    missing = [name for name in required if name not in arrays]
    if missing:
        raise RuntimeError(f"cannot summarize full output; missing {missing}")

    final_logits = arrays["final_logits"].reshape(-1)
    energy = arrays["energy"].reshape(-1)
    loss = arrays["loss"].reshape(-1)
    new_main = arrays["new_main_bn_params"].reshape(-1)
    new_ema = arrays["new_ema_bn_params"].reshape(-1)
    new_velocity = arrays["new_sgd_velocity"].reshape(-1)
    grad = arrays["flat_bn_grads"].reshape(-1)
    main = np.fromfile(repo_path("artifact_train/inputs/flat_bn_params.bin"), dtype=np.float32)
    velocity = np.fromfile(repo_path("artifact_full/inputs/sgd_velocity.bin"), dtype=np.float32)
    lr = float(np.fromfile(repo_path("artifact_full/inputs/lr.bin"), dtype=np.float32)[0])
    sgd_momentum = float(np.fromfile(repo_path("artifact_full/inputs/sgd_momentum.bin"), dtype=np.float32)[0])
    ema_decay = float(np.fromfile(repo_path("artifact_full/inputs/ema_decay.bin"), dtype=np.float32)[0])

    top_count = min(5, final_logits.size)
    top_indices = np.argsort(final_logits)[-top_count:][::-1]
    main_delta = new_main - main
    ema_delta = new_ema - main

    summary = {
        "steps": int(manifest.get("steps", 1)),
        "npu_dtype": manifest.get("npu_dtype", "f32"),
        "prediction": {
            "top1": int(top_indices[0]) if top_indices.size else None,
            "top5": [
                {"class": int(i), "logit": float(final_logits[i])}
                for i in top_indices
            ],
            "energy_logsumexp": float(energy[0]) if energy.size else None,
        },
        "tta_objective": {
            "loss": float(loss[0]) if loss.size else None,
            "note": (
                "PAFKIP uses loss_ind - alpha_ood * loss_ood, not cross entropy; "
                "negative values are possible when the OOD entropy-maximization term dominates."
            ),
        },
        "bn_training_state": {
            "bn_scalars": int(main.size),
            "changed_main_bn_scalars": int(np.count_nonzero(main_delta)),
            "changed_ema_bn_scalars": int(np.count_nonzero(ema_delta)),
            "main_delta": _stats(main_delta),
            "ema_delta": _stats(ema_delta),
            "grad": {
                **_stats(grad),
                "nonzero": int(np.count_nonzero(grad)),
            },
            "velocity": {
                **_stats(new_velocity),
                "nonzero": int(np.count_nonzero(new_velocity)),
            },
        },
    }
    if summary["steps"] == 1:
        expected_velocity = sgd_momentum * velocity + grad
        expected_main = main - lr * expected_velocity
        expected_ema = ema_decay * main + (1.0 - ema_decay) * expected_main
        summary["update_formula_checks"] = {
            "lr": lr,
            "sgd_momentum": sgd_momentum,
            "ema_decay": ema_decay,
            "velocity_max_abs_error": float(np.max(np.abs(new_velocity - expected_velocity))),
            "main_bn_update_max_abs_error": float(np.max(np.abs(new_main - expected_main))),
            "ema_update_max_abs_error": float(np.max(np.abs(new_ema - expected_ema))),
        }
    else:
        summary["update_formula_checks"] = (
            "skipped: multi-step recurrence needs per-step reference states"
        )
    print(json.dumps(summary, indent=2))


def run_target(args):
    from iree_run import compile_module, run_host

    manifest = load_manifest()
    vmfb = compile_module(manifest, FULL_LOOP_ARTIFACTS, args.target)
    if args.target == "host":
        run_host(vmfb, manifest, FULL_LOOP_ARTIFACTS)
        summarize_outputs(FULL_LOOP_ARTIFACTS / "host_outputs")


def _baremetal_input_args(manifest: dict) -> list[str]:
    args = []
    for item in manifest["inputs"]:
        spec = item["iree"]
        file_path = repo_path(item["file"])
        args.extend(["--input", f"{spec}=@{file_path}"])
    for item in manifest["outputs"]:
        args.extend(["--output-spec", item["iree"]])
    return args


def baremetal_build(args):
    from iree_run import compile_module

    artifacts = args.artifacts
    if not artifacts.is_absolute():
        artifacts = THIS_DIR / artifacts
    manifest = load_manifest(artifacts)
    target = getattr(args, "target", "saturn")
    vmfb = artifacts / f"{manifest.get('name', 'module')}_{target}.vmfb"
    if not args.reuse_vmfb or not vmfb.exists():
        vmfb = compile_module(manifest, artifacts, target)
    output_suffix = "baremetal" if target == "saturn" else f"{target}_baremetal"
    default_output = artifacts / f"{manifest.get('name', 'module')}_{output_suffix}.elf"
    output = args.output or default_output
    if not output.is_absolute():
        output = THIS_DIR / output
    runtime_root = args.runtime_root or default_runtime_root()
    toolchain_root = args.toolchain_root or os.environ.get("RISCV")
    if not toolchain_root:
        raise FileNotFoundError("RISCV toolchain root is missing; set $RISCV or pass --toolchain-root")

    bundle_tool = THIS_DIR / "third_party" / "iree" / "runtime" / "tools" / "iree-bundle-baremetal"
    cmd = [
        str(bundle_tool),
        "--module",
        str(vmfb),
        "--function",
        manifest["function"],
        "--runtime-root",
        str(runtime_root),
        "--toolchain-root",
        str(toolchain_root),
        "--march",
        args.march,
        "--mabi",
        args.mabi,
        "--output-print-limit",
        str(args.output_print_limit),
        "--output",
        str(output),
        "--runner",
        args.runner,
        *_baremetal_input_args(manifest),
    ]
    if getattr(args, "stack_shift", None) is not None:
        cmd.extend(["--stack-shift", str(args.stack_shift)])
    if args.optimize_size:
        cmd.append("--optimize-size")
    run_cmd(cmd)
    print(output)


def baremetal_run(args):
    from iree_run import spike_extension_for_target

    artifacts = args.artifacts
    if not artifacts.is_absolute():
        artifacts = THIS_DIR / artifacts
    manifest = load_manifest(artifacts)
    target = getattr(args, "target", "saturn")
    output_suffix = "baremetal" if target == "saturn" else f"{target}_baremetal"
    default_elf = artifacts / f"{manifest.get('name', 'module')}_{output_suffix}.elf"
    elf = args.elf or default_elf
    if not elf.is_absolute():
        elf = THIS_DIR / elf
    if args.rebuild or not elf.exists():
        baremetal_build(args)
    spike = shutil.which("spike")
    toolchain_root = args.toolchain_root or os.environ.get("RISCV")
    if not spike and toolchain_root:
        candidate = Path(toolchain_root) / "bin" / "spike"
        if candidate.exists():
            spike = str(candidate)
    if not spike:
        raise FileNotFoundError("spike not found in PATH; source the RISC-V/Saturn environment first")
    isa = (
        "rv64gc_zicsr_zifencei_zicntr_zihpm"
        if manifest.get("riscv_features") == "scalar"
        else "rv64gcv_zvl128b_zicsr_zifencei_zicntr_zihpm"
    )
    cmd = [spike]
    extension = spike_extension_for_target(target, manifest)
    if extension:
        cmd.append(f"--extension={extension}")
    cmd.extend(["-m4096", f"--isa={isa}", str(elf)])
    if args.timeout:
        cmd = ["timeout", str(args.timeout), *cmd]
    env = os.environ.copy()
    path_prefixes = []
    if toolchain_root:
        path_prefixes.append(str(Path(toolchain_root) / "bin"))
        chipyard_env_bin = Path(toolchain_root).parent / "bin"
        if chipyard_env_bin.exists():
            path_prefixes.append(str(chipyard_env_bin))
    if path_prefixes:
        env["PATH"] = os.pathsep.join(path_prefixes + [env.get("PATH", "")])
    result = run_capture(cmd, env=env, allow_timeout=bool(args.timeout))
    if not (args.timeout and result.returncode == 124):
        print(result.stdout)
        if args.summary:
            summarize_full_outputs(parse_baremetal_outputs(result.stdout, manifest), manifest)


def forward_build(args):
    args.artifacts = FORWARD_ARTIFACTS
    args.reuse_vmfb = not args.recompile
    baremetal_build(args)


def forward_run(args):
    args.artifacts = FORWARD_ARTIFACTS
    args.reuse_vmfb = not args.recompile
    baremetal_run(args)


def verify_host(args):
    cmd = [sys.executable, str(THIS_DIR / "tools" / "verify_stages.py")]
    if args.only:
        cmd.extend(["--only", *args.only])
    if args.skip_heavy:
        cmd.append("--skip-heavy")
    if args.target:
        cmd.extend(["--target", args.target])
    if args.reuse_vmfb:
        cmd.append("--reuse-vmfb")
    if args.compile_only:
        cmd.append("--compile-only")
    cmd.extend(["--atol", str(args.atol), "--rtol", str(args.rtol)])
    if args.single_tolerance:
        cmd.append("--single-tolerance")
    if args.json:
        cmd.extend(["--json", str(args.json)])
    run_cmd(cmd)


def verify_spike(args):
    stage_dir = STAGE_ARTIFACTS[args.stage]
    cmd = [
        sys.executable,
        str(THIS_DIR / "tools" / "verify_spike_stage.py"),
        str(stage_dir),
        "--target",
        args.target,
        "--runner",
        args.runner,
        "--print-limit",
        str(args.print_limit),
        "--timeout",
        str(args.timeout),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
    ]
    if args.elf:
        cmd.extend(["--elf", str(args.elf)])
    if args.log:
        cmd.extend(["--log", str(args.log)])
    if args.json:
        cmd.extend(["--json", str(args.json)])
    if args.reuse_vmfb:
        cmd.append("--reuse-vmfb")
    if args.skip_build:
        cmd.append("--skip-build")
    if args.skip_run:
        cmd.append("--skip-run")
    run_cmd(cmd)


def add_common_build_args(parser):
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--transform-seed", type=int, default=91)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--weights", choices=["none", "default"], default="none")
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    parser.add_argument("--npu-dtype", choices=["f32", "f16", "bf16"], default="f32")
    parser.add_argument("--force", action="store_true")


def add_baremetal_build_args(parser):
    parser.add_argument("--target", choices=["saturn", "flexinpu"], default="saturn")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--toolchain-root", type=Path)
    parser.add_argument("--march", default="rv64gcv_zvl128b")
    parser.add_argument("--mabi", default="lp64d")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-print-limit", type=int, default=16)
    parser.add_argument("--runner", choices=["direct", "tooling"], default="direct")
    parser.add_argument("--optimize-size", action="store_true")
    parser.add_argument(
        "--stack-shift",
        type=int,
        help="Override baremetal CRT stack/heap window size as log2(bytes).",
    ) # Verilator 상에서 DRAM 크기 조정 필요할 때


def add_baremetal_run_args(parser):
    add_baremetal_build_args(parser)
    parser.add_argument("--elf", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--timeout", type=int, default=0, help="optional timeout seconds for Spike")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Parse full printed outputs and summarize the PAFKIP training update.",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Single entrypoint for the ResNet50 PAFKIP-style IREE/Saturn TTA prototype."
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("prepare", help="export/compose the full all-BN EMA+KIP device loop")
    add_common_build_args(p)
    p.set_defaults(func=prepare)

    p = sub.add_parser("prepare-flexi-smoke", help="export a small f32/f16/bf16 Flexi-NPU matmul or conv stage")
    p.add_argument("--kind", choices=["matmul", "conv"], default="matmul")
    p.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="bf16")
    p.add_argument("--out-dtype", choices=["f32", "f16", "bf16"])
    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--init-value", type=float, default=0.0)
    p.add_argument("--out-dir", type=Path)
    p.set_defaults(func=prepare_flexi_smoke)

    p = sub.add_parser("prepare-aux", help="export aux stages, optionally in f16 for Flexi-NPU bring-up")
    p.add_argument("--out-dir", type=Path, default=AUX_ARTIFACTS)
    add_common_build_args(p)
    p.set_defaults(func=prepare_aux)

    p = sub.add_parser("compile", help="compile the full loop VMFB")
    p.add_argument("--target", choices=["host", "saturn", "flexinpu", "both"], default="host")
    p.set_defaults(func=compile_target)

    p = sub.add_parser("run", help="compile and run the full loop on the host")
    p.add_argument("--target", choices=["host"], default="host")
    p.set_defaults(func=run_target)

    p = sub.add_parser("forward-build", help="bundle ResNet50 forward logits into a baremetal ELF")
    add_baremetal_build_args(p)
    p.add_argument("--recompile", action="store_true", help="recompile the logits VMFB before bundling")
    p.set_defaults(func=forward_build)

    p = sub.add_parser("forward-run", help="run the ResNet50 forward-only baremetal ELF on Spike")
    add_baremetal_run_args(p)
    p.add_argument("--recompile", action="store_true", help="recompile the logits VMFB before bundling")
    p.set_defaults(func=forward_run)

    p = sub.add_parser("verify-host", help="verify decomposed stages with host IREE")
    p.add_argument("--only", nargs="+", choices=sorted(k for k in STAGE_ARTIFACTS if k != "full"))
    p.add_argument("--skip-heavy", action="store_true", help="skip ResNet logits/train stages")
    p.add_argument("--target", choices=["host", "saturn", "flexinpu", "both"], default="host")
    p.add_argument("--reuse-vmfb", action="store_true")
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--atol", type=float, default=5.0e-4)
    p.add_argument("--rtol", type=float, default=5.0e-4)
    p.add_argument("--single-tolerance", action="store_true")
    p.add_argument("--json", type=Path)
    p.set_defaults(func=verify_host)

    p = sub.add_parser("verify-spike", help="verify one stage on pk-free Spike baremetal")
    p.add_argument("stage", choices=sorted(k for k in STAGE_ARTIFACTS if k != "full"))
    p.add_argument("--target", choices=["saturn", "flexinpu"], default="saturn")
    p.add_argument("--runner", choices=["tooling", "direct"], default="direct")
    p.add_argument("--print-limit", type=int, default=0)
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--reuse-vmfb", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-run", action="store_true")
    p.add_argument("--elf", type=Path)
    p.add_argument("--log", type=Path)
    p.add_argument("--atol", type=float, default=5.0e-4)
    p.add_argument("--rtol", type=float, default=5.0e-4)
    p.add_argument("--json", type=Path)
    p.set_defaults(func=verify_spike)

    p = sub.add_parser("baremetal", help="build or run the pk-free Spike baremetal ELF")
    bare_sub = p.add_subparsers(dest="baremetal_cmd", required=True)

    b = bare_sub.add_parser("build", help="bundle a saturn VMFB and inputs into a baremetal ELF")
    b.add_argument("--artifacts", type=Path, default=FULL_LOOP_ARTIFACTS)
    add_baremetal_build_args(b)
    b.add_argument("--reuse-vmfb", action="store_true")
    b.set_defaults(func=baremetal_build)

    b = bare_sub.add_parser("run", help="run the baremetal ELF on Spike without pk")
    b.add_argument("--artifacts", type=Path, default=FULL_LOOP_ARTIFACTS)
    add_baremetal_run_args(b)
    b.add_argument("--reuse-vmfb", action="store_true")
    b.set_defaults(func=baremetal_run)

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
