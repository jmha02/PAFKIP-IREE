#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = THIS_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


FULL_LOOP_ARTIFACTS = THIS_DIR / "artifact_full"
TRAIN_STEP_ARTIFACTS = THIS_DIR / "artifact_train"
UPDATE_ARTIFACTS = THIS_DIR / "artifact_sgd"
AUX_ARTIFACTS = THIS_DIR / "artifact_aux"


def run_cmd(cmd: list[str]):
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


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
                "--momentum",
                str(args.momentum),
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
                "--momentum",
                str(args.momentum),
            ]
        )
    print(FULL_LOOP_ARTIFACTS)


def load_manifest() -> dict:
    manifest_path = FULL_LOOP_ARTIFACTS / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("full-loop artifacts are missing; run `python3 PAFKIP-IREE/run.py prepare` first")
    return json.loads(manifest_path.read_text())


def compile_target(args):
    from iree_run import compile_module

    manifest = load_manifest()
    targets = ["host", "saturn"] if args.target == "both" else [args.target]
    for target in targets:
        vmfb = compile_module(manifest, FULL_LOOP_ARTIFACTS, target)
        print(vmfb)


def summarize_outputs(output_dir: Path):
    import numpy as np

    final_logits = np.fromfile(output_dir / "final_logits.bin", dtype=np.float32)
    energy = np.fromfile(output_dir / "energy.bin", dtype=np.float32)
    loss = np.fromfile(output_dir / "loss.bin", dtype=np.float32)
    new_main = np.fromfile(output_dir / "new_main_bn_params.bin", dtype=np.float32)
    new_ema = np.fromfile(output_dir / "new_ema_bn_params.bin", dtype=np.float32)
    grad = np.fromfile(output_dir / "flat_bn_grads.bin", dtype=np.float32)
    main = np.fromfile(TRAIN_STEP_ARTIFACTS / "inputs" / "flat_bn_params.bin", dtype=np.float32)
    lr = np.fromfile(FULL_LOOP_ARTIFACTS / "inputs" / "lr.bin", dtype=np.float32)[0]
    momentum = np.fromfile(FULL_LOOP_ARTIFACTS / "inputs" / "momentum.bin", dtype=np.float32)[0]
    print(
        json.dumps(
            {
                "final_pred": int(np.argmax(final_logits)),
                "loss": float(loss.reshape(-1)[0]),
                "energy": float(energy.reshape(-1)[0]),
                "changed_main_bn_scalars": int(np.count_nonzero(new_main - main)),
                "changed_ema_bn_scalars": int(np.count_nonzero(new_ema - main)),
                "main_update_formula_max_abs": float(np.max(np.abs(new_main - (main - lr * grad)))),
                "ema_formula_max_abs": float(
                    np.max(np.abs(new_ema - (momentum * main + (1.0 - momentum) * new_main)))
                ),
            },
            indent=2,
        )
    )


def run_target(args):
    from iree_run import compile_module, run_host, run_spike

    manifest = load_manifest()
    vmfb = compile_module(manifest, FULL_LOOP_ARTIFACTS, args.target)
    if args.target == "host":
        run_host(vmfb, manifest, FULL_LOOP_ARTIFACTS)
        summarize_outputs(FULL_LOOP_ARTIFACTS / "host_outputs")
    else:
        run_spike(vmfb, manifest, FULL_LOOP_ARTIFACTS)
        summarize_outputs(FULL_LOOP_ARTIFACTS / "spike_outputs")


def compare(args):
    import numpy as np

    names = [
        "final_logits",
        "energy",
        "loss",
        "new_main_bn_params",
        "new_ema_bn_params",
        "flat_bn_grads",
    ]
    ok = True
    for name in names:
        host = np.fromfile(FULL_LOOP_ARTIFACTS / "host_outputs" / f"{name}.bin", dtype=np.float32)
        spike = np.fromfile(FULL_LOOP_ARTIFACTS / "spike_outputs" / f"{name}.bin", dtype=np.float32)
        diff = np.abs(host - spike)
        close = bool(np.allclose(host, spike, atol=args.atol, rtol=args.rtol))
        ok = ok and close
        print(
            f"{name}: close={close} max_abs={float(diff.max()) if diff.size else 0.0:.8g} "
            f"mean_abs={float(diff.mean()) if diff.size else 0.0:.8g}"
        )
    host_pred = int(np.argmax(np.fromfile(FULL_LOOP_ARTIFACTS / "host_outputs/final_logits.bin", dtype=np.float32)))
    spike_pred = int(np.argmax(np.fromfile(FULL_LOOP_ARTIFACTS / "spike_outputs/final_logits.bin", dtype=np.float32)))
    print(f"host_pred={host_pred} spike_pred={spike_pred}")
    if not ok or host_pred != spike_pred:
        raise AssertionError("host and Spike/Saturn outputs differ")


def add_common_build_args(parser):
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--momentum", type=float, default=0.999)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--weights", choices=["none", "default"], default="none")
    parser.add_argument("--force", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Single entrypoint for the ResNet50 PAFKIP-style IREE/Saturn TTA prototype."
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("prepare", help="export/compose the full all-BN EMA+KIP device loop")
    add_common_build_args(p)
    p.set_defaults(func=prepare)

    p = sub.add_parser("compile", help="compile the full loop VMFB")
    p.add_argument("--target", choices=["host", "saturn", "both"], default="host")
    p.set_defaults(func=compile_target)

    p = sub.add_parser("run", help="compile and run the full loop")
    p.add_argument("--target", choices=["host", "saturn"], default="host")
    p.set_defaults(func=run_target)

    p = sub.add_parser("compare", help="compare existing host_outputs and spike_outputs")
    p.add_argument("--atol", type=float, default=5.0e-3)
    p.add_argument("--rtol", type=float, default=5.0e-3)
    p.set_defaults(func=compare)

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        sys.argv.append("run")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
