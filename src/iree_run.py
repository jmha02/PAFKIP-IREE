#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parent
WORK_DIR = PACKAGE_DIR.parent


def tool_from_path_or_root(tool_name: str, relative_path: str) -> str:
    found = shutil.which(tool_name)
    if found:
        return found
    root = os.environ.get("FLEXI_ROOT") or os.environ.get("IREE_TOOLCHAIN_ROOT")
    if root:
        candidate = Path(root) / relative_path
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"{tool_name} not found in PATH. Source your IREE/Saturn environment "
        f"or set FLEXI_ROOT/IREE_TOOLCHAIN_ROOT to the checkout containing {relative_path}."
    )


def run(cmd, *, cwd=WORK_DIR, stdout=None):
    print("+ " + " ".join(str(c) for c in cmd))
    capture_stdout = stdout == subprocess.PIPE
    result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        stdout_text = ""
        if capture_stdout and result.stdout:
            stdout_text = "\n\nSTDOUT:\n" + result.stdout
        raise RuntimeError(
            "command failed\n"
            + " ".join(str(c) for c in cmd)
            + stdout_text
            + "\n\nSTDERR:\n"
            + result.stderr
        )
    if result.stderr:
        print(result.stderr)
    return result


def compile_module(manifest, out_dir: Path, target: str):
    iree_compile = tool_from_path_or_root("iree-compile", "iree-build/tools/iree-compile")
    module_name = manifest.get("name", "pafkip_train_step")
    vmfb = out_dir / f"{module_name}_{target}.vmfb"
    flags = [
        iree_compile,
        manifest["mlir"],
        "-o",
        str(vmfb),
        "--mlir-elide-elementsattrs-if-larger=8",
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-input-type=torch",
    ]
    if target == "host":
        flags.extend(
            [
                "--iree-hal-target-device=local",
                "--iree-llvmcpu-target-cpu=host",
                "--iree-llvmcpu-fail-on-large-vector=false",
                "--iree-opt-data-tiling=false",
            ]
        )
        if manifest.get("llvmcpu_vector_pproc_strategy"):
            flags.append(
                "--iree-llvmcpu-vector-pproc-strategy="
                + manifest["llvmcpu_vector_pproc_strategy"]
            )
        if manifest.get("llvmcpu_stack_allocation_limit"):
            flags.append(
                "--iree-llvmcpu-stack-allocation-limit="
                + str(manifest["llvmcpu_stack_allocation_limit"])
            )
    elif target == "saturn":
        scalar_riscv = manifest.get("riscv_features") == "scalar"
        flags.extend(
            [
                "--iree-llvmcpu-target-triple=riscv64-unknown-eabi-elf",
                "--iree-llvmcpu-target-cpu=generic-rv64",
                "--iree-llvmcpu-target-abi=lp64d",
                "--iree-llvmcpu-link-embedded",
                (
                    "--iree-llvmcpu-target-cpu-features=+m,+f,+d,+a"
                    if scalar_riscv
                    else "--iree-llvmcpu-target-cpu-features=+m,+f,+d,+a,+v,+zvl512b"
                ),
                "--riscv-insert-vsetvli-whole-vector-register-move-valid-vtype=false",
                "--iree-llvmcpu-fail-on-large-vector=false",
                "--iree-opt-data-tiling=false",
                f"--iree-hal-dump-executable-intermediates-to={out_dir / 'saturn_intms'}",
            ]
        )
        if not scalar_riscv:
            flags.append("--riscv-v-register-bit-width-lmul=4")
        if manifest.get("llvmcpu_vector_pproc_strategy"):
            flags.append(
                "--iree-llvmcpu-vector-pproc-strategy="
                + manifest["llvmcpu_vector_pproc_strategy"]
            )
        if manifest.get("llvmcpu_stack_allocation_limit"):
            flags.append(
                "--iree-llvmcpu-stack-allocation-limit="
                + str(manifest["llvmcpu_stack_allocation_limit"])
            )
    else:
        raise ValueError(target)
    run(flags)
    return vmfb


def invocation_args(manifest, output_dir: Path):
    args = [f"--function={manifest['function']}"]
    for item in manifest["inputs"]:
        args.append(f"--input={item['iree']}=@{item['file']}")
    output_paths = []
    for item in manifest["outputs"]:
        path = output_dir / f"{item['name']}.bin"
        output_paths.append(path)
        args.append(f"--output=@{path}")
    return args, output_paths


def run_host(vmfb: Path, manifest, out_dir: Path):
    exe = tool_from_path_or_root("iree-run-module", "iree-build/tools/iree-run-module")
    output_dir = out_dir / "host_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    args, paths = invocation_args(manifest, output_dir)
    run([exe, "--device=local-sync", f"--module={vmfb}", *args])
    return paths


def run_spike(vmfb: Path, manifest, out_dir: Path):
    spike = shutil.which("spike")
    if not spike:
        raise FileNotFoundError("spike not found in PATH; source your RISC-V/Saturn environment first")
    pk_dir = os.environ.get("IREE_RT_PK_DIR")
    if pk_dir:
        static_run_module = str(Path(pk_dir) / "tools/static-run-module")
    else:
        static_run_module = shutil.which("static-run-module")
        if not static_run_module:
            root = os.environ.get("FLEXI_ROOT") or os.environ.get("IREE_TOOLCHAIN_ROOT")
            if root:
                static_run_module = str(Path(root) / "runtime/pk/tools/static-run-module")
    if not static_run_module or not Path(static_run_module).exists():
        raise FileNotFoundError(
            "static-run-module not found. Build/source the RISC-V runtime, set "
            "IREE_RT_PK_DIR, or set FLEXI_ROOT/IREE_TOOLCHAIN_ROOT."
        )

    output_dir = out_dir / "spike_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    args, paths = invocation_args(manifest, output_dir)
    cmd = [
        spike,
        "-m4096",
        (
            "--isa=rv64gc_zicsr_zifencei_zicntr_zihpm"
            if manifest.get("riscv_features") == "scalar"
            else "--isa=rv64gcv_zvl512b_zicsr_zifencei_zicntr_zihpm"
        ),
        "pk",
        static_run_module,
        f"--module={vmfb}",
        *args,
    ]
    run(cmd, stdout=subprocess.PIPE)
    return paths


def compare_outputs(manifest, actual_paths, label: str):
    ok = True
    for item, actual_path in zip(manifest["outputs"], actual_paths):
        golden = np.fromfile(item["golden"], dtype=np.float32)
        actual = np.fromfile(actual_path, dtype=np.float32)
        if golden.shape != actual.shape:
            print(f"{label}:{item['name']} shape mismatch {actual.shape} != {golden.shape}")
            ok = False
            continue
        atol = 5.0e-4
        rtol = 5.0e-4
        close = np.allclose(actual, golden, atol=atol, rtol=rtol)
        max_abs = float(np.max(np.abs(actual - golden))) if golden.size else 0.0
        print(f"{label}:{item['name']} close={close} max_abs={max_abs:.6g}")
        ok = ok and close
    if not ok:
        raise AssertionError(f"{label} outputs differ from PyTorch golden")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=THIS_DIR / "artifacts")
    parser.add_argument("--target", choices=["host", "saturn", "both"], default="both")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.artifacts / "manifest.json").read_text())
    if args.target in ("host", "both"):
        host_vmfb = compile_module(manifest, args.artifacts, "host")
        if not args.skip_run:
            compare_outputs(manifest, run_host(host_vmfb, manifest, args.artifacts), "host")
    if args.target in ("saturn", "both"):
        saturn_vmfb = compile_module(manifest, args.artifacts, "saturn")
        if not args.skip_run:
            compare_outputs(manifest, run_spike(saturn_vmfb, manifest, args.artifacts), "spike")


if __name__ == "__main__":
    main()
