#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from paths import repo_path  # noqa: E402

OUTPUT_RE = re.compile(
    r"^\[IREE\]\[output\s+(\d+)\]\s+shape=.*\s+dtype=([A-Za-z0-9]+)\s+elements=(\d+)"
)
VALUE_RE = re.compile(r"^\s*\[\s*(\d+)\]\s+([-+a-zA-Z0-9.]+)\s*$")


def run(cmd: list[str], *, env=None, timeout=None, stdout=None):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        timeout=timeout,
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
        text=True if stdout is None else False,
        check=True,
    )


def load_manifest(stage_dir: Path) -> dict:
    return json.loads((stage_dir / "manifest.json").read_text())


def build_elf(
    stage_dir: Path,
    manifest: dict,
    *,
    target: str,
    elf: Path,
    print_limit: int,
    runner: str,
    reuse_vmfb: bool,
):
    env = os.environ.copy()
    riscv = env.get("RISCV")
    if riscv:
        env["PATH"] = os.pathsep.join(
            [
                str(Path(riscv).parent / "bin"),
                str(Path(riscv) / "bin"),
                env.get("PATH", ""),
            ]
        )
    cmd = [
        sys.executable,
        str(ROOT / "run.py"),
        "baremetal",
        "build",
        "--artifacts",
        str(stage_dir),
        "--target",
        target,
        "--runner",
        runner,
        "--output",
        str(elf),
        "--output-print-limit",
        str(print_limit),
    ]
    if reuse_vmfb:
        cmd.append("--reuse-vmfb")
    run(cmd, env=env)


def run_spike(elf: Path, log: Path, *, timeout: int):
    riscv = os.environ.get("RISCV")
    spike = shutil.which("spike")
    if not spike and riscv:
        candidate = Path(riscv) / "bin" / "spike"
        if candidate.exists():
            spike = str(candidate)
    if not spike:
        raise FileNotFoundError("spike not found in PATH; set RISCV or add spike to PATH")
    env = os.environ.copy()
    if riscv:
        env["PATH"] = os.pathsep.join([str(Path(riscv) / "bin"), env.get("PATH", "")])
    cmd = [
        spike,
        "-m4096",
        "--isa=rv64gcv_zvl512b_zicsr_zifencei_zicntr_zihpm",
        str(elf),
    ]
    if timeout:
        cmd = ["timeout", str(timeout), *cmd]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as f:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode == 124:
        raise TimeoutError(f"Spike timed out after {timeout}s: {log}")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def parse_spike_outputs(log: Path) -> dict[int, np.ndarray]:
    outputs: dict[int, dict[int, float]] = {}
    expected_counts: dict[int, int] = {}
    current = None
    for line in log.read_text(errors="ignore").splitlines():
        header = OUTPUT_RE.match(line)
        if header:
            current = int(header.group(1))
            expected_counts[current] = int(header.group(3))
            outputs[current] = {}
            continue
        match = VALUE_RE.match(line)
        if match and current is not None:
            outputs[current][int(match.group(1))] = float(match.group(2).lower())
    parsed = {}
    for index, values in outputs.items():
        expected = expected_counts[index]
        if len(values) != expected:
            raise RuntimeError(
                f"output {index} printed {len(values)} values, expected {expected}; "
                f"use --print-limit 0 for full golden comparison"
            )
        missing = [i for i in range(expected) if i not in values]
        if missing:
            raise RuntimeError(f"output {index} missing index {missing[0]}")
        parsed[index] = np.array([values[i] for i in range(expected)], dtype=np.float32)
    if not parsed:
        raise RuntimeError(f"no IREE outputs found in {log}")
    return parsed


def compare(manifest: dict, actual_by_index: dict[int, np.ndarray], *, atol: float, rtol: float):
    ok = True
    results = []
    for index, output in enumerate(manifest["outputs"]):
        if "golden" not in output:
            results.append({"name": output["name"], "ok": None, "reason": "no golden"})
            continue
        actual = actual_by_index[index]
        golden = np.fromfile(repo_path(output["golden"]), dtype=np.float32)
        shape_ok = actual.shape == golden.shape
        diff = np.abs(actual - golden) if shape_ok else np.array([], dtype=np.float32)
        close = bool(shape_ok and np.allclose(actual, golden, atol=atol, rtol=rtol))
        ok = ok and close
        results.append(
            {
                "name": output["name"],
                "ok": close,
                "actual_count": int(actual.size),
                "golden_count": int(golden.size),
                "max_abs": float(diff.max()) if diff.size else None,
                "mean_abs": float(diff.mean()) if diff.size else None,
            }
        )
    return ok, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/run one stage as baremetal Spike and compare printed outputs to golden."
    )
    parser.add_argument("stage_dir", type=Path)
    parser.add_argument("--target", choices=["saturn", "flexinpu"], default="saturn")
    parser.add_argument("--elf", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--runner", choices=["tooling", "direct"], default="direct")
    parser.add_argument("--print-limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument(
        "--reuse-vmfb",
        action="store_true",
        help="Reuse an existing target VMFB instead of recompiling before bundling.",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--atol", type=float, default=5.0e-4)
    parser.add_argument("--rtol", type=float, default=5.0e-4)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    stage_dir = args.stage_dir.resolve()
    manifest = load_manifest(stage_dir)
    elf_suffix = "baremetal_fullprint" if args.target == "saturn" else f"{args.target}_baremetal_fullprint"
    elf = args.elf or (stage_dir / f"{manifest['name']}_{elf_suffix}.elf")
    log = args.log or (ROOT / "logs" / f"spike_{manifest['name']}.log")

    if not args.skip_build:
        build_elf(
            stage_dir,
            manifest,
            target=args.target,
            elf=elf,
            print_limit=args.print_limit,
            runner=args.runner,
            reuse_vmfb=args.reuse_vmfb,
        )
    if not args.skip_run:
        run_spike(elf, log, timeout=args.timeout)
    else:
        report = {
            "ok": True,
            "stage_dir": str(stage_dir),
            "target": args.target,
            "elf": str(elf),
            "log": str(log),
            "skipped_run": True,
        }
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2))
        print(json.dumps({"ok": True, "stage": manifest["name"], "skipped_run": True}, indent=2))
        return
    actual = parse_spike_outputs(log)
    ok, outputs = compare(manifest, actual, atol=args.atol, rtol=args.rtol)
    report = {
        "ok": ok,
        "stage_dir": str(stage_dir),
        "elf": str(elf),
        "log": str(log),
        "outputs": outputs,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    for output in outputs:
        status = "ok" if output["ok"] else "FAIL"
        print(
            f"{output['name']}: {status} "
            f"max_abs={output.get('max_abs')} mean_abs={output.get('mean_abs')}"
        )
    print(json.dumps({"ok": ok, "stage": manifest["name"]}, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
