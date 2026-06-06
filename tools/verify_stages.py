#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from iree_run import compile_module, run_host  # noqa: E402
from paths import repo_path  # noqa: E402


STAGES = {
    "tta_views": ROOT / "artifact_aux" / "tta_views",
    "logits": ROOT / "artifact_aux" / "logits",
    "ema": ROOT / "artifact_aux" / "ema",
    "kip": ROOT / "artifact_aux" / "kip",
    "sgd": ROOT / "artifact_sgd",
    "train": ROOT / "artifact_train",
}

HEAVY_STAGES = {"logits", "train"}
DEFAULT_TOLERANCES = {
    ("train", "flat_bn_grads"): (1.0e-2, 1.0e-2),
}


def load_manifest(stage_dir: Path) -> dict:
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def compare_to_golden(
    stage: str,
    manifest: dict,
    actual_paths: list[Path],
    *,
    atol: float,
    rtol: float,
    use_default_tolerances: bool,
):
    results = []
    ok = True
    for item, actual_path in zip(manifest["outputs"], actual_paths):
        if "golden" not in item:
            results.append({"name": item["name"], "ok": None, "reason": "no golden"})
            continue
        golden = np.fromfile(repo_path(item["golden"]), dtype=np.float32)
        actual = np.fromfile(actual_path, dtype=np.float32)
        shape_ok = actual.shape == golden.shape
        if shape_ok and actual.size:
            diff = np.abs(actual - golden)
            max_abs = float(diff.max())
            mean_abs = float(diff.mean())
            max_index = int(diff.argmax())
        else:
            max_abs = 0.0
            mean_abs = 0.0
            max_index = -1
        item_atol, item_rtol = atol, rtol
        if use_default_tolerances:
            item_atol, item_rtol = DEFAULT_TOLERANCES.get(
                (stage, item["name"]), (atol, rtol)
            )
        close = bool(shape_ok and np.allclose(actual, golden, atol=item_atol, rtol=item_rtol))
        ok = ok and close
        results.append(
            {
                "name": item["name"],
                "ok": close,
                "shape_ok": bool(shape_ok),
                "actual_count": int(actual.size),
                "golden_count": int(golden.size),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "max_abs_index": max_index,
                "atol": item_atol,
                "rtol": item_rtol,
            }
        )
    return ok, results


def verify_stage(
    name: str,
    stage_dir: Path,
    *,
    target: str,
    reuse_vmfb: bool,
    run: bool,
    atol: float,
    rtol: float,
    use_default_tolerances: bool,
) -> dict:
    manifest = load_manifest(stage_dir)
    result = {
        "stage": name,
        "dir": str(stage_dir),
        "module": manifest.get("name"),
        "inputs": [item["name"] for item in manifest["inputs"]],
        "outputs": [item["name"] for item in manifest["outputs"]],
        "host": None,
        "saturn": None,
        "flexinpu": None,
    }

    if target in ("host", "both"):
        vmfb = stage_dir / f"{manifest.get('name', name)}_host.vmfb"
        if not (reuse_vmfb and vmfb.exists()):
            vmfb = compile_module(manifest, stage_dir, "host")
        host_result = {"vmfb": str(vmfb)}
        if run:
            actual_paths = run_host(vmfb, manifest, stage_dir)
            ok, outputs = compare_to_golden(
                name,
                manifest,
                actual_paths,
                atol=atol,
                rtol=rtol,
                use_default_tolerances=use_default_tolerances,
            )
            host_result.update(
                {
                    "ok": ok,
                    "output_dir": str(stage_dir / "host_outputs"),
                    "outputs": outputs,
                }
            )
        result["host"] = host_result

    if target in ("saturn", "both"):
        vmfb = stage_dir / f"{manifest.get('name', name)}_saturn.vmfb"
        if not (reuse_vmfb and vmfb.exists()):
            vmfb = compile_module(manifest, stage_dir, "saturn")
        result["saturn"] = {"vmfb": str(vmfb), "compiled": True}

    if target == "flexinpu":
        vmfb = stage_dir / f"{manifest.get('name', name)}_flexinpu.vmfb"
        if not (reuse_vmfb and vmfb.exists()):
            vmfb = compile_module(manifest, stage_dir, "flexinpu")
        result["flexinpu"] = {"vmfb": str(vmfb), "compiled": True}

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify each decomposed PAFKIP-IREE stage against its golden tensors."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(STAGES),
        help="Stage names to verify. Defaults to all stages.",
    )
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip ResNet stages: logits and train.",
    )
    parser.add_argument("--target", choices=["host", "saturn", "flexinpu", "both"], default="host")
    parser.add_argument("--reuse-vmfb", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--atol", type=float, default=5.0e-4)
    parser.add_argument("--rtol", type=float, default=5.0e-4)
    parser.add_argument(
        "--single-tolerance",
        action="store_true",
        help="Use --atol/--rtol for every output instead of stage-specific defaults.",
    )
    parser.add_argument("--json", type=Path, help="Optional report path.")
    args = parser.parse_args()

    names = args.only or list(STAGES)
    if args.skip_heavy:
        names = [name for name in names if name not in HEAVY_STAGES]

    reports = []
    all_ok = True
    for name in names:
        print(f"[verify] {name}", flush=True)
        report = verify_stage(
            name,
            STAGES[name],
            target=args.target,
            reuse_vmfb=args.reuse_vmfb,
            run=not args.compile_only and args.target in ("host", "both"),
            atol=args.atol,
            rtol=args.rtol,
            use_default_tolerances=not args.single_tolerance,
        )
        reports.append(report)
        host = report.get("host")
        if host and "ok" in host:
            all_ok = all_ok and bool(host["ok"])
            for output in host["outputs"]:
                status = "ok" if output["ok"] else "FAIL"
                print(
                    f"  {output['name']}: {status} "
                    f"max_abs={output.get('max_abs', 0.0):.6g} "
                    f"mean_abs={output.get('mean_abs', 0.0):.6g} "
                    f"tol=({output.get('atol')},{output.get('rtol')})"
                )
        if report.get("saturn"):
            print(f"  saturn vmfb: {report['saturn']['vmfb']}")
        if report.get("flexinpu"):
            print(f"  flexinpu vmfb: {report['flexinpu']['vmfb']}")

    final_report = {"ok": all_ok, "stages": reports}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(final_report, indent=2))
    print(json.dumps({"ok": all_ok, "stage_count": len(reports)}, indent=2))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
