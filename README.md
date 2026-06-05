# PAFKIP-IREE

FP32 PAFKIP-style ResNet50 test-time adaptation spike for IREE on a
RISC-V RVV/Saturn-style target.

The package is intentionally centered on one entrypoint:

```bash
python3 run.py --help
```

Older proxy-kernel and hand-written baremetal wrapper paths are not part of the
canonical flow. Baremetal execution is built with IREE's
`runtime/tools/iree-bundle-baremetal` from the patched IREE submodule.

## What Is Implemented

The prototype decomposes PAFKIP-style TTA into independently verifiable stages:

| Stage | Artifact | Purpose |
| --- | --- | --- |
| `tta_views` | `artifact_aux/tta_views` | deterministic PAFKIP image views |
| `logits` | `artifact_aux/logits` | ResNet50 forward with flat BN state |
| `train` | `artifact_train` | PAF loss and all-BN affine gradients |
| `sgd` | `artifact_sgd` | SGD with momentum over BN affine state |
| `ema` | `artifact_aux/ema` | EMA teacher update over BN affine state |
| `kip` | `artifact_aux/kip` | KIP final logits and energy |
| `full` | `artifact_full` | composed one-step TTA loop |

Only BatchNorm affine parameters are trainable. ResNet convolution and
classifier weights are frozen constants, so the on-device mutable state ABI is
the flat BN affine vector plus SGD velocity and EMA BN state.

## Setup

```bash
git clone --recursive <PAFKIP-IREE repo URL>
cd PAFKIP-IREE

export RISCV=/path/to/riscv-tools
export PATH="$RISCV/bin:$PATH"

tools/setup_from_scratch.sh
python3 tools/check_env.py
```

Required Python packages for re-exporting artifacts are `torch`, `torchvision`,
and `torch-mlir`. Existing generated artifacts can be reused without
re-exporting.

## Common Commands

Prepare or regenerate artifacts:

```bash
python3 run.py prepare
python3 run.py prepare --force
```

Run the composed one-step loop on host IREE:

```bash
python3 run.py run --target host
```

Compile the composed loop for RISC-V/Saturn:

```bash
python3 run.py compile --target saturn
```

Build and run ResNet50 forward-only baremetal Spike:

```bash
python3 run.py forward-build
python3 run.py forward-run --elf artifact_aux/logits/logits_baremetal.elf
```

Build and run the full composed TTA loop baremetal Spike:

```bash
python3 run.py baremetal build --artifacts artifact_full --reuse-vmfb
python3 run.py baremetal run --artifacts artifact_full --elf artifact_full/full_baremetal.elf
```

Verify decomposed stages on host:

```bash
python3 run.py verify-host --skip-heavy --reuse-vmfb
python3 run.py verify-host --only logits --target saturn --compile-only --reuse-vmfb
```

Verify one stage on baremetal Spike and compare against golden output:

```bash
python3 run.py verify-spike tta_views --reuse-vmfb
python3 run.py verify-spike ema --reuse-vmfb
python3 run.py verify-spike kip --reuse-vmfb
python3 run.py verify-spike sgd --reuse-vmfb
python3 run.py verify-spike logits --reuse-vmfb --atol 1e-2 --rtol 1e-2
```

`logits` is the full ResNet50 forward path and is much slower than the small
update stages under Spike.

## Verified Status

Current local verification:

- `tta_views`, `ema`, `kip`, and `sgd` pass baremetal Spike golden comparison.
- ResNet50 `logits` passes baremetal Spike golden comparison for all 1000 logits.
- ResNet50 forward assembly contains RVV vector instructions in matmul and
  matmul-like dispatches, including `vfmacc.*`, `vfmadd.*`, `vle32.v`, and
  `vse32.v`.
- The full composed loop runs on host IREE. Full-loop Spike execution is
  intentionally not used as the normal proof path because FP32 ResNet50
  forward/backward simulation is extremely slow.

Latest ResNet50 forward Spike comparison:

```text
output: logits[1000]
max_abs: 4.392862319946289e-05
mean_abs: 1.0494581147213466e-05
runtime: about 10h43m on this Spike setup
```

## Layout

```text
run.py                  single command entrypoint
src/                    export, compose, model, IREE helper code
tools/                  environment and verification helpers
docs/                   patch notes and setup notes
third_party/iree/       patched IREE submodule
PAFKIP/                 original PAFKIP reference submodule
artifact_*/             generated MLIR, VMFBs, inputs, goldens, outputs
logs/                   local run logs
```

`artifact_*/` and `logs/` are generated and ignored by git. Recreate them with
`python3 run.py prepare` and the relevant compile/verify commands.

## Patched IREE

The required IREE changes live in the `third_party/iree` submodule. The most
relevant areas are:

- NCHW convolution lowering for torchvision ResNet50.
- Large exported ResNet graph global-hoisting fix.
- BN/channel reduction tiling heuristic for RISC-V vector lowering.
- Embedded RISC-V/Saturn runtime binding and lifetime workarounds.
- `iree-bundle-baremetal` support for single-ELF Spike execution without pk.

Detailed notes are in:

```text
docs/IREE_WORKTREE_CHANGES.md
docs/FLEXI_WORKTREE_CHANGES.md
docs/PATCH_RISK_ASSESSMENT.md
docs/SETUP_FROM_SCRATCH.md
```
