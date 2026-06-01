# PAFKIP-IREE

Minimal FP32 spike for running a PAFKIP-style ResNet50 test-time adaptation
step through IREE on a RISC-V RVV/Saturn-style target.

This folder intentionally keeps only the current full-loop path. Older staged
final-BN experiments, CIFAR harnesses, and compiler repro scripts were removed
from this shareable package to avoid ambiguity.

## What Runs On Device

The canonical VMFB contains the compact all-BN EMA+KIP loop:

```text
inputs:
  image[1,3,224,224]
  main_bn_params[53120]
  ema_bn_params[53120]
  anchor_bn_params[53120]
  lr
  momentum

device work:
  ResNet50 logits with trainable all-BN affine state
  PAF loss and backward graph for all BN affine parameters
  SGD update of main BN state
  EMA update of BN state
  KIP final prediction

outputs:
  final_logits[1,1000]
  energy[1]
  loss
  new_main_bn_params[53120]
  new_ema_bn_params[53120]
  flat_bn_grads[53120]
```

Only BatchNorm affine parameters are trainable. ResNet convolution and
classifier weights are frozen constants, so EMA state only covers the BN affine
vector.

## Quickstart

`PAFKIP-IREE` does not require a fixed checkout path. It expects the IREE/Saturn
tools to be discoverable from the environment:

- `iree-compile`
- `iree-run-module`
- `spike`
- `static-run-module` for Saturn/Spike runs

Use whichever environment setup your checkout provides. If those binaries are
not on `PATH`, set `FLEXI_ROOT` or `IREE_TOOLCHAIN_ROOT` to the checkout that
contains `iree-build/` and `runtime/pk/`.

```bash
cd ~
# Example only:
#   source <your-flexi-or-iree-env>.sh
#   export FLEXI_ROOT=<path-to-your-flexi-checkout>

# Reuse existing artifacts. Add --force to regenerate Torch-MLIR exports.
python3 PAFKIP-IREE/run.py prepare

# Compile and run on IREE host.
python3 PAFKIP-IREE/run.py run --target host

# Compile and run on RISC-V/Saturn Spike. This is slow.
python3 PAFKIP-IREE/run.py run --target saturn

# Compare existing host and Spike/Saturn outputs.
python3 PAFKIP-IREE/run.py compare
```

## Current Verification

The renamed package was rechecked with:

```bash
cd ~
# Environment used for verification provided iree-compile, iree-run-module,
# spike, and static-run-module on PATH.
python3 PAFKIP-IREE/run.py prepare
python3 PAFKIP-IREE/run.py run --target host
python3 PAFKIP-IREE/run.py compile --target saturn
python3 PAFKIP-IREE/run.py compare
```

Host full-loop run:

```text
final_pred: 904
loss: -13.459149360656738
energy: 7.078167915344238
changed_main_bn_scalars: 49570 / 53120
changed_ema_bn_scalars: 26612 / 53120
main update formula max_abs: 7.275957614183426e-12
ema update formula max_abs: 1.1920928955078125e-07
```

Host vs Spike/Saturn:

```text
final_logits close=True max_abs=2.7123373e-05
energy close=True max_abs=4.7683716e-07
loss close=True max_abs=4.7683716e-06
new_main_bn_params close=True max_abs=3.5762787e-06
new_ema_bn_params close=True max_abs=2.2919266e-09
flat_bn_grads close=True max_abs=0.0036224388
host_pred=904 spike_pred=904
```

## Layout

```text
run.py
src/
  iree_run.py
  compose.py
  export_sgd.py
  export_aux.py
  export_train.py
  export_lib.py
  model.py
artifact_train/
artifact_sgd/
artifact_aux/
artifact_full/
```

The artifact directories contain generated MLIR, VMFBs, binary inputs, and
host/Spike outputs for the verified full-loop path.

## Required IREE/Runtime Changes

This spike depends on local compiler/runtime changes in the IREE/Flexi checkout
used to build the tools. The PAFKIP-relevant changes are:

```text
iree/compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/
  ConvertConvToIm2ColOp.cpp
    Adds channel-first/NCHW convolution handling in the im2col matmul lowering.
    This is needed for torchvision ResNet50 224x224 conv paths.

iree/compiler/src/iree/compiler/Dialect/Util/Transforms/
  HoistIntoGlobals.cpp
    Cleans dead ops deepest-first and removes nested ops from the worklist
    before erase. This avoids a Flow/global-hoisting crash seen on the large
    torch-mlir exported ResNet graph.

iree/compiler/src/iree/compiler/Codegen/LLVMCPU/
  KernelDispatch.cpp
    Adds a channel-wise projected-reduction heuristic for BN-style reductions.
    It forces scalar outer distribution on single-output-dimension reductions
    while preserving the reduction vector tile, avoiding invalid vector rank
    lowering for BN forward/backward reductions.

iree/runtime/src/iree/hal/
  buffer.c
  buffer_heap.c
  command_buffer.h
  command_buffer_validation.c
    Runtime workarounds for the embedded/local-sync RISC-V path: tolerate
    missing HAL metadata on VM rodata buffers, avoid transient-buffer discard
    before queued dispatch replay, force split heap metadata/data allocation,
    add a temporary payload guard region, and recover zero-length indirect
    binding spans from the backing buffer size.

iree/runtime/src/iree/hal/local/
  executable_library.h
iree/runtime/src/iree/modules/hal/
  module.c
    Raises dispatch/binding limits from small defaults to 255/256 so the large
    ResNet50 all-BN graph can marshal its bindings. `module.c` also prints a
    diagnostic when a non-empty binding points at a zero-length buffer.
```

There may also be unrelated local `iree/` row-bundle/dispatch-overlap changes
in the same checkout (`LLVMCPUExtractRowBundle.cpp`, `DispatchOverlapMerge.cpp`,
related pass registration/tests). Those are not required for this PAFKIP-IREE
full-loop path; they are separate Flexi/row-bundle profiling work.

Outside `iree/`, the relevant runtime changes are:

```text
runtime/src/runtime_entry.c
  Uses `iree_hal_buffer_view_allocate_buffer_copy` for baremetal inputs instead
  of importing host allocations directly. This gives the RISC-V/static runtime
  device-owned input buffers.

runtime/tools/iree-bundle-baremetal
  Reads VM function calling conventions by `internal_ordinal` when available,
  which is more robust for modules whose export order differs from signature
  table order.
```

After changing the IREE runtime files, rebuild the static runner before Spike
runs:

```bash
cd <path-to-your-flexi-or-runtime-checkout>
source <your-env>.sh
cmake --build runtime/pk --target static-run-module -j 16
```
