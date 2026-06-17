# PAFKIP-IREE

This repo is the RISC-V/Saturn version of the ResNet50 PAFKIP-style TTA spike.
The current handoff target is **RISC-V + Saturn RVV**.

## Setup

```bash
git clone --recursive https://github.com/jmha02/PAFKIP-IREE.git
cd PAFKIP-IREE

python3 -m pip install -r requirements.txt

export RISCV=/path/to/riscv-tools
export PATH="$RISCV/bin:$PATH"

tools/setup_from_scratch.sh
python3 tools/check_env.py
```

The setup we have actually tested is:

| Tool | Version / source |
| --- | --- |
| RISC-V GCC/G++ | `riscv64-unknown-elf-gcc 13.2.0 (gc891d8dc23e)` |
| RISC-V LLVM tools | `19.0.0git`, from the Chipyard `riscv-tools` install |
| Host LLVM/clang | `17.0.6` |
| CMake | `3.26.3` |
| Spike | `TDS-Simulator-Spike-Extension`, commit `cd1dfe2` |

Newer RISC-V GCC versions may work, but GCC 15/16 have not been the reference
environment for the baremetal ELF path.

## Generate Artifacts

Run this first. It exports the decomposed stages and the composed full loop.

```bash
python3 run.py prepare
```

If you want torchvision's pretrained ResNet50 weights:

```bash
python3 run.py prepare --weights default --force
```

## Stages

| Stage | Artifact | What it does |
| --- | --- | --- |
| `tta_views` | `artifact_aux/tta_views` | Builds the train/filter/anchor image views. |
| `logits` | `artifact_aux/logits` | Runs ResNet50 forward with the supplied BN parameters. |
| `train` | `artifact_train` | Computes the TTA loss and BN gradients. |
| `sgd` | `artifact_sgd` | Applies SGD momentum to the BN parameters. |
| `ema` | `artifact_aux/ema` | Updates the EMA BN parameters. |
| `kip` | `artifact_aux/kip` | Produces the final KIP logits and energy score. |
| `full` | `artifact_full` | Wires the stages into one PAFKIP-style TTA step. |

The BN parameter vector contains all ResNet50 BN gamma/beta values. Non-BN weights are fixed.
The TTA view stage is deterministic for now: the transform seed is fixed so that host, Spike, and golden outputs can be compared.

## Host Checks

Use host checks first.

```bash
python3 run.py verify-host --skip-heavy --reuse-vmfb
```

Run one stage on the host:

```bash
python3 run.py verify-host --only ema --reuse-vmfb
python3 run.py verify-host --only kip --reuse-vmfb
```

Compile-only checks for the RISC-V/Saturn target:

```bash
python3 run.py verify-host --target saturn --compile-only --reuse-vmfb
python3 run.py verify-host --only logits --target saturn --compile-only --reuse-vmfb
```

## Spike Checks

These run baremetal ELFs on Spike.
`verify-spike` is for checking one exported stage against its golden output.

```bash
python3 run.py verify-spike tta_views --reuse-vmfb
python3 run.py verify-spike ema --reuse-vmfb
python3 run.py verify-spike kip --reuse-vmfb
python3 run.py verify-spike sgd --reuse-vmfb
```

ResNet50 forward is much slower:

```bash
python3 run.py verify-spike logits --reuse-vmfb --atol 1e-2 --rtol 1e-2
```

Build and run the ResNet50 forward ELF directly:

```bash
python3 run.py forward-build --target saturn
python3 run.py forward-run --target saturn
```

## Full TTA Step

Compile the composed full loop:

```bash
python3 run.py compile --target saturn
```

Build the full baremetal ELF:

```bash
python3 run.py baremetal build \
  --target saturn \
  --artifacts artifact_full \
  --reuse-vmfb
```

Run the full loop on the host:

```bash
python3 run.py run --target host
```

Run the full loop on RISCV+Saturn Spike:

```bash
python3 run.py baremetal run \
  --target saturn \
  --artifacts artifact_full \
  --reuse-vmfb
```

This is the full baremetal path. It can take much longer than the small stage checks.

## Baremetal Simulator Notes

The generated ELF assumes:

- the core starts in M-mode at `0x80000000`
- the simulator supports `rv64gcv_zvl512b_zicsr_zifencei_zicntr_zihpm`
- stdout and exit are handled through the HTIF `tohost/fromhost` protocol
- enough DRAM is mapped above `0x80000000`

The command used by `run.py baremetal run` is:

```bash
spike -m4096 --isa=rv64gcv_zvl512b_zicsr_zifencei_zicntr_zihpm <elf>
```

This was tested with `TDS-Simulator-Spike-Extension`. In that Spike build,
`-m4096` creates a 4096 MiB target memory region at `DRAM_BASE=0x80000000`,
so the mapped range is `[0x80000000, 0x180000000)`. This is an ISS memory
setting; it is not the same thing as the DRAM size of a Verilator/RTL target.

For small-DRAM RTL simulations, rebuild the ELF with a smaller stack/heap
window. The default is `--stack-shift 30`, which reserves a 1 GiB window and is
intended for the 4 GiB Spike command above. For a 256 MiB RTL memory map, start
with `--stack-shift 26`:

```bash
python3 run.py baremetal build \
  --target saturn \
  --artifacts artifact_aux/ema \
  --reuse-vmfb \
  --stack-shift 26 \
  --output artifact_aux/ema/ema_baremetal_stack26.elf
```

If a simulator does not implement HTIF, the program may look stuck at the first
print, trap report, or exit. In that case the syscall/exit path must be adapted
to that simulator's UART or host interface.
