# PAFKIP-IREE

Current target: RISC-V/Saturn only.

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

## Generate Artifacts

```bash
python3 run.py prepare
```

Use torchvision ResNet50 pretrained weights when needed:

```bash
python3 run.py prepare --weights default --force
```

## Host Checks

```bash
python3 run.py verify-host --skip-heavy --reuse-vmfb
python3 run.py run --target host
```

## RISC-V/Saturn Compile

```bash
python3 run.py compile --target saturn
```

## ResNet50 Forward

```bash
python3 run.py verify-host --only logits --target saturn --compile-only --reuse-vmfb
python3 run.py forward-build --target saturn
python3 run.py forward-run --target saturn --reuse-vmfb
```

## Stage Spike Checks

```bash
python3 run.py verify-spike tta_views --reuse-vmfb
python3 run.py verify-spike ema --reuse-vmfb
python3 run.py verify-spike kip --reuse-vmfb
python3 run.py verify-spike sgd --reuse-vmfb
python3 run.py verify-spike logits --reuse-vmfb --atol 1e-2 --rtol 1e-2 # 약간 오차 존재. 해결 중
```

## Full Baremetal Build

```bash
python3 run.py baremetal build --target saturn --artifacts artifact_full --reuse-vmfb
```
