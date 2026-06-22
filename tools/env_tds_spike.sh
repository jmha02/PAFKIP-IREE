#!/usr/bin/env bash
# Source before running Saturn or FP16 FlexiNPU Spike checks.
export TDS_SPIKE_EXT_TOP="${TDS_SPIKE_EXT_TOP:-/root/TDS-Simulator-Spike-Extension}"
export RISCV="${RISCV:-/root/toolchains/tds-riscv-gcc15.2.0-medany}"
export CONDA_PREFIX="${TDS_SIM_CONDA_PREFIX:-/root/miniconda3/envs/tds-sim}"

export PATH="/root/PAFKIP-IREE/third_party/iree/build/tools:${TDS_SPIKE_EXT_TOP}/install/bin:${RISCV}/bin:${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${TDS_SPIKE_EXT_TOP}/install/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libpython3.11.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${TDS_SPIKE_EXT_TOP}/externals/tds-sim/srcs:${TDS_SPIKE_EXT_TOP}/flexi_npu/simulator:${PYTHONPATH:-}"
