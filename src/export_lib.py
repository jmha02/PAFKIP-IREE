import json
from pathlib import Path

import numpy as np
import torch
import torch_mlir.fx
from torch._decomp import get_decompositions
from torch.export import export
from torch_mlir.extras.fx_decomp_util import DEFAULT_DECOMPOSITIONS

from paths import repo_relative


DTYPE_TO_MLIR = {torch.float32: "f32"}


def shape_dtype_to_iree(tensor: torch.Tensor) -> str:
    shape = "x".join(str(d) for d in tensor.shape)
    return f"{shape}x{DTYPE_TO_MLIR[tensor.dtype]}" if shape else DTYPE_TO_MLIR[tensor.dtype]


def write_bin(path: Path, tensor: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tensor.detach().cpu().numpy()).astype(np.float32).tofile(path)


def export_one(out_dir, name, model, inputs, input_names, output_names, extra_manifest=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = out_dir / "inputs"
    golden_dir = out_dir / "golden"
    ep = export(model.eval(), inputs, strict=False)
    decomposition_table = get_decompositions([*DEFAULT_DECOMPOSITIONS, torch.ops.aten.as_strided])
    mlir = torch_mlir.fx.export_and_import(
        ep,
        output_type="torch",
        decomposition_table=decomposition_table,
    )
    mlir_path = out_dir / f"{name}.mlir"
    mlir_path.write_text(str(mlir))

    with torch.no_grad():
        outputs = model(*inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    manifest = {
        "name": name,
        "function": "main",
        "mlir": repo_relative(mlir_path),
        "inputs": [],
        "outputs": [],
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    for input_name, tensor in zip(input_names, inputs):
        path = input_dir / f"{input_name}.bin"
        write_bin(path, tensor)
        manifest["inputs"].append(
            {
                "name": input_name,
                "shape": list(tensor.shape),
                "dtype": "f32",
                "iree": shape_dtype_to_iree(tensor),
                "file": repo_relative(path),
            }
        )
    for output_name, tensor in zip(output_names, outputs):
        path = golden_dir / f"{output_name}.bin"
        write_bin(path, tensor)
        manifest["outputs"].append(
            {
                "name": output_name,
                "shape": list(tensor.shape),
                "dtype": "f32",
                "iree": shape_dtype_to_iree(tensor),
                "golden": repo_relative(path),
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {mlir_path}")
    return outputs
