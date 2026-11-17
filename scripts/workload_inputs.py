"""Turn one line of workload.jsonl into the inputs for `run`.

Used by verify.py. Follows SOL-ExecBench's rules for each input type
(src/sol_execbench/core/bench/io.py at commit a9fa080):

  safetensors  loaded from the file; shape and dtype must match the definition
  scalar       the literal value
  random       torch.randn in the input's dtype

SOL-ExecBench draws "random" inputs on the GPU with its own seed, so the
numbers here cannot be the same as there; the distribution is. Here the
random stream is fixed by the workload's uuid and a small seed number, so
every solution sees the same values, and several seeds can be run per test.
"""

import os
from collections import OrderedDict
from typing import Any

import torch
from safetensors.torch import load_file

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int32": torch.int32}

# SOL-ExecBench replaces randn with a name-based heuristic for these input
# names (io.py L246-247, L311-320). They must be stored, never "random".
SOL_HEURISTIC_NAMES = ("A", "A_log", "A_cumsum", "g")


def seed_from_uuid(uuid: str) -> int:
    return int(uuid.replace("-", "")[:15], 16)


def resolve_shape(shape: list[str] | None, axes: dict[str, int]) -> tuple[int, ...] | None:
    return None if shape is None else tuple(axes[name] for name in shape)


def load_inputs(problem_dir: str, definition: dict[str, Any], workload: dict[str, Any],
                seed: int = 0) -> "OrderedDict[str, Any]":
    """Inputs in definition order, as SOL-ExecBench would pass them to `run`.

    `seed` only changes the "random" inputs; stored inputs are the same for every seed.
    """
    axes = {name: spec["value"] for name, spec in definition["axes"].items() if spec["type"] == "const"}
    axes.update(workload["axes"])
    gen = torch.Generator().manual_seed(seed_from_uuid(workload["uuid"]) + 1_000_003 * (seed + 1))
    files: dict[str, dict[str, torch.Tensor]] = {}

    inputs: "OrderedDict[str, Any]" = OrderedDict()
    for name, spec in definition["inputs"].items():
        source = workload["inputs"].get(name, {"type": "random"})
        shape, dtype = resolve_shape(spec["shape"], axes), DTYPES[spec["dtype"]]

        if source["type"] == "scalar":
            inputs[name] = source["value"]
        elif source["type"] == "safetensors":
            path = os.path.join(problem_dir, source["path"])
            if path not in files:
                files[path] = load_file(path)
            tensor = files[path][source["tensor_key"]]
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"{name}: file has {tuple(tensor.shape)} {tensor.dtype}, "
                                 f"definition says {shape} {dtype}")
            inputs[name] = tensor
        elif source["type"] == "random":
            if not dtype.is_floating_point or name in SOL_HEURISTIC_NAMES or shape is None:
                raise ValueError(f"{name}: 'random' would not mean randn on SOL-ExecBench; store it instead")
            inputs[name] = torch.randn(shape, generator=gen).to(dtype)
        else:
            raise ValueError(f"{name}: unsupported input type {source['type']}")
    return inputs
