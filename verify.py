"""CPU verifier for kda_prefill_h8_d128_k_last, mirroring SOL-ExecBench's checks.

    .venv/bin/python verify.py                       # the reference against itself
    .venv/bin/python verify.py solutions/correct_chunked.py
    .venv/bin/python verify.py SOLUTION.py --seeds 5 --tests A,S2

A solution is a Python file with `run(...)` taking the inputs of
definition.json in order and returning (output, final_state, checkpoint_state).

For every workload in workload.jsonl and every seed:
  1. build the inputs and check the definition's constraints on them
  2. run the reference (the string embedded in definition.json) and the solution
  3. check the number of outputs, each shape and each dtype
  4. reject NaN or infinity in either side, and an all-zero output where the
     reference is not all zero
  5. compare with the workload's own tolerance: an element passes if
     |out - ref| <= max_atol + max_rtol * |ref|; the output passes if at least
     required_matched_ratio of its elements pass (and, if max_error_cap is
     set, no element is off by more than that)
Steps 4 and 5 follow src/sol_execbench/core/bench/correctness.py (commit a9fa080).
Like SOL-ExecBench, whole tensors are compared, with no masking, so
snapshot rows with no snapshot must be zero.
"""

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from workload_inputs import DTYPES, load_inputs, resolve_shape  # noqa: E402

PROBLEM_DIR = os.path.join(ROOT, "problem", "kda_prefill_h8_d128_k_last")


@dataclass
class OutputResult:
    name: str
    passed: bool
    reason: str = ""          # why it failed, if it did
    max_abs: float = 0.0
    max_rel: float = 0.0
    matched: float = 1.0
    used: float = 0.0         # worst |out - ref| / (atol + rtol * |ref|); above 1 is out of tolerance


def load_problem(problem_dir: str = PROBLEM_DIR) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(os.path.join(problem_dir, "definition.json"), encoding="utf-8") as f:
        definition = json.load(f)
    with open(os.path.join(problem_dir, "workload.jsonl"), encoding="utf-8") as f:
        workloads = [json.loads(line) for line in f if line.strip()]
    for workload in workloads:
        if "tolerance" not in workload:
            raise SystemExit(f"workload {workload.get('test_id', workload['uuid'])} has no tolerance in workload.jsonl")
    return definition, workloads


def reference_from(definition: dict[str, Any]) -> Callable[..., Any]:
    namespace: dict[str, Any] = {}
    exec(definition["reference"], namespace)
    return namespace["run"]


def solution_from(path: str) -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location("solution", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def all_axes(definition: dict[str, Any], workload: dict[str, Any]) -> dict[str, int]:
    axes = {name: spec["value"] for name, spec in definition["axes"].items() if spec["type"] == "const"}
    axes.update(workload["axes"])
    return axes


def broken_constraints(definition: dict[str, Any], workload: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    """The definition's constraint expressions that are false for these inputs."""
    names = {**all_axes(definition, workload), **inputs}
    return [c for c in definition.get("constraints", []) if not bool(eval(c, {"__builtins__": {}}, names))]


def compare(name: str, out: torch.Tensor, ref: torch.Tensor, tolerance: dict[str, Any]) -> OutputResult:
    """SOL-ExecBench's check_tensor_sanity + compute_error_stats for one output."""
    x, y = out.to(torch.float32), ref.to(torch.float32)
    if not bool(torch.isfinite(y).all()):
        return OutputResult(name, False, "reference has NaN or infinity")
    if not bool(torch.isfinite(x).all()):
        return OutputResult(name, False, "NaN" if bool(torch.isnan(x).any()) else "infinity")
    if float(y.norm()) > 0 and float(x.norm()) == 0:
        return OutputResult(name, False, "all zeros", max_abs=float(y.norm()))

    abs_error = (x - y).abs()
    atol, rtol = tolerance.get("max_atol", 1e-2), tolerance.get("max_rtol", 1e-2)
    bound = atol + rtol * y.abs()
    matched = 1.0 - float((abs_error > bound).sum()) / abs_error.numel()
    used = float((abs_error / bound).max()) if atol > 0 or bool((y != 0).all()) else float("inf")
    max_abs = float(abs_error.max())
    max_rel = float((abs_error / y.abs().clamp(min=atol)).max())

    reason = ""
    if matched < tolerance.get("required_matched_ratio", 0.99):
        reason = "outside tolerance"
    cap = tolerance.get("max_error_cap")
    if cap is not None and max_abs > cap:
        reason = "above max_error_cap"
    return OutputResult(name, reason == "", reason, max_abs, max_rel, matched, used)


def check_outputs(definition: dict[str, Any], workload: dict[str, Any],
                  outs: Any, refs: tuple[torch.Tensor, ...]) -> list[OutputResult]:
    """Steps 3-5 for one run of one workload."""
    specs = definition["outputs"]
    if not isinstance(outs, (tuple, list)) or len(outs) != len(specs):
        got = len(outs) if isinstance(outs, (tuple, list)) else type(outs).__name__
        return [OutputResult(name, False, f"expected {len(specs)} outputs, got {got}") for name in specs]

    axes = all_axes(definition, workload)
    results = []
    for (name, spec), out, ref in zip(specs.items(), outs, refs):
        shape, dtype = resolve_shape(spec["shape"], axes), DTYPES[spec["dtype"]]
        if not isinstance(out, torch.Tensor):
            results.append(OutputResult(name, False, f"not a tensor: {type(out).__name__}"))
        elif tuple(out.shape) != shape:
            results.append(OutputResult(name, False, f"shape {tuple(out.shape)}, expected {shape}"))
        elif out.dtype != dtype:
            results.append(OutputResult(name, False, f"dtype {out.dtype}, expected {dtype}"))
        else:
            results.append(compare(name, out, ref, workload["tolerance"]))
    return results


def verify_workload(definition: dict[str, Any], workload: dict[str, Any], reference: Callable[..., Any],
                    solution: Callable[..., Any], seeds: int, problem_dir: str = PROBLEM_DIR) -> list[OutputResult]:
    """Worst result per output over all seeds (a failure in any seed fails the output)."""
    worst: dict[str, OutputResult] = {}
    for seed in range(seeds):
        inputs = load_inputs(problem_dir, definition, workload, seed)
        broken = broken_constraints(definition, workload, inputs)
        if broken:
            return [OutputResult(name, False, f"invalid test input: {broken[0]}") for name in definition["outputs"]]

        refs = reference(*inputs.values())
        # The solution gets its own copies, so it cannot change what the reference saw.
        outs = solution(*[v.clone() if isinstance(v, torch.Tensor) else v for v in inputs.values()])

        for result in check_outputs(definition, workload, outs, refs):
            if seeds > 1 and not result.passed:
                result.reason += f" (seed {seed})"
            best = worst.get(result.name)
            if best is None:
                worst[result.name] = result
            elif best.passed:
                result.max_abs = max(result.max_abs, best.max_abs)
                result.max_rel = max(result.max_rel, best.max_rel)
                result.matched = min(result.matched, best.matched)
                result.used = max(result.used, best.used)
                worst[result.name] = result
    return list(worst.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("solution", nargs="?", default=os.path.join(PROBLEM_DIR, "reference.py"),
                        help="Python file with run(...); default: the reference itself")
    parser.add_argument("--seeds", type=int, default=3, help="random seeds per workload (default 3)")
    parser.add_argument("--tests", default="", help="comma-separated test ids to run (default: all)")
    args = parser.parse_args()

    definition, workloads = load_problem()
    reference, solution = reference_from(definition), solution_from(args.solution)
    wanted = set(filter(None, args.tests.split(",")))

    print(f"solution: {os.path.relpath(args.solution, ROOT)}   seeds per workload: {args.seeds}")
    print("'used' = the worst error as a share of what the tolerance allows (100% is the limit; "
          "shown as a multiple, e.g. 500x, when far above it)")
    print(f"{'test':<5}{'tokens':>7} {'output':<17}{'max abs err':>12}{'max rel err':>12}{'matched':>10}{'used':>13}  "
          f"{'atol':>8} {'rtol':>8} {'ratio':>6}  result")
    passed_tests, total = 0, 0
    worst_used: dict[str, float] = {}
    for workload in workloads:
        test_id = workload.get("test_id", workload["uuid"][:8])
        if wanted and test_id not in wanted:
            continue
        total += 1
        tol = workload["tolerance"]
        results = verify_workload(definition, workload, reference, solution, args.seeds)
        for i, r in enumerate(results):
            label = f"{test_id:<5}{workload['axes']['total_seq_len']:>7}" if i == 0 else " " * 12
            verdict = "PASS" if r.passed else f"FAIL: {r.reason}"
            used = f"{100 * r.used:.1f}%" if r.used < 10 else f"{r.used:,.0f}x"
            print(f"{label} {r.name:<17}{r.max_abs:>12.3e}{r.max_rel:>12.3e}{100 * r.matched:>9.4f}%{used:>13}  "
                  f"{tol.get('max_atol', 1e-2):>8.1e} {tol.get('max_rtol', 1e-2):>8.1e} "
                  f"{tol.get('required_matched_ratio', 0.99):>6}  {verdict}")
        passed_tests += all(r.passed for r in results)
        worst_used[test_id] = max(r.used for r in results)

    summary = ", ".join(f"{t} {100 * u:.0f}%" if u < 10 else f"{t} {u:,.0f}x" for t, u in worst_used.items())
    print(f"\ntolerance used, worst output per test: {summary}")
    print(f"{passed_tests} of {total} workloads passed" + ("" if passed_tests == total else "  -> FAIL"))
    sys.exit(0 if passed_tests == total else 1)


if __name__ == "__main__":
    main()
