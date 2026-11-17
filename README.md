# kda_prefill_h8_d128_k_last

Kimi Delta Attention (expected to use chunking) prefill from GLM-5.3-Flash, as served by vLLM on 8x H200 with tensor parallel 8 (64 / 8 = 8 KDA heads per GPU, head_dim 128). Written as a SOL-ExecBench problem (`definition.json` + `workload.jsonl` + safetensors data), with all checks runnable on CPU.

## What the kernel covers

Follows what happens inside vLLM's FlashKDA (chosen backend) prefill call: everything after the q/k/v short convolution and before the output norm and projection. The kernel takes raw gate logits `g` and raw `beta` logits, l2-normalizes q and k itself, and computes the gate as `-5 * sigmoid(exp(A_log) * (g + dt_bias))`, one value per channel.

## Inputs

| Name | Shape | dtype | Notes |
|---|---|---|---|
| `q` | `[total_seq_len, 8, 128]` | bfloat16 | raw; normalized inside |
| `k` | `[total_seq_len, 8, 128]` | bfloat16 | raw; normalized inside |
| `v` | `[total_seq_len, 8, 128]` | bfloat16 | |
| `g` | `[total_seq_len, 8, 128]` | bfloat16 | raw gate logits, one per channel |
| `beta` | `[total_seq_len, 8]` | bfloat16 | raw logits; sigmoid inside |
| `A_log` | `[8]` | float32 | layer weight |
| `dt_bias` | `[8, 128]` | float32 | layer weight |
| `scale` | scalar | float32 | always 128 ** -0.5 |
| `cu_seqlens` | `[num_seqs + 1]` | int32 | prompt boundaries |
| `initial_state` | `[num_seqs, 8, 128, 128]` | float32 | layout [value, key] |
| `checkpoint_offsets` | `[num_seqs]` | int32 | snapshot position per prompt; 0 = none |

## Outputs (in this order)

| Name | Shape | dtype |
|---|---|---|
| `output` | `[total_seq_len, 8, 128]` | bfloat16 |
| `final_state` | `[num_seqs, 8, 128, 128]` | float32 |
| `checkpoint_state` | `[num_seqs, 8, 128, 128]` | float32 |

## Snapshot (checkpoint) semantics

`checkpoint_state` has one row per prompt. `checkpoint_offsets[n] = c` means `checkpoint_state[n]` is prompt n's memory after its first `c` tokens of this call.

- `c` counts from the prompt's own first token (`cu_seqlens[n]`), not from
  the start of the packed token array.
- The snapshot is taken after the update for 0-based token `c - 1` and before
  the decay for token `c`.
- `c = 0` means no snapshot. Those rows must be all zeros; `verify.py` compares them.
- Allowed values: 0, or a multiple of 16 with `0 < c < prompt length`. This
  is the contract of the FlashKDA kernel vLLM builds.

## How to check a solution

Needs Python with `torch` and `safetensors` (`pip install -r requirements.txt`).

```
python verify.py solutions/correct_chunked.py       # passes all 11 tests
python verify.py solutions/incorrect_gdn_gate.py    # fails all 11 tests
python verify.py solutions/incorrect_junk_rows.py   # fails 10 of 11 tests (all but S1)
```

`incorrect_junk_rows.py` is the correct solution with one change: it does not zero its snapshot buffer, so rows of prompts with no snapshot hold junk. It shows that those rows really are checked. S1 passes because its only prompt has a snapshot, so it has no unused row.

This runs all 11 tests in `workload.jsonl` with 3 random seeds each (about 30 seconds on a laptop CPU), prints a PASS/FAIL line per output, and exits with code 0 if everything passed or 1 otherwise.

- `--seeds 5` changes the number of seeds per test.
- `--tests A,S2` runs only the named tests.

A solution is one Python file with a `run(...)` function that:

- takes the 11 inputs in the order of the Inputs table above;
- returns `output`, `final_state`, `checkpoint_state`, in that order, with the shapes and dtypes of the Outputs table;
- leaves `checkpoint_state` rows as zeros for prompts with no snapshot.

Files: the task is `problem/kda_prefill_h8_d128_k_last/` (`definition.json`, `workload.jsonl`, the test data in `blob/`, and `reference.py` for reading). `verify.py` uses the copy of the reference inside `definition.json`, and loads inputs with `scripts/workload_inputs.py`.

## Tests

| Test | Tokens | Prompts | Snapshots | What it checks |
|---|---|---|---|---|
| A | 1 | 1 | 0 | one-token prompt (non-blank memory, so the gate matters) |
| B | 16 | 1 | 0 | exact 16-block |
| C | 64 | 1 | 0 | exact 64-block |
| D | 1000 | 1 | 0 | partial blocks for both 16 and 64 |
| E | 8192 | 1 | 0 | long prompt, early tokens matter |
| F | 2000 | 1 | 0 | gate must be applied |
| G | 2598 | 4 | 0 | ragged batch |
| H | 800 | 2 | 0 | starting memory used and not swapped |
| S1 | 1000 | 1 | 1 | basic snapshot |
| S2 | 1600 | 3 | 2 | mid-64-block snapshot; relative positions |
| R | 10342 | 12 | 6 | realism: real prompt lengths, real gate weights |

"Snapshots" is the number of prompts with a snapshot. In A to H there are none, so `checkpoint_state` must be all zeros there.

## Tolerances

Each test has its own tolerance in `workload.jsonl`. They are calibrated on CPU against an fp32 chunked implementation (`solutions/correct_chunked.py`). They must be recalibrated against FlashKDA and FLA on GPU, which use bf16 block math and will not land as close to the reference.

## Simplifications versus the real call

- No leading batch-of-1 dimension (the real kernel takes `[1, T, H, D]`).
- Outputs are returned rather than written into pre-allocated buffers and a
  workspace.
- The lower bound (-5.0) is a fixed constant in the reference.
- Production leaves unused snapshot rows untouched, so they hold whatever the
  buffer held before. The grader compares the whole output and cannot skip
  rows, so this task requires those rows to be zeros. The kernel does not have
  to write them: a kernel that skips prompts with no snapshot, like FlashKDA,
  passes as long as its snapshot buffer starts as zeros. SOL-ExecBench
  allocates pre-allocated outputs with zeros (read from its source at commit
  `a9fa080`, not run). Only an uninitialized buffer fails, which is what
  `solutions/incorrect_junk_rows.py` shows.

## Assumptions

- FlashKDA being the default backend on H200 is read from vLLM's source, not observed at runtime.
- vLLM also requires snapshot offsets to be multiples of its prefix-cache hash block size, which in GLM-5.3-Flash's default configuration is a multiple of 16, so every offset is a multiple of 16.

## Open items

- `workload.jsonl` format: not yet checked against a real SOL-ExecBench workload file, including the extra `test_id` and `note` keys (and `hf_id` in `definition.json`).
- SM90 snapshot code path: not yet checked.
- cuLA block size: not yet confirmed.
- Whether SOL-ExecBench evaluates the value constraints (`constraints` in `definition.json`): not confirmed. `verify.py` does evaluate them.
- Tolerances: to be recalibrated on GPU (see Tolerances).

## Sources

- GLM-5.3-Flash config: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json
- vLLM recipe (default: 8x H200, `--tensor-parallel-size 8`): https://recipes.vllm.ai/zai-org/GLM-5.3-Flash
- vLLM GLM-5.3-Flash KDA layer: https://docs.vllm.ai/en/latest/api/vllm/models/glm5next/common/kda/
- vLLM's FlashKDA fork (has the snapshot arguments; commit `9d9e928`): https://github.com/vllm-project/FlashKDA
- FlashKDA upstream (no snapshot arguments): https://github.com/MoonshotAI/FlashKDA
- FlashKDA low-utilization issue: https://github.com/MoonshotAI/FlashKDA/issues/8
- cuLA: https://github.com/inclusionAI/cuLA
- KDA paper (Kimi Linear): https://arxiv.org/abs/2510.26692
- SOL-ExecBench: https://github.com/NVIDIA/SOL-ExecBench
- FlashInfer Trace format: https://flashinfer-bench.vercel.app/docs/flashinfer-trace/definition
- KernelBench (format considered): https://github.com/ScalingIntelligence/KernelBench
