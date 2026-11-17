"""Reference for kda_prefill_h8_d128_k_last: KDA prefill, one token at a time."""

# What this computes: vLLM's FlashKDA prefill call for GLM-5.3-Flash
# (`_flashkda_prefill`: raw g and beta, q/k l2-normalized in-kernel, bounded gate):
#   https://github.com/vllm-project/vllm/blob/ae7048c068d0382fc476d3afa64f6f1c63b38662/vllm/models/glm5next/common/kda.py#L409-L476
# The same recurrence in pure PyTorch, which this file is cross-checked against
# (`naive_recurrent_kda`, with `naive_kda_lowerbound_gate` from ../gate.py):
#   https://github.com/fla-org/flash-linear-attention/blob/7edb579d636139df2b31df580d497928121d35d4/fla/ops/kda/naive.py#L12-L66

import torch

# Gate lower bound: GLM-5.3-Flash config.json, linear_attn_config.gate_lower_bound.
LOWER_BOUND = -5.0

# l2norm epsilon, added to the sum of squares inside the square root:
#   vLLM  vllm/third_party/flash_linear_attention/ops/l2norm.py
#         L140 `eps: float = 1e-6`, L105 `rsqrt = tl.rsqrt(square_sum + eps)`
#   FlashKDA (vllm-project fork)  csrc/smxx/fwd_kernel1.cuh
#         L289 `rsqrtf(q_sq + 1e-6f)`, L290 `rsqrtf(k_sq + 1e-6f)`
L2NORM_EPS = 1e-6


def l2norm(x):
    return x / torch.sqrt((x * x).sum() + L2NORM_EPS)


def run(q, k, v, g, beta, A_log, dt_bias, scale, cu_seqlens, initial_state, checkpoint_offsets):
    total_seq_len, num_heads, head_dim = q.shape
    num_seqs = initial_state.shape[0]
    bounds = cu_seqlens.tolist()
    offsets = checkpoint_offsets.tolist()

    output = torch.zeros(total_seq_len, num_heads, head_dim, dtype=torch.float32, device=q.device)
    final_state = torch.zeros_like(initial_state)
    checkpoint_state = torch.zeros_like(initial_state)  # rows with offset 0 stay zero

    for n in range(num_seqs):
        for h in range(num_heads):
            # The memory starts over for every prompt and head: S is [V, K], fp32.
            S = initial_state[n, h].clone()

            for t in range(bounds[n], bounds[n + 1]):
                p = t - bounds[n] + 1  # 1-based position inside this prompt

                # Step 1: normalize q and k.
                q_t = l2norm(q[t, h].float())
                k_t = l2norm(k[t, h].float())
                v_t = v[t, h].float()

                # Step 2: write strength.
                b = torch.sigmoid(beta[t, h].float())

                # Step 3: log-decay, one value per key channel, in [-5, 0].
                a = LOWER_BOUND * torch.sigmoid(torch.exp(A_log[h]) * (g[t, h].float() + dt_bias[h]))

                # Step 4: decay the memory along the key dimension.
                S = S * torch.exp(a)[None, :]

                # Step 5: delta-rule update.
                S = S + b * torch.outer(v_t - S @ k_t, k_t)

                # Step 6: read out.
                output[t, h] = S @ (scale * q_t)

                # Step 7: snapshot after exactly `offsets[n]` tokens of this prompt.
                if offsets[n] == p:
                    checkpoint_state[n, h] = S

            final_state[n, h] = S

    return output.to(torch.bfloat16), final_state, checkpoint_state
