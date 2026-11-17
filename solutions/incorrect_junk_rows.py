"""INCORRECT on purpose: snapshot rows of prompts with no snapshot hold junk.

This is solutions/correct_chunked.py with one change, marked THE BUG in
`run`: checkpoint_state is not allocated with zeros, so the rows of prompts
with checkpoint_offsets == 0 are never written and keep whatever was in the
buffer. This is what production does (it leaves unused rows untouched), but
the task requires those rows to be zeros.

output, final_state and every real snapshot row are correct. It fails every
test that has a prompt with no snapshot. S1 has none (its one prompt has a
snapshot), so S1 passes.
"""

import torch

CHUNK = 64
LOWER_BOUND = -5.0
L2NORM_EPS = 1e-6


def l2norm(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + L2NORM_EPS)


def intra(q, k, gk, beta, scale):
    """Stage 2. All inputs are [H, C, ...] for one chunk of C tokens.

    Akk[t, j] (j < t) = beta_t * sum_d k_t[d] * k_j[d] * exp(gk_t[d] - gk_j[d]):
    how much token j's write is seen, after decay, by token t's key.
    Aqk[t, j] (j <= t) is the same with q_t in place of k_t, times scale.
    """
    num_heads, C, _ = q.shape
    # exp(gk_t - gk_j) for j <= t. The exponent is never positive there; the
    # clamp only keeps the unused j > t half from overflowing.
    decay = torch.exp((gk[:, :, None, :] - gk[:, None, :, :]).clamp(max=0.0))   # [H, t, j, K]
    k_decayed = k[:, None, :, :] * decay
    Aqk = scale * torch.einsum("htd,htjd->htj", q, k_decayed).tril()
    Akk = beta[:, :, None] * torch.einsum("htd,htjd->htj", k, k_decayed).tril(-1)

    eye = torch.eye(C, dtype=q.dtype, device=q.device).expand(num_heads, C, C)
    Akk_inv = torch.linalg.solve_triangular(eye + Akk, eye, upper=False, unitriangular=True)
    return Aqk, Akk_inv


def recompute_w_u(k, v, beta, Akk_inv, gk):
    """Stage 3."""
    w = Akk_inv @ (k * beta[:, :, None] * torch.exp(gk))
    u = Akk_inv @ (v * beta[:, :, None])
    kg = k * torch.exp(gk[:, -1:, :] - gk)
    return w, u, kg


def run(q, k, v, g, beta, A_log, dt_bias, scale, cu_seqlens, initial_state, checkpoint_offsets):
    bounds = cu_seqlens.tolist()
    offsets = checkpoint_offsets.tolist()

    # Elementwise preparation for all tokens at once, in float32, as [H, T, ...].
    q = l2norm(q.float()).transpose(0, 1)
    k = l2norm(k.float()).transpose(0, 1)
    v = v.float().transpose(0, 1)
    beta = torch.sigmoid(beta.float()).transpose(0, 1)
    gate = LOWER_BOUND * torch.sigmoid(torch.exp(A_log)[None, :, None] * (g.float() + dt_bias[None]))
    gate = gate.transpose(0, 1)

    output = torch.zeros_like(v)
    final_state = torch.zeros_like(initial_state)
    # THE BUG: the snapshot buffer is not zeroed. The fixed-seed random values
    # stand in for whatever an uninitialized buffer (torch.empty) would hold,
    # so that the failure is the same on every run.
    junk = torch.Generator().manual_seed(0)
    checkpoint_state = torch.randn(initial_state.shape, generator=junk).to(initial_state)

    for n in range(len(bounds) - 1):
        h = initial_state[n].clone()                       # [H, V, K]
        for start in range(bounds[n], bounds[n + 1], CHUNK):
            end = min(start + CHUNK, bounds[n + 1])        # the last chunk may be shorter
            chunk = slice(start, end)

            gk = gate[:, chunk].cumsum(dim=1)                                        # stage 1
            Aqk, Akk_inv = intra(q[:, chunk], k[:, chunk], gk, beta[:, chunk], scale)  # stage 2
            w, u, kg = recompute_w_u(k[:, chunk], v[:, chunk], beta[:, chunk], Akk_inv, gk)  # stage 3

            v_new = u - w @ h.transpose(-1, -2)                                      # stage 4
            output[:, chunk] = scale * (q[:, chunk] * torch.exp(gk)) @ h.transpose(-1, -2) + Aqk @ v_new  # stage 5

            # Snapshot after m tokens of this chunk, without leaving the chunk:
            # the same memory update as below, cut off at token m.
            m = offsets[n] - (start - bounds[n])
            if 0 < m <= end - start:
                kg_m = k[:, start:start + m] * torch.exp(gk[:, m - 1:m, :] - gk[:, :m, :])
                checkpoint_state[n] = h * torch.exp(gk[:, m - 1, :])[:, None, :] + v_new[:, :m].transpose(-1, -2) @ kg_m

            h = h * torch.exp(gk[:, -1, :])[:, None, :] + v_new.transpose(-1, -2) @ kg     # stage 4, memory
        final_state[n] = h

    return output.transpose(0, 1).to(torch.bfloat16), final_state, checkpoint_state
