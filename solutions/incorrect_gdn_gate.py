"""INCORRECT on purpose: a chunked GDN (Gated DeltaNet) kernel reused for KDA.

GDN is KDA's closest relative, and a chunked GDN prefill kernel already
exists, so reusing one is the natural shortcut. The two differ in one thing:
GDN has one decay gate per head and token, KDA has one per key channel.

THE BUG is in `run`, marked below: KDA's 128 per-channel gates are averaged
into one gate per head so that the GDN kernel can be used unchanged.
Everything else (the 64-token chunks, the triangular solve, prompt
boundaries, snapshots) is a correct GDN kernel, and with gates that are equal
across channels it would give the right answer.
"""

import torch

CHUNK = 64
LOWER_BOUND = -5.0
L2NORM_EPS = 1e-6


def l2norm(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + L2NORM_EPS)


def gdn_chunk(q, k, v, beta, g, h, scale, snapshot_at):
    """One chunk of the GDN recurrence for all heads. `g` is [H, C]: one gate per head and token.

    Returns the chunk's output, the memory after it, and (if 0 < snapshot_at
    <= C) the memory after that many tokens of the chunk.
    """
    num_heads, C, _ = q.shape
    gc = g.cumsum(dim=1)                                                        # [H, C]
    decay = torch.exp((gc[:, :, None] - gc[:, None, :]).clamp(max=0.0))         # [H, t, j]

    Aqk = (scale * (q @ k.transpose(-1, -2)) * decay).tril()
    Akk = (beta[:, :, None] * (k @ k.transpose(-1, -2)) * decay).tril(-1)
    eye = torch.eye(C, dtype=q.dtype, device=q.device).expand(num_heads, C, C)
    Akk_inv = torch.linalg.solve_triangular(eye + Akk, eye, upper=False, unitriangular=True)

    w = Akk_inv @ (k * (beta * torch.exp(gc))[:, :, None])
    u = Akk_inv @ (v * beta[:, :, None])
    v_new = u - w @ h.transpose(-1, -2)
    out = scale * (q * torch.exp(gc)[:, :, None]) @ h.transpose(-1, -2) + Aqk @ v_new

    def memory_after(m):
        kg = k[:, :m] * torch.exp(gc[:, m - 1:m] - gc[:, :m])[:, :, None]
        return h * torch.exp(gc[:, m - 1])[:, None, None] + v_new[:, :m].transpose(-1, -2) @ kg

    snapshot = memory_after(snapshot_at) if 0 < snapshot_at <= C else None
    return out, memory_after(C), snapshot


def run(q, k, v, g, beta, A_log, dt_bias, scale, cu_seqlens, initial_state, checkpoint_offsets):
    bounds = cu_seqlens.tolist()
    offsets = checkpoint_offsets.tolist()

    q = l2norm(q.float()).transpose(0, 1)
    k = l2norm(k.float()).transpose(0, 1)
    v = v.float().transpose(0, 1)
    beta = torch.sigmoid(beta.float()).transpose(0, 1)
    gate = LOWER_BOUND * torch.sigmoid(torch.exp(A_log)[None, :, None] * (g.float() + dt_bias[None]))

    # THE BUG: one shared gate per head (the mean over the 128 channels),
    # because the GDN kernel below takes a [T, H] gate.
    gate = gate.mean(dim=-1).transpose(0, 1)                                    # [H, T]

    output = torch.zeros_like(v)
    final_state = torch.zeros_like(initial_state)
    checkpoint_state = torch.zeros_like(initial_state)

    for n in range(len(bounds) - 1):
        h = initial_state[n].clone()
        for start in range(bounds[n], bounds[n + 1], CHUNK):
            end = min(start + CHUNK, bounds[n + 1])
            chunk = slice(start, end)
            output[:, chunk], h, snapshot = gdn_chunk(
                q[:, chunk], k[:, chunk], v[:, chunk], beta[:, chunk], gate[:, chunk], h, scale,
                snapshot_at=offsets[n] - (start - bounds[n]))
            if snapshot is not None:
                checkpoint_state[n] = snapshot
        final_state[n] = h

    return output.transpose(0, 1).to(torch.bfloat16), final_state, checkpoint_state
