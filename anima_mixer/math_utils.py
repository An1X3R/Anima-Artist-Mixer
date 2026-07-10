"""Small tensor helpers used by the cross-attention wrapper."""

import torch


def project_perpendicular(delta, base):
    # FP16 rounds 1e-8 to zero, so the old denominator guard could still
    # produce NaNs for zero or near-zero attention rows.
    delta_f32 = delta.to(torch.float32)
    base_f32 = base.to(torch.float32)
    base_norm_sq = (base_f32 * base_f32).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    proj_coef = (delta_f32 * base_f32).sum(dim=-1, keepdim=True) / base_norm_sq
    return (delta_f32 - proj_coef * base_f32).to(delta.dtype)


def limit_delta_norm(delta, base, cap_ratio):
    cap_ratio = float(cap_ratio)
    if cap_ratio <= 0.0:
        return delta
    base_norm = torch.linalg.vector_norm(base.to(torch.float32), dim=-1, keepdim=True)
    delta_norm = torch.linalg.vector_norm(delta.to(torch.float32), dim=-1, keepdim=True)
    limit = (base_norm * cap_ratio).clamp(min=1e-8)
    scale = (limit / delta_norm.clamp(min=1e-8)).clamp(max=1.0).to(delta.dtype)
    return delta * scale


def lowrank_rows_deterministic(rows, rank):
    """Project rows onto the top-k deterministic row-space directions."""
    if rank >= rows.shape[0]:
        return rows
    gram = rows @ rows.transpose(0, 1)
    evals, evecs = torch.linalg.eigh(gram)
    idx = torch.argsort(evals, descending=True)[:rank]
    basis = evecs[:, idx]
    return basis @ (basis.transpose(0, 1) @ rows)
