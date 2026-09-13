import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_attention_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,  # strides for Q: (B, H, N, d)
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        N, d,
        num_heads, sm_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    q_base = batch_idx * stride_qb + head_idx * stride_qh
    k_base = batch_idx * stride_kb + head_idx * stride_kh
    v_base = batch_idx * stride_vb + head_idx * stride_vh
    o_base = batch_idx * stride_ob + head_idx * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd

    k_ptrs = k_ptr + k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
    v_ptrs = v_ptr + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < d)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # --- Loop over K/V tiles ---
    for start_n in range(0, N, BLOCK_N):
        curr_offs_n = start_n + offs_n

        k_mask = (offs_d[:, None] < d) & (curr_offs_n[None, :] < N)
        v_mask = (curr_offs_n[:, None] < N) & (offs_d[None, :] < d)

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        scores = tl.dot(q, k) * sm_scale

        if CAUSAL:
            causal_mask = offs_m[:, None] >= curr_offs_n[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        seq_mask = curr_offs_n[None, :] < N
        scores = tl.where(seq_mask, scores, float("-inf"))

        m_tile = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_tile)

        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        p = tl.exp(scores - m_new[:, None])

        l_i = l_i + tl.sum(p, axis=1)
        p_cast = p.to(v.dtype)
        acc = acc + tl.dot(p_cast, v)

        m_i = m_new

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    out = acc / l_i[:, None]

    # Setup output pointers and store
    out_ptrs = out_ptr + o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    out_mask = (offs_m[:, None] < N) & (offs_d[None, :] < d)
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=out_mask)

def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False):
    # Shape validation checks
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"Expected 4D tensors for q, k, v (B, H, N, d), but got shapes: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
        )

    if q.size(-1) != k.size(-1):
        raise ValueError(
            f"Query feature dimension ({q.size(-1)}) must match Key feature dimension ({k.size(-1)})."
        )

    if k.size(-2) != v.size(-2):
        raise ValueError(
            f"Number of Key tokens ({k.size(-2)}) must match Value tokens ({v.size(-2)})."
        )

    if q.size(0) != k.size(0) or q.size(0) != v.size(0) or q.size(1) != k.size(1) or q.size(1) != v.size(1):
        raise ValueError(
            f"Batch and head dimensions must match across q, k, v. "
            f"Got q: {(q.size(0), q.size(1))}, k: {(k.size(0), k.size(1))}, v: {(v.size(0), v.size(1))}."
        )

    if causal and q.size(-2) != k.size(-2):
        raise ValueError(
            f"Causal masking requires N_q == N_k, but got N_q={q.size(-2)} and N_k={k.size(-2)}."
        )
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("fused_attention requires CUDA tensors; got devices "
                         f"q={q.device}, k={k.device}, v={v.device}.")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q, k, v must be on the same device, got "
                         f"q={q.device}, k={k.device}, v={v.device}.")

    B, H, N, d = q.shape
    sm_scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q)

    # Block configurations
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(d)

    grid = (
        triton.cdiv(N, BLOCK_M),
        B * H,
    )

    flash_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, d,
        H, sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        CAUSAL=causal,
    )

    return out