import torch
import math
def baseline_attention(q, k, v, causal=False):
    #q, k,v dimension ( B , H , N, d)
    if q.size(-1) != k.size(-1):
        raise ValueError(
            f"Query feature dimension ({q.size(-1)}) must match Key feature dimension ({k.size(-1)}) "
            f"to perform the dot product."
        )
    if k.size(-2) != v.size(-2):
        raise ValueError(
            f"Number of Key tokens ({k.size(-2)}) must match the number of Value tokens ({v.size(-2)}). "
            f"Each key must map exactly to one value."
        )
    if causal and q.size(-2) != k.size(-2):
        raise ValueError(
            f"Causal masking requires queries and keys to have the same sequence length, "
            f"but got N_q={q.size(-2)} and N_k={k.size(-2)}. Masking 'future' tokens only makes sense "
            f"when a sequence is attending to itself."
        )

    d = q.size(-1)
    N_q= q.size(-2)
    N_k = k.size(-2)

    scores = torch.matmul(q , k.transpose(-2,-1))/math.sqrt(d)  # shape (B, H, N, N)

    if causal:
        mask = torch.triu(torch.ones(N_q, N_k, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    return output
