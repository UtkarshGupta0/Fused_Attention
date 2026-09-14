# Fused Flash Attention in Triton — Writeup

## 1. Overview

This project implements a fused Flash Attention kernel in Triton, covering: a naive
PyTorch baseline (4.1), a tiled kernel with online softmax (4.2+4.3), a correctness
harness (4.4), causal masking with block-skipping (4.5), wall-clock/memory benchmarking
(4.6), and autotuning (4.7). The sections below focus on the design decisions and two
debugging investigations that were the most substantive parts of the work, rather than
walking through the spec item by item.

## 2. Core Algorithm: Online Softmax

Standard softmax requires the full row of attention scores to compute the row max and
normalization sum. A tiled kernel never has the full row in memory at once — it sees
one K/V block at a time — so the softmax must be computed incrementally.

For each new K/V tile, the kernel maintains three running values per query row: `m`
(running max), `l` (running sum of exponentials), and `acc` (running weighted output).
When a new tile's local max exceeds the running max, every value already accumulated
was computed relative to a max that's now known to be wrong — those values are
systematically **too large**, since subtracting a smaller max than the true max
overstates every exponent. The correction factor `alpha = exp(m_old - m_new)` is
therefore always ≤ 1, and rescaling `l` and `acc` by `alpha` before adding the new
tile's contribution keeps every quantity in the running accumulators expressed
relative to the same reference point at all times. Initialization is `m = -inf, l = 0,
acc = 0`, which makes the first tile's rescale a no-op by construction (`alpha =
exp(-inf - m_tile) = 0`, multiplying already-zero accumulators).

4.2 (tiling) and 4.3 (the rescale) are not separable milestones — a tiled kernel
without the rescale is not a simpler version of a working kernel, it is a kernel that
runs without crashing but produces silently incorrect numbers.

## 3. Causal Masking: Two Distinct Mechanisms

Causal masking in this kernel operates at two different granularities that are easy to
conflate but solve different problems:

- **Block-skipping** (coarse, performance): for a query block with max row index
  `(pid_m+1)*BLOCK_M - 1`, any K/V tile starting at `start_n` greater than that index
  is *entirely* masked — every position in it violates `i >= j`. The loop bound
  `hi = min(N, (pid_m+1)*BLOCK_M)` skips these tiles without ever loading them,
  cutting roughly half the memory traffic and matmuls for causal attention.
- **Element-wise masking** (fine, correctness): the tile that straddles the causal
  diagonal itself still contains a mix of valid and invalid positions (e.g. for
  `BLOCK_M=BLOCK_N=64, pid_m=0`, row `i=5` cannot see column `j=50`, but row `i=63`
  can see column `j=10` — same tile, different outcomes). This tile cannot be skipped,
  and the per-position `tl.where` mask inside the loop is still required for it.

Removing the fine-grained mask under the assumption that block-skipping subsumes it
would silently allow queries in the diagonal tile to attend to future tokens.

## 4. Investigation: Shared-Memory Resource Limit

Early correctness testing on a non-power-of-2 head dimension (`d=80`) failed with:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 98304, Hardware limit: 65536.
```

**Diagnosis.** `d=80` gets padded to `BLOCK_D=128` via `triton.next_power_of_2`
(required because `tl.arange` and Triton's compiler need power-of-2 tile dimensions).
The kernel holds a Q tile `(BLOCK_M, BLOCK_D)`, a K tile `(BLOCK_D, BLOCK_N)`, and a V
tile `(BLOCK_N, BLOCK_D)` in shared memory simultaneously. With `BLOCK_M=BLOCK_N=64`,
`BLOCK_D=128`, fp32 (4 bytes/element):

```
(64*128 + 128*64 + 64*128) * 4 bytes = 24,576 elements * 4 = 98,304 bytes
```

This matches the error exactly. The same shape in fp16 (2 bytes/element) requires
49,152 bytes — under the T4's 65,536-byte limit — which explains why the fp16 version
of this exact config passed while fp32 failed.

**Fix.** Block sizes are computed dynamically rather than hardcoded, querying the
actual device's shared memory budget (`torch.cuda.get_device_properties(q.device)
.shared_memory_per_block` — verified directly against `dir(props)` output on the
target GPU rather than assumed from an API guess, after an initial wrong guess
(`max_shared_memory_per_block`) silently fell back to a default via `getattr`) and
applying an 80% safety margin to leave headroom for Triton's own pipelining buffers
and alignment overhead:

```python
max_btile = int(shared_mem_per_block * 0.8) // (3 * BLOCK_D * element_bytes)
# decay from a preferred 64 down to the largest power-of-2 that fits
```

This keeps full 64×64 tiles for shapes that don't need to shrink (e.g. `d=64`) while
automatically scaling down only where the hardware forces it, and generalizes to GPUs
with larger shared memory budgets (A100/H100) without code changes.

## 5. Investigation: FP16 Performance Anomaly (and a Revised Conclusion)

**Initial finding.** Benchmarking with a fixed configuration (`BLOCK_M=BLOCK_N=64,
num_warps=4, num_stages=3`) showed FP16 running 8–13x *slower* than FP32 at large
sequence lengths (e.g. N=16384 causal: FP32 210ms vs FP16 2088ms) — the opposite of
the expected direction, since FP16 halves memory traffic and should enable Tensor
Core throughput.

**Initial investigation.** Reproduced in isolation with direct `torch.cuda.Event`
timing (ruling out benchmark-script artifacts). Directly printing the block sizes
chosen by the (at the time, manual) sizing logic showed FP16 got the *larger*, more
favorable tile size (64×64) while FP32 was clamped to 32×32 by the shared-memory
formula — ruling out "FP16 got worse tiles" as the explanation, since it should have
had an advantage, not a penalty. Cross-referencing Triton's issue tracker surfaced
documented community reports of FP16/BF16 underperforming FP32 in `tl.dot`-based
kernels, and FlashAttention's own documentation notes that Turing-generation GPUs
(the T4 used here) require separate, specialized handling rather than the mainline
kernel path. The working conclusion at that point was that this was a fundamental
Triton/Turing FP16 Tensor Core dispatch limitation.

**Revision after autotuning (4.7).** After implementing `@triton.autotune` with a
resource-aware config pruner and re-benchmarking, the FP16 penalty at N=4096 dropped
from ~10x to near parity (132ms → 13ms at that shape). Printing the winning configs
via `TRITON_PRINT_AUTOTUNING=1` showed FP32 and FP16 converged on the **identical**
optimal configuration (`BLOCK_M=32, BLOCK_N=32, num_warps=4, num_stages=2`) at the
shapes tested. This overturns the original conclusion: the anomaly was not a
fundamental FP16 hardware/dispatch limitation, but an artifact of one specific,
hand-picked fixed configuration (`num_stages=3`) that was poorly suited to the T4's
64KB shared-memory budget, and which happened to penalize FP16 far more severely than
FP32 for that specific config. The Triton issue-tracker reports of general FP16
underperformance may still reflect a real, narrower effect, but they are not the
primary explanation for what was observed here.

**Takeaway.** A reasonable-looking, manually chosen kernel configuration can produce
large, dtype-dependent performance artifacts that look like fundamental hardware
limitations but are actually just poor tuning for that specific hardware's resource
constraints — which is precisely the failure mode systematic autotuning (4.7) exists
to catch.

## 6. Results Summary

- **Correctness:** 16/16 configurations pass (`B,H,N,d` combinations spanning clean
  and non-clean block boundaries, non-power-of-2 head dimension, causal on/off,
  fp32/fp16), validated against a naive PyTorch reference with dtype-appropriate
  `torch.allclose` tolerances (fp32: 1e-4, fp16: 1e-2 — justified by mantissa
  precision: fp32 has ~7 decimal digits, fp16 ~3.3, and tiling reorders floating-point
  summation relative to the reference, so exact equality is not the right bar).
- **Memory scaling:** Baseline OOMs at N=16384 on all tested configurations; the
  Triton kernel continues to run, consistent with O(N) vs O(N²) memory scaling from
  never materializing the full N×N score matrix.
- **Latency:** Pre-autotuning, the kernel was competitive with but not consistently
  faster than PyTorch's baseline at small-to-medium N (baseline uses highly optimized
  cuBLAS kernels), only clearly winning once the baseline OOMs. Post-autotuning, FP16
  latency improved by up to ~10x at large N (see Section 5).

## 7. Known Limitations / Things Not Done
- backward pass was not implemented
- I tested the code only on the T4 google colab gpu as my laptop gpu was too old and no compatible drivers for triton were present for it
- It took em a lot of iteration as this was my first time working with triton so had to take a lot of help in debugging from Ai tools which is evident in my commit history