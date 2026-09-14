# Fused Flash Attention in Triton

A from-scratch Triton implementation of Flash Attention: a tiled kernel with online
softmax, causal masking with block-skipping, dynamic hardware-aware block sizing, and
autotuning — validated against a naive PyTorch baseline and benchmarked for latency
and memory scaling. See `WRITEUP.md` for the full technical writeup, including two
debugging investigations (a shared-memory resource bug, and a since-corrected FP16
performance anomaly).

## Repo Structure

```
.
├── README.md
├── WRITEUP.md
├── requirements.txt
├── src/
│   ├── baseline.py      # naive PyTorch reference implementation (4.1)
│   └── kernel.py        # Triton kernel: tiling, online softmax, causal
│                         #   masking, autotuning, and the Python wrapper (4.2–4.3, 4.5, 4.7)
├── tests/
│   └── test_correctness.py   # correctness harness vs. baseline (4.4)
├── benchmarks/
│   └── benchmark.py     # latency + memory sweep, OOM handling, plotting (4.6)
└── results/
    └── plots/           # generated benchmark plots (created on first run)
```

## Installation

Requires a CUDA-capable GPU with compute capability 7.0+ (Tensor Cores). Developed
and tested on a Colab T4 (compute capability 7.5). GPUs older than compute 7.0 (e.g.
Pascal-generation cards like the Quadro P2000) lack Tensor Cores and are not
supported — see `WRITEUP.md` for details on this incompatibility encountered during
development.

1. Install PyTorch with a CUDA build matching your system's CUDA toolkit version
   (check with `nvidia-smi` or `nvcc --version`). Example for CUDA 12.6:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

2. Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Verify:

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

## Usage

Run correctness tests (compares the Triton kernel against the PyTorch baseline across
multiple shapes, causal on/off, and fp32/fp16, with `torch.allclose`-based tolerances):

```bash
python -m tests.test_correctness
```

Run benchmarks (wall-clock latency and peak memory vs. sequence length, for both
causal and non-causal, fp32 and fp16 — saves plots to `results/plots/`):

```bash
python -m benchmarks.benchmark
```

Use the kernel directly:

```python
import torch
from src.kernel import fused_attention

q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)

out = fused_attention(q, k, v, causal=True)  # shape (2, 8, 1024, 64)
```

## Notes

- Both files are always run from the project root, since `src`, `tests`, and
  `benchmarks` are imported as packages (`python -m ...`, not `python
  tests/test_correctness.py` from inside the `tests/` directory).
- Block sizes (`BLOCK_M`, `BLOCK_N`) are chosen automatically per-call via
  `@triton.autotune`, pruned against the actual GPU's shared-memory budget
  (queried at runtime, not hardcoded), so behavior adapts across GPU generations.
- Causal masking and its block-skipping optimization live in the same file as the
  core kernel (`src/kernel.py`) rather than a separate file, since they share the
  same correctness-critical logic and should not risk drifting out of sync.