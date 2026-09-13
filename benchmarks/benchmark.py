import os
import torch
import triton
import triton.testing as tt
import matplotlib.pyplot as plt

from src.baseline import baseline_attention
from src.kernel import fused_attention


def benchmark_fn(fn, *args, **kwargs):
    return tt.do_bench(lambda: fn(*args, **kwargs))


def measure_memory(fn, *args, **kwargs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    mem_bytes = torch.cuda.max_memory_allocated()
    return mem_bytes / (1024 * 1024)


def run_benchmark_suite():
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]
    B, H, d = 2, 4, 64
    dtype = torch.float32

    results = {
        "N": seq_lengths,
        "baseline_time_ms": [],
        "kernel_time_ms": [],
        "baseline_mem_mb": [],
        "kernel_mem_mb": [],
        "baseline_oom_at": None,
    }

    print(
        f"{'N':<8} | {'Base Time (ms)':<15} | {'Kernel Time (ms)':<16} | {'Base Mem (MB)':<15} | {'Kernel Mem (MB)':<15}")
    print("-" * 80)

    for N in seq_lengths:
        q = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
        k = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
        v = torch.randn(B, H, N, d, device="cuda", dtype=dtype)

        if results["baseline_oom_at"] is None:
            try:
                base_time = benchmark_fn(baseline_attention, q, k, v, causal=True)
                base_mem = measure_memory(baseline_attention, q, k, v, causal=True)
                results["baseline_time_ms"].append(base_time)
                results["baseline_mem_mb"].append(base_mem)
                base_time_str = f"{base_time:.3f}"
                base_mem_str = f"{base_mem:.2f}"
            except torch.cuda.OutOfMemoryError:
                results["baseline_oom_at"] = N
                results["baseline_time_ms"].append(None)
                results["baseline_mem_mb"].append(None)
                base_time_str = "OOM"
                base_mem_str = "OOM"
                torch.cuda.empty_cache()
        else:
            results["baseline_time_ms"].append(None)
            results["baseline_mem_mb"].append(None)
            base_time_str = "OOM"
            base_mem_str = "OOM"

        kernel_time = benchmark_fn(fused_attention, q, k, v, causal=True)
        kernel_mem = measure_memory(fused_attention, q, k, v, causal=True)
        results["kernel_time_ms"].append(kernel_time)
        results["kernel_mem_mb"].append(kernel_mem)

        print(f"{N:<8} | {base_time_str:<15} | {kernel_time:.3f}{'':<10} | {base_mem_str:<15} | {kernel_mem:.2f}")

    return results


def plot_results(results):
    os.makedirs("results/plots", exist_ok=True)
    N_vals = results["N"]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)

    valid_base_t = [(n, t) for n, t in zip(N_vals, results["baseline_time_ms"]) if t is not None]
    if valid_base_t:
        base_n, base_t = zip(*valid_base_t)
        plt.plot(base_n, base_t, "o--", color="crimson", label="PyTorch Baseline")

    plt.plot(N_vals, results["kernel_time_ms"], "s-", color="royalblue", label="Triton Flash Attention")

    if results["baseline_oom_at"] is not None:
        oom_n = results["baseline_oom_at"]
        plt.axvline(x=oom_n, color="crimson", linestyle=":", alpha=0.7)
        plt.text(oom_n * 0.9, max(results["kernel_time_ms"]) * 0.5, f"Baseline OOM\n(N={oom_n})",
                 color="crimson", fontweight="bold", ha="right")

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence Length (N)", fontsize=11)
    plt.ylabel("Execution Time (ms) - Log Scale", fontsize=11)
    plt.title("Wall-Clock Execution Time vs. Sequence Length", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.subplot(1, 2, 2)

    valid_base_m = [(n, m) for n, m in zip(N_vals, results["baseline_mem_mb"]) if m is not None]
    if valid_base_m:
        base_n, base_m = zip(*valid_base_m)
        plt.plot(base_n, base_m, "o--", color="crimson", label="PyTorch Baseline (Naive O(N²))")

    plt.plot(N_vals, results["kernel_mem_mb"], "s-", color="royalblue", label="Triton Kernel (O(N))")

    if results["baseline_oom_at"] is not None:
        oom_n = results["baseline_oom_at"]
        plt.axvline(x=oom_n, color="crimson", linestyle=":", alpha=0.7)
        plt.text(oom_n * 0.9, max(results["kernel_mem_mb"]) * 0.5, f"Baseline OOM\n(N={oom_n})",
                 color="crimson", fontweight="bold", ha="right")

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence Length (N)", fontsize=11)
    plt.ylabel("Peak GPU Memory (MB) - Log Scale", fontsize=11)
    plt.title("Peak Memory Usage vs. Sequence Length", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plot_path = "results/plots/benchmark_scaling.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n[INFO] Benchmark plots successfully saved to '{plot_path}'")


if __name__ == "__main__":
    results = run_benchmark_suite()
    plot_results(results)