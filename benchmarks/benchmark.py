import os
import torch
import triton
import triton.testing as tt
import matplotlib.pyplot as plt
import itertools

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
    casual_modes = [True, False]
    dtype = [torch.float32, torch.float16]
    results ={}

    for dtype, causal in itertools.product(dtype, casual_modes):
        config_key = f"{'FP16' if dtype == torch.float16 else 'FP32'}_causal={causal}"
        print(f"\n--- Running Sweep: {config_key} ---")
        print(
            f"{'N':<8} | {'Base Time (ms)':<15} | {'Kernel Time (ms)':<16} | {'Base Mem (MB)':<15} | {'Kernel Mem (MB)':<15}")
        print("-" * 80)

        results[config_key] = {
            "N": seq_lengths,
            "baseline_time_ms": [],
            "kernel_time_ms": [],
            "baseline_mem_mb": [],
            "kernel_mem_mb": [],
            "baseline_oom_at": None,
        }

        for N in seq_lengths:
            q = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
            k = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
            v = torch.randn(B, H, N, d, device="cuda", dtype=dtype)

            # 1. PyTorch Baseline
            if results[config_key]["baseline_oom_at"] is None:
                try:
                    base_time = benchmark_fn(baseline_attention, q, k, v, causal=causal)
                    base_mem = measure_memory(baseline_attention, q, k, v, causal=causal)
                    results[config_key]["baseline_time_ms"].append(base_time)
                    results[config_key]["baseline_mem_mb"].append(base_mem)
                    b_t_str, b_m_str = f"{base_time:.3f}", f"{base_mem:.2f}"
                except torch.cuda.OutOfMemoryError:
                    results[config_key]["baseline_oom_at"] = N
                    results[config_key]["baseline_time_ms"].append(None)
                    results[config_key]["baseline_mem_mb"].append(None)
                    b_t_str, b_m_str = "OOM", "OOM"
                    torch.cuda.empty_cache()
            else:
                results[config_key]["baseline_time_ms"].append(None)
                results[config_key]["baseline_mem_mb"].append(None)
                b_t_str, b_m_str = "OOM", "OOM"

            # 2. Fused Triton Kernel
            kernel_time = benchmark_fn(fused_attention, q, k, v, causal=causal)
            kernel_mem = measure_memory(fused_attention, q, k, v, causal=causal)
            results[config_key]["kernel_time_ms"].append(kernel_time)
            results[config_key]["kernel_mem_mb"].append(kernel_mem)

            print(f"{N:<8} | {b_t_str:<15} | {kernel_time:.3f}{'':<10} | {b_m_str:<15} | {kernel_mem:.2f}")

    return results


def plot_results(results):
    os.makedirs("results/plots", exist_ok=True)
    N_vals = [256, 512, 1024, 2048, 4096, 8192, 16384]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Execution Time (Non-Causal)
    ax1 = axes[0, 0]
    res_fp32 = results["FP32_causal=False"]
    res_fp16 = results["FP16_causal=False"]

    valid_b_32 = [(n, t) for n, t in zip(N_vals, res_fp32["baseline_time_ms"]) if t is not None]
    if valid_b_32:
        ax1.plot(*zip(*valid_b_32), "o--", color="crimson", label="PyTorch Baseline (FP32)")
    ax1.plot(N_vals, res_fp32["kernel_time_ms"], "s-", color="royalblue", label="Triton Kernel (FP32)")
    ax1.plot(N_vals, res_fp16["kernel_time_ms"], "d-", color="seagreen", label="Triton Kernel (FP16)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_title("Latency: Non-Causal (Pure SRAM Tiling)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Sequence Length (N)")
    ax1.set_ylabel("Execution Time (ms)")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend()

    # Panel 2: Execution Time (Causal)
    ax2 = axes[0, 1]
    res_fp32_c = results["FP32_causal=True"]
    res_fp16_c = results["FP16_causal=True"]

    valid_b_32_c = [(n, t) for n, t in zip(N_vals, res_fp32_c["baseline_time_ms"]) if t is not None]
    if valid_b_32_c:
        ax2.plot(*zip(*valid_b_32_c), "o--", color="crimson", label="PyTorch Baseline (FP32)")
    ax2.plot(N_vals, res_fp32_c["kernel_time_ms"], "s-", color="royalblue", label="Triton Kernel (FP32)")
    ax2.plot(N_vals, res_fp16_c["kernel_time_ms"], "d-", color="seagreen", label="Triton Kernel (FP16)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title("Latency: Causal (Tiling + Block-Skipping)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Sequence Length (N)")
    ax2.set_ylabel("Execution Time (ms)")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()

    # Panel 3: Peak GPU Memory (Non-Causal)
    ax3 = axes[1, 0]
    valid_m_32 = [(n, m) for n, m in zip(N_vals, res_fp32["baseline_mem_mb"]) if m is not None]
    if valid_m_32:
        ax3.plot(*zip(*valid_m_32), "o--", color="crimson", label="PyTorch Baseline O(N²)")
    ax3.plot(N_vals, res_fp32["kernel_mem_mb"], "s-", color="royalblue", label="Triton FP32 O(N)")
    ax3.plot(N_vals, res_fp16["kernel_mem_mb"], "d-", color="seagreen", label="Triton FP16 O(N)")

    if res_fp32["baseline_oom_at"]:
        oom_n = res_fp32["baseline_oom_at"]
        ax3.scatter([oom_n], [valid_m_32[-1][1] * 2], color="crimson", marker="x", s=100, zorder=5)
        ax3.text(oom_n * 0.85, valid_m_32[-1][1] * 1.5, f"Baseline OOM\n(N={oom_n})", color="crimson",
                 fontweight="bold", ha="right")

    ax3.set_xscale("log", base=2)
    ax3.set_yscale("log")
    ax3.set_title("Peak Memory: Non-Causal", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Sequence Length (N)")
    ax3.set_ylabel("Peak Memory (MB)")
    ax3.grid(True, which="both", linestyle="--", alpha=0.5)
    ax3.legend()

    # Panel 4: Peak GPU Memory (Causal)
    ax4 = axes[1, 1]
    valid_m_32_c = [(n, m) for n, m in zip(N_vals, res_fp32_c["baseline_mem_mb"]) if m is not None]
    if valid_m_32_c:
        ax4.plot(*zip(*valid_m_32_c), "o--", color="crimson", label="PyTorch Baseline O(N²)")
    ax4.plot(N_vals, res_fp32_c["kernel_mem_mb"], "s-", color="royalblue", label="Triton FP32 O(N)")
    ax4.plot(N_vals, res_fp16_c["kernel_mem_mb"], "d-", color="seagreen", label="Triton FP16 O(N)")

    if res_fp32_c["baseline_oom_at"]:
        oom_n = res_fp32_c["baseline_oom_at"]
        ax4.scatter([oom_n], [valid_m_32_c[-1][1] * 2], color="crimson", marker="x", s=100, zorder=5)
        ax4.text(oom_n * 0.85, valid_m_32_c[-1][1] * 1.5, f"Baseline OOM\n(N={oom_n})", color="crimson",
                 fontweight="bold", ha="right")

    ax4.set_xscale("log", base=2)
    ax4.set_yscale("log")
    ax4.set_title("Peak Memory: Causal", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Sequence Length (N)")
    ax4.set_ylabel("Peak Memory (MB)")
    ax4.grid(True, which="both", linestyle="--", alpha=0.5)
    ax4.legend()

    plt.tight_layout()
    plot_path = "results/plots/benchmark_scaling_comprehensive.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n[INFO] Empirical plots saved to '{plot_path}'")



if __name__ == "__main__":
    results = run_benchmark_suite()
    plot_results(results)