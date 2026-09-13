import sys
import torch
import itertools
from src.baseline import baseline_attention
from src.kernel import fused_attention
import traceback


def run_single_test(B, H, N, d, causal, dtype, device="cuda"):
    torch.manual_seed(42)
    q = torch.randn((B, H, N, d), dtype=dtype, device=device)
    k = torch.randn((B, H, N, d), dtype=dtype, device=device)
    v = torch.randn((B, H, N, d), dtype=dtype, device=device)

    if dtype == torch.float32:
        rtol, atol = 1e-4, 1e-4
    else:
        rtol, atol = 1e-2, 1e-2

    out_ref = baseline_attention(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        causal=causal
    ).to(dtype)

    out_triton = fused_attention(q, k, v, causal=causal)

    is_close = torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol)

    diff = torch.abs(out_triton - out_ref)
    max_err, max_idx = torch.max(diff.view(-1), dim=0)
    mean_err = torch.mean(diff).item()

    unflattened_idx = torch.unravel_index(max_idx, out_triton.shape)
    worst_idx = tuple(t.item() for t in unflattened_idx)

    return {
        "passed": is_close,
        "has_nan": torch.isnan(out_triton).any().item(),
        "has_inf": torch.isinf(out_triton).any().item(),
        "max_error": max_err.item(),
        "mean_error": mean_err,
        "worst_idx": worst_idx,
        "val_triton": out_triton[worst_idx].item(),
        "val_ref": out_ref[worst_idx].item(),
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA is required to run Triton correctness tests.")
        sys.exit(1)

    shape_configs = [
        (2, 4, 128, 64),  # Clean block boundary
        (2, 4, 100, 64),  # N < BLOCK_N
        (1, 2, 130, 64),  # N > BLOCK_N
        (2, 4, 128, 80),  # Non-power-of-2 head dimension
    ]
    causal_options = [False, True]
    dtypes = [torch.float32, torch.float16]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    print("Starting Flash Attention Correctness Suite\n" + "=" * 70)

    for (B, H, N, d), causal, dtype in itertools.product(shape_configs, causal_options, dtypes):
        total_tests += 1
        config_str = f"B={B}, H={H}, N={N}, d={d} | causal={causal} | {dtype}"

        try:
            res = run_single_test(B, H, N, d, causal, dtype)
        except Exception as e:
            print(f"\n[FAIL] {config_str}\n  Kernel/Harness crashed with exception:")
            traceback.print_exc()
            failed_tests += 1
            continue

        if res["passed"]:
            print(f"[PASS] {config_str}")
            passed_tests += 1
        else:
            failed_tests += 1
            b_err, h_err, n_err, d_err = res["worst_idx"]
            print(f"\n[FAIL] {config_str}")
            print(f"  Diagnostics:")
            print(f"    NaN detected in output:       {res['has_nan']}")
            print(f"    Inf detected in output:       {res['has_inf']}")
            print(f"    Mean Absolute Error:          {res['mean_error']:.6e}")
            print(f"    Max Absolute Error:           {res['max_error']:.6e}")
            print(f"    Worst mismatch index:         (b={b_err}, h={h_err}, n={n_err}, d={d_err})")
            print(f"    Value at index (Triton):      {res['val_triton']:.6f}")
            print(f"    Value at index (Baseline):    {res['val_ref']:.6f}")
            print("-" * 70)

    print("=" * 70)
    print(f"Test Summary: {passed_tests}/{total_tests} passed, {failed_tests} failed.")

    if failed_tests > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()