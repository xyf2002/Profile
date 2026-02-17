#!/usr/bin/env python3
"""
Profile a single GEMM operation using torch.matmul with NVIDIA Nsys
Usage: nsys profile python3 profile_gemm_nsys.py --n 1024 --k 10240 --m 512
"""

import torch
import argparse
import time

def profile_single_gemm(n, k, m, iterations=10):
    """
    Profile a single GEMM operation for Nsys GPU metrics collection.
    Should have enough iterations for meaningful sampling.
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nProfiling GEMM for Nsys: n={n}, k={k}, m={m}")
    print(f"Matrix A shape: ({n}, {k})")
    print(f"Matrix B shape: ({k}, {m})")
    print(f"Output shape: ({n}, {m})")
    print(f"Total FLOPs: {2 * n * k * m / 1e9:.2f} GFLOPs")
    print(f"Iterations: {iterations}\n")
    
    # Create input tensors
    A = torch.randn(n, k, dtype=torch.float32, device=device)
    B = torch.randn(k, m, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(5):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Main profiling loop - Nsys will collect GPU metrics during this
    print("Running kernels for Nsys GPU metrics collection...")
    start_time = time.time()
    
    for i in range(iterations):
        result = torch.matmul(A, B)
        torch.cuda.synchronize()
        if i == 0:
            print(f"Result shape: {result.shape}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nProfiling complete!")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Average time per iteration: {elapsed/iterations:.3f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile GEMM with NVIDIA Nsys")
    parser.add_argument("--n", type=int, required=True, help="Matrix A rows")
    parser.add_argument("--k", type=int, required=True, help="Shared dimension")
    parser.add_argument("--m", type=int, required=True, help="Matrix B columns")
    parser.add_argument("--iter", type=int, default=10, help="Number of iterations")
    
    args = parser.parse_args()
    profile_single_gemm(args.n, args.k, args.m, args.iter)
