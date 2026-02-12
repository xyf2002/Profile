#!/usr/bin/env python3
"""
Profile a single GEMM operation using torch.matmul with NVIDIA NCU
Usage: ncu --set full python profile_gemm_single.py --n 1024 --k 10240 --m 512
"""

import torch
import argparse

def profile_single_gemm(n, k, m, iterations=10):
    """
    Profile a single GEMM operation
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nProfiling GEMM: n={n}, k={k}, m={m}")
    print(f"Matrix A shape: ({n}, {k})")
    print(f"Matrix B shape: ({k}, {m})")
    print(f"Output shape: ({n}, {m})")
    print(f"Total FLOPs: {2 * n * k * m / 1e9:.2f} GFLOPs\n")
    
    # Create input tensors
    A = torch.randn(n, k, dtype=torch.float32, device=device)
    B = torch.randn(k, m, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(5):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Profiling loop - NCU will capture this
    print("Running kernels for NCU profiling...")
    for i in range(iterations):
        result = torch.matmul(A, B)
        torch.cuda.synchronize()
        if i == 0:
            print(f"Result shape: {result.shape}")
    
    print("\nProfiling complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile GEMM with NVIDIA NCU")
    parser.add_argument("--n", type=int, required=True, help="Matrix A rows")
    parser.add_argument("--k", type=int, required=True, help="Shared dimension")
    parser.add_argument("--m", type=int, required=True, help="Matrix B columns")
    parser.add_argument("--iter", type=int, default=10, help="Iterations")
    
    args = parser.parse_args()
    profile_single_gemm(args.n, args.k, args.m, args.iter)
