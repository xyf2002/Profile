#!/usr/bin/env python3
"""
Profile GEMM operations using torch.matmul with NVIDIA NCU
Usage: ncu --set full python profile_gemm_ncu.py
"""

import torch
import csv
import sys
import os

def profile_gemm_from_csv(csv_file):
    """
    Read GEMM dimensions from CSV and profile each one
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Read CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        gemm_params = []
        for row in reader:
            # Skip empty rows
            if not row['n'] or not row['k'] or not row['m']:
                continue
            try:
                n = int(row['n'])
                k = int(row['k'])
                m = int(row['m'])
                gemm_params.append((n, k, m))
            except ValueError:
                continue
    
    print(f"Total GEMM configurations: {len(gemm_params)}")
    print()
    
    # Profile each GEMM configuration
    for idx, (n, k, m) in enumerate(gemm_params):
        print(f"[{idx+1}/{len(gemm_params)}] Profiling GEMM: n={n}, k={k}, m={m}")
        
        # Create input tensors
        A = torch.randn(n, k, dtype=torch.float32, device=device)
        B = torch.randn(k, m, dtype=torch.float32, device=device)
        
        # Warmup
        torch.cuda.synchronize()
        _ = torch.matmul(A, B)
        torch.cuda.synchronize()
        
        # Actual computation (this is what NCU will profile)
        result = torch.matmul(A, B)
        torch.cuda.synchronize()
        
        print(f"  Output shape: {result.shape}")
        print()

if __name__ == "__main__":
    csv_path = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(csv_path, "gemm_data.csv")
    
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found at {csv_file}")
        sys.exit(1)
    
    profile_gemm_from_csv(csv_file)
