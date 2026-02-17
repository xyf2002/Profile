#!/usr/bin/env python3
"""
Profile single or multiple GEMM operations using C cuBLAS wrapper
Supports both single GEMM (with --n, --k, --m) and batch mode (all configs from CSV)
Usage: 
  Single: python3 profile_cublasXtSgemm_ncu_v2.py --n 1024 --k 10240 --m 512
  Batch:  python3 profile_cublasXtSgemm_ncu_v2.py
"""

import ctypes
import numpy as np
import os
import sys
import csv
import argparse

# Find the compiled library
def find_cublas_wrapper():
    """Find the compiled libcublas_wrapper.so"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible locations
    possible_paths = [
        os.path.join(script_dir, "libcublas_wrapper.so"),
        os.path.join(script_dir, "build", "lib", "libcublas_wrapper.so"),
        os.path.join(script_dir, "..", "build", "lib", "libcublas_wrapper.so"),
        os.path.join(script_dir, "..", "lib", "libcublas_wrapper.so"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    raise FileNotFoundError(
        f"Could not find libcublas_wrapper.so\n"
        f"Please compile with CMake first:\n"
        f"  cd {script_dir}\n"
        f"  mkdir build && cd build\n"
        f"  cmake .. && make\n"
        f"Searched locations: {possible_paths}"
    )

# Load the wrapper library
try:
    wrapper_lib_path = find_cublas_wrapper()
    wrapper_lib = ctypes.CDLL(wrapper_lib_path)
except OSError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Define wrapper function signatures
cublas_init = wrapper_lib.cublas_init
cublas_init.argtypes = []
cublas_init.restype = ctypes.c_int

cublas_cleanup = wrapper_lib.cublas_cleanup
cublas_cleanup.argtypes = []
cublas_cleanup.restype = None

cublas_sgemm = wrapper_lib.cublas_sgemm
cublas_sgemm.argtypes = [
    ctypes.c_int,                      # transa
    ctypes.c_int,                      # transb
    ctypes.c_int,                      # m
    ctypes.c_int,                      # n
    ctypes.c_int,                      # k
    ctypes.c_float,                    # alpha
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # A
    ctypes.c_int,                      # lda
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # B
    ctypes.c_int,                      # ldb
    ctypes.c_float,                    # beta
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),  # C
    ctypes.c_int,                      # ldc
]
cublas_sgemm.restype = ctypes.c_int

cublas_get_error = wrapper_lib.cublas_get_error
cublas_get_error.argtypes = []
cublas_get_error.restype = ctypes.c_char_p


def run_single_gemm(n, k, m, iterations=10):
    """Profile a single GEMM operation"""
    print(f"Using cuBLAS wrapper library")
    print(f"Profiling: n={n}, k={k}, m={m}, iterations={iterations}\n")
    
    # Initialize cuBLAS
    ret = cublas_init()
    if ret != 0:
        error_msg = cublas_get_error().decode('utf-8')
        print(f"ERROR: Failed to initialize cuBLAS: {error_msg}")
        return False
    
    try:
        # Create input matrices
        A = np.random.randn(n, k).astype(np.float32)
        B = np.random.randn(k, m).astype(np.float32)
        C = np.zeros((n, m), dtype=np.float32)

        # Set parameters
        alpha = np.float32(1.0)
        beta = np.float32(0.0)
        transa = 0
        transb = 0
        lda = n
        ldb = k
        ldc = n

        # Warmup
        for _ in range(5):
            ret = cublas_sgemm(
                transa, transb,
                n, m, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc
            )

        # Main profiling loop
        for i in range(iterations):
            ret = cublas_sgemm(
                transa, transb,
                n, m, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc
            )
            if ret != 0:
                error_msg = cublas_get_error().decode('utf-8')
                print(f"WARNING: Iteration {i} failed: {error_msg}")

        print(f"✓ Completed: output shape {C.shape}")
        return True

    finally:
        cublas_cleanup()


def run_batch_gemm(csv_file):
    """Profile all GEMM operations from CSV"""
    print(f"Using cuBLAS wrapper library\n")

    # Initialize cuBLAS
    ret = cublas_init()
    if ret != 0:
        error_msg = cublas_get_error().decode('utf-8')
        print(f"ERROR: Failed to initialize cuBLAS: {error_msg}")
        return

    try:
        # Read CSV file
        gemm_params = []
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            
        # Skip header line
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                try:
                    n = int(parts[0])
                    k = int(parts[1])
                    m = int(parts[2])
                    if n > 0 and k > 0 and m > 0:
                        gemm_params.append((n, k, m))
                except (ValueError, IndexError):
                    continue

        print(f"Total GEMM configurations: {len(gemm_params)}\n")

        successful = 0
        failed = 0
        
        for idx, (n, k, m) in enumerate(gemm_params):
            print(f"[{idx+1}/{len(gemm_params)}] Profiling: n={n}, k={k}, m={m}")

            try:
                A = np.random.randn(n, k).astype(np.float32)
                B = np.random.randn(k, m).astype(np.float32)
                C = np.zeros((n, m), dtype=np.float32)

                alpha = np.float32(1.0)
                beta = np.float32(0.0)
                transa = 0
                transb = 0
                lda = n
                ldb = k
                ldc = n

                # Warmup
                for _ in range(5):
                    ret = cublas_sgemm(
                        transa, transb,
                        n, m, k,
                        alpha,
                        A, lda,
                        B, ldb,
                        beta,
                        C, ldc
                    )

                # Actual computation
                ret = cublas_sgemm(
                    transa, transb,
                    n, m, k,
                    alpha,
                    A, lda,
                    B, ldb,
                    beta,
                    C, ldc
                )

                if ret == 0:
                    print(f"  ✓")
                    successful += 1
                else:
                    error_msg = cublas_get_error().decode('utf-8')
                    print(f"  ✗ {error_msg}")
                    failed += 1

            except Exception as e:
                print(f"  ✗ Exception: {str(e)}")
                failed += 1

        print()
        print("=" * 50)
        print(f"Summary: {successful} successful, {failed} failed")
        print("=" * 50)

    finally:
        cublas_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile GEMM with C cuBLAS wrapper")
    parser.add_argument("--n", type=int, help="Matrix n dimension (single mode)")
    parser.add_argument("--k", type=int, help="Matrix k dimension (single mode)")
    parser.add_argument("--m", type=int, help="Matrix m dimension (single mode)")
    parser.add_argument("--iter", type=int, default=10, help="Iterations in single mode")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    csv_file = os.path.join(project_dir, "data", "gemm_data.csv")

    if args.n and args.k and args.m:
        # Single mode
        success = run_single_gemm(args.n, args.k, args.m, args.iter)
        sys.exit(0 if success else 1)
    else:
        # Batch mode
        if not os.path.exists(csv_file):
            print(f"ERROR: CSV file not found at {csv_file}")
            sys.exit(1)
        run_batch_gemm(csv_file)
