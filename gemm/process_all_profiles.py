#!/usr/bin/env python3
"""
Process all NVIDIA NCU profiling CSV files from profiled_data directory
and generate a comprehensive performance benchmark table.

This script:
1. Scans profiled_data directory for all CSV files
2. Extracts metrics from each file
3. Generates a formatted table matching the benchmark format
4. Outputs CSV and Markdown formats
"""

import pandas as pd
import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict


def extract_gemm_params(filename):
    """
    Extract GEMM parameters from filename format: gemm_nXXX_kXXX_mXXX.csv
    Returns: (n, k, m) as integers
    """
    match = re.search(r'gemm_n(\d+)_k(\d+)_m(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def extract_metrics(csv_path):
    """
    Extract GPU performance metrics from NCU profiling CSV.
    Returns: dictionary with performance metrics
    """
    metrics = {}
    try:
        # Skip the second row (units row)
        df = pd.read_csv(csv_path, skiprows=[1])
        
        if len(df) == 0:
            return None
        
        # SM Utilization - percentage of time SMs are actively issuing instructions
        # This metric directly measures SM busy time as a percentage
        if 'sm__issue_active.avg.pct_of_peak_sustained_elapsed' in df.columns:
            val = pd.to_numeric(df['sm__issue_active.avg.pct_of_peak_sustained_elapsed'].iloc[0], 
                              errors='coerce')
            if not pd.isna(val):
                val_float = min(float(val), 100)
                metrics['Avg SMs busy (%)'] = round(val_float, 1)
        
        # Compute Throughput - SM throughput percentage (Nsight Compute official metric)
        if 'sm__throughput.avg.pct_of_peak_sustained_elapsed' in df.columns:
            val = pd.to_numeric(df['sm__throughput.avg.pct_of_peak_sustained_elapsed'].iloc[0], 
                              errors='coerce')
            if not pd.isna(val):
                val_float = min(float(val), 100)
                metrics['Compute Throughput(%)'] = round(val_float, 1)
        
        # Memory Bandwidth - GPU compute memory throughput utilization
        # This measures the actual memory traffic including all cache levels
        # (More comprehensive than just DRAM bytes)
        if 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed' in df.columns:
            val = pd.to_numeric(df['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'].iloc[0], 
                              errors='coerce')
            if not pd.isna(val):
                val_float = min(float(val), 100)
                if val_float < 0.1 and val_float > 0:
                    metrics['Memory Bandwidth(%)'] = round(val_float, 2)
                else:
                    metrics['Memory Bandwidth(%)'] = round(val_float, 1)
        
        return metrics if metrics else None
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}", file=sys.stderr)
        return None


def process_profile_directory(profile_dir):
    """
    Process all CSV files in the profile directory.
    Returns: list of dictionaries with results
    """
    results = []
    profile_path = Path(profile_dir)
    
    if not profile_path.exists():
        print(f"Error: Directory not found: {profile_dir}", file=sys.stderr)
        return None
    
    csv_files = sorted(profile_path.glob('gemm_*.csv'))
    
    if not csv_files:
        print(f"Error: No CSV files found in {profile_dir}", file=sys.stderr)
        return None
    
    print(f"📊 Processing {len(csv_files)} profiling files...")
    print("-" * 80)
    
    for csv_file in csv_files:
        filename = csv_file.name
        n, k, m = extract_gemm_params(filename)
        
        if n is None:
            print(f"⚠ Skipping {filename}: Could not parse GEMM parameters")
            continue
        
        metrics = extract_metrics(str(csv_file))
        if not metrics:
            print(f"✗ {filename}: Failed to extract metrics")
            continue
        
        result = {
            'Model': f'GEMM',
            'N': n,
            'K': k,
            'M': m,
            'Workload': 'Training',
            'Batch size': 1,
            'FLOPs (GFLOPs)': round(2 * n * k * m / 1e9, 2),  # Calculate theoretical FLOPs
        }
        result.update(metrics)
        results.append(result)
        
        print(f"✓ n={n:>4}, k={k:>5}, m={m:>4} → Metrics extracted")
    
    print("-" * 80)
    return results


def create_summary_table(results):
    """
    Create a formatted summary table from results.
    """
    if not results:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Select columns in desired order
    columns = ['Model', 'N', 'K', 'M', 'Workload', 'Batch size', 
               'Avg SMs busy (%)', 'Compute Throughput(%)', 
               'Memory Bandwidth(%)']
    
    # Only include columns that exist
    columns = [col for col in columns if col in df.columns]
    df = df[columns]
    
    # Sort by N, K, M
    df = df.sort_values(['N', 'K', 'M']).reset_index(drop=True)
    
    return df


def save_outputs(df, output_csv):
    """
    Save results to CSV and Markdown formats.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"✓ CSV saved to: {output_path}")
    
    # Save Markdown
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w') as f:
        f.write("# GEMM Performance Benchmark Summary\n\n")
        f.write("## Performance Metrics Table\n\n")
        f.write("| " + " | ".join(df.columns) + " |\n")
        f.write("|" + "|".join(["---" for _ in df.columns]) + "|\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
        f.write("\n## Legend\n\n")
        f.write("- **N, K, M**: GEMM matrix dimensions (A: N×K, B: K×M, C: N×M)\n")
        f.write("- **Avg SMs busy (%)**: Streaming Multiprocessor utilization (% of time SMs are issuing instructions)\n")
        f.write("- **Compute Throughput(%)**: SM throughput utilization (Nsight Compute: sm_throughput)\n")
        f.write("- **Memory Bandwidth(%)**: GPU compute memory throughput utilization (Nsight Compute: gpu_compute_memory_throughput)\n")
    
    print(f"✓ Markdown saved to: {md_path}")
    
    return output_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Process all NVIDIA NCU profiling CSV files and create benchmark table"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='profiled_data',
        help='Directory containing profiling CSV files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='gemm_benchmark_results.csv',
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    # Process all profiles
    results = process_profile_directory(args.input_dir)
    
    if not results:
        print("\n❌ No results to process")
        sys.exit(1)
    
    # Create summary table
    summary_df = create_summary_table(results)
    
    if summary_df is None:
        print("\n❌ Failed to create summary table")
        sys.exit(1)
    
    # Save outputs
    csv_path, md_path = save_outputs(summary_df, args.output)
    
    # Display table
    print("\n" + "=" * 120)
    print("GEMM PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 120)
    print(summary_df.to_string(index=False))
    print("=" * 120)
    
    print(f"\n✓ Processing complete!")
    print(f"  CSV: {csv_path}")
    print(f"  MD:  {md_path}")


if __name__ == '__main__':
    main()
