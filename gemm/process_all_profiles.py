#!/usr/bin/env python3
"""
Process all NVIDIA NCU and NSys profiling CSV files from profiled_data and profiled_data_nsys directories
and generate a comprehensive performance benchmark table.

This script:
1. Scans profiled_data directory for NCU CSV files
2. Scans profiled_data_nsys directory for NSys CSV files
3. Extracts Avg SMs busy (%) from NSys data
4. Extracts other metrics from NCU data
5. Generates a formatted table matching the benchmark format
6. Outputs CSV and Markdown formats
"""

import pandas as pd
import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict


def get_times(start_times, dur, sms, threshold):
    """
    Calculate SM utilization over time using event-driven approach.
    Exact implementation from process_nsys.py for consistency.
    
    Args:
        start_times: kernel start times (in seconds)
        dur: kernel durations (in nanoseconds)
        sms: SM count needed for each kernel
        threshold: maximum number of SMs (max_sms)
    
    Returns:
        (new_times, new_sm): arrays of time points and SM usage
    """
    new_times = []
    new_sm = []
    times_sm_all = []
    
    sz = len(start_times)
    
    # Build events: at each kernel start/end, adjust total SM count
    for i in range(sz):
        sti = start_times[i]
        di = dur[i] * 1e-9  # Convert nanoseconds to seconds
        smi = sms[i]
        times_sm_all.append([sti, smi])
        times_sm_all.append([sti + di, -smi])
    
    # Sort events by time
    times_sm_all = sorted(times_sm_all)
    
    # Calculate cumulative SM usage at each time point
    cur = 0
    for x in times_sm_all:
        cur += x[1]
        new_times.append(x[0])
        new_sm.append(cur)
    
    # Apply threshold (cap at max_sms)
    total = len(new_sm)
    for i in range(total):
        new_sm[i] = min(new_sm[i], threshold)
    
    return new_times, new_sm


def extract_gemm_params(filename):
    """
    Extract GEMM parameters from filename format: gemm_nXXX_kXXX_mXXX.csv
    Returns: (n, k, m) as integers
    """
    match = re.search(r'gemm_n(\d+)_k(\d+)_m(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def extract_metrics_ncu(csv_path):
    """
    Extract GPU performance metrics from NCU profiling CSV (excluding Avg SMs busy).
    Returns: dictionary with performance metrics
    """
    metrics = {}
    try:
        # Skip the second row (units row)
        df = pd.read_csv(csv_path, skiprows=[1])
        
        if len(df) == 0:
            return None
        
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


def extract_metrics_nsys(csv_path, max_sms=80):
    """
    Extract Avg SMs busy (%) from NSys CUDA GPU trace CSV.
    Uses the exact same method as process_nsys.py for consistency.
    
    Args:
        csv_path: Path to NSys cuda_gpu_trace CSV file
        max_sms: Maximum number of SMs on GPU (default 80 for common GPUs)
    
    Returns:
        Dictionary with 'Avg SMs busy (%)' or None
    """
    try:
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            return None
        
        # Extract required columns from cuda_gpu_trace format
        # Format: Start (ns),Duration (ns),GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Name
        required_cols = ['Start (ns)', 'Duration (ns)', 'GrdX', 'GrdY', 'GrdZ']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Convert to numeric (matching process_nsys.py)
        start_time_all = pd.to_numeric(df['Start (ns)'], errors='coerce') * 1e-9  # Convert ns to seconds
        dur_all = pd.to_numeric(df['Duration (ns)'], errors='coerce')
        grdx = pd.to_numeric(df['GrdX'], errors='coerce')
        grdy = pd.to_numeric(df['GrdY'], errors='coerce')
        grdz = pd.to_numeric(df['GrdZ'], errors='coerce')
        
        # Calculate SM needed as number of blocks (Grid dimensions)
        # This is equivalent to SM count that could be needed
        sms_needed_all = grdx * grdy * grdz
        
        # Filter out invalid rows (NaN values or memory operations)
        valid_mask = ~(start_time_all.isna() | dur_all.isna() | sms_needed_all.isna())
        
        start_times = start_time_all[valid_mask].values.astype(float)
        dur = dur_all[valid_mask].values
        sms_needed = sms_needed_all[valid_mask].values
        
        if len(start_times) == 0:
            return None
        
        # Use event-driven approach (same as process_nsys.py)
        times, sm_used = get_times(start_times, dur, sms_needed, max_sms)
        
        if len(times) < 2:
            return None
        
        # Calculate weighted average SM utilization (same as process_nsys.py)
        all_time_weight = 0
        current_weight = 0
        for i in range(len(times) - 1):
            current_weight += sm_used[i] * (times[i + 1] - times[i])
            all_time_weight += max_sms * (times[i + 1] - times[i])
        
        if all_time_weight > 0:
            avg_sm_busy = (current_weight * 100) / all_time_weight
        else:
            avg_sm_busy = 0
        
        avg_sm_busy = min(float(avg_sm_busy), 100)
        
        return {'Avg SMs busy (%)': round(avg_sm_busy, 1)}
        
    except Exception as e:
        print(f"Error processing NSys {csv_path}: {e}", file=sys.stderr)
        return None


def process_profile_directory(ncu_dir, nsys_dir):
    """
    Process all CSV files from both NCU and NSys directories.
    Returns: list of dictionaries with combined results
    """
    results = []
    ncu_path = Path(ncu_dir)
    nsys_path = Path(nsys_dir)
    
    if not ncu_path.exists():
        print(f"Error: NCU Directory not found: {ncu_dir}", file=sys.stderr)
        return None
    
    if not nsys_path.exists():
        print(f"Error: NSys Directory not found: {nsys_dir}", file=sys.stderr)
        return None
    
    csv_files = sorted(ncu_path.glob('gemm_*.csv'))
    
    if not csv_files:
        print(f"Error: No CSV files found in {ncu_dir}", file=sys.stderr)
        return None
    
    print(f"📊 Processing {len(csv_files)} profiling files...")
    print("-" * 80)
    
    for csv_file in csv_files:
        filename = csv_file.name
        n, k, m = extract_gemm_params(filename)
        
        if n is None:
            print(f"⚠ Skipping {filename}: Could not parse GEMM parameters")
            continue
        
        # Get metrics from NCU
        metrics_ncu = extract_metrics_ncu(str(csv_file))
        if not metrics_ncu:
            print(f"✗ {filename}: Failed to extract NCU metrics")
            continue
        
        # Get metrics from NSys
        nsys_file = nsys_path / filename
        if not nsys_file.exists():
            print(f"⚠ {filename}: No matching NSys file found, skipping")
            continue
        
        metrics_nsys = extract_metrics_nsys(str(nsys_file))
        if not metrics_nsys:
            print(f"⚠ {filename}: Failed to extract NSys metrics")
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
        result.update(metrics_nsys)  # NSys: Avg SMs busy (%)
        result.update(metrics_ncu)   # NCU: other metrics
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
        f.write("- **Avg SMs busy (%)**: Streaming Multiprocessor utilization from NSys (event-driven weighted average using kernel start/end times and grid dimensions, same method as process_nsys.py)\n")
        f.write("- **Compute Throughput(%)**: SM throughput utilization from NCU (Nsight Compute: sm_throughput)\n")
        f.write("- **Memory Bandwidth(%)**: GPU compute memory throughput utilization from NCU (Nsight Compute: gpu_compute_memory_throughput)\n")
        f.write("\n### Data Sources\n\n")
        f.write("- **NSys Data**: NVIDIA Nsight Systems cuda_gpu_trace report for kernel execution timing and grid information\n")
        f.write("- **NCU Data**: NVIDIA Nsight Compute profiling for compute and memory metrics\n")
    
    print(f"✓ Markdown saved to: {md_path}")
    
    return output_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Process all NVIDIA NCU and NSys profiling CSV files and create benchmark table"
    )
    parser.add_argument(
        '--ncu-dir',
        type=str,
        default='profiled_data',
        help='Directory containing NCU profiling CSV files'
    )
    parser.add_argument(
        '--nsys-dir',
        type=str,
        default='profiled_data_nsys',
        help='Directory containing NSys profiling CSV files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='gemm_benchmark_results.csv',
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    # Process all profiles from both NCU and NSys
    results = process_profile_directory(args.ncu_dir, args.nsys_dir)
    
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
