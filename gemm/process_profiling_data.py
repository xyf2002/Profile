#!/usr/bin/env python3
"""
Process NVIDIA NCU profiling CSV data and generate a summary table.
Similar to the benchmarking table format: Model, Workload, Batch Size, Metrics
"""

import pandas as pd
import argparse
from pathlib import Path
import sys


def extract_metrics(df):
    """
    Extract key metrics from NCU profiling data.
    Returns a dictionary with calculated metrics.
    """
    metrics = {}
    
    try:
        # SM Utilization - Calculate as percentage of peak cycles
        if 'sm__cycles_active.avg' in df.columns:
            active_cycles = pd.to_numeric(df['sm__cycles_active.avg'].iloc[0], errors='coerce')
            if 'sm__cycles_elapsed.avg.per_second' in df.columns:
                elapsed_cycles = pd.to_numeric(df['sm__cycles_elapsed.avg.per_second'].iloc[0], errors='coerce')
                if not pd.isna(active_cycles) and not pd.isna(elapsed_cycles) and elapsed_cycles > 0:
                    sm_percent = (active_cycles / elapsed_cycles) * 100 if elapsed_cycles > 0 else 0
                    metrics['Avg SMs busy (%)'] = round(min(sm_percent, 100), 1)
        
        # Compute Throughput - SM instruction executed percentage
        if 'sm__inst_executed.avg.pct_of_peak_sustained_elapsed' in df.columns:
            val = pd.to_numeric(df['sm__inst_executed.avg.pct_of_peak_sustained_elapsed'].iloc[0], 
                              errors='coerce')
            if not pd.isna(val):
                metrics['Compute Throughput(%)'] = round(float(val), 1)
        
        # Memory Bandwidth - DRAM throughput percentage
        if 'dram__bytes_read.sum.pct_of_peak_sustained_elapsed' in df.columns:
            val_read = pd.to_numeric(df['dram__bytes_read.sum.pct_of_peak_sustained_elapsed'].iloc[0], 
                                    errors='coerce')
            if 'dram__bytes_write.sum.pct_of_peak_sustained_elapsed' in df.columns:
                val_write = pd.to_numeric(df['dram__bytes_write.sum.pct_of_peak_sustained_elapsed'].iloc[0], 
                                        errors='coerce')
                if not pd.isna(val_read) and not pd.isna(val_write):
                    # Total DRAM bandwidth utilization
                    total_dram = (float(val_read) + float(val_write)) / 2
                    metrics['Memory Bandwidth(%)'] = round(min(total_dram, 100), 1)
        
        # Memory Access - SM memory throughput
        if 'sm__memory_throughput.avg.pct_of_peak_sustained_elapsed' in df.columns:
            val = pd.to_numeric(df['sm__memory_throughput.avg.pct_of_peak_sustained_elapsed'].iloc[0], 
                              errors='coerce')
            if not pd.isna(val):
                metrics['Memory Throughput(%)'] = round(float(val), 1)
        
        # L2 Cache Hit Rate
        if 'lts__t_sector_hit_rate.pct' in df.columns:
            val = pd.to_numeric(df['lts__t_sector_hit_rate.pct'].iloc[0], errors='coerce')
            if not pd.isna(val):
                metrics['L2 Hit Rate(%)'] = round(float(val), 1)
        
    except Exception as e:
        print(f"Warning: Could not extract all metrics: {e}", file=sys.stderr)
    
    return metrics


def process_ncu_csv(csv_path, model_name="GEMM", workload="Training", batch_size=32):
    """
    Process a single NCU CSV file and extract metrics.
    
    Args:
        csv_path: Path to the NCU profiling CSV file
        model_name: Model identifier (e.g., "ResNet50", "BERT-large")
        workload: Type of workload ("Inference" or "Training")
        batch_size: Batch size used in the profiling
    
    Returns:
        Dictionary with model info and metrics
    """
    try:
        # Skip the first two rows (header names and units)
        df = pd.read_csv(csv_path, skiprows=[1])
        
        # Get the first data row (row 2 in file, index 1 after skiprows)
        if len(df) == 0:
            print(f"No data rows in {csv_path}", file=sys.stderr)
            return None
        
        metrics = extract_metrics(df)
        
        result = {
            'Model': model_name,
            'Workload': workload,
            'Batch size': batch_size,
        }
        result.update(metrics)
        return result
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def create_summary_table(csv_files_config):
    """
    Create a summary table from multiple NCU profiling CSV files.
    
    Args:
        csv_files_config: List of tuples (csv_path, model_name, workload, batch_size)
    
    Returns:
        DataFrame with summary table
    """
    results = []
    
    for csv_path, model_name, workload, batch_size in csv_files_config:
        result = process_ncu_csv(csv_path, model_name, workload, batch_size)
        if result:
            results.append(result)
    
    if not results:
        print("No data processed successfully", file=sys.stderr)
        return None
    
    summary_df = pd.DataFrame(results)
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Process NVIDIA NCU profiling data and create summary table"
    )
    parser.add_argument(
        'input_csv',
        type=str,
        help='Path to NCU profiling CSV file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='GEMM',
        help='Model name for the summary table'
    )
    parser.add_argument(
        '--workload',
        type=str,
        default='Training',
        choices=['Training', 'Inference'],
        help='Workload type'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size used in profiling'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for summary table'
    )
    
    args = parser.parse_args()
    
    # Process single file
    result = process_ncu_csv(
        args.input_csv,
        model_name=args.model,
        workload=args.workload,
        batch_size=args.batch_size
    )
    
    if result:
        summary_df = pd.DataFrame([result])
        
        # Display table
        print("\nSummary Table:")
        print("=" * 100)
        print(summary_df.to_string(index=False))
        print("=" * 100)
        
        # Save if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_path, index=False)
            print(f"\nSummary saved to: {output_path}")


if __name__ == '__main__':
    main()
