#!/bin/bash
# Profile all GEMM operations using NVIDIA Nsys (Systems Profiler)
# Measures GPU utilization including SM busy time
# Outputs statistics in CSV format

set -e

CSV_FILE="data/gemm_data.csv"
OUTPUT_DIR="profiled_data_nsys"
NSYS_SETTINGS="gpu-metrics-device=0"

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: $CSV_FILE not found!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "NVIDIA Nsys GPU Profiling for GEMM Operations"
echo "=================================================="
echo "Input file: $CSV_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Nsys metrics: GPU metrics collection"
echo "=================================================="
echo ""

counter=0
success=0
failed=0

# Skip header, read each line
while IFS=',' read -r n k m rest; do
    # Trim whitespace FIRST
    n=$(echo "$n" | xargs)
    k=$(echo "$k" | xargs)
    m=$(echo "$m" | xargs)
    
    # Skip header and empty lines (after trimming)
    if [[ "$n" == "n" ]] || [[ -z "$n" ]]; then
        continue
    fi
    
    counter=$((counter + 1))
    
    # Create output filenames
    base_name="gemm_n${n}_k${k}_m${m}"
    nsys_report="${OUTPUT_DIR}/${base_name}"
    
    echo "[${counter}] Profiling: n=$n, k=$k, m=$m"
    
    # Skip if already profiled
    if [ -f "${nsys_report}.sqlite" ]; then
        echo "  ✓ Already profiled (skipping)"
        success=$((success + 1))
        continue
    fi
    
    # Step 1: Run Nsys profiling
    echo "  → Running Nsys profiling..."
    nsys profile \
        -o "$nsys_report" \
        --gpu-metrics-devices 0 \
        python3 profile_gemm_nsys.py --n "$n" --k "$k" --m "$m" --iter 10 > /dev/null 2>&1
    
    if [ -f "${nsys_report}.nsys-rep" ]; then
        echo "    ✓ Nsys report generated"
        
        # Step 2: Convert to CSV using nsys stats
        echo "  → Converting to CSV..."
        if nsys stats "${nsys_report}.nsys-rep" \
            --report cuda_gpu_kern_sum \
            --format csv \
            -o "${nsys_report}_stats" 2>/dev/null; then
            
            # Rename CSV file to remove _stats_cuda_gpu_kern_sum suffix
            if [ -f "${nsys_report}_stats_cuda_gpu_kern_sum.csv" ]; then
                mv "${nsys_report}_stats_cuda_gpu_kern_sum.csv" "${nsys_report}.csv"
                echo "    ✓ CSV file created"
                success=$((success + 1))
                
                # Clean up .sqlite file to save space (keep .nsys-rep)
                rm -f "${nsys_report}.sqlite"
            else
                echo "    ✗ CSV file not found after conversion"
                failed=$((failed + 1))
            fi
        else
            echo "    ✗ CSV conversion failed"
            failed=$((failed + 1))
        fi
    else
        echo "    ✗ Profiling failed"
        failed=$((failed + 1))
    fi
    
    echo ""
    
done < "$CSV_FILE"

echo "=================================================="
echo "Profiling Summary"
echo "=================================================="
echo "Total processed: $counter"
echo "Success: $success"
echo "Failed: $failed"
echo "Output directory: $OUTPUT_DIR"
echo "=================================================="
echo ""
echo "Next step: Process results with process_profiling_data.py"
echo "  python3 process_profiling_data.py"
