#!/bin/bash
# Profile all GEMM operations from gemm_data.csv using NVIDIA NCU
# Generates .ncu-rep files and converts them to CSV
# All outputs saved to profiled_data directory

set -e

CSV_FILE="data/gemm_data.csv"
OUTPUT_DIR="profiled_data"
NCU_SETTINGS="detailed"  # Can be: detailed, default, full

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: $CSV_FILE not found!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "NVIDIA NCU Batch Profiling for GEMM Operations"
echo "=================================================="
echo "Input file: $CSV_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "NCU settings: $NCU_SETTINGS"
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
    ncu_rep_file="${OUTPUT_DIR}/${base_name}.ncu-rep"
    csv_file="${OUTPUT_DIR}/${base_name}.csv"
    
    echo "[${counter}] Profiling: n=$n, k=$k, m=$m"
    
    # Skip if already profiled
    if [ -f "$csv_file" ]; then
        echo "  ✓ Already profiled (skipping)"
        success=$((success + 1))
        continue
    fi
    
    # Step 1: Run NCU profiling
    echo "  → Running NCU profiling..."
    if ncu --set "$NCU_SETTINGS" \
           --target-processes all \
           -o "$ncu_rep_file" \
           python3 profile_gemm_single.py --n "$n" --k "$k" --m "$m" --iter 10 2>/dev/null; then
        echo "    ✓ NCU report generated"
    else
        echo "    ✗ Profiling failed"
        failed=$((failed + 1))
        continue
    fi
    
    # Step 2: Convert to CSV
    echo "  → Converting to CSV..."
    if ncu --import "$ncu_rep_file" \
            --csv \
            --page raw > "$csv_file" 2>/dev/null; then
        echo "    ✓ CSV file created"
        success=$((success + 1))
        
        # Optional: Remove .ncu-rep to save space
        # rm "$ncu_rep_file"
    else
        echo "    ✗ CSV conversion failed"
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
echo "Next step: Run the batch processor to analyze results"
echo "  python3 process_all_profiles.py --output benchmark_results.csv"
