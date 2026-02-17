#!/bin/bash
# Profile all GEMM operations using C cuBLAS wrapper with NVIDIA NCU
# Each GEMM dimension gets its own profile file (gemm_n*_k*_m*.ncu-rep)
# All outputs saved to ../profiled_data directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CSV_FILE="${PROJECT_DIR}/data/gemm_data.csv"
OUTPUT_DIR="${PROJECT_DIR}/profiled_data"
NCU_SETTINGS="detailed"  # Can be: detailed, default, full

# Check if C wrapper library is compiled
LIB_PATH="${SCRIPT_DIR}/build/lib/libcublas_wrapper.so"
if [ ! -f "$LIB_PATH" ]; then
    echo "ERROR: C wrapper library not compiled!"
    echo "Please compile first:"
    echo "  cd $SCRIPT_DIR"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && make"
    exit 1
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: $CSV_FILE not found!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "NVIDIA NCU Profiling with C cuBLAS Wrapper"
echo "=================================================="
echo "Input file: $CSV_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "C Library: $LIB_PATH"
echo "NCU settings: $NCU_SETTINGS"
echo "=================================================="
echo ""

counter=0
success=0
failed=0

# Skip header, read each line
while IFS=',' read -r n k m rest; do
    # Trim whitespace
    n=$(echo "$n" | xargs)
    k=$(echo "$k" | xargs)
    m=$(echo "$m" | xargs)
    
    # Skip header and empty lines
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
    
    # Step 1: Run NCU profiling with Python script (single mode)
    echo "  → Running NCU profiling..."
    if ncu --set "$NCU_SETTINGS" \
           --target-processes all \
           -o "$ncu_rep_file" \
           python3 "${SCRIPT_DIR}/profile_cublasXtSgemm_ncu_v2.py" --n "$n" --k "$k" --m "$m" --iter 10 2>/dev/null; then
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
echo "Next step: Analyze profiling data or use CSV files for analysis"
