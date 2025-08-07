#!/bin/bash

# Comprehensive RustMat vs Octave Benchmark Script
# Runs comparative performance tests across multiple domains

echo "==================================================="
echo "     Comprehensive RustMat vs GNU Octave Benchmarks"
echo "==================================================="

# Check if octave is installed
if ! command -v octave &> /dev/null; then
    echo "Error: GNU Octave is not installed or not in PATH"
    echo "Please install Octave to run comparative benchmarks"
    exit 1
fi

RUSTMAT="../target/release/rustmat"

# Check if rustmat binary exists
if [ ! -f "$RUSTMAT" ]; then
    echo "Warning: rustmat binary not found at $RUSTMAT"
    echo "Building from source in release mode..."
    cd ..
    cargo build --release
    cd benchmarks
fi

# Create results directory
mkdir -p results
RESULTS_FILE="results/benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

# Function to run a benchmark and capture timing
run_benchmark() {
    local name="$1"
    local script="$2"
    local runner="$3"
    local flags="$4"
    
    echo "Running $name with $runner..." | tee -a "$RESULTS_FILE"
    
    if [[ "$runner" == "octave" ]]; then
        time_output=$( { time octave --no-gui --eval "run('$script')" 2>/dev/null ; } 2>&1 )
    else
        time_output=$( { time "$RUSTMAT" $flags "$script" ; } 2>&1 )
    fi
    
    # Extract real time
    time_raw=$(echo "$time_output" | grep real | awk '{print $2}')
    time_seconds=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)
    
    echo "$name ($runner $flags): $time_raw ($time_seconds seconds)" | tee -a "$RESULTS_FILE"
    echo "$time_seconds"
}

# Arrays to store results
declare -A octave_results
declare -A rustmat_results
declare -A rustmat_jit_results

benchmarks=(
    "matrix_operations:Matrix Operations"
    "math_functions:Mathematical Functions" 
    "linear_algebra:Linear Algebra"
    "fft_performance:FFT Performance"
    "startup_time:Startup Time"
)

echo "Starting comprehensive benchmark suite..." | tee "$RESULTS_FILE"
echo "Date: $(date)" | tee -a "$RESULTS_FILE"
echo "System: $(uname -a)" | tee -a "$RESULTS_FILE"
echo "Octave version: $(octave --version | head -1)" | tee -a "$RESULTS_FILE"
echo "RustMat version: $($RUSTMAT --version 2>/dev/null || echo 'Development build')" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Run Octave benchmarks
echo ""
echo "=== Running Octave Benchmarks ===" | tee -a "$RESULTS_FILE"
for bench_info in "${benchmarks[@]}"; do
    IFS=':' read -r script_name display_name <<< "$bench_info"
    
    if [[ "$script_name" == "startup_time" ]]; then
        # Special handling for startup time - measure 5 runs
        echo "Measuring startup time (5 runs)..." | tee -a "$RESULTS_FILE"
        total=0
        for i in {1..5}; do
            time_seconds=$(run_benchmark "Startup Test $i" "$script_name.m" "octave" "")
            total=$(echo "$total + $time_seconds" | bc -l)
        done
        octave_results["$script_name"]=$(echo "scale=3; $total / 5" | bc -l)
        echo "Average startup time: ${octave_results[$script_name]} seconds" | tee -a "$RESULTS_FILE"
    else
        octave_results["$script_name"]=$(run_benchmark "$display_name" "$script_name.m" "octave" "")
    fi
    echo "" | tee -a "$RESULTS_FILE"
done

# Run RustMat benchmarks (no JIT)
echo ""
echo "=== Running RustMat Benchmarks (Interpreter Only) ===" | tee -a "$RESULTS_FILE"
for bench_info in "${benchmarks[@]}"; do
    IFS=':' read -r script_name display_name <<< "$bench_info"
    
    if [[ "$script_name" == "startup_time" ]]; then
        # Special handling for startup time - measure 5 runs
        echo "Measuring startup time (5 runs)..." | tee -a "$RESULTS_FILE"
        total=0
        for i in {1..5}; do
            time_seconds=$(run_benchmark "Startup Test $i" "$script_name.m" "rustmat" "--no-jit")
            total=$(echo "$total + $time_seconds" | bc -l)
        done
        rustmat_results["$script_name"]=$(echo "scale=3; $total / 5" | bc -l)
        echo "Average startup time: ${rustmat_results[$script_name]} seconds" | tee -a "$RESULTS_FILE"
    else
        rustmat_results["$script_name"]=$(run_benchmark "$display_name" "$script_name.m" "rustmat" "--no-jit")
    fi
    echo "" | tee -a "$RESULTS_FILE"
done

# Run RustMat benchmarks (with JIT)
echo ""
echo "=== Running RustMat Benchmarks (JIT Enabled) ===" | tee -a "$RESULTS_FILE"
for bench_info in "${benchmarks[@]}"; do
    IFS=':' read -r script_name display_name <<< "$bench_info"
    
    if [[ "$script_name" == "startup_time" ]]; then
        # Special handling for startup time - measure 5 runs
        echo "Measuring startup time (5 runs)..." | tee -a "$RESULTS_FILE"
        total=0
        for i in {1..5}; do
            time_seconds=$(run_benchmark "Startup Test $i" "$script_name.m" "rustmat" "")
            total=$(echo "$total + $time_seconds" | bc -l)
        done
        rustmat_jit_results["$script_name"]=$(echo "scale=3; $total / 5" | bc -l)
        echo "Average startup time: ${rustmat_jit_results[$script_name]} seconds" | tee -a "$RESULTS_FILE"
    else
        rustmat_jit_results["$script_name"]=$(run_benchmark "$display_name" "$script_name.m" "rustmat" "")
    fi
    echo "" | tee -a "$RESULTS_FILE"
done

# Generate summary report
echo ""
echo "==================================================="
echo "                 BENCHMARK SUMMARY                   "
echo "==================================================="

{
    echo ""
    echo "BENCHMARK SUMMARY"
    echo "================="
    printf "\n%-25s %-12s %-15s %-15s %-10s %-10s\n" "Benchmark" "Octave (s)" "RustMat (s)" "RustMat+JIT (s)" "Speedup" "JIT Speedup"
    echo "---------------------------------------------------------------------------------------------"
    
    for bench_info in "${benchmarks[@]}"; do
        IFS=':' read -r script_name display_name <<< "$bench_info"
        
        octave_time=${octave_results[$script_name]}
        rustmat_time=${rustmat_results[$script_name]}
        rustmat_jit_time=${rustmat_jit_results[$script_name]}
        
        if [[ -n "$octave_time" && -n "$rustmat_time" && -n "$rustmat_jit_time" ]]; then
            speedup=$(echo "scale=2; $octave_time / $rustmat_time" | bc -l)
            jit_speedup=$(echo "scale=2; $octave_time / $rustmat_jit_time" | bc -l)
            
            printf "%-25s %-12s %-15s %-15s %-10s %-10s\n" \
                "$display_name" \
                "$octave_time" \
                "$rustmat_time" \
                "$rustmat_jit_time" \
                "${speedup}x" \
                "${jit_speedup}x"
        fi
    done
    
    echo "---------------------------------------------------------------------------------------------"
    echo ""
    echo "Notes:"
    echo "- Lower times are better"
    echo "- Speedup shows RustMat performance vs Octave" 
    echo "- JIT Speedup shows RustMat+JIT performance vs Octave"
    echo "- Results saved to: $RESULTS_FILE"
    echo ""
    echo "System Information:"
    echo "- Date: $(date)"
    echo "- Hardware: $(uname -m) $(uname -s)"
    echo "- Octave: $(octave --version | head -1)"
    echo "- RustMat: $($RUSTMAT --version 2>/dev/null || echo 'Development build')"
    
} | tee -a "$RESULTS_FILE"

echo ""
echo "Benchmark completed! Results saved to: $RESULTS_FILE"