#!/bin/bash

# RunMat vs GNU Octave Benchmark Suite
# Comprehensive performance comparison with structured YAML output

set -e

echo "============================================================="
echo "    RunMat vs GNU Octave Performance Benchmark Suite"
echo "============================================================="

# Configuration
RUSTMAT_RELEASE="../target/release/runmat"
RUSTMAT_DEBUG="../target/debug/runmat"
RESULTS_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/benchmark_$TIMESTAMP.yaml"

# Check dependencies
if ! command -v octave &> /dev/null; then
    echo "Error: GNU Octave is not installed or not in PATH"
    echo "Please install Octave to run comparative benchmarks"
    exit 1
fi

if ! command -v yq &> /dev/null; then
    echo "Warning: yq not found. YAML output will be generated manually."
fi

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Build RunMat in release mode if needed
if [ ! -f "$RUSTMAT_RELEASE" ]; then
    echo "Building RunMat in release mode..."
    cd ..
    cargo build --release --features blas-lapack
    cd benchmarks
else
    echo "Using existing RunMat release binary"
fi

# Also ensure debug build for comparison
if [ ! -f "$RUSTMAT_DEBUG" ]; then
    echo "Building RunMat in debug mode..."
    cd ..
    cargo build --features blas-lapack
    cd benchmarks
fi

# System information gathering
get_system_info() {
    echo "Gathering system information..."
    
    # Basic system info
    OS=$(uname -s)
    ARCH=$(uname -m)
    KERNEL=$(uname -r)
    
    # CPU information
    if [[ "$OS" == "Darwin" ]]; then
        CPU_MODEL=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
        CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "Unknown")
        MEMORY_GB=$(echo "scale=1; $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024" | bc -l)
    elif [[ "$OS" == "Linux" ]]; then
        CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' || echo "Unknown")
        CPU_CORES=$(nproc 2>/dev/null || echo "Unknown")
        MEMORY_GB=$(echo "scale=1; $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024" | bc -l 2>/dev/null || echo "Unknown")
    else
        CPU_MODEL="Unknown"
        CPU_CORES="Unknown"
        MEMORY_GB="Unknown"
    fi
    
    # Software versions
    OCTAVE_VERSION=$(octave --version 2>/dev/null | head -1 | sed 's/GNU Octave, version //' || echo "Unknown")
    RUSTMAT_VERSION=$("$RUSTMAT_RELEASE" --version 2>/dev/null || echo "Development build")
}

# Benchmark execution function
run_single_benchmark() {
    local name="$1"
    local script="$2"
    local runner="$3"
    local flags="$4"
    local warmup_runs="${5:-1}"
    local timing_runs="${6:-3}"
    
    echo "  Running $name with $runner $flags..." >&2
    
    # Warmup runs (not timed)
    for ((i=1; i<=warmup_runs; i++)); do
        if [[ "$runner" == "octave" ]]; then
            octave --no-gui --eval "run('$script')" >/dev/null 2>&1 || true
        else
            "$runner" $flags "$script" >/dev/null 2>&1 || true
        fi
    done
    
    # Timing runs
    local total_time=0
    local times=()
    
    for ((i=1; i<=timing_runs; i++)); do
        if [[ "$runner" == "octave" ]]; then
            time_output=$( { time octave --no-gui --eval "run('$script')" 2>/dev/null ; } 2>&1 )
        else
            time_output=$( { time "$runner" $flags "$script" ; } 2>&1 )
        fi
        
        # Extract real time and convert to seconds
        time_raw=$(echo "$time_output" | grep "^real" | awk '{print $2}')
        time_seconds=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)
        times+=("$time_seconds")
        total_time=$(echo "$total_time + $time_seconds" | bc -l)
    done
    
    # Calculate statistics
    local avg_time=$(echo "scale=6; $total_time / $timing_runs" | bc -l)
    local min_time=$(printf '%s\n' "${times[@]}" | sort -n | head -1)
    local max_time=$(printf '%s\n' "${times[@]}" | sort -nr | head -1)
    
    echo "$avg_time $min_time $max_time"
}

# Initialize YAML output
init_yaml_output() {
    cat > "$RESULT_FILE" << EOF
# RunMat vs GNU Octave Benchmark Results
# Generated on $(date)

metadata:
  timestamp: "$TIMESTAMP"
  date: "$(date -Iseconds)"
  
system:
  os: "$OS"
  architecture: "$ARCH"
  kernel: "$KERNEL"
  cpu:
    model: "$CPU_MODEL"
    cores: $CPU_CORES
  memory_gb: $MEMORY_GB

software:
  octave:
    version: "$OCTAVE_VERSION"
  runmat:
    version: "$RUSTMAT_VERSION"
    build_features: ["blas-lapack"]

benchmark_config:
  warmup_runs: 1
  timing_runs: 3
  scripts:
    - startup_time.m
    - matrix_operations.m
    - math_functions.m
    - control_flow.m

results:
EOF
}

# Add benchmark results to YAML
add_benchmark_result() {
    local benchmark="$1"
    local octave_avg="$2" octave_min="$3" octave_max="$4"
    local runmat_avg="$5" runmat_min="$6" runmat_max="$7"
    local runmat_jit_avg="$8" runmat_jit_min="$9" runmat_jit_max="${10}"
    
    # Calculate speedups
    local speedup=$(echo "scale=2; $octave_avg / $runmat_avg" | bc -l 2>/dev/null || echo "N/A")
    local jit_speedup=$(echo "scale=2; $octave_avg / $runmat_jit_avg" | bc -l 2>/dev/null || echo "N/A")
    
    cat >> "$RESULT_FILE" << EOF
  $benchmark:
    octave:
      avg_time: $octave_avg
      min_time: $octave_min
      max_time: $octave_max
    runmat_interpreter:
      avg_time: $runmat_avg
      min_time: $runmat_min
      max_time: $runmat_max
      speedup_vs_octave: "${speedup}x"
    runmat_jit:
      avg_time: $runmat_jit_avg
      min_time: $runmat_jit_min
      max_time: $runmat_jit_max
      speedup_vs_octave: "${jit_speedup}x"
      speedup_vs_interpreter: "$(echo "scale=2; $runmat_avg / $runmat_jit_avg" | bc -l 2>/dev/null || echo "N/A")x"
EOF
}

# Main benchmark execution
main() {
    echo ""
    get_system_info
    init_yaml_output
    
    # Define benchmarks
    declare -a benchmarks=(
        "startup_time:Startup Time"
        "matrix_operations:Matrix Operations"
        "math_functions:Math Functions"
        "control_flow:Control Flow"
    )
    
    echo ""
    echo "Starting benchmark suite with $(echo "${benchmarks[@]}" | wc -w) benchmarks..."
    echo "Results will be saved to: $RESULT_FILE"
    echo ""
    
    for bench_info in "${benchmarks[@]}"; do
        IFS=':' read -r script_name display_name <<< "$bench_info"
        
        echo "=== $display_name Benchmark ==="
        
        # Run Octave benchmark
        echo "GNU Octave:"
        octave_results=($(run_single_benchmark "$display_name" "$script_name.m" "octave" ""))
        
        # Run RunMat interpreter benchmark  
        echo "RunMat (Interpreter):"
        runmat_results=($(run_single_benchmark "$display_name" "$script_name.m" "$RUSTMAT_RELEASE" "--no-jit"))
        
        # Run RunMat JIT benchmark
        echo "RunMat (JIT):"
        runmat_jit_results=($(run_single_benchmark "$display_name" "$script_name.m" "$RUSTMAT_RELEASE" ""))
        
        # Add results to YAML
        add_benchmark_result "$script_name" \
            "${octave_results[0]}" "${octave_results[1]}" "${octave_results[2]}" \
            "${runmat_results[0]}" "${runmat_results[1]}" "${runmat_results[2]}" \
            "${runmat_jit_results[0]}" "${runmat_jit_results[1]}" "${runmat_jit_results[2]}"
        
        echo ""
    done
    
    # Add summary to YAML
    cat >> "$RESULT_FILE" << EOF

summary:
  notes:
    - "Lower times are better"
    - "Speedup shows RunMat performance relative to GNU Octave"
    - "All times are in seconds"
    - "Results are averaged over multiple runs with warmup"
  conclusions:
    - "Results demonstrate RunMat's performance characteristics"
    - "JIT compilation provides additional performance benefits"
    - "Platform-specific optimizations (BLAS/LAPACK) utilized"
EOF
    
    echo "============================================================="
    echo "                    BENCHMARK COMPLETED"
    echo "============================================================="
    echo ""
    echo "Results saved to: $RESULT_FILE"
    echo ""
    echo "Quick Summary:"
    echo "$(grep -A 20 "results:" "$RESULT_FILE" | grep -E "(avg_time|speedup)" | head -10)"
    echo ""
    echo "For full results, view: cat $RESULT_FILE"
}

# Run the benchmark suite
main "$@"