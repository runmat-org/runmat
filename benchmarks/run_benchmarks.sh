#!/bin/bash

# RustMat vs Octave Benchmark Script
# Runs comparative performance tests

echo "==================================================="
echo "           RustMat vs GNU Octave Benchmarks"
echo "==================================================="

# Check if octave is installed
if ! command -v octave &> /dev/null; then
    echo "Error: GNU Octave is not installed or not in PATH"
    echo "Please install Octave to run comparative benchmarks"
    exit 1
fi

RUSTMAT="../target/debug/rustmat"

# Check if rustmat binary exists
if [ ! -f "$RUSTMAT" ]; then
    echo "Warning: rustmat binary not found at $RUSTMAT"
    echo "Building from source..."
    cd ..
    cargo build
    cd benchmarks
fi

# Arrays to store timing results
declare -a octave_times
declare -a rustmat_times
declare -a rustmat_jit_times

echo ""
echo "Running benchmarks with GNU Octave..."
echo "---------------------------------------------------"

echo ""
echo "Matrix Operations Benchmark (Octave):"
time_raw=$( { time octave --no-gui --eval "run('matrix_operations.m')" 2>/dev/null ; } 2>&1 | grep real | awk '{print $2}')
octave_times[0]=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)

echo ""
echo "Mathematical Functions Benchmark (Octave):"
time_raw=$( { time octave --no-gui --eval "run('math_functions.m')" 2>/dev/null ; } 2>&1 | grep real | awk '{print $2}')
octave_times[1]=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)

echo ""
echo "Startup Time Benchmark (Octave):"
echo "Measuring 5 cold starts..."
total=0
for i in {1..5}; do
    echo -n "Run $i: "
    time_raw=$( { time octave --no-gui --eval "run('startup_time.m')" 2>/dev/null ; } 2>&1 | grep real | awk '{print $2}')
    # Convert time format from "0m0.853s" to "0.853" for bc
    time=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)
    echo "$time_raw ($time seconds)"
    total=$(echo "$total + $time" | bc -l)
done
octave_times[2]=$(echo "scale=3; $total / 5" | bc -l)

echo ""
echo "Running benchmarks with RustMat..."
echo "---------------------------------------------------"

echo ""
echo "Matrix Operations Benchmark (RustMat):"
time_raw=$( { time "$RUSTMAT" --no-jit matrix_operations.m ; } 2>&1 | grep real | awk '{print $2}')
rustmat_times[0]=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)

echo ""
echo "Mathematical Functions Benchmark (RustMat):"
time_raw=$( { time "$RUSTMAT" --no-jit math_functions.m ; } 2>&1 | grep real | awk '{print $2}')
rustmat_times[1]=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)

echo ""
echo "Startup Time Benchmark (RustMat):"
echo "Measuring 5 cold starts..."
total=0
for i in {1..5}; do
    echo -n "Run $i: "
    time_raw=$( { time "$RUSTMAT" --no-jit startup_time.m ; } 2>&1 | grep real | awk '{print $2}')
    # Convert time format from "0m0.853s" to "0.853" for bc
    time=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)
    echo "$time_raw ($time seconds)"
    total=$(echo "$total + $time" | bc -l)
done
rustmat_times[2]=$(echo "scale=3; $total / 5" | bc -l)

echo ""
echo "Running JIT-enabled benchmarks with RustMat..."
echo "---------------------------------------------------"

echo ""
echo "Matrix Operations Benchmark (RustMat + JIT):"
time_raw=$( { time "$RUSTMAT" matrix_operations.m ; } 2>&1 | grep real | awk '{print $2}')
rustmat_jit_times[0]=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)

echo ""
echo "Mathematical Functions Benchmark (RustMat + JIT):"
time_raw=$( { time "$RUSTMAT" math_functions.m ; } 2>&1 | grep real | awk '{print $2}')
rustmat_jit_times[1]=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)

echo ""
echo "Startup Time Benchmark (RustMat + JIT):"
echo "Measuring 5 cold starts..."
total=0
for i in {1..5}; do
    echo -n "Run $i: "
    time_raw=$( { time "$RUSTMAT" startup_time.m ; } 2>&1 | grep real | awk '{print $2}')
    # Convert time format from "0m0.853s" to "0.853" for bc
    time=$(echo "$time_raw" | sed 's/m/ * 60 + /' | sed 's/s//' | bc -l)
    echo "$time_raw ($time seconds)"
    total=$(echo "$total + $time" | bc -l)
done
rustmat_jit_times[2]=$(echo "scale=3; $total / 5" | bc -l)

echo ""
echo "==================================================="
echo "                 Benchmark Summary                   "
echo "==================================================="
printf "\n%-25s %-12s %-12s %-12s\n" "Benchmark" "Octave" "RustMat" "RustMat+JIT"
printf "%-25s %-12s %-12s %-12s\n" "Matrix Operations" "${octave_times[0]}" "${rustmat_times[0]}" "${rustmat_jit_times[0]}"
printf "%-25s %-12s %-12s %-12s\n" "Mathematical Functions" "${octave_times[1]}" "${rustmat_times[1]}" "${rustmat_jit_times[1]}"
printf "%-25s %-12s %-12s %-12s\n" "Avg Startup Time" "${octave_times[2]}" "${rustmat_times[2]}" "${rustmat_jit_times[2]}"
echo "==================================================="
echo "Times shown in seconds. Lower is better."
echo "Run 'rustmat --info' for detailed system information"
echo "==================================================="