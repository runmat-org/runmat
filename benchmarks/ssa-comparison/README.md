# SSA Optimization A/B Benchmark

Compares RunMat JIT performance with and without SSA IR optimizations.

## What it tests

- **SSA None**: Direct bytecode→Cranelift (legacy path)
- **SSA Speed**: Simplify + DCE + CSE (standard)
- **SSA Aggressive**: All passes including LICM

## Key optimizations tested

1. **Constant Folding**: `2 + 3` → `5` at compile time
2. **CSE**: Eliminate redundant `x + y` computations
3. **DCE**: Remove dead code
4. **LICM**: Hoist loop-invariant code to preheader

## Running

```bash
# Run the benchmark
cargo run --release -p runmat -- benchmarks/ssa-comparison/ssa_bench.m

# Or with different SSA levels via env (when implemented):
# SSA_OPT_LEVEL=none cargo run --release -p runmat -- benchmarks/ssa-comparison/ssa_bench.m
```
