# RunMat Performance Benchmarks

This directory contains reproducible, cross-language benchmarks and shareable articles comparing RunMat against common alternatives for representative workloads.

## Structure

- harness/: shared Python utilities to run implementations, time them, and collect results
- <case>/
  - runmat.m: MATLAB-syntax script for RunMat
  - octave: benchmark reuses runmat.m via `octave -qf runmat.m`
  - python_numpy.py: NumPy implementation
  - python_torch.py: PyTorch implementation (uses GPU if available)
  - julia.jl: Julia implementation
  - ARTICLE.md: public-facing writeup for the case

## Benchmark Harness Usage (example – 4k image processing)

```bash
python3 ./.harness/run_bench.py --case 4k-image-processing --iterations 3 --output ../results/4k_image_processing.json
```

## Notes
- The harness auto-detects available interpreters (RunMat, Python, Octave, Julia) and skips missing ones.
- For RunMat, the harness prefers a `runmat` binary on PATH; if not present, it falls back to `cargo run -q -p runmat --release --`, which requires a Rust toolchain and will be slower.
- Reported metric is wall-clock time (ms) per run. Individual implementations may also print additional timing info; the harness records wall-clock consistently across languages.