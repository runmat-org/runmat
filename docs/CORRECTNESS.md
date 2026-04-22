---
title: "Correctness & Trust"
description: "How RunMat validates numerical accuracy: named reference implementations, parity tests for every numerical builtin, and tolerances chosen per operation rather than by a universal rule."
keywords:
  - RunMat correctness
  - numerical accuracy
  - numerical correctness
  - parity testing
  - validated numerical libraries
  - GPU CPU parity
  - LAPACK
  - rustfft
  - floating-point tolerance
  - atol rtol
ogTitle: "Correctness & Trust"
ogDescription: "Every numerical path in RunMat traces to a named dependency, a pinned version, and a runnable test. See the coverage table, parity test links, and tolerance methodology."
jsonLd:
  "@context": "https://schema.org"
  "@graph":
    - "@type": "FAQPage"
      "@id": "https://runmat.com/docs/correctness#faq"
      mainEntity:
        - "@type": "Question"
          name: "How does RunMat validate numerical accuracy?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Three mechanisms. First, inherited correctness from established Rust crates (rustfft for FFT, nalgebra for SVD and dense linear algebra, num-complex for complex arithmetic). Second, optional FFI to platform-native BLAS and LAPACK (Apple Accelerate on macOS, OpenBLAS on Linux), the same libraries NumPy, SciPy, and MATLAB's own solvers rely on. Third, in-repo solvers validated against external references or RunMat's own CPU path. RunMat ships ~450 total builtins; the ~110 that live under crates/runmat-runtime/src/builtins/math/ are the numerical-math subset where tolerance-based validation applies, and those ship with ~1,635 co-located test functions plus ~40 integration-level parity tests covering GPU-vs-CPU, fusion engine, feature-gated FFI, and RNG statistics."
        - "@type": "Question"
          name: "What reference implementations does RunMat validate against?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Established open-source Rust crates (rustfft, nalgebra, num-complex), platform-native BLAS and LAPACK (Apple Accelerate on macOS, OpenBLAS elsewhere), closed-form or analytic solutions for elementary math and reductions, and RunMat's own CPU paths when validating GPU kernels. Every parity test cites its reference in the test file itself."
        - "@type": "Question"
          name: "How do you validate GPU results?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Every GPU-accelerated builtin is parity-tested against RunMat's CPU path. Each test picks a tolerance from the numerical properties of its specific operation: absolute for most cases, relative scaling for operations where magnitudes grow (SYRK, large-k matmul in f32). See the coverage table for the exact atol and rtol per builtin. If a GPU path drifts past its test's bound, CI fails the build before it ships."
        - "@type": "Question"
          name: "What tolerances does RunMat use and why?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Tolerances are chosen per operation, not by a universal rule. For f64, most tests bound absolute error between 1e-9 and 1e-12: tighter for closed-form operations, looser for factorizations that accumulate rounding. For f32, most tests bound absolute error between 1e-5 and 1e-6; operations where absolute magnitudes grow use relative bounds like 5e-4 * max(|reference|, 1) instead. The coverage table lists the exact atol and rtol for each category, and every test file cites its own tolerance inline with a comment explaining the choice."
        - "@type": "Question"
          name: "Why doesn't a == b return true when a and b look equal?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Exact equality on floating-point values almost never does what you want. IEEE 754 arithmetic produces slightly different last bits depending on operation order, SIMD width, and fused multiply-adds. Use abs(a - b) < 1e-9 for scalars or an isclose-style tolerance check for arrays. This is the same behaviour as NumPy, MATLAB, Julia, and PyTorch."
        - "@type": "Question"
          name: "Are all builtins validated?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Yes, for the builtins where tolerance-based validation is the right question. Of RunMat's ~450 total builtins, ~110 are numerical-math builtins (under crates/runmat-runtime/src/builtins/math/); all but two common.rs helper modules in that subset carry co-located unit tests, and a smaller GPU-accelerated, LAPACK-wrapped, or cross-cutting reduction subset additionally has integration-level parity tests. The remaining ~340 builtins handle string, array, plotting, I/O, and OOP; these are validated by behavioural tests rather than tolerance-based parity, since 'fprintf returned the right text' doesn't have an atol. We never silently remove a row: if validation regresses, the status updates."
        - "@type": "Question"
          name: "Can I run the validation tests myself?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Yes. Every parity test ships in the public repository and runs with standard cargo test. No MATLAB license, no external data files. A WGPU-compatible GPU is required for GPU parity tests; CPU tests run anywhere. Commands are listed in the Reproduce it yourself section."
        - "@type": "Question"
          name: "What happens when a dependency is updated?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Our Cargo.toml pins each numerical crate to a specific version. When we bump a version, the corresponding parity tests re-run in CI. If tolerances break, the bump does not ship."
        - "@type": "Question"
          name: "Does RunMat use the same random number generator as MATLAB?"
          acceptedAnswer:
            "@type": "Answer"
            text: "No. RunMat's rand and randn use a custom linear congruential generator (RunMatLCG). Sequences are deterministic given a seed, but they will not match MATLAB's Mersenne Twister or NumPy's PCG64. If your workflow depends on reproducing a specific random sequence from another tool, this is a known difference."
        - "@type": "Question"
          name: "Why don't you guarantee bit-identical reproduction of MATLAB output?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Accuracy and bit-identical reproducibility are different contracts. IEEE 754 arithmetic produces slightly different last bits depending on operation order, SIMD width, thread count, and fused multiply-adds; two MATLAB installations on different hardware can disagree on the last bit for the same code. Intel's math library calls this the CNR tradeoff (Conditional Numerical Reproducibility): you can have peak performance or bit-identical output, not both for free. RunMat provides validated numerical accuracy within documented tolerances, not bit-identical reproduction of any particular binary."
        - "@type": "Question"
          name: "How do I report a numerical bug?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Open an issue at github.com/runmat-org/runmat/issues with a minimal MATLAB-syntax reproduction, the expected output (and its source), and the observed output. Numerical bugs are treated as blockers."
    - "@type": "HowTo"
      "@id": "https://runmat.com/docs/correctness#reproduce"
      name: "Reproduce RunMat's numerical parity tests locally"
      description: "Clone RunMat and run the CPU and GPU parity tests that back every numerical builtin listed on this page."
      totalTime: "PT15M"
      tool:
        - "@type": "HowToTool"
          name: "Rust toolchain (cargo)"
        - "@type": "HowToTool"
          name: "WGPU-compatible GPU (required only for GPU parity tests)"
      step:
        - "@type": "HowToStep"
          position: 1
          name: "Clone the RunMat repository"
          text: "git clone https://github.com/runmat-org/runmat && cd runmat"
        - "@type": "HowToStep"
          position: 2
          name: "Run the FFT GPU-vs-CPU parity test"
          text: "cargo test -p runmat-accelerate --features wgpu --test fft_staged"
        - "@type": "HowToStep"
          position: 3
          name: "Run the fused GPU reduction parity test"
          text: "cargo test -p runmat-accelerate --features wgpu --test fused_square_mean_all_parity"
        - "@type": "HowToStep"
          position: 4
          name: "Run the statistics reduction parity test"
          text: "cargo test -p runmat-runtime --test reduction_parity"
        - "@type": "HowToStep"
          position: 5
          name: "Run the optional BLAS/LAPACK FFI parity test"
          text: "cargo test -p runmat-runtime --features blas-lapack --test blas_lapack"
        - "@type": "HowToStep"
          position: 6
          name: "Run the full GPU parity suite"
          text: "cargo test -p runmat-accelerate --features wgpu"
    - "@type": "TechArticle"
      "@id": "https://runmat.com/docs/correctness#article"
      dateModified: "2026-04-17"
      about:
        - "@type": "Thing"
          name: "Numerical correctness"
        - "@type": "Thing"
          name: "Floating-point tolerance"
        - "@type": "Thing"
          name: "GPU vs CPU parity testing"
        - "@type": "Thing"
          name: "LAPACK"
      citation:
        - "@type": "SoftwareSourceCode"
          name: "rustfft"
          codeRepository: "https://github.com/ejmahler/RustFFT"
          url: "https://crates.io/crates/rustfft"
        - "@type": "SoftwareSourceCode"
          name: "nalgebra"
          codeRepository: "https://github.com/dimforge/nalgebra"
          url: "https://crates.io/crates/nalgebra"
        - "@type": "SoftwareSourceCode"
          name: "lapack"
          url: "https://crates.io/crates/lapack"
        - "@type": "SoftwareSourceCode"
          name: "blas"
          url: "https://crates.io/crates/blas"
        - "@type": "SoftwareSourceCode"
          name: "accelerate-src"
          url: "https://crates.io/crates/accelerate-src"
        - "@type": "SoftwareSourceCode"
          name: "openblas-src"
          url: "https://crates.io/crates/openblas-src"
        - "@type": "SoftwareSourceCode"
          name: "num-complex"
          url: "https://crates.io/crates/num-complex"
---

# Correctness & Trust

Every numerical builtin in RunMat traces back to a named dependency, a pinned version, and a test you can run on your own machine. When a result looks wrong, you don't file a support ticket and wait. You read the source, find the crate, and check the tolerance. This page is the map: what backs each category of numerical computation, how we validate it, and where the gaps are.

This page is a living inventory of RunMat's ~110 numerical-math builtins (the subset of RunMat's ~450 total builtins where tolerance-based validation applies), the reference implementations they're validated against, and the tolerances those tests enforce. The other ~340 builtins (string, array, plotting, I/O, OOP) are validated by behavioural tests rather than by numerical tolerance.

---

## How we define numerical accuracy

Numerical accuracy means a builtin produces the same answer as a known-correct reference, within a tolerance chosen for that specific operation. "Known-correct" is concrete: an established open-source library ([rustfft](https://crates.io/crates/rustfft) for FFT, [nalgebra](https://crates.io/crates/nalgebra) for SVD, LAPACK for eigendecomposition), a closed-form expression, a hand-derived expected output, or (for GPU paths) RunMat's own CPU implementation.

What we commit to:

- Every numerical builtin has a named reference implementation and an automated test that compares against it.
- The tolerance for each test is chosen from the numerical properties of the operation (condition number, accumulation depth, precision) and documented inline in the test source.
- Regressions are caught in CI before a release ships, not in a customer's pipeline.

Accuracy and reproducibility are two different contracts. RunMat provides the first: every builtin is validated against a named reference within a tolerance chosen for that operation. RunMat does not currently provide bit-identical reproducibility across machines, backends, or thread counts. No mainstream numerical runtime does, without an explicit performance-costing mode. Intel's math library calls this the *CNR tradeoff* (Conditional Numerical Reproducibility): you can have peak performance or bit-identical output, not both for free.

A note on MATLAB: because IEEE 754 arithmetic produces different last-bit results depending on operation ordering, SIMD width, and compiler flags, no two runtimes produce bit-identical output for the same code. This holds even for two MATLAB installations on different hardware. RunMat targets *numerical* equivalence with auditable references, not bit-identical reproduction of a closed-source binary. More in [Design Philosophy](/docs/design-philosophy).

---

## Correctness tiers

RunMat's builtins fall into three correctness tiers, tested differently:

- **Crate-backed builtins.** When a builtin delegates to a well-established Rust crate ([rustfft](https://crates.io/crates/rustfft) for FFT, [nalgebra](https://crates.io/crates/nalgebra) for SVD), we inherit that crate's correctness. Our tests confirm RunMat's calling convention, array layout, and output format reproduce the crate's results, not re-prove the crate's internals.
- **LAPACK-backed builtins.** Optional FFI to platform-native BLAS and LAPACK ([lapack](https://crates.io/crates/lapack) 0.19, [blas](https://crates.io/crates/blas) 0.22, Apple Accelerate via [accelerate-src](https://crates.io/crates/accelerate-src) on macOS, [openblas-src](https://crates.io/crates/openblas-src) elsewhere). These are the same libraries NumPy, SciPy, and MATLAB's own solvers rely on.
- **In-repo solvers.** Some factorizations (LU, QR, Cholesky) and small in-place routines are implemented directly in the RunMat runtime. For these, we test against a known-correct reference (an external crate, LAPACK, or hand-derived expected values) to a documented floating-point tolerance.
- **GPU paths.** Every GPU-accelerated builtin is parity-tested against RunMat's own CPU path. Each parity test picks a tolerance from the numerical properties of its operation: typically `1e-9` to `1e-12` for f64 and `1e-5` to `1e-6` for f32, with relative bounds for operations where absolute magnitudes grow. See the coverage table for the exact `atol` and `rtol` per builtin, or the [FFT worked example](#worked-example-how-we-validate-the-fft) for a full GPU-vs-CPU parity walkthrough.

---

### Validation spans the entire math tree

The [FFT walkthrough below](#worked-example-how-we-validate-the-fft) is the cleanest example because it crosses a CPU/GPU boundary against an external reference (`rustfft`), which makes every piece of the methodology visible in one test file. The same pattern runs across RunMat's numerical tree:

- RunMat ships **~450 total builtins** across math, linear algebra, array ops, string, plotting, I/O, and OOP. The ~110 builtins where tolerance-based numerical validation applies all live under [`crates/runmat-runtime/src/builtins/math/`](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math). String handling, file I/O, figure rendering, and the rest are validated by behavioural tests instead; they don't have an "`atol`" to document.
- Those ~110 numerical-math modules ship with co-located `#[cfg(test)]` suites, totalling roughly **1,635 `#[test]` functions**.
- Categories covered include **elementary math** (`exp`, `log`, `abs`, `sqrt`, `gamma`, hyperbolics), **trigonometry** (`sin`/`cos`/`tan` and their inverse and hyperbolic variants), **reductions** (`mean`, `std`, `var`, `median`, `min`, `max`, `prod`, `cumsum`, `diff`, `all`, `any`), **rounding** (`mod`, `rem`, `fix`, `floor`, `ceil`, `round`), **signal processing** (`conv`, `conv2`, `filter`, `deconv`, window functions), **polynomials** (`polyval`, `polyfit`, `polyder`, `polyint`, `roots`), and **every linear algebra factorization, solver, and structural query**.
- **~40 integration-level parity tests** under [`crates/runmat-accelerate/tests/`](https://github.com/runmat-org/runmat/tree/main/crates/runmat-accelerate/tests) and [`crates/runmat-runtime/tests/`](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/tests) exercise cross-cutting paths: GPU-vs-CPU parity, output-pool reuse, fusion-engine correctness, feature-gated BLAS/LAPACK FFI, and statistical properties of the RNG.

Use the coverage table below to find the exact reference, test file, and tolerance for any category you care about.

---

## Coverage table

Each row lists a category of builtins, the reference implementation it's tested against with a link to the parity test on `main`, and the tolerance bounds that test enforces. The **`atol`** (absolute tolerance) and **`rtol`** (relative tolerance, scaled by `max(|reference|, 1)`) columns mirror NumPy's `assert_allclose` vocabulary; dashed cells mean that tolerance type doesn't apply for the operation. *Validated* means an automated parity test exists. *CPU-only validated* means CPU coverage is in place but GPU parity isn't yet. *GPU-vs-CPU validated* means only the cross-backend parity test exists (no external-reference test yet).

| Category | Source | atol | rtol | Status |
|---|---|---|---|---|
| **Elementary math** | [`libm`](https://crates.io/crates/libm) + in-repo; [`num-complex`](https://crates.io/crates/num-complex) 0.4 for complex<br />→ [`elementwise/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise) + [`complex.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/tests/complex.rs) | `1e-12` (F64)<br />`1e-5` (F32) | — | Validated |
| **Trigonometry** | [`libm`](https://crates.io/crates/libm) + in-repo<br />→ [`trigonometry/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/trigonometry) | `1e-12` (F64)<br />`1e-5` (F32) | — | Validated |
| **Rounding / modulo** | In-repo<br />→ [`rounding/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/rounding) | Exact (int) or `1e-12` | — | Validated |
| **FFT** | [`rustfft`](https://crates.io/crates/rustfft) 6.4 (CPU); staged WGPU kernel (GPU)<br />→ [`fft_staged.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/tests/fft_staged.rs) + [`fft/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/fft) | `1e-9` (F64)<br />`1e-3` (F32) | — | Validated |
| **Signal** | In-repo + GPU provider hooks; [`num-complex`](https://crates.io/crates/num-complex) 0.4 for complex<br />→ [`signal/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/signal) | `1e-10` – `1e-12` (exact on int) | — | Validated |
| **Polynomials** | In-repo Horner's method; companion-matrix roots<br />→ [`poly/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/poly) | `1e-10` – `1e-12` | — | Validated |
| **Reductions** | Runtime reduction infra (CPU + GPU)<br />→ [`reduction_parity.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/tests/reduction_parity.rs) + [`reduction/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/reduction) | `1e-7` (mean), `1e-6` (sum), `1e-10` – `1e-12` (in-module) | — | Validated |
| **Cumulative reductions** | In-repo<br />→ [`reduction/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/reduction) | Exact (int); `1e-12` (float) | — | Validated |
| **Linear solve** | [`nalgebra`](https://crates.io/crates/nalgebra) 0.32 `SVD` + `DMatrix`<br />→ [`linalg/solve/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/linalg/solve) | `1e-7` (residual), `1e-4` – `1e-5` (GPU/CPU) | — | Validated |
| **SVD** | [`nalgebra`](https://crates.io/crates/nalgebra) 0.32 `linalg::SVD`<br />→ [`svd.rs` tests](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/svd.rs) | `1e-10` – `1e-12` | — | Validated |
| **LU factorization** | In-repo partial-pivot solver<br />→ [`lu.rs` tests](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/lu.rs) | `1e-9` (reconstruction) | — | GPU-vs-CPU validated |
| **QR factorization** | In-repo Householder solver<br />→ [`qr.rs` tests](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/qr.rs) | `1e-9` (reconstruction) | — | GPU-vs-CPU validated |
| **Eigendecomposition** | LAPACK `dgeev`/`zgeev` via [`lapack`](https://crates.io/crates/lapack) 0.19 FFI<br />→ [`eig.rs` tests](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/eig.rs) | `1e-10` (`assert_matrix_close`) | — | CPU-only validated |
| **Cholesky** | LAPACK `dpotrf` via [`lapack`](https://crates.io/crates/lapack) 0.19 (feature-gated) + in-repo fallback<br />→ [`chol.rs` tests](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/chol.rs) + [GPU tests](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/src/backend/wgpu/tests.rs) | `1e-12` (in-module), `1e-6` (GPU) | — | Validated |
| **Linalg ops (non-factor)** | In-repo + [`nalgebra`](https://crates.io/crates/nalgebra) 0.32<br />→ [`linalg/ops/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/linalg/ops) | `1e-9` – `1e-12` | — | Validated |
| **Linalg structure** | In-repo<br />→ [`linalg/structure/` module tests](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/linalg/structure) | Pattern / boolean (exact) | — | Validated |
| **Vector algebra** | In-repo + WGPU provider<br />→ [`cross.rs` tests](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/cross.rs) | `1e-9` – `1e-12` | — | CPU-only validated |
| **Matrix multiply (GPU)** | WGPU fused kernels<br />→ [`matmul_residency.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/tests/matmul_residency.rs), [`matmul_epilogue.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/tests/matmul_epilogue.rs), [`matmul_small_k.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/tests/matmul_small_k.rs) | `1e-9` (F64 linear), `5e-5` (nonlinear epilogue), `1e-6` (residency) | `5e-4 * max(|ref|, 1)` (F32 vec4) | Validated |
| **SYRK (GPU)** | WGPU kernel<br />→ [`syrk.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/tests/syrk.rs) | `1e-9` (F64) | `1e-3 * max(|want|, 1)` (F32) | Validated |
| **Fused reductions (GPU)** | WGPU fusion engine<br />→ [`fused_square_mean_all_parity.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/tests/fused_square_mean_all_parity.rs), [`fused_reduction_sum_square.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/tests/fused_reduction_sum_square.rs), [`fused_reduction_sum_mul.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/tests/fused_reduction_sum_mul.rs) | `1e-6` | — | Validated |
| **BLAS / LAPACK (optional)** | FFI via [`blas`](https://crates.io/crates/blas) 0.22, [`lapack`](https://crates.io/crates/lapack) 0.19; [`accelerate-src`](https://crates.io/crates/accelerate-src) (macOS) / [`openblas-src`](https://crates.io/crates/openblas-src) (Linux)<br />→ [`blas_lapack.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/tests/blas_lapack.rs) | `1e-10` | — | Available |
| **Random number generation** | Custom `RunMatLCG` (in-repo)<br />→ [`rng.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/tests/rng.rs) | Statistical: `|mean|` < 0.01, `|variance − 1|` < 0.02 at n=1M | — | Validated |
| **ODE solvers** | — | — | — | Not yet shipped |
| **Optimization** | — | — | — | Not yet shipped |
| **Interpolation** | — | — | — | Not yet shipped |

---

## Worked example: how we validate the FFT

The FFT is one category out of many validated this way; the same pattern applies to every GPU-accelerated and reference-backed builtin in the coverage table. We walk through it here because it crosses a CPU/GPU boundary against a well-known external reference, which makes every piece of the methodology visible in a single test file.

On the CPU, RunMat calls [`rustfft`](https://crates.io/crates/rustfft) 6.4, the same crate that underlies much of the Rust numerical ecosystem (~15M downloads). On the GPU, RunMat ships its own staged WGPU kernel. The parity test at [`crates/runmat-accelerate/tests/fft_staged.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/tests/fft_staged.rs) runs the same inputs through both paths and compares them elementwise, with the tolerance chosen by the active provider's precision:

```rust
fn tolerance(precision: ProviderPrecision) -> f64 {
    match precision {
        ProviderPrecision::F32 => 1e-3,
        ProviderPrecision::F64 => 1e-9,
    }
}

// ...for each output element (got_re, got_im) vs (expected_re, expected_im):
assert!(
    (got_re - er).abs() <= tol && (got_im - ei).abs() <= tol,
    "FFT GPU/CPU drift at idx={idx}: got=({got_re},{got_im}) expected=({er},{ei}) tol={tol}"
);
```

If the GPU kernel drifts (from a compiler upgrade, a shader rewrite, or an environment variable change), CI fails on the next pull request. If `rustfft` drifts against its own prior behaviour after a version bump, the same test catches it. Other GPU-accelerated builtins follow the same pattern with tolerances matched to their own numerical properties (see the coverage table): fused reductions bound absolute error at `1e-6`, linear matmul epilogues at `1e-9`, and F32 matmul uses relative bounds like `5e-4 * max(|reference|, 1)` because the absolute answer grows with problem size.

---

## Reproduce it yourself

Every parity test listed above runs with standard `cargo test`. No MATLAB license or external data files required. A WGPU-compatible GPU is needed for GPU parity tests; CPU-only tests run anywhere.

```bash
# FFT: GPU staged kernel vs rustfft CPU reference
cargo test -p runmat-accelerate --features wgpu --test fft_staged

# Fused reduction: GPU x.*x then mean(...,'all') vs closed-form
cargo test -p runmat-accelerate --features wgpu --test fused_square_mean_all_parity

# Statistics: mean/sum reduction parity
cargo test -p runmat-runtime --test reduction_parity

# BLAS/LAPACK wrappers (requires system BLAS)
cargo test -p runmat-runtime --features blas-lapack --test blas_lapack

# All GPU parity tests at once
cargo test -p runmat-accelerate --features wgpu
```

Every command above is copy-pasteable against a fresh `git clone` of [runmat-org/runmat](https://github.com/runmat-org/runmat).

---

## How RunMat compares

| Trait | RunMat | MATLAB | Octave | NumPy / SciPy |
|---|---|---|---|---|
| **Numerical backend visible?** | Yes; every crate, version, and test is public | No; closed-source binary | Yes | Yes |
| **Tolerance documented per builtin?** | Yes, on this page and per-function pages | No public methodology | Per-test in source | Per-test in source |
| **GPU path validated against CPU?** | Yes; every GPU builtin has a parity test | Parallel Computing Toolbox; no public parity methodology | No native GPU | Requires CuPy/JAX with their own validation |
| **Parity tests runnable by users?** | Yes, via `cargo test` | No | Yes (Octave CI) | Yes |
| **Validation against reference implementation?** | `rustfft`, LAPACK, `nalgebra`, own CPU | Internal only | Own reference | LAPACK / reference BLAS |
| **Open-source runtime?** | Yes | No | Yes | Yes |

An engineer facing an audit cares most about the last row: can they prove the answer is right to a third party? With MATLAB, the evidence chain stops at the vendor. With RunMat, it continues to the test file in the repository.

---

## Known feature gaps

Things RunMat doesn't currently implement, or implements with a narrower scope than MATLAB:

- **Generalised eigenvalue problems.** `eig(A, B)` is not yet implemented. The `'balance'`/`'nobalance'` options are accepted for forward compatibility but are currently no-ops.
- **LU, QR in-repo vs LAPACK cross-check.** These factorizations are implemented in-repo. GPU-vs-CPU parity is covered; a planned cross-check against LAPACK for the same inputs is not yet in CI.
- **Fusion-only GPU reductions.** Some reduction parities (sum, mean) are validated through the fusion engine's end-to-end parity tests (e.g. [`fused_square_mean_all_parity.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-accelerate/tests/fused_square_mean_all_parity.rs)) rather than standalone kernels. Coverage is real but the seams aren't obvious from a single file.
- **`RunMatLCG` is not Mersenne Twister.** `rand`/`randn` use a custom LCG. Sequences are deterministic per seed but will not match MATLAB's or NumPy's output for the same seed.
- **ODE, optimization, interpolation.** Not shipped. Each will receive its own validation row when it lands.

---

## Axes of non-reproducibility

Even a builtin that is "validated" here will not produce *bit-identical* output across every environment, because IEEE 754 floating-point arithmetic is non-associative and modern hardware reorders work for performance. These are the same axes [PyTorch's numerical-accuracy note](https://pytorch.org/docs/stable/notes/numerical_accuracy.html), [NumPy's discussion of reproducibility](https://numpy.org/doc/stable/reference/random/bit_generators/index.html), and Intel's [CNR documentation](https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-conditional-numerical-reproducibility-cnr.html) call out. RunMat validates results *within* a tolerance, not to the last bit, because of these:

- **Parallel reduction order.** Summing a vector across 2 threads versus 4 threads reorders adds. The last-bit result can change, the mathematical answer does not.
- **SIMD width.** Running a kernel with AVX2 (8-wide F32) versus AVX-512 (16-wide F32) changes accumulation grouping.
- **GPU backend and device.** Apple Metal, NVIDIA, and AMD expose different shader compilers and different available instructions. Two GPUs can return different last bits for the same WGPU kernel.
- **Fused multiply-add (FMA).** A path that uses `fma(a, b, c)` produces a different last bit than `a * b + c` computed as two separate rounded operations. CPU, GPU, and compiler all make this choice independently.
- **RNG algorithm.** `RunMatLCG` is deterministic per seed but will not match Mersenne Twister (MATLAB) or PCG64 (NumPy) output for the same seed. Changing RNG is a breaking change we have not made.
- **Floating-point precision mode.** Running in F32 mode versus F64 mode produces different tolerances on every operation. The coverage table lists both where applicable.

None of these are bugs. They are the shape of floating-point computation on modern hardware, and they are why every mainstream numerical runtime documents tolerances rather than promising bit-identical output.

---

## CI commitment

Every new numerical builtin ships with a parity test before it merges. The test must name its reference (external crate, LAPACK routine, analytic solution, or CPU path) and document its tolerance. When the test merges, this page gains a row. We will never silently remove a row. If a validation regresses or a dependency changes, the row's status updates to reflect that.

External crate versions are pinned in the workspace `Cargo.toml`. When we bump a version, the corresponding parity test re-runs in CI. If it breaks, the bump doesn't ship.

---

## Glossary

- **Tolerance.** The maximum allowed difference between two numerical results before a test fails. RunMat tests use two forms: **`atol`** (absolute tolerance, a fixed floating-point value like `1e-9`) and **`rtol`** (relative tolerance, a bound that scales with the magnitude of the reference, e.g. `5e-4 * max(|reference|, 1)`). Choice of form depends on whether the operation's absolute answer grows with input size. Same vocabulary as NumPy's [`numpy.testing.assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html).
- **Parity test.** An automated test that computes the same quantity two ways (e.g. GPU and CPU, or RunMat and `rustfft`) and asserts the results agree within a documented tolerance.
- **IEEE 754.** The floating-point arithmetic standard implemented in every modern CPU and GPU. Defines f32/f64 representation, rounding modes, and exceptional values.
- **LAPACK.** Linear Algebra PACKage. The standard library of dense matrix routines underneath NumPy, SciPy, MATLAB, Julia, and most of the rest of scientific computing. RunMat wraps it via FFI.
- **BLAS.** Basic Linear Algebra Subprograms. The low-level matrix/vector kernels LAPACK is built on. Apple Accelerate and OpenBLAS are two widely-shipped implementations.
- **WGPU.** The Rust implementation of the WebGPU API. RunMat's GPU path targets WGPU so the same kernels run on Apple Metal, NVIDIA CUDA-compatible drivers, AMD, and any other WGPU-capable device.

---

## FAQ

### How does RunMat validate numerical accuracy?

Three mechanisms, stacked:

1. **Inherited correctness from established libraries.** FFTs come from [`rustfft`](https://crates.io/crates/rustfft) 6.4, SVD from [`nalgebra`](https://crates.io/crates/nalgebra) 0.32, complex arithmetic from [`num-complex`](https://crates.io/crates/num-complex) 0.4. These crates have their own test suites, their own reference implementations, and millions of downloads in production.
2. **Optional FFI to platform-native BLAS and LAPACK.** Apple Accelerate on macOS, OpenBLAS on Linux. The same libraries NumPy, SciPy, and MATLAB's own solvers rely on.
3. **In-repo solvers validated against external references or RunMat's own CPU path.** RunMat ships ~450 total builtins; the ~110 that live under `crates/runmat-runtime/src/builtins/math/` are the numerical-math subset where tolerance-based validation applies. Those ship with ~1,635 co-located `#[test]` functions plus ~40 integration-level parity tests covering GPU-vs-CPU, fusion engine, feature-gated FFI, and RNG statistics.

Every row in the [coverage table](#coverage-table) cites its reference implementation, parity test file, and enforced tolerance. For one category walked through end-to-end (inputs, reference, tolerance choice, CI hook), see the [FFT worked example](#worked-example-how-we-validate-the-fft).

### What reference implementations does RunMat validate against?

Established open-source Rust crates ([`rustfft`](https://crates.io/crates/rustfft), [`nalgebra`](https://crates.io/crates/nalgebra), [`num-complex`](https://crates.io/crates/num-complex)), platform-native BLAS and LAPACK (Apple Accelerate on macOS, OpenBLAS elsewhere), closed-form or analytic solutions for elementary math and reductions, and RunMat's own CPU paths when validating GPU kernels. Every parity test cites its reference in the test file itself.

### How do you validate GPU results?

Every GPU-accelerated builtin is parity-tested against RunMat's CPU path. Each test picks a tolerance from the numerical properties of its specific operation: absolute bounds for most cases, relative scaling for operations where magnitudes grow (SYRK, large-k F32 matmul). See the [coverage table](#coverage-table) for the exact `atol` and `rtol` per builtin. If a GPU path drifts past its test's bound, CI fails the build before it ships.

### What tolerances does RunMat use and why?

Tolerances are chosen per operation, not by a universal rule. For F64, most tests bound absolute error between `1e-9` and `1e-12`: tighter for closed-form operations (elementary math, reductions on well-conditioned inputs), looser for factorizations that accumulate rounding across many operations. For F32, most tests bound absolute error between `1e-5` and `1e-6`; operations where absolute magnitudes grow with problem size (SYRK, large-k matmul) use relative bounds like `5e-4 * max(|reference|, 1)` instead. The [coverage table](#coverage-table) lists exact `atol` and `rtol` per category, and every test file cites its own tolerance inline with a comment explaining the choice.

### Why doesn't `a == b` return `true` when `a` and `b` look equal?

Exact equality on floating-point values almost never does what you want. IEEE 754 arithmetic produces slightly different last bits depending on operation order, SIMD width, and fused multiply-adds, so two paths that compute the "same" mathematical answer routinely disagree in the 52nd bit. The idiomatic check is:

```matlab
abs(a - b) < 1e-9              % scalars
all(abs(a - b) < 1e-9, 'all')  % arrays (or use an isclose-style helper)
```

This is the same behaviour as NumPy, MATLAB, Julia, and PyTorch. If exact equality is what you need, you're probably comparing integers stored as floats; cast to `int64` first and compare those.

### Are all builtins validated?

Yes, for the builtins where tolerance-based validation is the right question. Of RunMat's ~450 total builtins, ~110 are numerical-math builtins (under `crates/runmat-runtime/src/builtins/math/`); all but two `common.rs` helper modules in that subset carry co-located unit tests, and a smaller GPU-accelerated, LAPACK-wrapped, or cross-cutting reduction subset additionally has integration-level parity tests. The remaining ~340 builtins handle string, array, plotting, I/O, and OOP; these are validated by behavioural tests rather than tolerance-based parity, since "fprintf returned the right text" doesn't have an `atol`. The [coverage table](#coverage-table) above lists every numerical builtin category with its current validation status. We never silently remove a row: if validation regresses, the status updates.

### Can I run the validation tests myself?

Yes. Every parity test ships in the public repository and runs with standard `cargo test`. No MATLAB license, no external data files. A WGPU-compatible GPU is required for GPU parity tests; CPU tests run anywhere. Commands are in the [Reproduce it yourself](#reproduce-it-yourself) section above.

### What happens when a dependency is updated?

Our `Cargo.toml` pins each numerical crate to a specific version. When we bump a version, the corresponding parity tests re-run in CI. If tolerances break, the bump does not ship.

### Does RunMat use the same random number generator as MATLAB?

No. RunMat's `rand` and `randn` use a custom linear congruential generator (`RunMatLCG`). Sequences are deterministic given a seed, but they will not match MATLAB's Mersenne Twister or NumPy's PCG64. If your workflow depends on reproducing a specific random sequence from another tool, this is a known difference.

### Why don't you guarantee bit-identical reproduction of MATLAB output?

Accuracy and bit-identical reproducibility are different contracts. IEEE 754 arithmetic produces different last-bit results depending on operation ordering, SIMD width, thread count, and fused multiply-adds; two MATLAB installations on different hardware can disagree in the last bit for the same code. Intel's math library calls this the CNR tradeoff (Conditional Numerical Reproducibility): you can have peak performance or bit-identical output, not both for free. RunMat provides validated numerical accuracy within documented tolerances, not bit-identical reproduction of any particular binary.

### How do I report a numerical bug?

Open an issue at [github.com/runmat-org/runmat/issues](https://github.com/runmat-org/runmat/issues/new) with a minimal MATLAB-syntax reproduction, the expected output (and its source), and the observed output. Numerical bugs are treated as blockers.

---

## Report a numerical issue

If a RunMat builtin returns a result you believe is wrong, we want to hear about it. [Open an issue](https://github.com/runmat-org/runmat/issues/new) with:

1. A minimal MATLAB-syntax snippet that reproduces the problem.
2. The expected output and where that expectation comes from (MATLAB version, a textbook formula, a hand calculation, another tool).
3. The observed RunMat output, including the RunMat version (`runmat --version`) and GPU backend if relevant.

A numerical regression is a higher-priority bug class than a performance regression. We fix them first.

---

*Last reviewed: 2026-04-17 · Source: [`docs/CORRECTNESS.md`](https://github.com/runmat-org/runmat/blob/main/docs/CORRECTNESS.md)*
