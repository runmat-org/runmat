---
title: "Correctness & Trust"
description: "How RunMat validates numerical correctness: GPU/CPU parity tests, LAPACK-backed solvers, documented tolerances, and CI-gated regression checks."
keywords:
  - RunMat correctness
  - MATLAB alternative accuracy
  - numerical correctness
  - GPU CPU parity
  - LAPACK
  - rustfft
  - floating-point tolerance
  - MATLAB numerical validation
ogTitle: "Correctness & Trust"
ogDescription: "Every numerical path in RunMat is traceable, testable, and reproducible. See the full coverage table, parity test links, and methodology."
jsonLd:
  "@context": "https://schema.org"
  "@graph":
    - "@type": "FAQPage"
      "@id": "https://runmat.com/docs/correctness#faq"
      mainEntity:
        - "@type": "Question"
          name: "Is RunMat bit-exactly equivalent to MATLAB?"
          acceptedAnswer:
            "@type": "Answer"
            text: "No, and no runtime that claims otherwise is telling the truth. IEEE 754 floating-point arithmetic produces different least-significant bits depending on operation ordering, fused multiply-adds, SIMD width, and compiler flags. RunMat targets mathematical correctness within documented tolerances (1e-9 for f64, 1e-3 for f32), not bitwise reproduction of MATLAB's binary."
        - "@type": "Question"
          name: "What reference implementations does RunMat validate against?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Established open-source Rust crates (rustfft, nalgebra, num-complex), platform-native BLAS and LAPACK (Apple Accelerate on macOS, OpenBLAS elsewhere), and RunMat's own CPU paths when validating GPU kernels. Every parity test cites its reference in the test file itself."
        - "@type": "Question"
          name: "How do you validate GPU results?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Every GPU-accelerated builtin is parity-tested against RunMat's CPU path. The GPU kernel must reproduce the CPU result within a precision-dependent tolerance: 1e-9 for f64 and 1e-3 for f32. If a GPU path drifts, CI fails the build before it ships."
        - "@type": "Question"
          name: "What tolerances does RunMat use and why?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Double precision (f64) uses 1e-9 for element-wise comparisons; single precision (f32) uses 1e-3. These are consistent with the representable precision of each format after accumulated rounding across realistic workloads (matmul, FFT, reductions). Some tests use looser or tighter bounds documented inline with each test."
        - "@type": "Question"
          name: "Are all builtins validated?"
          acceptedAnswer:
            "@type": "Answer"
            text: "No. The coverage table on this page lists every numerical builtin with its current validation status. Some functions are validated GPU-vs-CPU; some are CPU-only validated; some are not yet shipped. We never silently remove a row: if validation regresses, the status updates."
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

**What this page is:** a living inventory of the external crates and internal solvers behind RunMat's numerical builtins, the parity tests that validate them, and the tolerances those tests enforce.

**What this page isn't:** a claim of bit-exact equivalence with MATLAB. We validate against open-source reference implementations and our own CPU paths, not against a proprietary binary. Where we haven't validated something yet, we say so.

---

## Why we don't chase bit-exact MATLAB parity

IEEE 754 arithmetic produces different least-significant bits for the same mathematical expression depending on operation ordering, fused multiply-adds, SIMD width, and compiler flags. A parallel reduction with two threads and a parallel reduction with four threads can disagree in the last bit. That's physics, not a bug.

Bit-exact parity with MATLAB is therefore impossible in principle. Chasing it in practice means shipping a black box nobody can debug. Our design philosophy prioritizes *auditable correctness over opaque compatibility*. Read more in [Design Philosophy](/docs/design-philosophy).

What we commit to instead:

- Numerical results within a documented tolerance of an open-source reference.
- Every builtin's implementation is readable, and every validation test is runnable, in public.
- Regressions are caught in CI before a release ships, not in a customer's pipeline.

---

## Correctness tiers

RunMat's builtins fall into three correctness tiers, tested differently:

- **Crate-backed builtins.** When a builtin delegates to a well-established Rust crate ([rustfft](https://crates.io/crates/rustfft) for FFT, [nalgebra](https://crates.io/crates/nalgebra) for SVD), we inherit that crate's correctness. Our tests confirm RunMat's calling convention, array layout, and output format reproduce the crate's results, not re-prove the crate's internals.
- **LAPACK-backed builtins.** Optional FFI to platform-native BLAS and LAPACK ([lapack](https://crates.io/crates/lapack) 0.19, [blas](https://crates.io/crates/blas) 0.22, Apple Accelerate via [accelerate-src](https://crates.io/crates/accelerate-src) on macOS, [openblas-src](https://crates.io/crates/openblas-src) elsewhere). These are the same libraries NumPy, SciPy, and MATLAB's own solvers rely on.
- **In-repo solvers.** Some factorizations (LU, QR, Cholesky) and small in-place routines are implemented directly in the RunMat runtime. For these, we test against a known-correct reference (an external crate, LAPACK, or hand-derived expected values) to a documented floating-point tolerance.
- **GPU paths.** Every GPU-accelerated builtin is parity-tested against RunMat's own CPU path. The GPU kernel must reproduce the CPU result within a precision-dependent tolerance: `1e-9` for f64, `1e-3` for f32. If the GPU path drifts, CI catches it before a release ships.

---

## Coverage table

Every row in this table carries a live GitHub link on the `dev` branch. Rows tagged *Validated* have an automated parity test. Rows tagged *CPU-only validated* have CPU correctness coverage but no GPU parity test yet. Rows tagged *Follow-up* are shipped but lack a dedicated test; they are tracked and will move to *Validated* as tests land.

| Category | Builtins | Implementation | Parity test | Tolerance | Status |
|---|---|---|---|---|---|
| **FFT** | [`fft`](/docs/reference/builtins/fft), [`ifft`](/docs/reference/builtins/ifft), [`fft2`](/docs/reference/builtins/fft2) | [`rustfft`](https://crates.io/crates/rustfft) 6.2 (CPU); staged WGPU kernel (GPU) | [`fft_staged.rs`](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-accelerate/tests/fft_staged.rs) | `1e-9` (f64), `1e-3` (f32) | Validated |
| **SVD** | [`svd`](/docs/reference/builtins/svd) | [`nalgebra`](https://crates.io/crates/nalgebra) 0.32 `linalg::SVD` | [`svd.rs` module tests](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/src/builtins/math/linalg/factor/svd.rs) | Per-test | Validated |
| **Eigendecomposition** | [`eig`](/docs/reference/builtins/eig) | LAPACK `dgeev`/`zgeev` via [`lapack`](https://crates.io/crates/lapack) 0.19 FFI | [`eig.rs` module tests](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/src/builtins/math/linalg/factor/eig.rs) | Per-test | CPU-only validated |
| **Linear solve** | [`linsolve`](/docs/reference/builtins/linsolve), `\` (mldivide) | [`nalgebra`](https://crates.io/crates/nalgebra) 0.32 `SVD` + `DMatrix` | [`linalg` module tests](https://github.com/runmat-org/runmat/tree/dev/crates/runmat-runtime/src/builtins/math/linalg) | Per-test | Validated |
| **LU factorization** | [`lu`](/docs/reference/builtins/lu) | In-repo partial-pivot solver | [`lu.rs` module tests](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/src/builtins/math/linalg/factor/lu.rs) | Per-test | GPU-vs-CPU validated |
| **QR factorization** | [`qr`](/docs/reference/builtins/qr) | In-repo solver | [`qr.rs` module tests](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/src/builtins/math/linalg/factor/qr.rs) | Per-test | GPU-vs-CPU validated |
| **Cholesky** | [`chol`](/docs/reference/builtins/chol) | LAPACK `dpotrf` via [`lapack`](https://crates.io/crates/lapack) 0.19 FFI (feature-gated) + in-repo fallback | [`chol.rs` module tests](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/src/builtins/math/linalg/factor/chol.rs) | Per-test | CPU-only validated |
| **Statistics** | [`mean`](/docs/reference/builtins/mean), [`std`](/docs/reference/builtins/std), [`var`](/docs/reference/builtins/var), [`median`](/docs/reference/builtins/median) | Runtime reduction infrastructure | [`reduction_parity.rs`](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/tests/reduction_parity.rs); [`std.rs` tests](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/src/builtins/math/reduction/std.rs) | `1e-6` to `1e-9` | Validated |
| **Signal** | [`filter`](/docs/reference/builtins/filter), [`conv`](/docs/reference/builtins/conv), [`conv2`](/docs/reference/builtins/conv2) | In-repo + GPU provider hooks; [`num-complex`](https://crates.io/crates/num-complex) 0.4 for complex paths | [`conv2.rs` module tests](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/src/builtins/math/signal/conv2.rs) | Per-test | Validated (module-level) |
| **Polynomials** | [`polyval`](/docs/reference/builtins/polyval) | In-repo Horner's method | [`polyval.rs` module tests](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/src/builtins/math/poly/polyval.rs) | `1e-9` (f64) | CPU-only validated |
| **Vector algebra** | [`cross`](/docs/reference/builtins/cross), [`dot`](/docs/reference/builtins/dot) | In-repo + WGPU provider | [`cross.rs` module tests](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/src/builtins/math/linalg/ops/cross.rs) | Per-test | CPU-only validated |
| **Matrix multiply (GPU)** | `*` (matmul) | WGPU fused kernels | [`matmul_residency.rs`](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-accelerate/tests/matmul_residency.rs), [`matmul_epilogue.rs`](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-accelerate/tests/matmul_epilogue.rs) | `1e-6` / `1e-9` | Validated |
| **Fused reductions (GPU)** | `mean(...,'all')` on fused expressions | WGPU fusion engine | [`fused_square_mean_all_parity.rs`](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-accelerate/tests/fused_square_mean_all_parity.rs) | `1e-6` | Validated |
| **BLAS / LAPACK (optional)** | Enabled via `blas-lapack` feature flag | FFI to system BLAS/LAPACK via [`blas`](https://crates.io/crates/blas) 0.22, [`lapack`](https://crates.io/crates/lapack) 0.19; [`accelerate-src`](https://crates.io/crates/accelerate-src) 0.3 on macOS, [`openblas-src`](https://crates.io/crates/openblas-src) 0.10 elsewhere | [`blas_lapack.rs`](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-runtime/tests/blas_lapack.rs) | Exact for small integers | Available |
| **RNG** | [`rand`](/docs/reference/builtins/rand), [`randn`](/docs/reference/builtins/randn) | Custom `RunMatLCG` (in-repo) | `rng.rs` integration test | Exact (seeded) | Validated |
| **ODE solvers** | `ode45`, `ode23` | — | — | — | Not yet shipped |
| **Optimization** | `fminunc`, `fminsearch`, `lsqcurvefit` | — | — | — | Not yet shipped |
| **Interpolation** | `interp1`, `interp2` | — | — | — | Not yet shipped |

**Coverage snapshot:** 14 of 17 numerical-builtin categories shipped; 11 validated, 3 CPU-only validated, 1 follow-up, 3 not yet shipped.

---

## Worked example: how we validate the FFT

The FFT is the load-bearing kernel for half of RunMat's signal and image workloads. It is also the cleanest illustration of how we validate GPU-accelerated paths against a CPU reference.

On the CPU, RunMat calls [`rustfft`](https://crates.io/crates/rustfft) 6.2, the same crate that underlies much of the Rust numerical ecosystem (~15M downloads). On the GPU, RunMat ships its own staged WGPU kernel. The parity test at [`crates/runmat-accelerate/tests/fft_staged.rs`](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-accelerate/tests/fft_staged.rs) runs the same inputs through both paths, computes `Y = rustfft(x)` on the CPU and `X = wgpu_fft(x)` on the GPU, and asserts:

```rust
let max_err = (X - Y).iter().map(|z| z.norm()).fold(0.0_f64, f64::max);
assert!(max_err < 1e-9, "FFT GPU/CPU drift: {max_err}");
```

If the GPU kernel drifts, whether from a compiler upgrade or a shader rewrite, CI fails on the next pull request. If `rustfft` drifts against its own prior behaviour after a version bump, the same test catches it. This is the pattern every GPU-accelerated numerical builtin follows.

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

## Known limitations

We flag real caveats here instead of hiding them per-function:

- **Fusion-only GPU reductions.** Many reduction parities (sum, mean) are validated through the fusion engine's end-to-end parity tests (e.g. [`fused_square_mean_all_parity.rs`](https://github.com/runmat-org/runmat/blob/dev/crates/runmat-accelerate/tests/fused_square_mean_all_parity.rs)) rather than standalone kernels. Coverage is real but the seams aren't obvious from a single file.
- **`RunMatLCG` is not Mersenne Twister.** `rand`/`randn` use a custom LCG. Sequences are deterministic per seed but will not match MATLAB's or NumPy's output for the same seed.
- **Generalised eigenvalue problems.** `eig(A, B)` is not yet implemented. The `'balance'`/`'nobalance'` options are accepted for forward compatibility but are currently no-ops.
- **LU, QR in-repo vs LAPACK.** These factorizations are implemented in-repo. GPU-vs-CPU parity is covered; a planned cross-check against LAPACK for the same inputs is not yet in CI.
- **ODE, optimization, interpolation.** Not shipped. Each will receive its own validation row when it lands.

---

## CI commitment

Every new numerical builtin ships with a parity test before it merges. The test must name its reference (external crate, LAPACK routine, analytic solution, or CPU path) and document its tolerance. When the test merges, this page gains a row. We will never silently remove a row. If a validation regresses or a dependency changes, the row's status updates to reflect that.

External crate versions are pinned in the workspace `Cargo.toml`. When we bump a version, the corresponding parity test re-runs in CI. If it breaks, the bump doesn't ship.

---

## Glossary

- **Tolerance.** The maximum allowed absolute or relative difference between two numerical results before a test fails. Expressed as a float, e.g. `1e-9`.
- **Parity test.** An automated test that computes the same quantity two ways (e.g. GPU and CPU, or RunMat and `rustfft`) and asserts the results agree within a documented tolerance.
- **IEEE 754.** The floating-point arithmetic standard implemented in every modern CPU and GPU. Defines f32/f64 representation, rounding modes, and exceptional values.
- **LAPACK.** Linear Algebra PACKage — the standard library of dense matrix routines underneath NumPy, SciPy, MATLAB, Julia, and most of the rest of scientific computing. RunMat wraps it via FFI.
- **BLAS.** Basic Linear Algebra Subprograms — the low-level matrix/vector kernels LAPACK is built on. Apple Accelerate and OpenBLAS are two widely-shipped implementations.
- **WGPU.** The Rust implementation of the WebGPU API. RunMat's GPU path targets WGPU so the same kernels run on Apple Metal, NVIDIA CUDA-compatible drivers, AMD, and any other WGPU-capable device.

---

## FAQ

### Is RunMat bit-exactly equivalent to MATLAB?

No, and no runtime that claims otherwise is telling the truth. IEEE 754 floating-point arithmetic produces different least-significant bits depending on operation ordering, fused multiply-adds, SIMD width, and compiler flags. RunMat targets mathematical correctness within documented tolerances (`1e-9` for f64, `1e-3` for f32), not bitwise reproduction of MATLAB's binary.

### What reference implementations does RunMat validate against?

Established open-source Rust crates ([`rustfft`](https://crates.io/crates/rustfft), [`nalgebra`](https://crates.io/crates/nalgebra), [`num-complex`](https://crates.io/crates/num-complex)), platform-native BLAS and LAPACK (Apple Accelerate on macOS, OpenBLAS elsewhere), and RunMat's own CPU paths when validating GPU kernels. Every parity test cites its reference in the test file itself.

### How do you validate GPU results?

Every GPU-accelerated builtin is parity-tested against RunMat's CPU path. The GPU kernel must reproduce the CPU result within a precision-dependent tolerance: `1e-9` for f64 and `1e-3` for f32. If a GPU path drifts, CI fails the build before it ships.

### What tolerances does RunMat use and why?

Double precision (f64) uses `1e-9` for element-wise comparisons; single precision (f32) uses `1e-3`. These are consistent with the representable precision of each format after accumulated rounding across realistic workloads (matmul, FFT, reductions). Some tests use looser or tighter bounds documented inline with each test.

### Are all builtins validated?

No. The [coverage table](#coverage-table) on this page lists every numerical builtin with its current validation status. Some functions are validated GPU-vs-CPU; some are CPU-only validated; some are not yet shipped. We never silently remove a row: if validation regresses, the status updates.

### Can I run the validation tests myself?

Yes. Every parity test ships in the public repository and runs with standard `cargo test`. No MATLAB license, no external data files. A WGPU-compatible GPU is required for GPU parity tests; CPU tests run anywhere. Commands are in the [Reproduce it yourself](#reproduce-it-yourself) section above.

### What happens when a dependency is updated?

Our `Cargo.toml` pins each numerical crate to a specific version. When we bump a version, the corresponding parity tests re-run in CI. If tolerances break, the bump does not ship.

### Does RunMat use the same random number generator as MATLAB?

No. RunMat's `rand` and `randn` use a custom linear congruential generator (`RunMatLCG`). Sequences are deterministic given a seed, but they will not match MATLAB's Mersenne Twister or NumPy's PCG64. If your workflow depends on reproducing a specific random sequence from another tool, this is a known difference.

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

*Last reviewed: 2026-04-17 · Source: [`docs/CORRECTNESS.md`](https://github.com/runmat-org/runmat/blob/dev/docs/CORRECTNESS.md) · Coverage: 14 of 17 numerical-builtin categories shipped (11 validated, 3 CPU-only, 1 follow-up).*
