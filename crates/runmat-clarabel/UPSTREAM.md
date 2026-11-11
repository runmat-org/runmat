# Clarabel.rs in RunMat

This publishable `runmat-clarabel` package retains Clarabel.rs's source layout,
numerical implementations, upstream authors and Apache-2.0 license. The RunMat
package version is 0.7.1; the numerical upstream release is **Clarabel 0.11.1**.
The original solver banner/version remains 0.11.1, not the RunMat package version.

Upstream: https://github.com/oxfordcontrol/Clarabel.rs/tree/v0.11.1

Exact source: https://static.crates.io/crates/clarabel/clarabel-0.11.1.crate

Official registry archive SHA-256:
`d950c870fad4feed732bc3f7da489d223ba3911eac0522e6512eed7e659cfafe`.
The downloaded archive was verified against the official sparse registry entry
before extraction. The original upstream manifest is retained byte-for-byte as
`UPSTREAM-Cargo.toml`, and the original upstream license is retained. Cargo reserves
the filename `Cargo.toml.orig` for its own packaged manifest, so the upstream
manifest uses a distinct name in this publishable derivative. The upstream
registry archive's VCS provenance is likewise retained as
`UPSTREAM-vcs-info.json`; `.cargo_vcs_info.json` is reserved for Cargo's current
package provenance.

RunMat changes are limited to publishable package identity/workspace MSRV,
truthful upstream version presentation, the cooperative solver API and its tests,
feature-correct gating of JSON-only error types/imports, and explicit lifetime
syntax required by the workspace compiler, and equivalent standard-library
idioms required by workspace Clippy. Workspace profiles
and dependency resolution are selected at the workspace root rather than by
ignored nested package settings.
The default artifact is a Rust library. Mixed Rust-library and `cdylib` outputs
use shared, unhashed filenames, causing feature/profile variants to overwrite
one another in a shared Cargo target directory. Shared-library consumers select
`cargo rustc --lib --crate-type cdylib` with their required optional features.
The Python/Julia source and feature definitions remain available.
Rust source formatting follows the workspace toolchain; these mechanical
whitespace/layout changes do not alter the numerical algorithm or literals.
The existing synchronous solve body is a single private macro shared by the
unchanged synchronous API and the new owned `solve_with_progress` async method.
Numerical update order, kernels, scaling, residuals, termination checks and
solution post-processing are not independently reimplemented. The async method
awaits a lending progress callback between iterations, excluding host-turn time
from solver timers. It owns the solver across suspension, so cancellation/error
drops the same workspace rather than exposing a partially progressed borrowed
solver. No restart/replay, worker protocol, async runtime, boxed per-iteration
callback future or new numerical algorithm is introduced.

The async API requires the RunMat workspace's Rust 1.90 baseline. Runtime uses
the pure Rust core with default features disabled; optional upstream features
and dependencies remain intact rather than being pruned to satisfy a build.

Copyright and attribution remain with the upstream contributors as recorded in
LICENSE.md, CITATION.bib and the original source. New RunMat extension comments
identify the derivative changes. Publication of this package is separate from
the upstream Clarabel crate; consumers do not rely on a root-only Cargo patch.
