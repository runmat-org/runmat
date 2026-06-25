---
title: "Build System"
category: "Development"
section: "14.1"
last_updated: "May 28, 2026"
---

# Build System

RunMat builds from a single Cargo workspace. The workspace keeps the language pipeline, execution engines, runtime builtins, acceleration layer, plotting, CLI, LSP, snapshotting, filesystem, and WASM bindings in one versioned graph.

The TypeScript package in `bindings/ts` contains bindings for the WASM runtime to run in the browser, along with the LSP bundle, generated builtin metadata, and startup snapshot used by JavaScript consumers.

## Workspace Layout

The root workspace uses Cargo resolver v2. Workspace dependency versions live in the root `Cargo.toml`; internal crates are pinned to the same RunMat version.

| Area | Crates |
| --- | --- |
| Language pipeline | `runmat-lexer`, `runmat-parser`, `runmat-hir`, `runmat-mir`, `runmat-static-analysis` |
| Execution | `runmat-vm`, `runmat-turbine`, `runmat-core` |
| Runtime | `runmat-runtime`, `runmat-builtins`, `runmat-filesystem`, `runmat-time`, `runmat-config` |
| Performance systems | `runmat-accelerate`, `runmat-accelerate-api`, `runmat-gc`, `runmat-gc-api`, `runmat-plot`, `runmat-snapshot` |
| Host surfaces | `runmat` CLI, `runmat-lsp`, `runmat-wasm`, `runmat-server-client`, `runmat-telemetry`, `runmat-logging` |

The `runmat` binary lives in `crates/runmat-cli`. It depends on the compiler, VM, runtime, plotting, acceleration, filesystem, config, telemetry, and session crates, so a default CLI build is the broadest native build target.

## Feature Flags

The CLI default feature set enables the normal local developer experience:

| Feature | Effect |
| --- | --- |
| `gui` | Enables native plotting GUI support through `runmat-plot`. |
| `blas-lapack` | Enables high-performance BLAS/LAPACK operations in `runmat-runtime`. |
| `wgpu` | Enables the WGPU acceleration path. |
| `jit` | Enables the Turbine JIT tier through `runmat-core`. |

Additional flags matter for specific builds:

| Feature | Use |
| --- | --- |
| `blas-only` | Enables BLAS without LAPACK. |
| `vendored-openssl` | Builds release/cross targets without relying on target-system OpenSSL discovery. |
| `plot-web` | Used by the WASM runtime for browser plotting support. |
| `runmat-wasm/gpu` | Default WASM feature that enables WebGPU-backed acceleration where available. |
| `runmat-lsp/wasm` | Browser-oriented LSP build with native defaults disabled. |

`runmat-accelerate` declares backend feature names for CUDA, ROCm, Metal, Vulkan, OpenCL, and WGPU. The wired backend in the current workspace is WGPU; the other feature names are placeholders for backend-specific integration.

## Native Dependencies

The native build touches numerical, graphics, and networking libraries.

| Dependency | Platform behavior |
| --- | --- |
| BLAS/LAPACK | macOS uses Apple's Accelerate framework. Linux and Windows use OpenBLAS/LAPACK through system packages or vcpkg. |
| OpenSSL | Linux release builds use system OpenSSL. Non-Linux release targets enable `vendored-openssl`. |
| WGPU/GUI stack | Native plotting and GPU builds pull WGPU, windowing, EGL/GL, Wayland/X11, udev, and related platform packages. |
| ZeroMQ | CI and runner provisioning install ZeroMQ packages for environments that need the server/client stack. |

`crates/runmat-runtime/build.rs` participates in BLAS/LAPACK discovery when `blas-lapack` is enabled. It honors the standard library hints used by local packages and vcpkg: `VCPKG_ROOT`, `VCPKGRS_TRIPLET`, `VCPKG_DEFAULT_TRIPLET`, `OPENBLAS_DIR`, `BLAS_LIB_DIR`, `BLAS_LIBS`, `LAPACK_LIB_DIR`, and `LAPACK_LIBS`.

On Ubuntu-like systems, the important local packages are:

```bash
sudo apt-get install -y libopenblas-dev liblapack-dev libzmq3-dev pkg-config libssl-dev
```

The full Linux runner also installs GUI/GPU headers and libraries such as X11, Wayland, EGL, GL, udev, and dbus because CI builds all targets and features.

## Common Rust Builds

Use the default build for normal local development:

```bash
cargo build
cargo build -p runmat
```

The default CLI feature set includes `occt-native` for STEP, IGES, and BREP CAD topology import. If `RUNMAT_OCCT_ROOT` or `RUNMAT_OCCT_INCLUDE_DIR`/`RUNMAT_OCCT_LIB_DIR` do not point to an existing OCCT installation, the build uses bundled OCCT and requires CMake on `PATH` or through Cargo's `CMAKE` environment overrides. On macOS, install it with `brew install cmake`.

Developers without CMake or OCCT can build the CLI with the default local feature set minus OCCT CAD topology import:

```bash
cargo build-no-occt
```

This alias enables `gui`, `blas-lapack`, `wgpu`, and `jit`, but excludes `occt-native`.

Use release mode when checking CLI performance or benchmark behavior:

```bash
cargo build -p runmat --release
```

Release and cross-build jobs use locked dependencies and explicit feature sets:

```bash
cargo build --locked --release --bin runmat --features blas-lapack
cargo build --locked --release --bin runmat --features blas-lapack,vendored-openssl
```

Linux release builds use the first form. Windows and macOS release builds use the vendored OpenSSL form.

## WASM And TypeScript Build

The WebAssembly target is `wasm32-unknown-unknown`. The CI path first generates the runtime builtin registry for WASM, then builds and tests the bindings:

```bash
rustup target add wasm32-unknown-unknown
scripts/regenerate-wasm-registry.sh
cargo build -p runmat-wasm --target wasm32-unknown-unknown --features occt-wasm-host
scripts/test-wasm-headless.sh
```

The TypeScript package owns the distributable browser artifacts:

```bash
cd bindings/ts
npm ci
npm run build
```

`npm run build` cleans previous artifacts, generates builtin metadata, builds the web WASM package, builds the WASM LSP package, emits TypeScript, creates the standard-library snapshot, and syncs WASM artifacts into `dist`.

The WASM registry has an ordering constraint: proc macros write registry entries while `runmat-runtime` compiles for `wasm32-unknown-unknown`. Always use `scripts/regenerate-wasm-registry.sh`; it generates the production `plot-web,occt-wasm-host` registry into a temporary file, marks it complete only after cargo succeeds, then atomically replaces `generated_wasm_registry.rs`. Normal WASM builds validate the source fingerprint, target/features, completion marker, and entry count, and fail if the registry is missing, partial, stale, or generated for another runtime configuration.

## Release Helpers

Release versioning is handled by `scripts/cut-release.sh <version>`. The script validates the version and clean branch state, updates workspace crate versions and the TypeScript package version, runs `cargo check -q`, commits, tags, and pushes.

Native release artifacts are built by GitHub Actions for the supported release triples listed in [Supported Architectures](/docs/runtime/development/supported-architectures). WASM package publication is handled by the `wasm-bindings` workflow.
