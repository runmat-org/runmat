# Agent Instructions

## No Allow

- Avoid adding `#[allow(...)]` attributes or lint suppressions.
- If an allow is absolutely unavoidable, exhaust alternatives first and add a brief inline comment explaining why the allow is necessary at that location.

## Documentation Co-Updates

- When making changes to an area of the codebase, update `docs/*.md` files if any relevant documentation has changed.

## Cursor Cloud specific instructions

### Project overview

RunMat is a high-performance MATLAB-compatible runtime written as a Cargo workspace (~30 Rust crates) under `crates/`. The main binary is `runmat` (crate `runmat-cli`). See `README.md` for full architecture.

### Build & run

- **Build:** `CXX=g++ cargo build --workspace`
- **Run CLI:** `./target/debug/runmat` (REPL) or pipe scripts: `echo '...' | ./target/debug/runmat`
- **Run a `.m` file:** `./target/debug/runmat run script.m`
- The `CXX=g++` env var is required because the default `c++` (clang) cannot locate the GCC C++ stdlib headers needed to compile the vendored ZeroMQ dependency (`zmq-sys`).

### Lint

- `cargo fmt --all -- --check`
- `CXX=g++ cargo clippy --workspace --all-targets --all-features`

### Tests

- `CXX=g++ cargo test --workspace`
- One pre-existing test failure in `runmat-ignition::matrix_slicing::empty_slice_from_two_arg_colon` is a known issue (not environment-related).
- GPU/wgpu warnings (Vulkan surface extensions, EGL/DRI3 errors) are expected in headless environments without a GPU—they do not affect correctness.

### System dependencies (already installed in the VM snapshot)

`libopenblas-dev`, `libzmq3-dev`, `libssl-dev`, `pkg-config`, `cmake`, and X11/Wayland/EGL/Mesa/GTK3 dev packages for the `gui` feature. A `libstdc++.so` symlink at `/usr/lib/x86_64-linux-gnu/libstdc++.so` pointing to the GCC 13 version is also required for linking.

### Key non-obvious notes

- The `gui` feature (default-on) compiles but cannot display windows in headless environments. Plotting tests that require a display are skipped automatically.
- The Rust toolchain version (1.90.0) is pinned via `rust-toolchain.toml`—do not upgrade without updating that file.
- Default features for the CLI binary are `gui`, `blas-lapack`, `wgpu`, `jit` (see `crates/runmat-cli/Cargo.toml`).