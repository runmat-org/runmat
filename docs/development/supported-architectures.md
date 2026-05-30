---
title: "Supported Architectures"
category: "Development"
section: "14.2"
last_updated: "May 28, 2026"
---

# Supported Architectures

RunMat has two major build families: native binaries for the CLI/runtime and WebAssembly artifacts for browser and TypeScript hosts. Native releases are built for Windows, macOS, and Linux; WASM builds target `wasm32-unknown-unknown`.

## Release Targets

The native release matrix builds these targets:

| Release artifact | Rust target |
| --- | --- |
| Windows x86_64 | `x86_64-pc-windows-msvc` |
| macOS Intel | `x86_64-apple-darwin` |
| macOS Apple Silicon | `aarch64-apple-darwin` |
| Linux x86_64 | `x86_64-unknown-linux-gnu` |

CI also cross-builds the same native targets before release. The main test matrix runs on Linux x86_64 and macOS; the release workflow additionally validates Windows before packaging.

## Toolchain

Rust is pinned to `1.90.0` with `rustfmt` and `clippy`. CI validates the exact toolchain on self-hosted Linux and Windows runners, and installs the same channel on macOS.

Windows MSVC builds use a larger linker stack through `.cargo/config.toml`. That setting covers `x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc`; the ARM64 Windows target is configured but not part of the current release matrix.

## Native Dependencies By Platform

| Platform | Dependency strategy |
| --- | --- |
| Linux x86_64 | System OpenBLAS, LAPACK, ZeroMQ, pkg-config, OpenSSL, and GUI/GPU development packages. |
| macOS x86_64/aarch64 | Accelerate for BLAS/LAPACK, Homebrew ZeroMQ, WGPU through Metal. |
| Windows x86_64 MSVC | Prebuilt ZeroMQ plus vcpkg OpenBLAS/LAPACK. Release builds use vendored OpenSSL. |

The CLI default build enables GUI, BLAS/LAPACK, WGPU, and JIT. If a platform has trouble with a native dependency, reduce the feature set while isolating the failure:

```bash
cargo build -p runmat --no-default-features
cargo build -p runmat --no-default-features --features jit
```

## WebAssembly

The WASM target is:

```text
wasm32-unknown-unknown
```

`runmat-wasm` is compiled as a `cdylib` and `rlib`, then packaged by `wasm-pack` for the TypeScript distribution. The package build emits browser WASM artifacts, LSP WASM artifacts, TypeScript definitions, builtin metadata, and a startup snapshot.

WASM builds differ from native builds in a few important ways:

| Area | WASM behavior |
| --- | --- |
| Thread-local state | Uses a custom single-thread `WasmTlsCell` instead of native `thread_local!`. |
| Filesystem | Uses browser or host callbacks behind the provider API; the default current directory is `/`. |
| Environment | Provides a synthetic environment such as `HOME=/`, `PATH=/`, and `USER=user`. |
| Input and async work | Awaits through the JavaScript async path instead of spawning native helper threads. |
| GPU | Requires browser WebGPU and async WGPU provider initialization. |

The headless WASM test scripts are built around Chrome on macOS and Linux. ChromeDriver resolution currently covers Darwin arm64, Darwin x86_64, and Linux x86_64.

## GPU Backend Support

RunMat's active GPU backend is WGPU. On native platforms, WGPU can route through the platform graphics stack, including Metal on macOS, DirectX 12 on Windows, and Vulkan or similar backends on Linux.

WASM GPU support depends on browser WebGPU availability. The WASM runtime initializes the WGPU provider asynchronously; synchronous WGPU provider registration is not available on `wasm32`.

Precision support depends on the active provider. Providers advertise supported precision, and RunMat guards kernels accordingly:

| Precision | Behavior |
| --- | --- |
| `F32` | Accepted on all GPU providers. |
| `F64` | Requires provider support for double precision. |
| `U8`, `U16` | Rejected by the current precision guard for GPU kernels. |

`RUNMAT_ALLOW_PRECISION_DOWNCAST=1` permits implicit downcast from double precision to provider-native precision and emits a warning.

## CI Coverage

The main Unix CI jobs run:

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo check --all-targets --all-features
RUST_TEST_THREADS=1 cargo test --all-targets --all-features
```

The cross-build job compiles all release targets in release mode. The WASM workflow builds `runmat-wasm` for `wasm32-unknown-unknown`, then runs the headless WASM test script.
