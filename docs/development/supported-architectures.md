---
title: "Supported Architectures"
category: "Development"
section: "14.2"
last_updated: "May 29, 2026"
---

# Supported Architectures

RunMat has two major build families: native binaries for the CLI/runtime and WebAssembly artifacts for browser and TypeScript hosts. Native releases are built for Windows, macOS, and Linux; WASM builds target `wasm32-unknown-unknown`.

## Packaged Targets

The release pipeline packages native CLI/runtime binaries for these targets:

| Release artifact | Rust target |
| --- | --- |
| Windows x86_64 | `x86_64-pc-windows-msvc` |
| macOS Intel | `x86_64-apple-darwin` |
| macOS Apple Silicon | `aarch64-apple-darwin` |
| Linux x86_64 | `x86_64-unknown-linux-gnu` |

CI also cross-builds the same native targets before release. The main test matrix runs on Linux x86_64 and macOS; the release workflow additionally validates Windows before packaging.

## CPU Architecture Support

RunMat is written in Rust and much of the parser, compiler, VM, runtime, builtin library, and session engine is portable across CPU architectures that support Rust `std`. Packaged and CI-covered support is currently narrower than theoretical source-build support.

| Target family | Support tier | Package | CI coverage | JIT | GPU path | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `x86_64-pc-windows-msvc` | Packaged native | Yes | Cross-build + release validation | Supported | WGPU / DX12 | Full CLI/runtime target. |
| `x86_64-apple-darwin` | Packaged native | Yes | macOS tests + cross-build | Supported | WGPU / Metal | Full CLI/runtime target. |
| `aarch64-apple-darwin` | Packaged native | Yes | macOS tests + cross-build | Supported | WGPU / Metal | Apple Silicon; best-supported ARM target today. |
| `x86_64-unknown-linux-gnu` | Packaged native | Yes | Linux tests + cross-build | Supported | WGPU / Vulkan or host backend | Full CLI/runtime target when native dependencies are present. |
| `aarch64-pc-windows-msvc` | Configured native | No | Build configuration only | AArch64 backend, unvalidated | WGPU / DX12 if the native stack works | `.cargo/config.toml` includes linker stack sizing, but this target is not packaged. |
| `aarch64-unknown-linux-gnu` | Experimental source build | No | No routine coverage | AArch64 backend, unvalidated | Experimental WGPU / Linux graphics stack | Source-build target; treat as experimental until CI and packaging cover it. |
| `armv7-*`, `arm-unknown-linux-*` | Untested source build | No | No routine coverage | Not supported | Platform-specific, untested | Interpreter-only builds may be possible, but native dependencies and GPU support vary by board. |
| `wasm32-unknown-unknown` | Packaged WASM | Yes | WASM build + browser tests | Not supported | Browser WebGPU | Browser and TypeScript package target, not a native CLI target. |

The JIT tier is more constrained than the interpreter. Turbine currently reports JIT support for `x86_64` and `aarch64`; other architectures should be expected to run interpreter-only even if the rest of the runtime builds.

## Embedded And Edge Devices

RunMat is not a `no_std` runtime and is not designed for bare-metal microcontrollers. It assumes an operating system, heap allocation, filesystem or filesystem-provider access, atomics, async/runtime support, and in several configurations native libraries such as BLAS/LAPACK, TLS, windowing, or GPU drivers.

Practical embedded support depends on the class of device:

| Device class | RunMat fit |
| --- | --- |
| Cortex-M / bare-metal MCUs | Not supported. These targets lack the OS, allocator, filesystem, and process model RunMat expects. |
| Cortex-R / RTOS-style systems | Not supported unless they provide a Rust `std` environment close to Linux or another supported OS. |
| Cortex-A Linux SBCs and edge boxes | Plausible as source builds, especially ARM64 Linux, but not packaged today. Start with CPU/interpreter validation before enabling JIT, BLAS/LAPACK, plotting, or WGPU. |
| Android devices | Not currently packaged for the RunMat CLI. Browser/WASM use may work through WebGPU where the browser and device support it; native Android packaging is not part of the current build matrix. |

For ARM Linux boards, the safest bring-up order is:

```bash
cargo check -p runmat-parser
cargo check -p runmat-core --no-default-features
cargo check -p runmat-runtime --no-default-features
```

Then add features and host layers back deliberately: JIT first on AArch64, BLAS/LAPACK after native math libraries are installed, and WGPU/GUI only after the graphics stack is known to work. The full CLI is a release-oriented package and should be brought up after the core crates compile.

## Toolchain

Rust is pinned to `1.90.0` with `rustfmt` and `clippy`. CI validates the exact toolchain on self-hosted Linux and Windows runners, and installs the same channel on macOS.

Windows MSVC builds use a larger linker stack through `.cargo/config.toml`. That setting covers `x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc`; the ARM64 Windows target is configured but not currently packaged.

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

### ARM And Mali GPUs

Mali GPUs are not targeted directly by RunMat, but they are reachable as an acceleration provider through the platform GPU API exposed to WGPU, usually Vulkan on Linux/Android or browser WebGPU in a supported browser.

That means Mali support depends on the full driver stack:

- The device must expose the backend WGPU can use on that platform.
- The adapter must support the WebGPU/WGPU limits and features RunMat kernels require.
- Shader compilation, buffer limits, storage-buffer support, and precision support are provider-reported at runtime.
- CPU fallback remains the expected behavior when GPU initialization fails or when a workload is too small for offload to win.

On ARM Linux systems with Mali GPUs, treat GPU acceleration as experimental until validated on the exact board, kernel, Mesa/vendor driver, windowing/headless setup, and workload. A CPU-only RunMat build can still be useful on these devices even when WGPU is unavailable.

## CI Coverage

The main Unix CI jobs run:

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo check --all-targets --all-features
RUST_TEST_THREADS=1 cargo test --all-targets --all-features
```

The cross-build job compiles all packaged native targets in release mode. The WASM workflow builds `runmat-wasm` for `wasm32-unknown-unknown`, then runs the headless WASM test script.
