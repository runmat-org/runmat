# Build System

RunMat builds from a single Cargo workspace. The workspace keeps the language pipeline, execution engines, runtime builtins, acceleration layer, plotting, CLI, LSP, snapshotting, filesystem, and WASM bindings in one versioned graph.

The TypeScript package in `bindings/ts` contains bindings for the WASM runtime to run in the browser, along with the LSP bundle and generated builtin metadata used by JavaScript consumers.

## Workspace Layout

The root workspace uses Cargo resolver v2. Workspace dependency versions live in the root `Cargo.toml`; internal crates are pinned to the same RunMat version.

| Area | Crates |
| --- | --- |
| Language pipeline | `runmat-lexer`, `runmat-parser`, `runmat-hir`, `runmat-mir`, `runmat-static-analysis` |
| Execution | `runmat-vm`, `runmat-native-codegen`, `runmat-native-executor`, `runmat-jit`, `runmat-aot-runtime`, `runmat-aot`, `runmat-core` |
| Runtime | `runmat-runtime`, `runmat-builtins`, `runmat-filesystem`, `runmat-time`, `runmat-config` |
| Performance systems | `runmat-accelerate`, `runmat-accelerate-api`, `runmat-gc`, `runmat-gc-api`, `runmat-plot` |
| Host surfaces | `runmat` CLI, `runmat-lsp`, `runmat-wasm`, `runmat-server-client`, `runmat-telemetry`, `runmat-logging` |

The `runmat` binary lives in `crates/runmat-cli`. It depends on the compiler, VM, runtime, plotting, acceleration, filesystem, config, telemetry, and session crates, so a default CLI build is the broadest native build target.

## Feature Flags

The CLI default feature set enables the normal local developer experience:

| Feature | Effect |
| --- | --- |
| `gui` | Enables native plotting GUI support through `runmat-plot`. |
| `blas-lapack` | Enables high-performance BLAS/LAPACK operations in `runmat-runtime`. |
| `wgpu` | Enables the WGPU acceleration path. |
| `jit` | Enables Core's host-native adaptive JIT selection. Machine-code lowering lives in `runmat-native-codegen`, the shared invocation host lives in `runmat-native-executor`, and adaptive compilation and publication live in `runmat-jit`. |

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
| HDF5 | Native runtime builds require HDF5 headers and libraries for `h5read`, `h5write`, and related builtins. Linux uses the distro package, macOS uses Homebrew, and Windows uses vcpkg. |

`crates/runmat-runtime/build.rs` participates in BLAS/LAPACK discovery when `blas-lapack` is enabled. It honors the standard library hints used by local packages and vcpkg: `VCPKG_ROOT`, `VCPKGRS_TRIPLET`, `VCPKG_DEFAULT_TRIPLET`, `OPENBLAS_DIR`, `BLAS_LIB_DIR`, `BLAS_LIBS`, `LAPACK_LIB_DIR`, and `LAPACK_LIBS`.

On Ubuntu-like systems, the important local packages are:

```bash
sudo apt-get install -y libhdf5-dev libopenblas-dev liblapack-dev libzmq3-dev pkg-config libssl-dev
```

On macOS, install the corresponding native dependencies with:

```bash
brew install hdf5 zeromq
```

On Windows, install `hdf5`, `openblas`, `lapack-reference` (or the CI-specific LAPACK package), `zeromq`, and `opencascade` through the same vcpkg triplet. Set `HDF5_DIR` to the absolute vcpkg installed-prefix directory so `hdf5-metno-sys` finds the matching headers, import library, and runtime DLL.

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

### Native standalone runtime

Build the CLI and its matching compile-time runtime pack with the two-phase helper. For a CPU-only development build on macOS or Linux:

```bash
scripts/build-runmat-with-aot-runtime.sh --no-default-features --features jit
```

On Windows:

```powershell
scripts/build-runmat-with-aot-runtime.ps1 --no-default-features --features jit
```

The helpers default to the release profile. Their output directory contains:

| Output | Purpose |
| --- | --- |
| `runmat` or `runmat.exe` | CLI with the trusted pack manifest embedded |
| `runmat-aot-runtime.zip` | Separate compile-time runtime, native manifest, licenses and notices |
| `runmat-aot-runtime.json` | Distribution metadata that pins the pack digest and size |
| `THIRD_PARTY_NOTICES.txt` | Canonical Runtime third-party notices |

Ordinary `cargo build` provides script execution without a compile-time distribution manifest. Use the helper when developing `runmat compile`. For the compile-time runtime, the CLI embeds only the small distribution manifest. The temporary packaging directory is removed after the build; Cargo's static library and build cache remain in the target directory.

#### Verify the built pair

Pass the binary and pack to the smoke check:

```bash
scripts/verify-runmat-with-aot-runtime.sh target/release/runmat target/release/runmat-aot-runtime.zip
```

The PowerShell verifier accepts the binary and optional pack paths in the same order. These checks provision the explicit pack offline, compile a small program and verify its output. Replace `release` with the selected profile output directory, and include the target triple directory when using `--target`.

For the broader installation and cache checks, run:

```bash
node scripts/distribution/accept-aot-runtime-pack.mjs \
  target/release/runmat target/release/runmat-aot-runtime.zip \
  /path/to/new-aot-acceptance.json
```

This check uses synthetic source and a temporary `RUNMAT_AOT_CACHE_DIR`. It exercises ordinary execution without a pack, offline and unattended misses, concurrent installation, bounded retention, offline compilation, standalone output and corrupt-pack refusal. The receipt binds the binary and pack digests. The temporary cache is removed after the check.

Published download retrieval and interactive terminal consent need separate release checks; this offline cohort does not exercise them.

#### Select execution backends

The CPU-only example keeps the stock console and callable factories. To build a target WGPU backend, use `--no-default-features --features jit,wgpu`. The CUDA combinations are `jit,cuda` and `jit,wgpu,cuda`. Both helpers accept these flags.

The helper forwards accelerator choices to the runtime archive and preserves the full requested feature set for the CLI. Backend support is read from evidence in the target archive. The stock factories provide console and callable services; they provide no graphical interaction or additional service grants. Custom embedding archives select their qualified factory symbols by building the runtime crate directly.

These explicit development feature choices are separate from the production CLI feature set. The compile-time pack split leaves FEA and geometry in ordinary execution.

#### What the helper builds

The first phase builds `runmat-aot-runtime` as a static library using the shared Runtime, native executor and `Value` implementation. The linked process-image and JIT entrypoints use the same verified executor contract. The standalone library retains the HIR/MIR operation schema needed to decode Native IR; parser, lowering, static analysis, Core composition, native object emission, VM and adaptive JIT implementation remain outside it.

Cargo reports the target's ordered native link requirements. The packer validates and compresses that archive, then adds its native manifest, first-party licenses and canonical third-party notices. Distribution output uses paired `--pack-out` and `--pack-manifest-out` arguments with explicit `--license` and `--free-use` inputs. The helpers supply the checkout's canonical license files.

The second phase builds the CLI with the distribution manifest embedded. That manifest pins the complete pack digest and size, native ABI, target, runtime/catalog identities, capabilities and link tokens. `runmat aot install --from PATH` verifies an explicitly supplied pack; compilation loads a verified cached pack or asks to install the exact distribution pinned by the CLI.

#### Source and link contracts

Core supplies the immutable executable unit, source identities and binding names. Canonical MIR owns whole-program reachability, and the builtin catalog supplies link, placement, provider and extension contracts. AOT consumes those inputs; the CLI renders the resulting plan through `--explain-link` or `--link-plan-json PATH`.

`native-specialized` uses exact selected bindings, ordinary archive extraction and platform dead stripping when capabilities and complete reachability/catalog analysis permit it. Otherwise it force-loads the archive for inventory-backed builtin discovery. `closed-world` requires that complete proof and rejects unknown or non-canonical targets. Selected composition references collision-free catalog binding symbols and installs the selected set as the invocation-scoped runtime authority.

`scripts/check-closed-world-binary.sh` compares an executable with its JSON link plan and rejects extra builtin bindings or retained compiler, VM and JIT symbols.

Native objects carry a versioned architecture, operating system, pointer width, ABI and object-format contract. Incompatible hosts reject them, and target identity contributes to recipe and artifact IDs. Portable bytecode and executable units can run on native or browser hosts. Compiled execution bundles retain the materialized program and frozen package identities without reconstructing project source; dynamic source access uses the separate source-project closure.

#### Build settings

Use `--profile` or `RUNMAT_BUILD_PROFILE` to select a Cargo profile. An explicit `--target` overrides `RUNMAT_BUILD_TARGET`, which overrides `CARGO_BUILD_TARGET`. The runtime archive and CLI use the deployment target; the packer executes on the Rust compiler's build-host target. Both helpers honor `CARGO_TARGET_DIR`.

Darwin phases default Rust and native C/CMake compilation to deployment target `11.0`. Set `MACOSX_DEPLOYMENT_TARGET` or `CMAKE_OSX_DEPLOYMENT_TARGET` to choose another minimum; an unset variable inherits the supplied one, and conflicting explicit values are rejected. The deployment archive and CLI apply this policy when their target is Darwin. The packer applies it independently when its build host is Darwin. Other phases retain the original environment.

The release workflow selects `11.0` before macOS dependency builds and cache restoration. Its deployment-policy cache prefix excludes objects built under older policies for both macOS architectures. Execution support on older macOS versions requires its own qualification evidence.

Runtime, analysis-FEA and Builtins use eight release codegen units so LLVM can optimize them in parallel. Optimization levels and feature selection retain their configured values.

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

The WASM registry has an ordering constraint: proc macros write registry entries while `runmat-runtime` compiles for `wasm32-unknown-unknown`. Always use `scripts/regenerate-wasm-registry.sh`; it generates the production `plot-web` registry into a temporary file, marks it complete only after cargo succeeds, then atomically replaces `generated_wasm_registry.rs`. Normal WASM builds validate the source fingerprint, target/features, completion marker, and entry count, and fail if the registry is missing, partial, stale, or generated for another runtime configuration.

## Release Helpers

Prepare release versions on `dev` with `scripts/prepare-release.sh <version>`. The script updates workspace crates, `Cargo.lock`, and the TypeScript package, validates the resulting state, and leaves the changes uncommitted for normal review.

After that versioned commit reaches `main`, run `scripts/cut-release.sh <version>`. The cut script requires a clean checkout that exactly matches `origin/main`, verifies that every workspace crate and TypeScript package already has the requested version, runs a locked build check, and tags that reviewed commit. It does not modify or commit release files.

Native release artifacts are built by GitHub Actions for the supported release triples listed in [Supported Architectures](/docs/runtime/development/supported-architectures). WASM package publication is handled by the `wasm-bindings` workflow.
