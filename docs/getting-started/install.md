# Installation

The quickest way to get started with RunMat is to use RunMat Desktop: [Download RunMat Desktop](/download/latest)

If you prefer to use the command line, you can install RunMat via package managers or build from source.

## Command Line Installation

RunMat can be installed via pre-compiled binaries, package managers, or built from source.

### Linux runtime libraries

Pre-compiled Linux CLI binaries use system BLAS/LAPACK, HDF5, OpenSSL and C++ runtime libraries. Install packages matching the archive's shared-library ABIs before running the CLI or its installer. For Debian 12:

```sh
sudo apt-get update
sudo apt-get install -y curl libopenblas0 libhdf5-103-1 libssl3 libstdc++6
```

Other distributions use their corresponding ABI-compatible packages. Building executables with `runmat compile` also requires a system linker and native development libraries; see [Native Compilation](/docs/runtime/compiler/native-compilation).

### Automated Installers

- Linux/macOS: `curl -fsSL https://runmat.com/install.sh | sh`
- Windows (PowerShell): `iwr https://runmat.com/install.ps1 | iex`

### Package Managers

- Homebrew (macOS/Linux): `brew install runmat-org/tap/runmat`
- Cargo (Rust): `cargo install runmat --features gui`

### Build from Source

- Clone repository: `git clone https://github.com/runmat-org/runmat.git`
- Build with GUI features: `cargo build --release --features gui`

Note: Building from source requires the Rust toolchain. The `gui` feature enables the `wgpu`-based plotting and windowing system.

### Next Steps

Read the [Configuration Reference](/docs/runtime/getting-started/config) to learn how to configure RunMat, or the [Hello World](/docs/runtime/getting-started/hello-world) example to get started with RunMat.
