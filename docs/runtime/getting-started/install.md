---
title: "Installation"
category: "Getting Started"
section: "1.1"
last_updated: "May 28, 2026"
---

# Installation

The quickest to get started with RunMat is to use RunMat Desktop: [Download RunMat Desktop](/download/latest)

If you prefer to use the command line, you can install RunMat via package managers or build from source.

## Command Line Installation

RunMat can be installed via pre-compiled binaries, package managers, or built from source.

### Automated Installers

- Linux/macOS: `curl -fsSL https://runmat.com/install.sh | sh`
- Windows (PowerShell): `iwr https://runmat.com/install.ps1 | iex`

### Package Managers

- Homebrew (macOS/Linux): `brew install runmat-org/tap/runmat`
- Cargo (Rust): `cargo install runmat --features gui`

### Build from Source

Building from source requires the Rust toolchain. The `gui` feature enables the `wgpu`-based plotting and windowing system.

```bash
git clone https://github.com/runmat-org/runmat.gitcd runmat && cargo build --release --features gui
```

# Next Steps

Read the [Configuration Reference](/docs/runtime/getting-started/config) to learn how to configure RunMat, or the [Hello World](/docs/runtime/getting-started/hello-world) example to get started with RunMat.