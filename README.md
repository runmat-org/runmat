# 🚀 RunMat: Modern Free MATLAB Compatible Runtime
### A GPU-accelerated, open-source MATLAB runtime with automatic fusion, by the creators of [Dystr](https://dystr.com)

[![Build Status](https://img.shields.io/github/actions/workflow/status/runmat-org/runmat/ci.yml?branch=main)](https://github.com/runmat-org/runmat/actions)
[![License](https://img.shields.io/badge/license-MIT%20with%20Attribution-blue.svg)](LICENSE.md)
[![Crates.io](https://img.shields.io/crates/v/runmat.svg)](https://crates.io/crates/runmat)
[![Downloads](https://img.shields.io/crates/d/runmat.svg)](https://crates.io/crates/runmat)

**[🌐 Website](https://runmat.org) • [📖 Documentation](https://runmat.org/docs)**

---

## What is RunMat?

RunMat is a **modern, GPU-accelerated runtime** for MATLAB® and GNU Octave code that eliminates license fees, vendor lock-in, and performance bottlenecks. Built from the ground up in Rust with a **V8-inspired architecture**, it delivers:

- 🚀 **GPU-accelerated execution** with automatic fusion and intelligent CPU/GPU routing (10x-1000x speedups)
- ⚡ **Instant startup** (5ms vs 900ms+ in Octave) via advanced snapshotting
- 🔥 **Fusion engine** that collapses operation chains into single GPU dispatches
- 📚 **Extensive built-in library** (200+ functions) covering arrays, linear algebra, FFT/signal processing, statistics, strings, and I/O
- 📦 **Package system** supporting both native Rust packages and MATLAB source packages
- 🎨 **GPU-accelerated plotting** that's beautiful and responsive
- 📊 **Native Jupyter support** with rich interactive widgets
- 🛡️ **Memory safety** and **zero crashes** guaranteed by Rust
- 💰 **$0 licensing costs** - completely free and open source

## 📊 Performance Benchmarks

RunMat delivers exceptional performance through its tiered execution engine and GPU acceleration. Our comprehensive benchmark suite compares RunMat against NumPy, PyTorch, Julia, and Octave across real-world workloads.

### CPU Performance (JIT vs Octave)

<table>
<tr>
<th>Benchmark</th>
<th>GNU Octave 9.4</th>
<th>RunMat (JIT)</th>
<th>Speedup</th>
</tr>
<tr>
<td>Startup time (cold)</td>
<td>915ms</td>
<td>5ms</td>
<td><strong>183x faster</strong></td>
</tr>
<tr>
<td>Matrix operations</td>
<td>822ms</td>
<td>5ms</td>
<td><strong>164x faster</strong></td>
</tr>
<tr>
<td>Mathematical functions</td>
<td>868ms</td>
<td>5ms</td>
<td><strong>163x faster</strong></td>
</tr>
<tr>
<td>Control flow (loops)</td>
<td>876ms</td>
<td>6ms</td>
<td><strong>155x faster</strong></td>
</tr>
</table>

### GPU Acceleration Performance

RunMat's fusion engine and auto-offload system deliver significant speedups on GPU-accelerated workloads:

- **10× or greater speedups** for image processing, computer vision, and DSP workloads with long elementwise chains
- **100× or greater speedups** for quant finance, simulation, and batched linear algebra workloads
- **1000× or greater speedups** for massively parallel telemetry and independent channel processing

### Benchmark Suite

Our reproducible benchmark suite includes:

- **4K Image Processing** - Per-pixel normalization, radiometric correction, and gamma correction
- **PCA** - Principal component analysis on large datasets
- **Batched NLMS** - Adaptive filtering across multiple channels
- **Monte Carlo Analysis** - Risk path simulation and statistical analysis
- **Batched IIR Smoothing** - Signal smoothing across large batches

*Benchmarks run on Apple M2 Max with BLAS/LAPACK optimization and GPU acceleration. See [benchmarks/](benchmarks/) for reproducible test scripts, detailed results, and comparisons against NumPy, PyTorch, and Julia.*

---

### Why Engineers and Scientists Love RunMat

<table>
<tr>
<td width="50%">

**🔬 For Researchers & Academics**
- Run existing MATLAB scripts without expensive licenses
- Reproducible science with open-source tools
- Fast iteration cycles for algorithm development
- Publication-quality plots that render beautifully

</td>
<td width="50%">

**⚙️ For Engineers & Industry**
- Embed scientific computing in production systems
- No vendor lock-in or licensing audits
- Deploy to cloud/containers without restrictions
- Modern CI/CD integration out of the box

</td>
</tr>
</table>

## 🎯 Quick Start

### Installation

```bash
# Quick install (Linux/macOS)
curl -fsSL https://runmat.org/install.sh | sh

# Quick install (Windows PowerShell)
iwr https://runmat.org/install.ps1 | iex

# Or install from crates.io
cargo install runmat --features gui

# Or build from source
git clone https://github.com/runmat-org/runmat.git
cd runmat && cargo build --release --features gui
```

#### Linux prerequisite

For BLAS/LAPACK acceleration on Linux, install the system OpenBLAS package before building:

```bash
sudo apt-get update && sudo apt-get install -y libopenblas-dev
```

### Run Your First Script

```bash
# Start the interactive REPL
runmat

# Or run an existing .m file
runmat script.m

# Or pipe a script into RunMat
echo "a = 10; b = 20; c = a + b" | runmat

# Check GPU acceleration status
runmat accel-info

# Benchmark a script
runmat benchmark script.m --iterations 5 --jit

# View system information
runmat info
```

### Jupyter Integration

```bash
# Register RunMat as a Jupyter kernel
runmat --install-kernel

# Launch JupyterLab with RunMat support
jupyter lab
```

### GPU-Accelerated Example

```matlab
% RunMat automatically uses GPU when beneficial
x = rand(10000, 1, 'single');
y = sin(x) .* x + 0.5;  % Automatically fused and GPU-accelerated
mean(y)  % Result computed on GPU
```

## 🌟 See It In Action

### MATLAB Compatibility
```matlab
% Your existing MATLAB code just works
A = [1 2 3; 4 5 6; 7 8 9];
B = A' * A;
eigenvals = eig(B);
plot(eigenvals);
```

### GPU-Accelerated Fusion
```matlab
% RunMat automatically fuses this chain into a single GPU kernel
% No kernel code, no rewrites—just MATLAB syntax
x = rand(1024, 1, 'single');
y = sin(x) .* x + 0.5;        % Fused: sin, multiply, add
m = mean(y, 'all');            % Reduction stays on GPU
fprintf('m=%.6f\n', double(m)); % Single download at sink
```

### Performance That Scales
```matlab
% Matrix operations that fly - 150x+ faster than Octave
n = 1000;
A = randn(n, n);
B = randn(n, n);
tic; C = A * B; toc  % Executes in ~5ms vs 800ms+ in Octave

% GPU acceleration for large workloads
imgs = rand(16, 2160, 3840, 'single');  % 4K image batch
mu = mean(imgs, [2 3]);                 % GPU-accelerated
sigma = sqrt(mean((imgs - mu).^2, [2 3]));
out = ((imgs - mu) ./ sigma) * 1.0123 - 0.02;  % Fused GPU kernel
```

### Beautiful, Interactive Plotting
```matlab
% Create a stunning 3D surface plot with GPU acceleration
[X, Y] = meshgrid(-2:0.1:2, -2:0.1:2);
Z = X .* exp(-X.^2 - Y.^2);
surf(X, Y, Z);  % GPU-accelerated rendering
```

## 🏗️ Architecture: V8-Inspired Performance

RunMat's **tiered execution engine** delivers both fast startup and blazing runtime performance. The architecture combines CPU JIT compilation with GPU acceleration through an intelligent fusion engine.

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **🎯 runmat-ignition** | Baseline interpreter for instant startup | HIR-to-bytecode compiler + stack-based interpreter |
| **⚡ runmat-turbine** | Optimizing JIT compiler for hot code | Cranelift backend |
| **🧠 runmat-gc** | High-performance memory management | Generational GC with pointer compression |
| **🚀 runmat-accelerate** | GPU acceleration subsystem | Fusion engine + auto-offload planner + wgpu provider |
| **🔥 Fusion Engine** | Collapses operation chains into single GPU dispatches | WGSL kernel generation + pipeline caching |
| **🎨 runmat-plot** | Interactive plotting engine | GPU-accelerated via wgpu |
| **📦 runmat-snapshot** | Fast startup system | Binary blob serialization |
| **🔧 runmat-runtime** | 200+ builtin functions | BLAS/LAPACK integration + GPU-accelerated operations |

### GPU Acceleration Architecture

RunMat Accelerate provides cross-platform GPU support through a portable backend:

- **Cross-platform GPU support**: Metal (macOS), DirectX 12 (Windows), Vulkan (Linux) via wgpu
- **Fusion engine**: Automatically detects and fuses elementwise operation chains and reductions
- **Auto-offload planner**: Intelligently routes operations between CPU and GPU based on workload characteristics
- **Residency management**: Keeps tensors on GPU to minimize host↔device transfers
- **Pipeline caching**: Compiles WGSL shaders once and reuses them for subsequent runs

## 🚀 GPU Acceleration: Fusion & Auto-Offload

RunMat automatically accelerates your MATLAB code on GPUs without requiring kernel code or rewrites. The system works through four stages:

### 1. Capture the Math
RunMat builds an "acceleration graph" that captures the intent of your operations—shapes, operation categories, dependencies, and constants. This graph provides a complete view of what your script computes.

### 2. Decide What Should Run on GPU
The fusion engine detects long chains of elementwise operations and linked reductions, planning to execute them as combined GPU programs. The auto-offload planner estimates break-even points and routes work intelligently:
- **Fusion detection**: Combines multiple operations into single GPU dispatches
- **Auto-offload heuristics**: Considers element counts, reduction sizes, and matrix multiply saturation
- **Residency awareness**: Keeps tensors on device once they're worth it

### 3. Generate GPU Kernels
RunMat generates portable WGSL (WebGPU Shading Language) kernels that work across platforms:
- **Metal** on macOS
- **DirectX 12** on Windows  
- **Vulkan** on Linux

Kernels are compiled once and cached for subsequent runs, eliminating recompilation overhead.

### 4. Execute Efficiently
The runtime minimizes host↔device transfers by:
- Uploading tensors once and keeping them resident
- Executing fused kernels directly on GPU memory
- Only gathering results when needed (e.g., for `fprintf` or display)

### Example: Automatic GPU Fusion

```matlab
% This code automatically fuses into a single GPU kernel
x = rand(1024, 1, 'single');
y = sin(x) .* x + 0.5;  % Fused: sin, multiply, add
m = mean(y, 'all');      % Reduction stays on GPU
fprintf('m=%.6f\n', double(m));  % Single download at sink
```

RunMat detects the elementwise chain (`sin`, `.*`, `+`), fuses them into one GPU dispatch, keeps `y` resident on GPU, and only downloads `m` when needed for output.

For more details, see [Introduction to RunMat GPU](docs/INTRODUCTION_TO_RUNMAT_GPU.md) and [How RunMat Fusion Works](docs/HOW_RUNMAT_FUSION_WORKS.md).

## 🎨 Modern Developer Experience

### Rich REPL with Intelligent Features
```bash
runmat> .info
🦀 RunMat v0.1.0 - High-Performance MATLAB Runtime
⚡ JIT: Cranelift (optimization: speed)
🧠 GC: Generational (heap: 45MB, collections: 12)
🚀 GPU: wgpu provider (Metal/DX12/Vulkan)
🎨 Plotting: GPU-accelerated (wgpu)
📊 Functions loaded: 200+ builtins + 0 user-defined

runmat> .stats
Execution Statistics:
  Total: 2, JIT: 0, Interpreter: 2
  Average time: 0.12ms

runmat> accel-info
GPU Acceleration Provider: wgpu
Device: Apple M2 Max
Backend: Metal
Fusion pipeline cache: 45 hits, 2 misses
```

### First-Class Jupyter Support
- Rich output formatting with LaTeX math rendering
- Interactive widgets for parameter exploration  
- Seamless plotting integration
- Full debugging support with breakpoints

### Extensible Architecture
```rust
// Adding a new builtin function is trivial
#[runtime_builtin("myfunction")]
fn my_custom_function(x: f64, y: f64) -> f64 {
    x.powf(y) + x.sin()
}
```

### Advanced CLI Features

RunMat includes a comprehensive CLI with powerful features:

```bash
# Check GPU acceleration status
runmat accel-info

# Benchmark a script
runmat benchmark my_script.m --iterations 5 --jit

# Create a snapshot for faster startup
runmat snapshot create -o stdlib.snapshot

# GC statistics and control
runmat gc stats
runmat gc major

# System information
runmat info
```

See [CLI Documentation](docs/CLI.md) for the complete command reference.

## 📦 Package System

RunMat's package system enables both systems programmers and MATLAB users to extend the runtime. The core stays lean while packages provide domain-specific functionality.

### Native Packages (Rust)

High-performance built-ins implemented in Rust:

```rust
#[runtime_builtin(
    name = "norm2",
    category = "math/linalg",
    summary = "Euclidean norm of a vector.",
    examples = "n = norm2([3,4])  % 5"
)]
fn norm2_builtin(a: Value) -> Result<Value, String> {
    let t: Tensor = (&a).try_into()?;
    let s = t.data.iter().map(|x| x * x).sum::<f64>().sqrt();
    Ok(Value::Num(s))
}
```

Native packages get type-safe conversions, deterministic error IDs, and zero-cost documentation generation.

### Source Packages (MATLAB)

MATLAB source packages compile to RunMat bytecode:

```matlab
% +mypackage/norm2.m
function n = norm2(v)
    n = sqrt(sum(v .^ 2));
end
```

Both package types appear identically to users—functions show up in the namespace, reference docs, and tooling (help, search, doc indexing).

### Package Management

```bash
# Declare dependencies in .runmat
[packages]
linalg-plus = { source = "registry", version = "^1.2" }
viz-tools = { source = "git", url = "https://github.com/acme/viz-tools" }

# Install packages
runmat pkg install

# Publish your package
runmat pkg publish
```

*Note: Package manager CLI is currently in development. See [Package Manager Documentation](docs/PACKAGE_MANAGER.md) for design details.*

## 💡 Design Philosophy

RunMat follows a **minimal core, fast runtime, open extension model** philosophy:

### Core Principles

- **Full language support**: The core implements the complete MATLAB grammar and semantics, not a subset
- **Extensive built-ins**: The standard library aims for complete base MATLAB built-in coverage (200+ functions)
- **Tiered execution**: Ignition interpreter for fast startup, Turbine JIT for hot code
- **GPU-first math**: Fusion engine automatically turns MATLAB code into fast GPU workloads
- **Small, portable runtime**: Single static binary, fast startup, modern CLI, Jupyter kernel support
- **Toolboxes as packages**: Signal processing, statistics, image processing, and other domains live as packages

### What RunMat Is

- A modern, high-performance runtime for MATLAB code
- A minimal core with a thriving package ecosystem
- GPU-accelerated by default with intelligent CPU/GPU routing
- Open source and free forever

### What RunMat Is Not

- A reimplementation of MATLAB-in-full (toolboxes are packages)
- A compatibility layer (we implement semantics, not folklore)
- An IDE (use any editor: Cursor, VSCode, IntelliJ, etc.)

RunMat keeps the core small and uncompromisingly high-quality; everything else is a package. This enables:
- Fast iteration without destabilizing the runtime
- Domain experts shipping features without forking
- A smaller trusted compute base, easier auditing
- Community-driven package ecosystem

See [Design Philosophy](docs/DESIGN_PHILOSOPHY.md) for the complete design rationale.

## 🌍 Who Uses RunMat?

<div align="center">
<table>
<tr>
<td align="center" width="25%">
<strong>🎓 Universities</strong><br/>
Teaching numerical methods<br/>without license fees
</td>
<td align="center" width="25%">
<strong>🔬 Research Labs</strong><br/>
Reproducible science with<br/>open-source tools
</td>
<td align="center" width="25%">
<strong>🏭 Engineering Teams</strong><br/>
Embedded scientific computing<br/>in production systems
</td>
<td align="center" width="25%">
<strong>🚀 Startups</strong><br/>
Rapid prototyping without<br/>expensive toolchain costs
</td>
</tr>
</table>
</div>

## 🤝 Join the Revolution

RunMat is more than just software—it's a movement toward **open, fast, and accessible scientific computing**. We're building the future of numerical programming, and we need your help.

### 🛠️ How to Contribute

<table>
<tr>
<td width="33%">

**🚀 For Rust Developers**
- Implement new builtin functions
- Optimize the JIT compiler  
- Enhance the garbage collector
- Build developer tooling

[**Contribute Code →**](https://github.com/runmat-org/runmat/discussions)

</td>
<td width="33%">

**🔬 For Domain Experts**
- Add mathematical functions
- Improve MATLAB compatibility
- Write comprehensive tests
- Create benchmarks

[**Join Discussions →**](https://github.com/runmat-org/runmat/discussions)

</td>
<td width="33%">

**📚 For Everyone Else**
- Report bugs and feature requests
- Improve documentation
- Create tutorials and examples
- Spread the word

[**Get Started →**](https://github.com/runmat-org/runmat/issues/labels/good-first-issue)

</td>
</tr>
</table>

### 💬 Connect With Us

- **GitHub Discussions**: [Share ideas and get help](https://github.com/runmat-org/runmat/discussions)  
- **Twitter**: [@dystr_ai](https://x.com/dystr_ai) for updates and announcements
- **Newsletter**: [Subscribe](https://runmat.org/newsletter) for monthly updates

## 📜 License

RunMat is licensed under the **MIT License with Attribution Requirements**. This means:

✅ **Free for everyone** - individuals, academics, most companies  
✅ **Open source forever** - no vendor lock-in or license fees  
✅ **Commercial use allowed** - embed in your products freely  
⚠️ **Attribution required** - credit "RunMat by Dystr" in public distributions  
⚠️ **Special provisions** - large scientific software companies must keep modifications open source  

See [LICENSE.md](LICENSE.md) for complete terms or visit [runmat.org/license](https://runmat.org/license) for FAQs.

---

**Built with ❤️ by [Dystr Inc.](https://dystr.com) and the RunMat community**

⭐ **Star us on GitHub** if RunMat helps your work!

[**🚀 Get Started**](https://runmat.org/docs/getting-started) • [**🐦 Follow @dystr**](https://x.com/dystr_ai)

---

*MATLAB® is a registered trademark of The MathWorks, Inc. RunMat is not affiliated with, endorsed by, or sponsored by The MathWorks, Inc.*