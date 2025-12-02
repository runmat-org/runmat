# 🚀 RunMat: The fastest runtime for your math
### RunMat automatically **fuses operations and intelligently routes between CPU and GPU**. MATLAB syntax. No kernel code, no rewrites.

[![Build Status](https://img.shields.io/github/actions/workflow/status/runmat-org/runmat/ci.yml?branch=main)](https://github.com/runmat-org/runmat/actions)
[![License](https://img.shields.io/badge/license-MIT%20with%20Attribution-blue.svg)](LICENSE.md)
[![Crates.io](https://img.shields.io/crates/v/runmat.svg)](https://crates.io/crates/runmat)
[![Downloads](https://img.shields.io/crates/d/runmat.svg)](https://crates.io/crates/runmat)

**[🌐 Website](https://runmat.org) • [📖 Documentation](https://runmat.org/docs)**

---

### **Status: Pre-release (v0.2)**
RunMat is an early build. The core runtime and GPU engine already pass thousands of tests, but some plotting features are still missing or buggy. Expect a few rough edges. Feedback and bug reports help us decide what to fix next.

---

## What is RunMat?

With RunMat you write your math in clean, readable MATLAB-style syntax. RunMat automatically fuses your operations into optimized kernels and runs them on the best place — CPU or GPU. On GPU, it can often match or beat hand-tuned CUDA on many dense numerical workloads

It runs on whatever GPU you have — NVIDIA, AMD, Apple Silicon, Intel — through native APIs (Metal / DirectX 12 / Vulkan). No device management. No vendor lock-in. No rewrites.

Core ideas:

- **MATLAB syntax, not a new language**  
- **Fast on CPU and GPU**, with one runtime  
- **No device flags** — Fusion automatically chooses CPU vs GPU based on data size and transfer cost heuristics

## ✨ Features at a glance

- **MATLAB language**

  - Familiar `.m` files, arrays, control flow  
  - Many MATLAB / Octave scripts run with few or no changes  

- **Fusion: automatic CPU+GPU choice**

  - Builds an internal graph of array ops  
  - Fuses elementwise ops and reductions into bigger kernels  
  - Chooses CPU or GPU per kernel based on shape and transfer cost  
  - Keeps arrays on device when that is faster  

- **Modern CPU runtime**

  - Ignition interpreter for fast startup  
  - Turbine JIT (Cranelift) for hot paths  
  - Generational GC tuned for numeric code  
  - Memory-safe by design (Rust)

- **Cross-platform GPU backend**

  - Uses wgpu / WebGPU  
  - Supports **Metal (macOS), DirectX 12 (Windows), Vulkan (Linux)**  
  - Falls back to CPU when workloads are too small for GPU to win  

- **Plotting and tooling (pre-release)**

  - Simple 2D line and scatter plots work today  
  - Plots that use filled shapes or meshes (box plots, violin plots, surfaces, many 3D views) are **not wired up yet**  
  - 3D plots and better camera controls are **on the roadmap**  
  - VS Code / Cursor extensions are also **on the roadmap**  


- **Open source**

  - MIT License with attribution  
  - Small binary, CLI-first design 

--- 

## 📊 Performance highlights

These are large workloads where **Fusion chooses GPU**.  
Hardware: **Apple M2 Max**, **Metal**, each point is the mean of 3 runs.




### 4K Image Pipeline Perf Sweep (B = batch size)
| B | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|---|---:|---:|---:|---:|---:|
| 4  | 142.97 | 801.29 | 500.34 | 3.50× | 5.60× |
| 8  | 212.77 | 808.92 | 939.27 | 4.41× | 3.80× |
| 16 | 241.56 | 907.73 | 1783.47 | 7.38× | 3.76× |
| 32 | 389.25 | 1141.92 | 3605.95 | 9.26× | 2.93× |
| 64 | 683.54 | 1203.20 | 6958.28 | 10.18× | 1.76× |


![4K image pipeline speedup](https://web.runmatstatic.com/4k-image-processing_speedup-b.svg)

### Monte Carlo Perf Sweep 
| Paths (simulations) | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|--------------------:|-----------:|-------------:|-----------:|---------------:|-----------------:|
| 250k   | 108.58 |   824.42 |  4,065.87 | 37.44× | 7.59× |
| 500k   | 136.10 |   900.11 |  8,206.56 | 60.30× | 6.61× |
| 1M     | 188.00 |   894.32 | 16,092.49 | 85.60× | 4.76× |
| 2M     | 297.65 | 1,108.80 | 32,304.64 |108.53× | 3.73× |
| 5M     | 607.36 | 1,697.59 | 79,894.98 |131.55× | 2.80× |



![Monte Carlo speedup](https://web.runmatstatic.com/monte-carlo-analysis_speedup-b.svg)

### Elementwise Math Perf Sweep (points)
| points | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|---|---:|---:|---:|---:|---:|
| 1M   | 145.15 | 856.41  |   72.39 | 0.50× | 5.90× |
| 2M   | 149.75 | 901.05  |   79.49 | 0.53× | 6.02× |
| 5M   | 145.14 | 1111.16 |  119.45 | 0.82× | 7.66× |
| 10M  | 143.39 | 1377.43 |  154.38 | 1.08× | 9.61× |
| 100M | 144.81 | 16,404.22 | 1,073.09 | 7.41× | 113.28× |
| 200M | 156.94 | 16,558.98 | 2,114.66 | 13.47× | 105.51× |
| 500M | 137.58 | 17,882.11 | 5,026.94 | 36.54× | 129.97× |
| 1B | 144.40 | 20,841.42 | 11,931.93 | 82.63× | 144.34× |

![Elementwise math speedup](https://web.runmatstatic.com/elementwise-math_speedup-b.svg)

On smaller arrays Fusion keeps work on CPU so you still get low overhead and a fast JIT. 

*Benchmarks run on Apple M2 Max with BLAS/LAPACK optimization and GPU acceleration. See [benchmarks/](benchmarks/) for reproducible test scripts, detailed results, and comparisons against NumPy, PyTorch, and Julia.*


---



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

### Plotting
```matlab
% Simple 2D line plot (works in the pre-release)
x = linspace(0, 2*pi, 1000);
y = sin(x);

plot(x, y);
grid on;
title("Sine wave");
```

---

## 🧱 Architecture: CPU+GPU performance

RunMat uses a tiered CPU runtime plus a fusion engine that automatically picks CPU or GPU for each chunk of math.

### Key components

| Component              | Purpose                                  | Technology / Notes                                                  |
| ---------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| ⚙️ runmat-ignition   | Baseline interpreter for instant startup | HIR → bytecode compiler, stack-based interpreter                    |
| ⚡ runmat-turbine     | Optimizing JIT for hot code              | Cranelift backend, tuned for numeric workloads                      |
| 🧠 runmat-gc         | High-performance memory management       | Generational GC with pointer compression                            |
| 🚀 runmat-accelerate | GPU acceleration subsystem               | Fusion engine + auto-offload planner + `wgpu` backend               |
| 🔥 Fusion engine       | Collapses op chains, chooses CPU vs GPU  | Builds op graph, fuses ops, estimates cost, keeps tensors on device |
| 🎨 runmat-plot       | Plotting layer (pre-release)                          | 2D line/scatter plots work today; 3D, filled shapes, and full GPU plotting are on the roadmap |
| 📸 runmat-snapshot   | Fast startup snapshots                   | Binary blob serialization / restore                                 |
| 🧰 runmat-runtime    | Core runtime + 200+ builtin functions    | BLAS/LAPACK integration and other CPU/GPU-accelerated operations    |


### Why this matters

- **Tiered CPU execution** gives quick startup and strong single-machine performance.  
- **Fusion engine** removes most manual device management and kernel tuning.  
- **GPU backend** runs on NVIDIA, AMD, Apple Silicon, and Intel through Metal / DirectX 12 / Vulkan, with no vendor lock-in.



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

The RunMat package manager is still in active design—no CLI commands ship in the current toolchain yet. The [Package Manager Documentation](docs/PACKAGE_MANAGER.md) captures the proposed workflow (dependency manifests, registry + git sources, publishing flow) and will be updated once the implementation begins.

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

RunMat is built for array-heavy math in many domains.

Examples: 

<div align="center">
<table>
<tr>
<td align="center" width="25%">
<strong>Imaging / geospatial</strong><br/>
4K+ tiles, normalization, radiometric correction, QC metrics
</td>
<td align="center" width="25%">
<strong>Quant / simulation</strong><br/>
Monte Carlo risk, scenario analysis, covariance, factor models
</td>
<td align="center" width="25%">
<strong>Signal processing / control</strong><br/>
Filters, NLMS, large time-series jobs
</td>
<td align="center" width="25%">
<strong>Researchers and students</strong><br/>
MATLAB background, need faster runs on laptops or clusters
</td>
</tr>
</table>
</div>

If you write math in MATLAB and hit performance walls on CPU, RunMat is built for you.

## 🤝 Join the mission

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
- **Twitter**: [@dystreng](https://x.com/dystreng) for updates and announcements

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

⭐ **Star us on GitHub** if RunMat is useful to you.

[**🚀 Get Started**](https://runmat.org/docs/getting-started) • [**🐦 Follow @dystr**](https://x.com/dystrEng)

---

*MATLAB® is a registered trademark of The MathWorks, Inc. RunMat is not affiliated with, endorsed by, or sponsored by The MathWorks, Inc.*
