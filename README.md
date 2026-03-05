<p align="center">
  <strong><h1 align="center">RunMat</h1></strong>
</p>

<p align="center">
  <strong>Open-source runtime for math. MATLAB syntax. CPU + GPU. No license fees.</strong>
</p>

<p align="center">
  RunMat automatically fuses operations and intelligently routes between CPU and GPU.<br/>
  Works across Windows, macOS, Linux, and WebAssembly, across NVIDIA, AMD, Apple Silicon, and Intel GPUs.<br/>
  No kernel code. No rewrites. No device flags. No vendor lock-in.
</p>

<p align="center">
  <a href="https://github.com/runmat-org/runmat/actions"><img src="https://img.shields.io/github/actions/workflow/status/runmat-org/runmat/ci.yml?branch=main" alt="Build Status"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT%20with%20Attribution-blue.svg" alt="License"></a>
  <a href="https://crates.io/crates/runmat"><img src="https://img.shields.io/crates/v/runmat.svg" alt="Crates.io"></a>
  <a href="https://crates.io/crates/runmat"><img src="https://img.shields.io/crates/d/runmat.svg" alt="Downloads"></a>
</p>

<p align="center">
  <a href="https://runmat.com/sandbox"><strong>Try it now — no install needed</strong></a> · <a href="https://runmat.com/docs">Docs</a> · <a href="https://runmat.com/blog">Blog</a> · <a href="https://runmat.com">Website</a>
</p>

<p align="center"><em>Status: Pre-release (v0.3) — core runtime and GPU engine pass thousands of tests. Expect a few rough edges.</em></p>

---

## What is RunMat?

With RunMat you write your math in clean, readable MATLAB-style syntax. RunMat automatically fuses your operations into optimized kernels and runs them on the best available hardware — CPU or GPU. On GPU, it can often match or beat hand-written CUDA on many dense numerical workloads.

It runs on whatever GPU you have — NVIDIA, AMD, Apple Silicon, Intel — through native APIs (Metal / DirectX 12 / Vulkan). No device management. No vendor lock-in. No rewrites.

```matlab
x  = 0:0.01:4*pi;
y0 = sin(x) .* exp(-x / 10);
y1 = y0 .* cos(x / 4) + 0.25 .* (y0 .^ 2);
y2 = tanh(y1) + 0.1 .* y1;

plot(x, y2);
```

Points in the graph below correspond to the number of elements in the `x` vector above:

![Elementwise math speedup](https://web.runmatstatic.com/elementwise-math_speedup-b.svg)

Core ideas:

- **MATLAB input language compatibility, not a new language**  
- **Fast on CPU and GPU**, with one runtime  
- **No device flags** — Fusion automatically chooses CPU vs GPU based on data size and transfer cost heuristics

---

## Ways to Use RunMat

The open-source runtime in this repo powers every RunMat surface:

<div align="center">
<table>
<tr>
<td align="center" width="20%">
<h3>🌐 Browser</h3>
No install needed<br/><br/>
Runs via WebAssembly + WebGPU.<br/>
Your code never leaves your machine.<br/><br/>
<a href="https://runmat.com/sandbox"><strong>Try now →</strong></a>
</td>
<td align="center" width="20%">
<h3>⌨️ CLI</h3>
Open source (this repo)<br/><br/>
Run <code>.m</code> files, benchmark,<br/>
integrate into CI/CD.<br/><br/>
<code>cargo install runmat</code>
</td>
<td align="center" width="20%">
<h3>📦 NPM</h3>
Embed anywhere<br/><br/>
Full runtime — execution, GPU,<br/>
plotting — in any web app.<br/><br/>
<a href="https://www.npmjs.com/package/runmat"><code>npm install runmat</code></a>
</td>
<td align="center" width="20%">
<h3>🖥️ Desktop</h3>
Coming soon<br/><br/>
Native IDE with local files<br/>
and full GPU acceleration.<br/><br/>
&nbsp;
</td>
<td align="center" width="20%">
<h3>☁️ Cloud</h3>
Free tier available<br/><br/>
Versioning, collaboration,<br/>
team management.<br/><br/>
<a href="https://runmat.com/pricing"><strong>Pricing →</strong></a>
</td>
</tr>
</table>
</div>

---

## ✨ Features at a glance

- **MATLAB input language compatibility, not a new language**

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
  - Supports **Metal (macOS), DirectX 12 (Windows), Vulkan (Linux), WebGPU (browser)**  
  - Falls back to CPU when workloads are too small for GPU to win  

- **Async-capable runtime**

  - Evaluation is built on Rust futures — non-blocking by design, not bolted on  
  - GPU readback, interactive input, and long-running scripts never block the host  
  - Language-level `async`/`await` with cooperative tasks is on the roadmap  
  - MATLAB has no equivalent — RunMat scripts can run interactively in a browser without freezing the page  

- **WebAssembly target + NPM package**

  - The full runtime compiles to WASM and ships as part of this repo (`runmat-wasm`)  
  - Available as [`runmat` on NPM](https://www.npmjs.com/package/runmat) — embed execution, GPU acceleration, and plotting into any web app  
  - GPU acceleration works in the browser via WebGPU  
  - Powers the [browser sandbox](https://runmat.com/sandbox) — your code runs locally, never on a server  

- **Plotting**

  - Interactive 2D and 3D plots  
  - Line, scatter, and surface plots supported today  
  - Some advanced plot types (box plots, violin plots) are still in progress  

- **Open-source runtime**

  - The full runtime, GPU engine, JIT, GC, and plotting — everything in this repo — is MIT licensed  
  - Small binary, CLI-first design  

--- 

## 📊 Performance

Up to **131x faster than NumPy** and **7x faster than PyTorch** on Monte Carlo simulations. Hardware: Apple M2 Max, Metal. Median of 3 runs.

![Monte Carlo speedup](https://web.runmatstatic.com/monte-carlo-analysis_speedup-b.svg)

<details>
<summary><strong>Monte Carlo raw data</strong></summary>

| Paths (simulations) | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|--------------------:|-----------:|-------------:|-----------:|---------------:|-----------------:|
| 250k   | 108.58 |   824.42 |  4,065.87 | 37.44× | 7.59× |
| 500k   | 136.10 |   900.11 |  8,206.56 | 60.30× | 6.61× |
| 1M     | 188.00 |   894.32 | 16,092.49 | 85.60× | 4.76× |
| 2M     | 297.65 | 1,108.80 | 32,304.64 |108.53× | 3.73× |
| 5M     | 607.36 | 1,697.59 | 79,894.98 |131.55× | 2.80× |

</details>

<details>
<summary><strong>4K Image Pipeline</strong> — up to 10x faster than NumPy</summary>

![4K image pipeline speedup](https://web.runmatstatic.com/4k-image-processing_speedup-b.svg)

| B | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|---|---:|---:|---:|---:|---:|
| 4  | 142.97 | 801.29 | 500.34 | 3.50× | 5.60× |
| 8  | 212.77 | 808.92 | 939.27 | 4.41× | 3.80× |
| 16 | 241.56 | 907.73 | 1783.47 | 7.38× | 3.76× |
| 32 | 389.25 | 1141.92 | 3605.95 | 9.26× | 2.93× |
| 64 | 683.54 | 1203.20 | 6958.28 | 10.18× | 1.76× |

</details>

<details>
<summary><strong>Elementwise Math</strong> — up to 144x faster than PyTorch at 1B elements</summary>

![Elementwise math speedup](https://web.runmatstatic.com/elementwise-math_speedup-b.svg)

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

</details>

On smaller arrays Fusion keeps work on CPU so you still get low overhead and a fast JIT. 

*See [benchmarks/](benchmarks/) for reproducible test scripts, detailed results, and comparisons against NumPy, PyTorch, and Julia.*


---

## 🎯 Quick Start

### Installation

```bash
# Quick install (Linux/macOS)
curl -fsSL https://runmat.com/install.sh | sh

# Quick install (Windows PowerShell)
iwr https://runmat.com/install.ps1 | iex

# Homebrew (macOS/Linux)
brew install runmat-org/tap/runmat

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
```

### CLI Features

```bash
# Check GPU acceleration status
runmat accel-info

# Benchmark a script
runmat benchmark script.m --iterations 5 --jit

# Create a snapshot for faster startup
runmat snapshot create -o stdlib.snapshot

# View system information
runmat info
```

See [CLI Documentation](https://runmat.com/docs/cli) for the complete command reference.

### Jupyter Integration

```bash
# Register RunMat as a Jupyter kernel
runmat --install-kernel

# Launch JupyterLab with RunMat support
jupyter lab
```

---

## 🧱 Architecture: CPU+GPU performance

RunMat uses a tiered CPU runtime plus a fusion engine that automatically picks CPU or GPU for each chunk of math. All of the components below are open source and live in this repository.

### Key components

| Component              | Purpose                                  | Technology / Notes                                                  |
| ---------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| ⚙️ runmat-ignition   | Baseline interpreter for instant startup | HIR → bytecode compiler, stack-based interpreter                    |
| ⚡ runmat-turbine     | Optimizing JIT for hot code              | Cranelift backend, tuned for numeric workloads                      |
| 🧠 runmat-gc         | High-performance memory management       | Generational GC with pointer compression                            |
| 🚀 runmat-accelerate | GPU acceleration subsystem               | Fusion engine + auto-offload planner + `wgpu` backend               |
| 🔥 Fusion engine       | Collapses op chains, chooses CPU vs GPU  | Builds op graph, fuses ops, estimates cost, keeps tensors on device |
| 🎨 runmat-plot       | Plotting layer                           | Interactive 2D/3D plots; some advanced plot types still in progress |
| 🌐 runmat-wasm       | WebAssembly build of the runtime         | Runs in any browser; powers the sandbox at runmat.com               |
| 📸 runmat-snapshot   | Fast startup snapshots                   | Binary blob serialization / restore                                 |
| 🧰 runmat-runtime    | Core runtime + 300+ builtin functions    | BLAS/LAPACK integration and other CPU/GPU-accelerated operations    |


### Why this matters

- **Tiered CPU execution** gives quick startup and strong single-machine performance.  
- **Fusion engine** removes most manual device management and kernel tuning.  
- **GPU backend** runs on NVIDIA, AMD, Apple Silicon, and Intel through Metal / DirectX 12 / Vulkan, with no vendor lock-in.

---

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
x = rand(1024, 1, 'single');
y = sin(x) .* x + 0.5;        % Fused: sin, multiply, add
m = mean(y, 'all');            % Reduction stays on GPU
fprintf('m=%.6f\n', double(m)); % Single download at sink
```

RunMat detects the elementwise chain (`sin`, `.*`, `+`), fuses them into one GPU dispatch, keeps `y` resident on GPU, and only downloads `m` when needed for output.

For more details, see [Introduction to RunMat GPU](https://runmat.com/docs/accelerate/fusion-intro).

---

## 💡 Design Philosophy

RunMat follows a **minimal core, fast runtime, open extension model** philosophy:

- **Full language support**: The core implements the complete MATLAB grammar and semantics, not a subset
- **Extensive built-ins**: The standard library aims for complete base MATLAB built-in coverage (300+ functions)
- **Tiered execution**: Ignition interpreter for fast startup, Turbine JIT for hot code
- **GPU-first math**: Fusion engine automatically turns MATLAB code into fast GPU workloads
- **Small, portable runtime**: Single static binary, fast startup, modern CLI, Jupyter kernel support
- **Toolboxes as packages**: Signal processing, statistics, image processing, and other domains live as packages — the package manager is [in active design](https://runmat.com/docs/package-manager)

RunMat keeps the core small and uncompromisingly high-quality; everything else is a package. Use any editor you like, or the built-in [browser IDE](https://runmat.com/sandbox) and upcoming desktop app.

See [Design Philosophy](https://runmat.com/docs/design-philosophy) for the complete design rationale.

---

## 🌍 Who Uses RunMat?

RunMat is built for array-heavy math in many domains.

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

---

## 📚 Quick links

- **Getting started**
  - [Installation](https://runmat.com/docs/getting-started)
  - [Browser sandbox](https://runmat.com/sandbox)
  - [CLI reference](docs/CLI.md)
  - [Configuration](docs/CONFIG.md)

- **Language & runtime**
  - [Language reference](docs/LANGUAGE.md)
  - [Language coverage](docs/LANGUAGE_COVERAGE.md)
  - [Built-in function library](docs/LIBRARY.md)
  - [Design philosophy](docs/DESIGN_PHILOSOPHY.md)

- **GPU acceleration**
  - [Introduction to RunMat GPU](docs/INTRODUCTION_TO_RUNMAT_GPU.md)
  - [GPU behavior notes](docs/GPU_BEHAVIOR_NOTES.md)
  - [Fusion & auto-offload](https://runmat.com/docs/accelerate/fusion-intro)

- **Plotting**
  - [Plotting guide](docs/PLOTTING.md)

- **Runtime architecture**
  - [Architecture overview](docs/ARCHITECTURE.md)
  - [Async design](docs/ARCH_ASYNC.md)
  - [Filesystem](docs/FILESYSTEM.md)
  - [Roadmap](docs/ROADMAP.md)

- **Embedding & integration**
  - [NPM package (`runmat`)](bindings/ts/README.md)
  - [Browser sandbox guide](docs/DESKTOP_BROWSER_GUIDE.md)

- **Contributing**
  - [Contributing guide](docs/CONTRIBUTING.md)
  - [Developer setup](docs/DEVELOPING.md)

- **Blog**
  - [Introducing RunMat](https://runmat.com/blog/introducing-runmat)
  - [Why Rust](https://runmat.com/blog/why-rust)
  - [MATLAB Alternatives 2026](https://runmat.com/blog/matlab-alternatives)
  - [How to Use GPUs in MATLAB](https://runmat.com/blog/how-to-use-gpu-in-matlab)
  - [In Defense of MATLAB Whiteboard-Style Code](https://runmat.com/blog/in-defense-of-matlab-whiteboard-style-code)

---

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

---

## 📜 License

The RunMat runtime is open source and licensed under the **MIT License with Attribution Requirements**. This means:

✅ **Free for everyone** - individuals, academics, most companies  
✅ **Open source forever** - the runtime will always be free and open  
✅ **Commercial use allowed** - embed in your products freely  
⚠️ **Attribution required** - credit "RunMat by Dystr" in public distributions  
⚠️ **Special provisions** - large scientific software companies must keep modifications open source  

RunMat Cloud and the desktop app are separate products built on top of this open-source runtime. See [runmat.com/pricing](https://runmat.com/pricing) for details.

See [LICENSE.md](LICENSE.md) for complete terms or visit [runmat.com/license](https://runmat.com/license) for FAQs.

---

**Built with ❤️ by [Dystr Inc.](https://dystr.com) and the RunMat community**

⭐ **Star us on GitHub** if RunMat is useful to you.

[**🚀 Get Started**](https://runmat.com/docs/getting-started) • [**📖 Docs**](https://runmat.com/docs) • [**📝 Blog**](https://runmat.com/blog) • [**💰 Pricing**](https://runmat.com/pricing) • [**🐦 Follow @dystr**](https://x.com/dystrEng)

---

*MATLAB® is a registered trademark of The MathWorks, Inc. RunMat is not affiliated with, endorsed by, or sponsored by The MathWorks, Inc.*
