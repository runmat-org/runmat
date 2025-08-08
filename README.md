# 🚀 RunMat: Modern Free MATLAB Compatible Runtime
### A blazing-fast, open-source MATLAB/Octave runtime, by the creators of [Dystr](https://dystr.com)

[![Build Status](https://img.shields.io/github/actions/workflow/status/runmat-org/runmat/ci.yml?branch=main)](https://github.com/runmat-org/runmat/actions)
[![License](https://img.shields.io/badge/license-MIT%20with%20Attribution-blue.svg)](LICENSE.md)
[![Crates.io](https://img.shields.io/crates/v/runmat.svg)](https://crates.io/crates/runmat)
[![Downloads](https://img.shields.io/crates/d/runmat.svg)](https://crates.io/crates/runmat)

**[🌐 Website](https://runmat.org) • [📖 Documentation](https://runmat.org/docs)**

---

## What is RunMat?

RunMat is a **modern, high-performance runtime** for MATLAB® and GNU Octave code that eliminates license fees, vendor lock-in, and performance bottlenecks. Built from the ground up in Rust with a **V8-inspired architecture**, it delivers:

- 🚀 **150-180x faster execution** than Octave through JIT compilation
- ⚡ **Instant startup** (5ms vs 900ms+ in Octave) via advanced snapshotting
- 🎨 **GPU-accelerated plotting** that's beautiful and responsive
- 📊 **Native Jupyter support** with rich interactive widgets
- 🛡️ **Memory safety** and **zero crashes** guaranteed by Rust
- 💰 **$0 licensing costs** - completely free and open source

## 📊 Performance Benchmarks

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

*Benchmarks run on Apple M2 Max with BLAS/LAPACK optimization. See [benchmarks/](benchmarks/) for reproducible test scripts and detailed results.*

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

### Run Your First Script

```bash
# Start the interactive REPL
runmat

# Or run an existing .m file
runmat script.m

# Or pipe a script into RunMat
echo "a = 10; b = 20; c = a + b" | runmat
```

### Jupyter Integration

```bash
# Register RunMat as a Jupyter kernel
runmat --install-kernel

# Launch JupyterLab with RunMat support
jupyter lab
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

### Performance That Scales
```matlab
% Matrix operations that fly - 150x+ faster than Octave
n = 1000;
A = randn(n, n);
B = randn(n, n);
tic; C = A * B; toc  % Executes in ~5ms vs 800ms+ in Octave
```

### Beautiful, Interactive Plotting (experimental)
```matlab
% Create a stunning 3D surface plot
[X, Y] = meshgrid(-2:0.1:2, -2:0.1:2);
Z = X .* exp(-X.^2 - Y.^2);
surf(X, Y, Z);
```

## 🏗️ Architecture: V8-Inspired Performance

RunMat's **tiered execution engine** delivers both fast startup and blazing runtime performance.

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **🎯 runmat-ignition** | Baseline interpreter for instant startup | HIR-to-bytecode compiler + stack-based interpreter |
| **⚡ runmat-turbine** | Optimizing JIT compiler for hot code | Cranelift backend |
| **🧠 runmat-gc** | High-performance memory management | Generational GC with pointer compression |
| **🎨 runmat-plot** | Interactive plotting engine | GPU-accelerated via wgpu |
| **📦 runmat-snapshot** | Fast startup system | Binary blob serialization |
| **🔧 runmat-runtime** | 50+ builtin functions | BLAS/LAPACK integration |

## 🎨 Modern Developer Experience

### Rich REPL with Intelligent Features
```bash
runmat> .info
🦀 RunMat v0.1.0 - High-Performance MATLAB Runtime
⚡ JIT: Cranelift (optimization: speed)
🧠 GC: Generational (heap: 45MB, collections: 12)
🎨 Plotting: GPU-accelerated (wgpu)
📊 Functions loaded: 52 builtins + 0 user-defined
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

[**Contribute Code →**](CONTRIBUTING.md)

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