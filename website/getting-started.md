# Getting Started with RunMat

Get up and running with RunMat in minutes. This guide will walk you through installation, basic usage, and your first interactive session.

## Installation

### Linux & macOS

```bash
curl -fsSL https://runmat.org/install.sh | sh
```

### Windows

```powershell
iwr https://runmat.org/install.ps1 | iex
```

[More Installation Options](/download)

## Your First RunMat Session

### 1. Start the Interactive REPL

Open your terminal and start RunMat:

```bash
$ runmat
RunMat v0.0.1 by Dystr (https://dystr.com)
High-performance MATLAB/Octave runtime with JIT compilation and GC
JIT compiler: enabled (Cranelift optimization level: Speed)
Garbage collector: "default"
No snapshot loaded - standard library will be compiled on demand
Type 'help' for help, 'exit' to quit, '.info' for system information
runmat>
```

### 2. Try Basic Calculations

Start with simple arithmetic and variables:

```matlab
runmat> x = 5
ans = 5
runmat> y = 3.14
ans = 3.14
runmat> result = x * y + 2
ans = 17.7
```

### 3. Work with Matrices

Create and manipulate matrices using familiar MATLAB syntax:

```matlab
runmat> A = [1, 2, 3; 4, 5, 6]
ans = [1 2 3; 4 5 6]
runmat> B = A * 2
ans = [2 4 6; 8 10 12]
runmat> C = A + B
ans = [3 6 9; 12 15 18]
```

### 4. Create Your First Plot

Generate beautiful plots with GPU acceleration:

```matlab
runmat> x = [0, 1, 2, 3, 4, 5]
ans = [0 1 2 3 4 5]
runmat> y = [0, 1, 4, 9, 16, 25]
ans = [0 1 4 9 16 25]
runmat> plot(x, y)
[Interactive plot window opens]
```

✅ Interactive window with zoom, pan, and rotate controls

## Running MATLAB Scripts

### Execute .m Files

Run existing MATLAB/Octave scripts directly:

```bash
# Run a script file
runmat script.m

# Run with specific options
runmat run --jit-threshold 100 simulation.m
```

Most MATLAB and GNU Octave scripts will run without modification. Check our [compatibility guide](/docs/language-coverage) for details.

## Jupyter Notebook Integration

### 1. Install RunMat as a Jupyter Kernel

Make RunMat available as a kernel in Jupyter notebooks:

```bash
runmat --install-kernel
RunMat Jupyter kernel installed successfully!
Kernel directory: ~/.local/share/jupyter/kernels/runmat
```

✅ One-time setup that works with existing Jupyter installations

### 2. Start Jupyter and Select RunMat

Launch Jupyter and create notebooks with the RunMat kernel:

```bash
# Start Jupyter Notebook
jupyter notebook

# Or Jupyter Lab
jupyter lab

# Then select "RunMat" when creating a new notebook
```

✅ Full MATLAB syntax support with 150x faster execution than GNU Octave

### 3. Verify Installation

Check that the RunMat kernel is properly installed:

```bash
jupyter kernelspec list
Available kernels:
  python3    /usr/local/share/jupyter/kernels/python3
  runmat    ~/.local/share/jupyter/kernels/runmat
```

If you don't see RunMat listed, ensure Jupyter is installed and try running the install command again.

## Next Steps

### Learn the Fundamentals

Dive deeper into RunMat's features and capabilities.

[How RunMat Works](/docs/how-it-works) →

### Explore Examples

See RunMat in action with real-world examples.

*Coming Soon*

## Need Help?

Join our community and get support from other RunMat users and developers.

- [GitHub Discussions](https://github.com/runmat-org/runmat/discussions)
- [Report Issues](https://github.com/runmat-org/runmat/issues)


