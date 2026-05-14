# MATLAB Compatibility

RunMat is a high-performance runtime for MATLAB-syntax code. It covers the core MATLAB language — variables, operators, control flow, functions, N-D indexing, full `classdef` OOP, packages, imports, and exceptions — along with 400+ built-in functions and automatic GPU acceleration across all major vendors. No license required.

**What "compatible" means here:** RunMat targets the core language grammar and built-in functions that engineers use daily. Many `.m` scripts run without changes. Scripts that depend on MATLAB toolboxes, Simulink, MEX/Java/Python interop, or specialized file formats (.slx, .mlapp) are outside this scope. Where a script uses a function RunMat doesn't ship yet, the built-in agent can often help adapt the code — see [Agent-assisted migration](#agent-assisted-migration) below.

This page summarizes what works, what doesn't, and where to look for details. For the full feature-by-feature matrix, see [Language Coverage](/docs/language-coverage). For individual functions, see the [Built-in Function Reference](/docs/matlab-function-reference).

## Language coverage

RunMat targets MATLAB's core language grammar and semantics. Engineers familiar with MATLAB can be productive immediately — no new language to learn.

| Category | Status | Highlights |
| :--- | :---: | :--- |
| Variables & data types | ✅ | `double`, `single`, char arrays, string arrays, logicals, integers (`int8`…`uint64`), complex numbers, `global`, `persistent` |
| Operators | ✅ | Arithmetic, element-wise, relational, logical (element-wise and short-circuit), transpose (`'` and `.'`), colon ranges |
| Control flow | ✅ | `if/elseif/else`, `for`, `while`, `switch/case/otherwise`, `break`, `continue`, `return`, `try/catch/end`, `rethrow` |
| Functions | ✅ | Named functions, multiple returns (`[a,b]=f()`), anonymous functions with closures, `varargin`/`varargout`, `nargin`/`nargout` |
| Indexing & slicing | ✅ | N-D numeric indexing, logical indexing, `end` arithmetic, struct field access, cell content indexing, function/cell expansion into slice targets |
| OOP (`classdef`) | ✅ | Properties (including `Dependent`), methods (static/instance), events (`addlistener`/`notify`), handle classes, enumerations, operator overloading, metaclass operator `?Class` |
| Packages & imports | ✅ | `import pkg.*`, `import pkg.name`, MATLAB-parity precedence (locals > user > specific > wildcard > `Class.*`) |
| Scripting & syntax | ✅ | `.m` scripts, `%` and `%{ %}` comments, line continuation `...`, semicolon suppression, command-form calls |
| Exceptions | ✅ | `MException` with MATLAB-compatible identifiers and messages across indexing, arity, and OOP error paths |

Full details: [Language Coverage](/docs/language-coverage)

## Built-in functions

RunMat includes 400+ core MATLAB built-in functions covering math, linear algebra, array creation and manipulation, string operations, file I/O, and more. Notable additions include `peaks` (GPU-accelerated), `clear`/`clc`/`close all` session management, and full `mldivide` (backslash) linear system solving.

Browse the complete list: [Built-in Function Reference](/docs/matlab-function-reference)

## Plotting

RunMat includes 40+ plotting builtins with GPU-first rendering, interactive 3D camera, theming, and scene persistence.

**2D chart types:** `plot`, `scatter`, `bar`, `histogram`, `hist`, `area`, `stairs`, `stem`, `errorbar`, `pie`, `contour`, `contourf`, `image`, `imagesc`, `imshow`, `quiver`

**3D chart types:** `plot3`, `surf`, `surfc`, `mesh`, `meshc`, `scatter3`

**Log-scale:** `semilogx`, `semilogy`, `loglog`

**Figure management:** `figure`, `subplot`, `hold`, `clf`, `cla`, `close`, `title`, `sgtitle`, `xlabel`, `ylabel`, `zlabel`, `legend`, `colorbar`, `colormap`, `axis`, `grid`, `box`, `shading`, `view`, `drawnow`, `pause`

**Handle graphics:** `get`, `set`

Rendering is GPU-first: vertex buffers are built on-device and rendered through WebGPU in the browser or Metal/Vulkan/DX12 natively. Interactive 3D camera supports rotate, pan, and zoom with reversed-Z depth.

Advanced/specialized chart types (`polar`, `heatmap`, `geobubble`, `wordcloud`, `stackedplot`, `swarmchart`) and full annotation objects are not yet supported.

## Toolbox coverage

RunMat focuses on core MATLAB — the language, operators, data types, and general-purpose built-in functions. MATLAB's add-on toolboxes are not included. Engineers whose workflows depend on specific toolboxes should check the table below.

| MATLAB Toolbox | RunMat Status | Notes |
| :--- | :---: | :--- |
| Core MATLAB | ✅ | ~99% language, 400+ functions, full OOP, GPU |
| Simulink | ❌ | RunMat is script-based only; no block-diagram modeling |
| Signal Processing | ❌ | Not implemented |
| Control System | ❌ | Not implemented |
| Image Processing | ❌ | Basic array ops and `image`/`imagesc`/`imshow` work on image data |
| Statistics & Machine Learning | ❌ | Core stats functions (`mean`, `std`, `var`, `median`, `sort`, `hist`, etc.) are available as builtins |
| Optimization | ❌ | Not implemented |
| Symbolic Math | ❌ | Not implemented |
| Parallel Computing | ⚠️ | Different model — RunMat provides automatic GPU acceleration without explicit `gpuArray` or `parfor` |

## Compatibility modes

RunMat supports three compatibility modes, configured in `.runmat`:

- **`compat = "runmat"`** (default) — accepts MATLAB command syntax (`hold on`, `axis equal`, etc.) with RunMat error namespaces
- **`compat = "matlab"`** — same as `runmat` but error identifiers use `MATLAB:` prefix for closer parity
- **`compat = "strict"`** — disables command-style implicit forms; all calls must use explicit parenthesized syntax

Details: [Language Reference](/docs/language)

## Where RunMat goes beyond MATLAB

- **GPU-native tensor execution** on any vendor (NVIDIA, AMD, Intel, Apple) via Metal, Vulkan, DirectX 12, and WebGPU — not CUDA-only
- **Automatic device planning and fusion** — the runtime picks CPU vs GPU based on data size and fuses compatible operations into fewer kernels
- **Zero-temporary slice expansion** — function/cell outputs write directly to destination slices without intermediate copies
- **Memory safety** — Rust implementation eliminates entire classes of memory bugs
- **Browser-native** — full IDE runs client-side via WebAssembly with no server, no login, no quotas
- **Built-in versioning** — automatic per-save file history, snapshots, and git export without requiring engineers to learn git

## Where RunMat goes beyond GNU Octave

- Full `classdef` OOP (properties, methods, events, handle classes, enumerations, operator overloading, metaclass `?Class`)
- `import pkg.*` / `import pkg.name` with MATLAB-parity precedence
- N-D `end` arithmetic across dimensions in both gather and scatter
- Function/cell expansion into slice targets with dynamic packing
- Uniform `MException` identifier/message model
- Automatic GPU acceleration (Octave has no GPU support)
- String arrays (`"..."`) with MATLAB-parity indexing and comparison

## Known limitations

- **No Simulink** — RunMat is a script/function runtime, not a block-diagram simulation environment
- **No GUI frameworks** — GUIDE and App Designer are not supported
- **No MATLAB toolboxes** — Signal Processing, Control System, Image Processing, etc. are not included (see table above)
- **Plotting gaps** — specialized chart types (`polar`, `heatmap`, etc.) and full Handle Graphics property coverage are still being added
- **File I/O** — core functions (`load`, `save`, `fopen`, `fclose`, `fprintf`, `fscanf`, `fread`, `fwrite`, `readmatrix`, `writematrix`) are available; some advanced I/O functions are not yet implemented
- **MEX / Java / Python interop** — not supported

## Agent-assisted migration

RunMat includes a built-in agent that connects to the same runtime session you're working in. When migrating existing `.m` scripts, the agent can help in three ways:

1. **Diagnostics and unsupported calls.** Run your script, let it fail, and ask the agent to look at the errors. It can read the diagnostics, identify which functions are missing, and propose replacements using RunMat's existing built-ins or a small local helper.
2. **Command-form cleanup.** For teams moving toward `strict` compatibility mode, the agent can convert command-style calls (`hold on`, `axis tight`) to their explicit parenthesized form (`hold("on")`, `axis("tight")`).
3. **Toolbox-adjacent patterns.** Some toolbox functions have straightforward equivalents in core MATLAB math. The agent can often rewrite these calls using RunMat primitives (e.g., reimplementing a simple signal processing function from its mathematical definition).

Every proposed edit is a reviewable diff — you accept, reject, or accept in part. The agent does not silently modify your files.

**What the agent cannot do:** It cannot add Simulink support, MEX/Java/Python interop, or GUI frameworks. Scripts that depend heavily on these will need manual porting or are outside RunMat's scope.

## Performance

RunMat uses a tiered execution model inspired by Google's V8: code starts running immediately in an interpreter, then hot paths are compiled into optimized machine code.

- **Monte Carlo (5M paths):** ~2.8x faster than PyTorch, ~130x faster than NumPy
- **Elementwise chains (1B elements):** ~100x+ faster than PyTorch when fusion eliminates memory traffic
- **Startup:** ~5 ms via snapshot-based initialization

Full benchmark data: [Benchmarks](/benchmarks)

## Will my MATLAB code run in RunMat?

The fastest way to find out:

1. Open the [browser sandbox](https://runmat.com/sandbox)
2. Paste your `.m` script
3. Hit Run — no install, no account, no upload

Code execution happens locally in your browser — nothing is uploaded to run your script. If you sign in for cloud storage or use the built-in agent, data is sent to RunMat's servers and the configured LLM provider respectively.

If something doesn't work, [open an issue](https://github.com/runmat-org/runmat/issues) with a minimal reproducer and we'll add a conformance test.
