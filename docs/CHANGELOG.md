# RunMat Changelog

_What's new across the RunMat runtime, cloud, and sandbox. For technical runtime details and commit diffs, see [GitHub Releases](https://github.com/runmat-org/runmat/releases) (runtime only)._

---

## [v0.4.0](https://github.com/runmat-org/runmat/compare/v0.3.2...HEAD)

### Runtime

#### Added

New [plot types](/docs/matlab-function-reference#plotting):
- Add `stem` with GPU-accelerated rendering
- Add `errorbar` visualization
- Add `area` filled area plots
- Add `contour` / `contourf` with GPU-resident path rendering
- Add `plot3` 3D line plots
- Add `imagesc` scaled image display
- Add `pie` charts
- Add `histogram` plotting
- Add `semilogx`, `semilogy`, `loglog` log-scale axes
- Add `plot(y)` shorthand — single-argument plot

Plotting infrastructure:
- Add graphics handles and per-axes subplot state
- Add axes helpers: `xlim`, `ylim`, `zlim`, `caxis`/`clim`
- Add common signature forms: `bar(x,y)`, `surf(Z)`, `mesh(Z)`, `surfc(Z)`, `meshc(Z)`, `stairs(y)`
- Add `get`/`set` for plot handles with shared property access
- Add annotation and legend builtins: `title`, `xlabel`, `ylabel`, `zlabel`, `legend`
- Add 3D view controls: `view(az, el)`
- Add world-space text annotations

Numerical computing:
- Implement correct matrix division semantics — `\`, `/`, `.\`, `./` now have distinct, correct semantics through parser, HIR, bytecode, VM, and GPU paths. `mldivide` and `mrdivide` support triangular solves, dense square solves, tall least-squares, wide minimum-norm solves, transpose variants, and Cholesky-backed solves with GPU-resident F32 execution
- Implement full FFT/IFFT family — `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn` now fully implemented with underlying RustFFT for CPU paths and high-performance staged GPU shaders supporting power-of-two, radix-3, radix-5, mixed compositions, and Bluestein fallback for non-smooth/prime lengths. Complex array indexing now works throughout (`Y = fft(x); Y = Y(1:N/2)`)

Other:
- Add `clear`, `clc`, `close all` — session management commands now work in browser and native
- Add duration display and datetime interop support
- Add [`peaks`](/docs/matlab-function-reference) builtin — GPU-accelerated with mixed-residency tensor support and type inference

#### Fixed
- Fix struct field indexing — `s.arr(k)` and `s.arr(1:n)` now work with MATLAB-compatible semantics
- Fix implicit struct creation — assigning to `r.x = 10` on an uninitialized variable now materializes a struct
- Fix `fprintf` compatibility for literal dollar signs and casted numeric inputs
- Fix `atan2` stack underflow that crashed scripts calling `atan2` with compound expressions
- Fix leading-dot floats — `.5` parsed as `0.5`
- Fix GPU tensor to f64 conversion for iterative solve workflows
- Fix range-indexing type errors
- Fix GPU memory cleanup on QR and `mrdivide` errors
- Fix `matmul` parameter caching
- Fix histogram and bar charts not displaying
- Fix right bar chart persisting after generating new charts
- Fix `contourf` Z dimension mismatch
- Fix `area` X/Y length mismatch
- Fix `image` surface data handling
- Fix `stairs` 1-argument form
- Fix Z-axis label styling
- Fix `plot` 3-argument overload
- Fix stem shader offset when baseline is hidden


### Sandbox

#### Added
- Add notebook editor infrastructure — markdown cells alongside code cells (not yet user-facing)
- Add RunMat Agent infrastructure — typed protocol, config, ops, and event model (not yet user-facing)

#### Fixed
- Fix artifact loading to return newest entry when an ID is reused
- Fix browser terminal `clear`/`clc` handling
- Fix runtime state leaking between notebook run sessions

#### Changed
- Improve path handling with safe segment escaping for dot components
- Improve auth flow — email verification now redirects directly to the sandbox with an auth modal
- Improve Auth0 email login flow
- Improve stdout deduplication and clear event handling

### Cloud

#### Added
- Add incremental streaming for the RunMat agent
- Add durable persistence and replay for run artifacts

---

## [v0.3.2](https://github.com/runmat-org/runmat/compare/v0.3.1...v0.3.2)

_March 24, 2026_

Deployment improvements for macOS, Windows, and Linux — no runtime or engine changes.

---

## [v0.3.1](https://github.com/runmat-org/runmat/compare/v0.3.0...v0.3.1)

_March 24, 2026_

Infrastructure changes only — no runtime or engine changes.

---

## [v0.3.0](https://github.com/runmat-org/runmat/compare/v0.2.8...v0.3.0)

_March 24, 2026_

### Runtime

#### Added
- WebAssembly (WASM) compile target — RunMat now runs entirely in the [browser](/docs/desktop-browser-guide) with WebGPU acceleration, published as the `runmat` npm package
- Add fused GPU rendering pipeline for 2D and 3D plots with zero-copy surface data path
- Add 3D depth camera with reversed-Z and dynamic clip planes
- Add type inference — context-aware shape resolvers track tensor shapes through the compiler
- Add builtins: `int32`, `uint16`, `isgpuarray`, `magic`, `empty`, `frewind`, `rand(m,n,p)`, `uint8` elementwise ops, `isequal`, `logical`, `cellfun` handle support, `fullfile`, `erase`, `atanh`, `tempname`
- Add full call stack in error diagnostics with stack depth limit and source location tracking
- Add [MATLAB compatibility mode](/docs/language)

#### Changed
- Migrate the entire VM, all builtins, and the GPU provider to async
- Change default error namespace to `RunMat` (from `MATLAB`)

#### Fixed
- Fix slice indexing shape preservation, `conv2` kernel flip, `chol` multi-output control flow, `load` auto-assignment, `dlmread`/`csvread` header parsing, `cummax`/`cummin` alignment, `fgets` character limit, native auto-init race condition, and GPU compute dispatch stack layout

### Sandbox

#### Added
- Launch browser sandbox at [runmat.com/sandbox](https://runmat.com/sandbox)

#### Fixed
- Fix save file behavior
- Fix figure state reset on re-run
- Fix variable inspector pagination — large vectors now paginate correctly
- Fix variable inspector caching — stale values no longer shown after re-runs
- Fix data file viewer error handling
#### Changed
- Show struct values before tensor values in the output panel
- Improve variable inspector layout

### Cloud

#### Added
- Add organizations, projects, and team memberships

---

## [v0.2.8](https://github.com/runmat-org/runmat/compare/v0.2.7...v0.2.8)

_December 22, 2025_

#### Added
- Add complex number support — complex arithmetic works throughout the runtime
- Add non-conjugate transpose (`.'`) — apostrophe handling now distinguishes from conjugate transpose
- Add Homebrew installation — `brew install runmat` (see [CLI docs](/docs/cli))

#### Changed
- Replace manual REPL input with `rustyline` for line editing, command history, and formatting
- Persist `ans` variable across expressions
- Display struct content in the REPL
- Improve multi-line formatting for `ls` and matrices
- Suppress output for assignments ending in `;`

#### Fixed
- Fix `save()` stack overflow bug

---

## [v0.2.7](https://github.com/runmat-org/runmat/compare/v0.2.6...v0.2.7)

_December 2, 2025_

#### Added
- Add anonymous telemetry to help improve RunMat. No code is ever captured — only internal error codes. See [Telemetry](/docs/telemetry) for details

#### Fixed
- Fix CLI version display

---

## [v0.2.6](https://github.com/runmat-org/runmat/compare/v0.0.4...v0.2.6)

_November 21 – 24, 2025_

Covers [v0.2.0](https://github.com/runmat-org/runmat/compare/v0.0.4...v0.2.0) through v0.2.6.

#### Added

_[Accelerate](/docs/accelerate/fusion-intro) (GPU backend):_
- Add wgpu-based GPU backend — Metal (macOS), DirectX 12 (Windows), Vulkan (Linux)
- Add cost model — runtime profiling routes work to CPU or GPU based on data size and transfer cost
- Add f32/f64 compute shaders for a broad set of builtins — `ones`, `zeros`, `rand`, reductions, and many more now dispatch to the GPU automatically

_[Fusion engine](/docs/fusion-guide):_
- Add computation graph pattern scanning — the runtime analyzes your code's computation graph, matches sequences against a library of fusible patterns, and replaces them with optimized GPU kernels automatically
- Add 5–6 initial fusion operations including elementwise math chains, where multiple operations collapse into a single GPU kernel eliminating intermediate memory traffic

---

## [v0.0.3](https://github.com/runmat-org/runmat/compare/v0.0.2...v0.0.3)

_August 25, 2025_

#### Added
- Add Jupyter kernel support — ZMQ transport so RunMat can serve as a Jupyter kernel
- Add PowerShell installer for Windows

---

## [v0.0.2](https://github.com/runmat-org/runmat/compare/v0.0.1...v0.0.2)

_August 19, 2025_

#### Added
- Implement interpretation for the core MATLAB language grammar and semantic surface, across the parser, HIR, and VM. See [Language Coverage](/docs/language-coverage) for the full matrix of implemented coverage

---

## [v0.0.1](https://github.com/runmat-org/runmat/releases/tag/v0.0.1)

_August 10, 2025_

#### Added
- Initial release — lexer, architecture plan, project scaffolding
- Add release workflow with macOS signing, cross-compilation, crates.io publishing

