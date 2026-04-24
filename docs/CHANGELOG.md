# RunMat Changelog

_What's new across the RunMat runtime, cloud, and sandbox. For technical runtime details and commit diffs, see [GitHub Releases](https://github.com/runmat-org/runmat/releases) (runtime only)._

---

## [v0.4.4](https://github.com/runmat-org/runmat/compare/v0.4.1...v0.4.4)

_April 24, 2026_

### Runtime

#### Added
- Add `cross` — MATLAB-compatible vector cross products across row vectors, column vectors, matrices, and higher-rank tensors, with GPU-resident execution for real-valued inputs when the active provider supports it
- Add `gradient` — numerical gradients with MATLAB-compatible matrix output ordering, scalar spacing support, complex host support, and WGPU-backed GPU residency for scalar-spacing gradients
- Add `trapz` and `cumtrapz` — trapezoidal and cumulative trapezoidal integration with scalar spacing, coordinate-vector spacing, explicit dimension selection, complex inputs, and GPU input fallback that re-uploads real-valued outputs for downstream GPU work
- Add `sgtitle` — figure-level titles for subplot layouts with text styling, explicit figure-handle targeting, `get`/`set` support, scene replay, and vector/native export support
- Add CLI artifact capture for script runs — `--artifacts-dir`, `--artifacts-manifest`, `--capture-figures`, `--figure-size`, and `--max-figures`

#### Changed
- Rename `runmat-ignition` to `runmat-vm` and split the VM, compiler, interpreter dispatch, indexing, runtime state, and acceleration paths into smaller modules
- Refactor the lexer, parser, HIR, core session, WASM runtime, and CLI layers for maintainability without changing their public behavior
- Expand TypeScript/WASM figure metadata to include `sgTitle` and `sgTitleStyle`, matching the new figure-level title support

#### Fixed
- Fix WGPU provider cleanup for intermediate GPU tensor handles so error paths release temporary device resources correctly
- Fix `gradient` shape handling for empty tensors and align WGPU gradient parameters for the backend layout
- Fix SVG/vector-export text positioning so subplot titles and figure-level titles scale correctly with font size
- Fix `sgtitle` serialization and child enumeration, and allow numeric values to be used as title text

### Sandbox

#### Changed
- Reorganize the browser/WASM runtime into dedicated API, plotting, session, stream, filesystem, GPU, snapshot, state, and wire-format modules
- Improve replay smoke coverage and GPU gradient coverage for the WASM target

### Cloud

#### Changed
- Rework the CLI around task-oriented command modules for script execution, REPL, Jupyter kernels, snapshots, configuration, acceleration diagnostics, remote projects, remote files, and telemetry transport
- Update remote-project CLI documentation around authentication, project selection, filesystem history, snapshots, git sync, retention policies, and `remote run`

### Docs

#### Added
- Add a numerical correctness guide covering tolerance choices, CPU/GPU parity, FFT validation, builtin coverage, and reproducibility expectations
- Add builtin reference pages for `cross`, `gradient`, `trapz`, `cumtrapz`, and `sgtitle`
- Add an alphabetical builtin index to the website reference
- Add a MATLAB FFT guide blog post

#### Changed
- Expand builtin reference metadata and examples for plotting, numerical, string, I/O, and OOP builtins
- Refresh download, MATLAB Online, MATLAB alternatives, GPU, plotting, and CLI documentation
- Improve website navigation, card layouts, heading styles, markdown rendering, Mermaid rendering reliability, image preload behavior, and SEO metadata

---

## [v0.4.1](https://github.com/runmat-org/runmat/compare/v0.4.0...v0.4.1)

_April 15, 2026_

### Runtime

#### Added
- Expand `input()` compatibility — numeric prompts now accept logical values, named constants (`pi`, `inf`, `nan`), and matrix/vector literals with MATLAB-compatible output types. Complex expressions such as `sqrt(2)` and `ones(3)` now evaluate through the full MATLAB pipeline when an eval hook is available
- Add configurable CLI credential storage for `runmat login` — `auto`, `secure`, `file`, and `memory`

#### Changed
- Improve CLI auth persistence — `auto` mode now prefers secure keyring storage and falls back to file-backed credentials with restricted permissions when secure storage is unavailable
- Extract remote auth and public API client logic into a dedicated `runmat-server-client` crate

#### Fixed
- Fix native `input()` evaluation stack overflow by isolating nested prompt evaluation from the outer interpreter call stack
- Fix GPU-backed selector indexing — GPU-resident index tensors now materialize correctly before linear and dimensional indexing

### Sandbox

#### Fixed
- Fix workspace/variable display after `clear()` — variables assigned later in the same execution block now reappear correctly in workspace snapshots
- Fix browser `input()` handling after session resets — reinitializing the WASM session now preserves the installed async input handler

### Cloud

#### Added
- Expand the public API with project git import, project import status, usage notices, upgrade requests, chunk upload targets, data manifest access, run indexes, storage summaries, storage-category deletion, telemetry ingest, and web intake submission endpoints

---

## [v0.4.0](https://github.com/runmat-org/runmat/compare/v0.3.2...v0.4.0)

_April 13, 2026_

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
- Add signal processing helpers — `hann`, `hamming`, `blackman` windowing functions and `nextpow2`, all with GPU acceleration. Supports `symmetric`/`periodic` modes and output type selection

Other:
- Add `clear`, `clc`, `close all` — session management commands now work in browser and native
- Add `datetime` — MATLAB-compatible construction, formatting, string conversion, and subtraction
- Add `duration` — display, arithmetic, and datetime interop
- Add [`peaks`](/docs/matlab-function-reference) builtin — GPU-accelerated with mixed-residency tensor support and type inference

#### Changed
- Expand GPU fusion coverage — `sign`, `fix`, `hypot`, `pow2`, `asinh`, `acosh`, `atanh`, `mod`, `rem` now fuse into single GPU kernels automatically

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
- Add durable persistence and replay for run artifacts

---

## [v0.3.2](https://github.com/runmat-org/runmat/compare/v0.3.0...v0.3.2)

_March 24, 2026_

Deployment and infrastructure improvements for macOS, Windows, and Linux — no runtime or engine changes. Covers v0.3.1 through v0.3.2.

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
