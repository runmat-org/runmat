# RunMat Design Philosophy

A clear, minimal core. A fast, pragmatic runtime. An open extension model.

RunMat is not a reimplementation of MATLAB-in-full. It is a modern runtime that runs MATLAB code very fast, is pleasant to reason about, and is easy to extend in both the Rust and MATLAB languages. We keep the core small and uncompromisingly high-quality; everything else is a package.

This document explains the “why” and the “how” of that design.

## TL;DR

- We separate the language from the runtime. The runtime stays slim; the language surface grows via packages.
- We implement a minimal, well-specified subset of semantics in the core (arrays, indexing, control flow, functions, errors). Built-in functions in core are deliberately few.
- We expose a first-class package system: native (Rust) packages and source (MATLAB) packages. Both can add functions, operators, types and documentation.
- We emphasize performance (tiered execution, careful GC, predictable memory layout) and clarity (simple rules, strict errors), not historical quirks.
- We value stability and composability over maximal compatibility. Where MATLAB's legacy is ambiguous or inconsistent, we choose consistency.

## Historical precedents that work

- The UNIX philosophy: keep the core small, compose through well-defined interfaces.
- Linux kernel + modules: a tight core with a massive extension surface.
- LLVM: a small, carefully specified IR and powerful back-ends hosting a universe of front-ends.
- Node.js + npm, Rust + crates.io, Python + PyPI, Julia + Pkg: a tiny “waist” with a thriving package ecosystem.
- HotSpot (JVM) / V8 (JS): tiered execution (interpreter feeding an optimizing JIT) as the pragmatic choice for fast dynamic languages.

RunMat stands on the shoulders of these patterns.

## The minimal core

The core runtime implements:

- Values and arrays: numeric scalars, column-major dense tensors, logical and string types, cells/structs sufficient for MATLAB-style programming.
- Semantics you can reason about: deterministic evaluation order, explicit error identifiers, predictable promotion rules, clear indexing (N-D slicing, logical masks, `end` arithmetic) with well-defined error behavior.
- Execution engine: an interpreter (Ignition) with an optimizing JIT (Turbine) via Cranelift. The interpreter produces profiling signals that drive JIT compilation.
- Memory model: a generational, mark-and-sweep GC with safe handles, write barriers, and conscious promotion heuristics; no raw pointers exposed to user code.
- A handful of built-ins in core: just enough to bootstrap the system, test semantics, and enable packages (arrays, math primitives, string utilities, error handling). Everything else belongs in packages.

What we deliberately do not put in the core:

- A grab-bag of thousands of functions “for compatibility”.
- Toolboxes. They live as packages.
- Every historical corner-case. Compatibility is measured by principled semantics and clearly documented differences, not by folklore.

## Language vs runtime vs IDE

MATLAB is a language, a large proprietary standard library, an IDE, and an ecosystem of toolboxes. We treat each separately.

- The language is a syntax and semantics for arrays, functions, control flow, and errors. That we support in the core.
- The library (built-ins) is open-ended. RunMat ships a slim standard library and lets packages provide the rest. Documentation is generated from the runtime (not a hand-maintained spreadsheet of parity; parity is not a product goal).
- RunMat is not an IDE. It is a runtime that can be used with any IDE (such as Cursor, VSCode, or IntelliJ).

This separation keeps the core maintainable and lets the community move fast without destabilizing the runtime.

## Packages: two paths, one experience

RunMat's package system is designed for both systems programmers and MATLAB users.

- Native packages (Rust):
  - Author functions in Rust using `#[runtime_builtin(...)]` macros.
  - The macro captures metadata (name, category, summary, examples) and registers the function via inventory at startup.
  - You get type-safe conversions (`TryFrom<&Value>`), deterministic error IDs, and zero-cost documentation generation.
  - Ship to the RunMat registry (or your own), versioned with semver; users declare dependencies in `.runmat`.

- Source packages (MATLAB):
  - Author `.m`/package code.
  - Package metadata declares functions and doc. The builder compiles to RunMat bytecode or runs through the interpreter.
  - No Rust required; great for domain packages and teaching materials.

Both package kinds show up identically to users: functions appear in the namespace, show up in reference docs, and participate in the same tooling (help, search, doc indexing).

## Compilation model you can extend

- Front end: parser and HIR focused on clarity and predictable desugaring, with explicit nodes for MATLAB-specific constructs (e.g., `subsref`/`subsasgn`, varargout/varargin).
- Tiered execution: interpreter first, then JIT for hot paths; pay-as-you-go performance.
- Stable IR and ABI for built-ins: packages target a stable “waist” (Value/Type, call ABI, error model). This keeps packages forward-compatible as the core evolves.
- Accelerate providers: BLAS/LAPACK and GPU back-ends are pluggable. Packages can request capabilities but do not hard-code back-ends.

## Performance principles

- Value layout is intentional (column-major, contiguous dense tensors, predictable strides) to enable vectorized kernels and BLAS handoff.
- JIT uses Cranelift for robust codegen; we avoid speculative heroics that hurt predictability.
- GC favors pause-time goals appropriate for a technical REPL and batch mode alike; safe handles prevent accidental use-after-free.
- Clear slow-path fallbacks (e.g., mixed types) and visible, actionable error messages.

## Documentation and UX

- Function reference is generated from the runtime metadata. What you can call is what you can read about.
- Search and filters are built for the working engineer (categories, examples, snippets), not a marketing scoreboard.
- Docs embrace differences: we call out where RunMat intentionally diverges from MATLAB for clarity or performance.

## 100% MATLAB compatibility

- There is a long tail of MATLAB's historical individual function behavior. This is not a product goal.
- The goal of the RunMat project is not to clone MATLAB. It is a modern, minimal, fast runtime to execute MATLAB code, with a package manager that can add any nuanced behavior.
- Compatibility is not a product strategy. Great performance, a clean model, and a thriving package ecosystem are.
- To that end, RunMat implements the the entire set of the MATLAB language's grammar and core semantics, but only implements a slim standard library and package system, rather than implementing every function in MATLAB's ecosystem.
- If you need a specific MATLAB function or behavior, you can write it yourself in Rust or MATLAB and publish it as a package (or download it from the registry if it's already there).
- In short, RunMat is a modern, minimal executor of MATLAB code semantics, with a small standard library and package system. There are similarities to how your code will execute in MATLAB versus RunMat (e.g. the interpretation of language semantics is the same), but in RunMat, the implementations for the majority of functions are up to the package ecosystem rather than monolithicly in the core.

## Stability, versioning, and safety

- Semantic versioning for the core and registry packages. Narrow, stable interfaces at the “waist” (Value/Type, ABI, error IDs).
- Deterministic builds: the package manager resolves to explicit versions and writes a lockfile.
- Sandboxing and review culture for native packages; source packages run under the same semantics as the core.

## What this enables

- A community that can ship domain math, IO, plotting, and GPU packages without waiting on core releases.
- A smaller trusted compute base (TCB), easier auditing, faster iteration.
- A runtime that can target new back-ends (SIMD profiles, GPUs, accelerators) without breaking package authors.

## Roadmap (high-level)

- v0.x: polish core semantics, stabilize Value/Type/ABI, publish initial package examples, ship the registry and `.runmat` flow, keep JIT/GC focused and fast, finish plotting library. 
- v1.0: locked waist (stable ABI), registry GA, hardened GC and profiling, documented performance guide, full reference generated from runtime metadata.

## Appendix: Frequently asked questions

- “Why Rust?” Memory safety, excellent tooling, fearless concurrency in future features, and easy FFI to BLAS/LAPACK and GPU stacks.
- “Isn't MATLAB huge?” Yes, and that's why we don't pull it into the core. We provide a minimal substrate and an open package system, as we believe that is where a modern runtime should draw the line.
- “Will package X be as fast as MATLAB's toolbox?” Often faster; we build on top-tier BLAS/LAPACK and an IR suited for JIT. Native Rust packages can write tight kernels when needed.
- “How do I add a builtin?” Add a Rust function with `#[runtime_builtin(...)]`, specify metadata, test it, and publish. The docs and search will automatically include it.

RunMat is for people who want a fast, pleasant MATLAB-style environment that evolves like a modern open-source system. If that sounds like you, welcome.
