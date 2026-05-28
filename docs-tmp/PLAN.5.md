# Plan 5: Project Composition, Source Discovery, And Entrypoints

## Objective

Move RunMat from host/path-centric execution to explicit project composition using `runmat.toml`.

Plans 0-4 establish semantic HIR, MIR, analysis facts, VM layout, runtime execution, async boundaries, semantic workspace export, and complete MATLAB core semantics. Plan 5 makes those internal concepts externally addressable through a reproducible project/package model.

## Desired Resting State

RunMat can load a project manifest, discover local source roots, resolve declared entrypoints, and make dependencies available to import/name resolution without treating source-level `import` as the dependency mechanism.

The system supports:

- `[package]`
- `[sources]`
- `[dependencies]`
- `[[entrypoints]]`
- project-relative source discovery
- MATLAB-compatible source layout discovery
- named runnable targets
- local path dependencies as the first dependency form
- canonical package/module identity
- diagnostics in project terms rather than loader internals
- incremental invalidation based on source/manifest/dependency changes

## Core Invariants

- Config declares what packages and source roots are available.
- Source-level `import` controls name visibility and ergonomics.
- `import` does not fetch, discover, or declare dependencies.
- Scripts remain source authoring style only; internally they lower to synthetic entry functions.
- Entry execution starts from explicit `HirEntrypoint`s.
- `HirEntrypoint` records origin and execution/export policy; it does not duplicate `FunctionKind`.
- Canonical identities remain qualified even when source uses unqualified imports.
- MATLAB layout conventions such as local functions, package folders, class folders, private folders, and path precedence are source discovery semantics, not runtime string lookup hacks.
- Manifest/source/dependency changes invalidate relevant compiler products.

## Primary Crates

- `runmat-config`
- `runmat-cli`
- `runmat-core`
- `runmat-hir`

## Secondary Crates

- `runmat-runtime`
- `runmat-vm`
- `runmat-kernel`
- `runmat-wasm`
- `runmat-lsp`

## Manifest Shape

Initial canonical shape:

```toml
[package]
name = "thermal-demo"
version = "0.1.0"

[sources]
roots = ["src", "lib", "examples"]

[dependencies]
linalg = { path = "../linalg" }
signal = { path = "../signal" }

[[entrypoints]]
name = "main"
path = "src/main"

[[entrypoints]]
name = "demo"
path = "examples/thermal_surface_demo"

[[entrypoints]]
name = "server"
module = "app.server"
function = "main"
```

## Implementation Plan

1. Add project manifest parsing in `runmat-config`.

This should be separate from existing runtime/tooling config. Existing `.runmat.*` runtime configuration and new `runmat.toml` project manifests should not be conflated.

2. Add manifest validation.

Validation rules:

- `[package].name` is required
- `[sources].roots` is required and non-empty
- source roots are project-relative
- source roots exist or produce clear diagnostics
- every dependency has a supported locator field
- only path dependencies are initially required
- no duplicate dependency names
- no duplicate entrypoint names
- each entrypoint uses exactly one valid target form
- path entrypoints resolve under the project root

3. Add project discovery.

Discovery should find `runmat.toml` from explicit CLI arguments, current-working-directory upward search, or host-provided paths for kernel/wasm/editor contexts.

4. Add source root scanning and source index construction.

Source roots should discover script-like files, function files, class files, package folders (`+pkg`), class folders (`@ClassName`), private folders, and package/module path structure where present. Discovery should build a project source index without lowering every file eagerly unless needed.

5. Define canonical package and module identity.

Initial rule:

- package identity defaults to `[package].name`
- local source files receive module names based on source-root-relative paths
- `+pkg` folders contribute package-qualified module/class/function identity
- `@ClassName` folders contribute nominal class identity and method membership
- `private` folders participate in MATLAB-compatible visibility and precedence where supported
- dependency table keys are canonical top-level package names unless explicit alias/package remapping is later added

6. Resolve entrypoints.

Path entrypoints infer `.m` if omitted and lower through the same `HirEntrypoint` target/origin/policy model as REPL/snippet execution. Module/function entrypoints resolve through canonical module identity.

7. Integrate CLI and core entrypoint selection.

CLI/core should support running by project entrypoint name, explicit source path compatibility, and clear errors when multiple entrypoints exist and none is selected.

8. Add dependency source availability.

Initial support:

- local path dependencies
- dependency manifest loading
- dependency source roots added to the composition graph
- import/name resolution can see dependency packages only when declared

9. Move import resolution onto the project composition graph.

`import` should resolve names against current package/module symbols, configured source roots, declared dependencies, builtins, and runtime class metadata. Package-qualified calls should resolve as package functions/classes rather than ordinary member access.

Data files such as `.mat` files are not modules, but discovery should provide enough source-root/project context for workspace-effecting builtins such as `load` to find project-local data through host/runtime policy.

10. Wire incrementality.

Project composition should provide dependency edges and cache keys for HIR, MIR, analysis, and summaries.

## Tests

Add tests for manifest parsing, missing sections, invalid source roots, duplicate dependencies, duplicate entrypoints, invalid target forms, path entrypoint resolution, function-file entrypoints, script-like entrypoints, same-file local function forward resolution, `+pkg` package functions/classes, `@ClassName` class folders, private-folder visibility where supported, dependency import success/failure, project-local data lookup for `load`, and incremental invalidation on manifest/source changes.

## Acceptance Criteria

- `runmat.toml` project manifests are parsed and validated.
- Source roots and local path dependencies are discovered.
- MATLAB source layout conventions are represented in the source index and resolver products.
- Named entrypoints can be selected and executed.
- Source-level scripts still work but lower through `HirEntrypoint`.
- Import resolution is driven by the composition graph rather than ad hoc runtime path search as the primary semantic model.
- CLI and core execution support project entrypoints without breaking REPL/snippet execution.
- Project composition provides cache/invalidation inputs for incremental compilation.

## Explicit Non-Goals

- Do not implement registry or git dependencies yet.
- Do not implement full package publishing.
- Do not require users to rewrite MATLAB-style source trees into a new module syntax.
- Do not complete runtime class/builtin metadata generation; that is Plan 6.
- Do not complete accelerate/fusion/GC lifetime hardening; that is Plan 7.
