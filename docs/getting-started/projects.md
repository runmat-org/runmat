# Projects

A RunMat project is a folder of MATLAB-syntax code anchored by a `runmat.toml` or `runmat.json` manifest. The manifest tells RunMat where the source lives, which other projects it depends on, and which workflows can be run by name. RunMat discovers it by walking up from the source file or working directory, so the same project resolves consistently across the CLI, Desktop, and the LSP.

Typically, you should use a project with a manifest once a single script is no longer enough on its own, for example when it calls helper functions in nearby files, organizes code into packages or classes, hides private functions, pins runtime settings, or defines a named workflow that should run the same way everywhere.

## Direct Files And Named Entrypoints

In RunMat, you can always run a `.m` file directly:

```bash
runmat src/analyze_sales.m
runmat run src/analyze_sales
```

The second form can infer `.m` for local paths. Direct file execution is the simplest way to run one script and does not require an entrypoint declaration.

Named entrypoints are useful when a project has one or more canonical entrypoints that you want to be able to run by name:

```bash
runmat run analysis
```

Named entrypoints are declared in the project manifest:

```toml
[package]
name = "sales_report"
version = "0.1.0"

[sources]
roots = ["src"]

[entrypoints.analysis]
path = "src/analyze_sales.m"
```

`path` targets point at a source file. Like with direct file execution, the `.m` extension may be omitted. An entrypoint can also target a discovered module/function pair:

```toml
[entrypoints.summary]
module = "stats"
function = "summarize"
```

Exactly one target form is allowed for each entrypoint: `path`, or `module` plus `function`.

## Source Roots

`[sources].roots` defines where RunMat scans for project source files:

```toml
[sources]
roots = ["src", "tests"]
```

Source roots are relative to the manifest directory. They must exist and cannot use parent-directory traversal. Files outside configured roots can still be run directly by path, but they are not part of the project source index used for module/function entrypoints and cross-file symbol discovery.

Source roots make functions and symbols in another directory discoverable by name at compile time. RunMat uses them for project discovery, named entrypoints, module/function resolution, package/class/private indexing, LSP analysis, and dependency composition.

In this way, it is similar to `addpath` in MATLAB:

```matlab
addpath("src")
addpath("lib")
```

You can also use `addpath` (including `addpath(genpath(...))`) while a RunMat session is running. Eligible `.m` functions become callable immediately through direct calls, `feval`, function handles, and callback builtins, and the added path persists across later REPL inputs in that session. `which` and callable resolution use the same ordered session search path; later `addpath`, `rmpath`, or `path` changes therefore affect subsequent runtime selection.

`addpath` remains a runtime/environment operation. A path computed or mutated while the program runs cannot generally provide the LSP and compiler with a single statically proven target, so those calls retain conservative dynamic facts until execution. Prefer `[sources].roots` for stable project sources: RunMat can index those sources ahead of time and preserve cross-file navigation, diagnostics, output arity, type/shape analysis, and optimized direct-call compilation.

## Local Dependencies

Local dependencies make another RunMat project available during composition:

```text
sales-report/
  runmat.toml
  src/analyze_sales.m
  deps/
    shared-tools/
      runmat.toml
      src/+format/titleCase.m
```

Root manifest:

```toml
[package]
name = "sales_report"

[sources]
roots = ["src"]

[dependencies]
tools = { path = "deps/shared-tools", version = "0.1.0" }
```

Dependency manifest:

```toml
[package]
name = "shared_tools"
version = "0.1.0"

[sources]
roots = ["src"]
```

The dependency alias participates in project symbol discovery. A source file from the dependency can be resolved by its own qualified name, by its package-qualified name, or through the root dependency alias when imports or function handles need that form.

## Git Dependencies And Locking

Git dependencies use a credential-free HTTPS or SSH repository URL, one selector, and an optional repository subdirectory:

```toml
[dependencies]
tools = { git = "https://github.com/acme/shared-tools.git", tag = "v1.4.0", subdir = "runmat", version = "^1.4" }
```

Use exactly one of `rev`, `tag`, or `branch`. RunMat resolves a mutable tag or branch to an exact commit and verified tree digest, records that immutable identity in `runmat.lock`, and reuses it until `runmat package update` is requested explicitly. Path dependencies inside a Git package are resolved as exact subdirectory trees from the same commit, so monorepo layouts remain immutable and checkout-independent.

Normal `run`, REPL, `check`, benchmark, bytecode, and package commands share one resolver and one frozen graph. `--locked` requires the existing lock to match the current manifest, selected target/groups/features, path contents, and immutable dependencies. `--offline` permits only already cached content. `--frozen` combines locked and offline behavior and prohibits network access, selector updates, and lockfile mutation.

```bash
runmat package resolve
runmat package fetch
runmat package update
runmat package tree
runmat package why tools
runmat --locked check src/main.m
runmat --offline run src/main.m
runmat --frozen run src/main.m
```

Git credentials come from the host credential provider and are never written to the manifest, lockfile, cache identity, or diagnostics. Native RunMat stores normalized shared Git object databases separately from verified content-addressed snapshots. Browser RunMat uses the authenticated Server Git snapshot gateway, validates the returned inventory in portable Rust, publishes the same canonical blobs/tree to IndexedDB transactionally, and mounts the verified tree read-only through the configured virtual filesystem.

## RunMat Server Project Dependencies

A project hosted by RunMat Server can be used directly as an immutable package source:

```toml
[dependencies]
tools = { project = "proj_0123456789abcdef0123456789abcdef", service = "https://api.runmat.com", snapshot = "stable", version = "^1.4" }
```

`service` is a credential-free HTTPS origin; when omitted, RunMat uses the active configured Server origin. `snapshot` accepts either a mutable tag such as `stable` (and defaults to `main`) or an exact `snap_...` ID. Resolution and explicit update may resolve a tag. RunMat then records the exact Server origin, project ID, snapshot ID, and canonical tree digest in `runmat.lock`; normal locked execution requests that exact identity and never live-mounts the remote project. Server identity is part of source identity, so equal project or snapshot strings from different Servers cannot alias.

Native, browser, and WASM clients validate the Server inventory with the same portable tree algorithm and publish it transactionally into the shared immutable cache described below. Authentication is sent only to the explicitly configured matching Server origin. An acquisition denied by current permissions, a deleted snapshot, corrupt content, or an interrupted transfer cannot publish a cache entry. Previously authorized plaintext already copied into the local cache remains usable by exact locked identity during offline execution after later permission loss or remote deletion: revoking Server access cannot revoke bytes already delivered to a client. Mutable tags still require online resolution and cannot be used to bypass a frozen or locked graph.

## Shared Package Cache

Inspect and collect the shared package cache with:

```bash
runmat package cache status
runmat package cache status --json
runmat package cache gc
runmat package cache gc --target-bytes 1073741824
runmat package cache prune
```

GC and prune never delete objects protected by a pin or active lease. Native sessions and browser project resolvers acquire renewable graph leases, release them on clean disposal, and rely on expiry after a process, tab, or worker disappears. Cache writes publish metadata and payloads together through revision compare-and-swap; interrupted staging is discarded, incomplete dependency closures are not exposed, digest mismatches become explicit corruption records, and cache eviction is reported as a recoverable miss. Native processes coordinate immutable materialization and physical-tree collection with narrow process locks. Browser tabs and workers use IndexedDB transactions and retry stale revisions rather than overwriting newer state.

The native cache location follows the platform cache directory. `RUNMAT_PACKAGE_CACHE_DIR` can select an explicit cache root for hermetic CI or embedding; do not place credentials in that path or variable.

`runmat package vendor` copies the resolved dependency closure into a project-local `vendor` directory by default and atomically records `runmat-vendor.json` at the workspace root with the exact graph, source identities, and project-relative copy locations:

```bash
runmat package vendor
runmat package vendor --output third_party/runmat
```

Frozen execution requires this verified vendor manifest for every live path dependency. RunMat resolves those dependencies from their vendored copies, preserves their locked source identities, and rejects a stale graph, missing copy, source mismatch, or content/manifest tampering. Immutable Git and RunMat Server project dependencies may instead replay from their exact cached snapshots; `--frozen` still performs no network or lockfile mutation.

## Complete Project Example

This project has one top-level script, a sibling helper function, a private helper, a package function, and a class folder:

```text
sales-report/
  runmat.toml
  src/
    analyze_sales.m
    normalizeRows.m
    private/
      localScale.m
    +stats/
      summarize.m
    @Report/
      Report.m
```

`runmat.toml`:

```toml
[package]
name = "sales_report"
version = "0.1.0"

[sources]
roots = ["src"]

[entrypoints.analysis]
path = "src/analyze_sales"
```

`src/analyze_sales.m`:

```matlab
sales = [100 120 130; 80 95 105];

scaled = localScale(sales);
normalized = normalizeRows(scaled);

[totals, averages] = stats.summarize(normalized);

report = Report("sales", totals);
headline = report.title();

disp(headline);
```

`src/normalizeRows.m`:

```matlab
function out = normalizeRows(x)
    rowTotals = sum(x, 2);
    out = x ./ rowTotals;
end
```

`src/private/localScale.m`:

```matlab
function y = localScale(x)
    y = x * 100;
end
```

`src/+stats/summarize.m`:

```matlab
function [totals, averages] = summarize(x)
    totals = sum(x, 1);
    averages = mean(x, 1);
end
```

`src/@Report/Report.m`:

```matlab
classdef Report
    properties
        Name
        Totals
    end

    methods
        function obj = Report(name, totals)
            obj.Name = name;
            obj.Totals = totals;
        end

        function text = title(obj)
            text = "Report: " + obj.Name;
        end
    end
end
```

Run the project entrypoint:

```bash
cd sales-report
runmat run analysis
```

The top-level script variables become candidates for the session workspace after execution. Locals inside `normalizeRows`, `localScale`, `stats.summarize`, and `Report.title` stay inside their function frames.

## What Projects Do Not Change

Project composition does not replace MATLAB source rules. Packages still use `+pkg` folders, classes still use class files and `@ClassName` folders, and private functions remain private to their source area.

`import` controls name visibility inside source code. Dependencies control which external project symbols are available to the resolver. Keeping those responsibilities separate lets RunMat preserve MATLAB-style code while giving hosts a stable project boundary.

## Related Docs

- [Configuration Reference](/docs/runtime/getting-started/config)
- [Command Line Interface](/docs/runtime/getting-started/cli)
- [Module Composition](/docs/runtime/compiler/modules)
