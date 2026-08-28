# Packages

RunMat packages make a MATLAB-syntax project reproducible across the CLI, Desktop, browser/WASM, the LSP, and CI. A package is described by `runmat.toml`, resolved into one deterministic graph, recorded in `runmat.lock`, and consumed through a serialized frozen-project handoff. Consumers never independently reinterpret the manifest or choose different dependency versions.

## Dependency Sources

RunMat supports four dependency sources. Every dependency has a local alias and exactly one locator.

```toml
[dependencies]
local_tools = { path = "deps/local-tools", version = "^1.2" }
git_tools = { git = "https://github.com/acme/tools.git", tag = "v1.2.3", version = "^1.2" }
cloud_tools = { project = "proj_0123456789abcdef0123456789abcdef", snapshot = "stable", version = "^1.2" }
registry_tools = { package = "acme/tools", version = "^1.2" }
```

Path dependencies identify project-relative source trees. Git dependencies require exactly one `rev`, `tag`, or `branch` selector and may select a `subdir`. RunMat Server project dependencies resolve a tag or exact snapshot ID at a credential-free HTTPS service origin. Registry dependencies use `organization/name`, optionally with an explicit registry name.

Development and test-only dependencies use `[dev-dependencies]` and `[test-dependencies]`. Target-specific forms live under `[target."<predicate>".dependencies]`, `[target."<predicate>".dev-dependencies]`, and `[target."<predicate>".test-dependencies]`. Optional dependencies and requested/default features are solver inputs, not post-resolution filtering.

## Locking And Updates

`runmat package resolve` creates or reconciles `runmat.lock`. The lock records exact package instances, immutable source identities, selected features/groups/target capabilities, dependency edges, and a graph digest. Commit it with the project.

```bash
runmat package resolve
runmat package tree
runmat package why registry_tools
runmat package fetch
runmat package update
```

Ordinary execution reuses a compatible lock. `--locked` rejects manifest, path-content, target, group, feature, or dependency drift. `--offline` prohibits network acquisition but may fill mounts from verified local content. `--frozen` combines locked and offline behavior and prohibits lock mutation. Mutable Git branches/tags, Server snapshot tags, and registry version ranges are reconsidered only by an allowed resolution or explicit update.

## Source Roots And Toolbox Layouts

Each `[sources].roots` entry is one MATLAB lookup root. Package folders (`+name`), class folders (`@Name`), and `private` folders beneath it keep MATLAB visibility rules. Ordinary nested directories become qualified module segments rather than silently behaving like `addpath(genpath(...))`.

When migrating a toolbox that historically adds many ordinary directories, list every stable lookup directory as a source root or keep truly dynamic session setup in `addpath`. Prefer manifest roots for stable code because the compiler and LSP can index those functions, preserve source identity, navigate across files, infer output arity/types/shapes, and compile direct calls. See the [migration guide](./migration.md).

## Browser And WASM

Browser RunMat uses the same Rust manifest, solver, lock, graph, integrity, cache-policy, and handoff types as native RunMat. JavaScript supplies virtual filesystem access, authenticated fetch, IndexedDB transactions, WebCrypto recipient keys, immutable mounts, and worker lifecycle only.

Execution-bundle construction replaces host-local handoff paths with canonical logical source objects while retaining the same frozen graph and catalog. Native and browser workers verify and rebase that handoff onto worker-owned storage rather than resolving dependencies again. Browser hosts must materialize private closures in the worker’s ephemeral filesystem provider and dispose that provider with the worker; plaintext private package bodies are not added to the persistent package cache.

The browser runtime reads `runmat.lock` beside the manifest, resolves or replays the graph, writes a canonical generated lock, installs the frozen handoff in the active WASM session, and exposes the graph/source revision. Complete cached closures can reopen with locked offline policy after a page reload. A missing or evicted object is a recoverable cache miss; denied persistent-storage permission is reported but does not weaken transaction correctness.

Git dependencies use the RunMat Server Git snapshot gateway in browsers, while native hosts may use direct Git. Both validate the exact commit and canonical tree digest. Native-only capability requirements such as native libraries, MEX, JVM, or subprocesses fail during browser resolution with a dependency-path diagnostic instead of failing later during execution.

## Related Documentation

- [Hosted Registry](./registry.md)
- [Publishing](./publishing.md)
- [Package Security](./security.md)
- [CI, Offline Use, And Vendoring](./ci.md)
- [Migrating Existing Toolboxes](./migration.md)
- [Projects](../getting-started/projects.md)
