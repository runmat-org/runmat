# Migrating Existing Toolboxes

Existing MATLAB toolboxes can migrate incrementally. Start by making the current lookup roots explicit, then split stable external code into dependencies, generate and commit a lock, and add target/capability declarations for host-specific functionality.

## Replace Setup Scripts With Stable Roots

A common setup function recursively calls `addpath`:

```matlab
addpath(genpath("src"))
addpath(genpath("vendor"))
```

Each ordinary directory placed on the MATLAB path is a distinct lookup root. Represent stable directories explicitly:

```toml
[package]
name = "existing-toolbox"
version = "1.0.0"

[sources]
roots = [
  "src",
  "src/analysis",
  "src/io",
  "src/utilities",
  "vendor/helper"
]
```

Do not collapse arbitrary ordinary subdirectories into one root when the code expects their functions to be unqualified. Beneath a root, `+package`, `@Class`, and `private` keep their MATLAB meanings; ordinary subdirectories become qualified module segments. Generated manifests should sort and normalize roots so every host freezes the same source catalog.

Keep `addpath` for genuinely dynamic, user-selected, or session-local folders. Runtime calls now make eligible functions callable immediately, including through `feval`, function handles, and callbacks, but dynamic path changes intentionally remain conservative for static analysis.

## Introduce Dependencies

Move copied stable code into a path dependency first:

```toml
[dependencies]
helper = { path = "deps/helper", version = "^1" }
```

Once the helper has its own package identity, use an exact Git source or publish it to the registry. Keep the alias stable while changing the locator; callers and dependency explanations continue to use the alias.

```toml
[dependencies]
helper = { git = "https://github.com/acme/helper.git", tag = "v1.4.0", version = "^1.4" }
```

Run `runmat package resolve`, inspect `runmat package tree`, verify important choices with `runmat package why`, and commit `runmat.lock`. Use `runmat package update` only when intentionally selecting newer mutable versions.

## Declare Platform Requirements

Pure MATLAB packages generally need no capability declaration. Packages that require native libraries, MEX, JVM, subprocesses, WebGPU, browser filesystem, network, workers, or shared memory should declare the requirement so unsupported hosts reject the dependency during resolution with the selecting dependency path.

Split optional host integration behind features or target-specific dependencies. This lets the pure core resolve and run in WASM while native wrappers remain explicit:

```toml
[capabilities]
optional = ["native-library"]

[target.'capability:native-library'.dependencies]
native_bridge = { package = "acme/native-bridge", version = "^1", optional = true }
```

## Validate The Migration

For a representative workflow:

1. Resolve and commit the graph with `runmat package resolve`.
2. Run `runmat --locked check` on entrypoints and key library functions.
3. Execute with interpreter and JIT configurations.
4. Reopen with `--frozen` after warming the cache.
5. Vendor the closure and repeat frozen execution where hermetic source control is required.
6. Open the same project in Desktop/browser, confirm the graph/source revision, navigate into dependency sources, reload, and repeat with locked offline policy.
7. Test intentional unsupported capability paths and verify they fail at resolution rather than later execution.

Large-toolbox validation should record the upstream repository and commit, source/test file counts, exact root mapping, selected clean analysis and execution probes, lock/graph/source revisions, cache restoration result, browser result, and known language compatibility exclusions. Do not claim the entire upstream test suite is supported based on one smoke function.
