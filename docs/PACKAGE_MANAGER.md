# RunMat Package Manager (Draft)

Status: Draft — this document describes the design we are implementing. CLI subcommands exist as stubs today and will be enabled incrementally.

RunMat's package manager makes the core small and everything else composable. It supports:

- Native packages (Rust): high-performance built-ins implemented in Rust using `#[runtime_builtin]` macros.
- Source packages (MATLAB): `.m` packages compiled or interpreted by the runtime.
- Registries, semver, dependency resolution, and a lockfile for reproducibility.
- Runtime doc generation from package metadata.

The goal is a tiny, stable “waist” (Value/Type/ABI, error model, doc metadata) so packages can evolve independently of the core.

---

## Why a package system?

- Keep the core runtime lean and stable.
- Enable domain experts to ship features without forking the runtime.
- Make it easy to mix Rust and MATLAB code in one environment.
- Ensure reproducibility and discoverability via registries.

This mirrors what works for modern ecosystems (crates.io, npm, PyPI, Julia Pkg) and avoids tying product velocity to core releases.

---

## Concepts and artifacts

- `.runmat` — project config (TOML). Declares dependencies and registries. See `docs/CONFIG.md`.
- `runmat.lock` — lockfile with resolved versions and sources.
- Registries — default `https://packages.runmat.org` plus user registries.
- Packages — either native (Rust) or source (MATLAB). Both carry metadata and docs.

### Package spec (declared in `.runmat`)

```toml
[packages]
# registry package
linalg-plus = { source = "registry", version = "^1.2" }
# git repo
viz-tools   = { source = "git", url = "https://github.com/acme/viz-tools", rev = "main" }
# local path during development
my-local    = { source = "path", path = "../my-local" }

[packages.registries]
# default registry is implicit, but additional registries can be provided in config
```

The schema is defined in `runmat/src/config.rs` (`PackagesConfig`, `PackageSpec`).

---

## Native packages (Rust)

Native packages use stable runtime interfaces and macros to register built-ins. They compile to dynamic libraries that RunMat loads at startup.

### Minimal crate layout

```
my_native_pkg/
├─ Cargo.toml
└─ src/lib.rs
```

### `Cargo.toml`

```toml
[package]
name = "my_native_pkg"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
runmat-builtins = { path = "../../crates/runmat-builtins" }
runmat-macros   = { path = "../../crates/runmat-macros" }
runmat-gc-api   = { path = "../../crates/runmat-gc-api" }
```

### `src/lib.rs`

```rust
use runmat_macros::runtime_builtin;
use runmat_builtins::{Value, Tensor};

#[runtime_builtin(
    name = "norm2",
    category = "math/linalg",
    summary = "Euclidean norm of a vector.",
    examples = "n = norm2([3,4])  % 5",
    keywords = "norm,l2,euclidean",
)]
fn norm2_builtin(a: Value) -> Result<Value, String> {
    let t: Tensor = (&a).try_into()?; // 1-D or 2-D vector accepted
    let s = t.data.iter().map(|x| x * x).sum::<f64>().sqrt();
    Ok(Value::Num(s))
}
```

That's it. The macro infers the type signature for docs, registers the builtin via `inventory`, and ensures errors are surfaced consistently.

### Building and testing locally

```sh
cargo build -p my_native_pkg
# RunMat will load cdylibs from the resolved package set at startup
```

During development, declare your package in the project's `.runmat` as a `path` source so RunMat loads your local build.

### Publishing (planned flow)

```sh
runmat pkg publish
```

- Validates metadata and docs.
- Uploads a platform-agnostic source artifact (crate) to the registry.
- Registry builds platform binaries (cdylib) in CI and serves them to clients.

The first release will support path/git sources and local dev; registry publishing follows shortly after.

---

## Source packages (MATLAB)

Source packages are plain `.m` packages with a manifest, compiled or interpreted by RunMat.

### Layout

```
awesome_signals/
├─ runmat.toml          # optional, or declared from project .runmat
└─ src/
   ├─ hann.m
   └─ welch_psd.m
```

### Example `hann.m`

```matlab
function w = hann(n)
  if nargin < 1
    error('MATLAB:narginchk','hann requires n');
  end
  i = (0:n-1)';
  w = 0.5 - 0.5*cos(2*pi*i/(n-1));
end
```

### Declaring the package (project `.runmat`)

```toml
[packages]
awesome-signals = { source = "path", path = "./vendor/awesome_signals" }
```

At install time RunMat can optionally compile `.m` files to bytecode caches for faster startup, or load them on demand through the interpreter.

---

## How RunMat loads packages

1. Resolve dependencies:
   - Combine `.runmat` + registry index to a concrete set of versions.
   - Write/update `runmat.lock`.
2. Build and stage artifacts:
   - Native: build cdylibs (or download registry-built artifacts) into a local package cache.
   - Source: stage `.m` (and optional bytecode) under a project cache.
3. Startup:
   - Load cdylibs and call the package's registration entry (inventory-based, version-checked ABI).
   - Add source package search paths; register doc metadata.

The “waist” is minimal: Value/Type conversions, call ABI, error model, and doc metadata. This keeps the loading protocol stable.

---

## CLI

```text
runmat pkg add <name>[@version]     # add dependency (registry)
runmat pkg remove <name>            # remove dependency
runmat pkg install                  # resolve, build/fetch, stage
runmat pkg update                   # upgrade per semver ranges
runmat pkg publish                  # publish native/source package
```

Current state: subcommands print a “coming soon” message; resolution/build/staging are implemented incrementally behind the scenes.

---

## Examples

### 1) Add a Rust builtin

```rust
// crates/my_stats/src/lib.rs
use runmat_macros::runtime_builtin;
use runmat_builtins::{Value};

#[runtime_builtin(
    name = "zscore",
    category = "stats",
    summary = "Standardize data to zero mean and unit variance.",
    examples = "z = zscore([1,2,3])",
)]
fn zscore_builtin(a: Value) -> Result<Value, String> {
    let t: runmat_builtins::Tensor = (&a).try_into()?;
    let n = t.data.len().max(1);
    let mean = t.data.iter().sum::<f64>() / n as f64;
    let var = t.data.iter().map(|x| (x-mean)*(x-mean)).sum::<f64>() / n as f64;
    let std = var.sqrt().max(1e-12);
    let out: Vec<f64> = t.data.iter().map(|x| (x-mean)/std).collect();
    Ok(Value::Tensor(runmat_builtins::Tensor::new(out, t.shape).unwrap()))
}
```

Project `.runmat`:

```toml
[packages]
my-stats = { source = "path", path = "./crates/my_stats" }
```

Run:

```sh
runmat pkg install
runmat -e "z = zscore([1,2,3])"
```

### 2) Add a MATLAB builtin in a source package

```
my_signal_pkg/
├─ src/hilbert.m
└─ runmat.toml
```

`runmat.toml` (optional metadata):

```toml
[package]
name = "my_signal_pkg"
version = "0.1.0"
```

Project `.runmat`:

```toml
[packages]
my-signal-pkg = { source = "path", path = "./vendor/my_signal_pkg" }
```

---

## Versioning, features, and lockfile

- Packages use semver. Project `.runmat` specifies ranges; `runmat.lock` pins exact versions.
- Features are opt-in flags defined by the package (e.g., `features = ["gpu"]`).
- Reproducibility: CI runs with `runmat.lock` checked in.

---

## Security and safety

- Native code runs with the host process privileges; treat it like any native library.
- Guidelines: no unsafe FFI without justification, no long-running global constructors, respect the runtime's error model.
- Registry policy will include human review for new native packages and automated checks.

---

## Roadmap

- Alpha: path + git sources, local build/stage, docs integration.
- Beta: registry read, lockfile format 1, partial publish.
- GA: registry write (publish), prebuilt artifacts, delta updates, perf metrics, signed indices.

---

## FAQ

- “Will my MATLAB toolbox work unchanged?” If it's pure `.m`, likely with small edits. If it calls a large number of niche built-ins, those calls will need to be resolved by a package that provides compatible functions.
- “How fast are native packages?” As fast as you can write Rust (plus JIT acceleration for mixed workloads). Many built-ins will be memory-bound.
- “How do docs appear?” The macro metadata + exporter pipeline adds your functions to the docs automatically.
- “What about GPU?” Use the Accelerate provider API from native packages; source packages can call into GPU-enabled built-ins provided by native packages.

This document will evolve as we ship the package manager. Feedback and proposals welcome.
