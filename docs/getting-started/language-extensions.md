# MATLAB Language Extensions

RunMat runs MATLAB-language source code and also provides capabilities beyond MATLAB's documented language and builtin surface. These additions are called language extensions. They let a program keep MATLAB syntax while using RunMat's concurrency model, typed data systems, broader numerical library, and device-aware execution.

RunMat extensions are enabled by default can be disabled by changing to the `matlab` compatability mode, as described below.

## Extensions Available in RunMat

RunMat extensions fall into five main groups. The examples below introduce the major capabilities; individual builtin pages remain the source of truth for the exact signatures, classes, outputs, and current limitations of each function.

### Async Functions, Futures, and Spawn Handles

RunMat adds `async function`, `await(...)`, and `spawn(...)` to MATLAB-language source. Calling an async function creates a lazy future; `await` resolves it. `spawn` produces a single-use handle that can be awaited later.

```matlab
async function y = loadResult()
  y = readmatrix("result.csv");
end

future = loadResult();
result = await(future);
```

Top-level `await` depends on host policy. The current `spawn` implementation resolves the future before returning its handle rather than scheduling background work. See [Async Execution](/docs/runtime/execution/async) for the execution model and current beta limits.

### RunMat-Native Data, Geometry, and Utility APIs

RunMat includes namespaced APIs designed for workflows that extend beyond an in-memory MATLAB workspace.

The `data.*`, `Dataset.*`, `DataArray.*`, and `DataTransaction.*` APIs store large typed arrays in chunks, read and write slices, commit transactional updates, and work through local or remote filesystem providers. Declared `single` and integer classes remain typed on disk and when slices are read back.

```matlab
schema.arrays.samples = struct( ...
    "dtype", "uint16", ...
    "shape", [1000000 64], ...
    "chunk", [4096 64]);

ds = data.create("training.data", schema);
samples = Dataset.array(ds, "samples");
DataArray.write(samples, { [1 4096], ":" }, batch);
```

See the [Datasets API](/docs/runtime/fs/datasets) for schemas, slicing, transactions, snapshots, import, export, and remote storage.

RunMat also provides `geometry.*` and `fea.*` for loading CAD or mesh geometry, constructing multiphysics studies, running parameter sweeps, and retaining solver evidence. The subsystem can be driven from `.m` code or RunMat's declarative `.fea` format. See [FEA on Geometry](/docs/fea) for its beta status and supported physics families.

Small RunMat-native utilities include `urlencode` and `urldecode`, which provide URL text encoding directly to MATLAB-language programs.

### Broader Numerical Classes

Many MATLAB builtins document only `single` or `double` data, or accept integers only in narrow roles. RunMat extends these functions when the input has a clear numerical meaning and the conversion or output class can be defined precisely.

RunMat provides 632 input forms across 379 builtin names that are not yet supported in MATLAB. They include integer inputs passed to floating-point statistics functions, exact integer dimensions and controls, integer coordinates in graphics, and operations that preserve integer storage. For example, RunMat accepts `uint16` observations directly in `corr` and converts them when the floating-point statistical calculation begins:

```matlab
samples = uint16(readmatrix("sensor-readings.csv"));
R = corr(samples);  % R is double
```

RunMat also supports exact integer values in sparse matrices. Their class is retained through supported sparse construction, indexing, assignment, structural operations, and conversion back to full storage:

```matlab
S = sparse(uint32([0 7; 11 0]));
A = full(S);
class(A)  % uint32
```

Other builtin extensions admit logical, character, complex, or resident inputs where the behavior is useful and unambiguous. Each input class is gated independently; support for one extended class does not imply that every other class is accepted.

### Additional Builtin Forms and Output Controls

RunMat adds signatures and options that compose naturally with the existing function. Common examples include `"like"` output prototypes, additional callable forms, scalar expansion, useful option aliases, and data shapes beyond the compatible signature.

```matlab
prototype = gpuArray.zeros(1, 1, "single");
[X, Y] = meshgrid(x, y, "like", prototype);
p = randperm(100, 10, "like", prototype);
```

Here the prototype selects output precision and residency without changing the mathematical operation. Other examples include explicit missing-value flags for `all` and `any`, text callables for selected higher-order functions, matrix and multidimensional forms in selected I/O APIs, and additional graphics selectors and aliases.

The generated builtin catalog currently records 1,274 distinct RunMat-only extension identifiers across 537 builtin names. A shared extension can appear on several related functions, and a single builtin can expose independent extensions for its data, controls, options, and device behavior.

### GPU-Aware Forms and Placement

RunMat extends selected builtins to accept explicit `gpuArray` inputs, preserve resident outputs, or use a `"like"` prototype to select a device. These forms make device placement visible in the program and are documented on the affected builtin page.

Automatic acceleration is different. JIT compilation, kernel fusion, automatic GPU residency, and transparent gathering for compatible host execution do not change the program's accepted syntax or result, so they remain available as runtime optimizations in every compatibility mode. An explicit `gpuArray` expresses user-visible device intent and may have a separate extension policy when MATLAB does not document that form.

Extensions are supported interfaces rather than permissive accidents. Builtin extensions carry stable identifiers and compatibility metadata, and RunMat checks their policy before avoidable provider access, file I/O, graphics changes, or other side effects.

## Choosing a Compatibility Mode

Set the mode in `runmat.toml`:

```toml
[runtime.language]
compat = "runmat"
```

| Mode | Extension behavior |
| --- | --- |
| `runmat` | Enables supported RunMat language and builtin extensions. This is the default. |
| `matlab` | Excludes extensions that fall outside the MATLAB compatibility target and uses MATLAB-oriented error identifiers where supported. |
| `strict` | Tightens permissive syntax such as command-style calls. It does not select a historical MATLAB release or a different numeric model. |

Compatibility mode is a policy for the whole execution request. It does not disable transparent runtime optimizations, and it does not change documented numeric results merely because RunMat uses a different execution path.

## Reading Function Documentation

Each builtin reference separates ordinary behavior from RunMat-only forms. When a call is an extension, the page identifies the affected argument or behavior and the mode required to use it. Independent extensions on the same function remain independent: enabling `runmat` mode permits the implemented forms, but one extension does not silently broaden another argument.

When `matlab` mode rejects an extension, RunMat reports a compatibility error before performing provider access, file I/O, graphics mutation, or another avoidable side effect. Stable extension identifiers in builtin metadata allow editors and other tooling to explain the rejected form.

For the broader language and builtin coverage model, including the current compatibility target, see [MATLAB Language Compatibility](/docs/runtime/matlab-compatibility). For configuration details, see the [Configuration Reference](/docs/runtime/getting-started/config).
