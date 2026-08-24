# Native Compilation

`runmat compile` compiles a MATLAB-syntax `.m` entrypoint into an executable for the current host:

```bash
runmat compile simulation.m -o simulation
./simulation
```

On Windows:

```powershell
runmat compile simulation.m -o simulation.exe
.\simulation.exe
```

The executable contains native program code and a matching compiler-free RunMat runtime. It runs without the RunMat CLI, project source, or a separately installed RunMat SDK.

## Choose an execution form

RunMat can execute a project in a live session or link it into a native process image:

| Form | Compilation | Source availability during execution | Function discovery |
| --- | --- | --- | --- |
| `runmat run`, REPL, or Desktop | VM execution with adaptive JIT compilation | Project source remains available | Session lookup and runtime path changes are supported |
| `runmat compile --policy=native-specialized` | Native object linked before launch | Program source closure is fixed at compile time | Runtime builtin discovery remains available |
| `runmat compile --policy=closed-world` | Native object linked before launch | Program source closure is fixed at compile time | Runtime calls must resolve to the selected builtin bindings |

Use a live session when a program adds source paths, defines functions interactively, or selects source files at runtime. Use native compilation when the entrypoint and its project/package source closure are known before launch.

## Compilation pipeline

Native compilation starts from the compiler product used by VM and JIT execution:

```text
MATLAB-syntax source + project/package closure
                    │
                    ▼
         parser → HIR → MIR → analysis
                    │
                    ▼
          immutable executable unit
                    │
                    ▼
             verified Native IR
                    │
                    ▼
          Cranelift native object
                    │
                    ├──────────────┐
                    ▼              ▼
              program object   matching RunMat runtime archive
                    │              │
                    └──────┬───────┘
                           ▼
                     system linker
                           │
                           ▼
                 host-native executable
```

Project composition resolves entry-file functions, configured source roots, package dependencies, package and class folders, private functions, and imports. HIR and MIR preserve source bindings, requested output counts, control flow, call identities, effects, and analysis facts. Core packages those products into an immutable executable unit bound to the program and environment revision.

`runmat-native-codegen` lowers the executable unit into verified Native IR. Cranelift then emits a Mach-O object on macOS, an ELF object on Linux, or a COFF object on Windows. The system linker combines that object with the runtime archive embedded in the installed `runmat` binary.

Adaptive JIT and ahead-of-time compilation share executable-unit construction, Native IR lowering, the native ABI, runtime operations, and `runmat-native-executor`. The JIT publishes machine-code entrypoints into a running session. `runmat compile` emits relocatable objects and links them into an executable before launch.

See [Mid-Level IR (MIR)](/docs/runtime/compiler/mir) for the compiler representation and [JIT Compiler](/docs/runtime/jit) for adaptive native execution.

## Entrypoints and source closure

The command accepts a `.m` script or function entrypoint. The AOT process launcher supplies no function arguments, so a function entrypoint must have no required inputs.

RunMat composes source from:

- the entry file;
- functions declared in that file;
- directories listed under `[sources].roots` in `runmat.toml`; and
- dependencies resolved through the project and package lock.

For example:

```toml
[sources]
roots = ["src", "lib"]
```

The linked executable contains a fixed program-source closure. A runtime `addpath(...)` call can change path state but cannot add another `.m` file to an executable that has already been linked. Add stable source directories to `[sources].roots`; keep programs that load source dynamically on `runmat run`.

See [Projects](/docs/runtime/getting-started/projects) and [Module Composition](/docs/runtime/compiler/modules) for source lookup and package composition.

## Runtime composition policies

Select runtime composition with `--policy`.

### `native-specialized`

`native-specialized` is the default:

```bash
runmat compile simulation.m --policy=native-specialized -o simulation
```

RunMat compiles the retained program functions and preserves builtin discovery through the runtime registry. The linker force-loads the runtime archive so inventory-backed builtin registration remains available to the process.

This policy accepts programs whose source closure is fixed but whose builtin targets cannot all be reduced to a finite set during analysis.

### `closed-world`

`closed-world` requires a bounded program and builtin target set:

```bash
runmat compile simulation.m --policy=closed-world -o simulation
```

Each reachable catalog-backed builtin contributes a stable native binding symbol. The executable installs the selected bindings as its invocation-scoped lookup authority. Calls cannot fall back to the process-global builtin registry.

Archive extraction and platform dead stripping can omit unreferenced runtime members. Compilation returns a diagnostic when analysis finds an unknown call target, an unbounded dynamic call such as `feval`, a builtin without a canonical native binding, or an unresolved target-conditional binding. RunMat does not switch policies after such a failure.

### Reserved policies

The CLI reserves two additional names:

| Policy | Intended artifact | Current behavior |
| --- | --- | --- |
| `dynamic-runtime` | Executable with an embedded frontend and dynamic source loader | Returns a capability diagnostic |
| `portable` | Target-independent executable artifact | Returns a capability diagnostic |

Neither policy is implemented by the current host-native workflow.

## Inspect the link plan

Use `--explain-link` to print the program and runtime nodes retained by compilation:

```bash
runmat compile simulation.m --explain-link -o simulation
```

The explanation includes program functions, builtins, classes, provider and extension boundaries, runtime families, and the edge that retained each node. Reachability edges distinguish direct, finite-dynamic, and unknown targets. A closed-world explanation also lists each selected builtin binding and native symbol.

Use `--link-plan-json` to write the deterministic plan for CI or artifact inspection:

```bash
runmat compile simulation.m \
  --policy=closed-world \
  --link-plan-json build/simulation.link.json \
  -o build/simulation
```

The JSON includes the program graph, runtime archive, builtin catalog, target, policy, capabilities, reachability, and binding identities used by the compilation. An existing plan is protected; pass `--force` to replace it.

## Optimization and output

The native object supports three optimization settings:

| Setting | Cranelift policy | Use |
| --- | --- | --- |
| `none` | No optimization | Compilation and linker diagnosis |
| `size` | Optimize for speed and size | Reduce generated program-object size |
| `speed` | Optimize for speed | Default |

Select a setting with `--optimization`:

```bash
runmat compile simulation.m --optimization=size -o simulation
```

Without `-o`, the output uses the entrypoint stem in the current directory and adds `.exe` on Windows. RunMat refuses to replace an existing output unless `--force` is present. Forced replacement preserves the previous executable until the new link succeeds.

Temporary objects, the decoded runtime archive, and the linker response file are created in a private directory beside the requested output and removed after linking. Pass `--keep-temps` to retain that directory for diagnosis.

## Runtime archive and compatibility checks

Official native CLI packages embed a compressed static archive built from `runmat-aot-runtime`. The archive contains `Value`, garbage collection, runtime operations, builtin implementations, and the shared native executor. It excludes the parser, HIR/MIR lowering, static analysis, Core composition, VM, adaptive JIT policy, and native object emitter.

The archive manifest records the target, object format, native ABI, schema versions, runtime and builtin-catalog identities, archive digests, linker requirements, and supported runtime capabilities. Before linking, RunMat checks the archive against the compiler and current host. A target, ABI, schema, runtime, archive, or catalog mismatch stops compilation.

`cargo build` can produce a CLI without the embedded archive. To develop or test `runmat compile`, use the two-phase helper documented under [Native standalone runtime](/docs/runtime/development/build-system#native-standalone-runtime). The archive is an internal build product. RunMat does not expose it as a public SDK or discover it from the installation directory.

## Platforms and linkers

`runmat compile` currently produces an executable for the host that runs the compiler. Packaged native compilation covers:

| Host | Rust target | Object and executable format |
| --- | --- | --- |
| Windows x86-64 | `x86_64-pc-windows-msvc` | COFF / PE |
| macOS Intel | `x86_64-apple-darwin` | Mach-O |
| macOS Apple Silicon | `aarch64-apple-darwin` | Mach-O |
| Linux x86-64 | `x86_64-unknown-linux-gnu` | ELF |

Cross-compilation is not part of this workflow. The program object, runtime archive, system linker, ABI, and enabled native libraries must describe the host target.

RunMat searches for `cc`, `clang`, or `gcc` on macOS and Linux, and `link.exe` or `lld-link.exe` on Windows. Set `--linker PATH` or `RUNMAT_LINKER` to select a specific driver. Native libraries enabled in the RunMat build, such as HDF5 or BLAS/LAPACK, must be available through the platform's standard linker search paths.

WebAssembly hosts cannot launch Mach-O, ELF, or PE executables or allocate host-native executable memory. Browser execution continues through the portable VM and browser runtime. The reserved `portable` policy will require a separate distribution and execution contract.

See [Supported Architectures](/docs/runtime/development/supported-architectures) for packaged targets and native dependency details.

## Diagnose compilation failures

| Failure | Check |
| --- | --- |
| Entry file is rejected | `runmat compile` currently accepts `.m` entrypoints |
| A project function cannot be resolved | Add its stable directory to `[sources].roots` or its package to the project lock |
| `closed-world` reports an unknown or unbounded target | Remove the dynamic call, provide a statically resolvable target, or select `native-specialized` explicitly |
| The RunMat build has no native compile runtime | Build with the two-phase AOT runtime helper or install an official native package |
| No supported linker is found | Install a platform C/C++ toolchain or set `--linker` or `RUNMAT_LINKER` |
| The linker cannot find a native library | Install the development library used by the RunMat build and expose its standard search path |
| The output or JSON link plan already exists | Choose another path or pass `--force` |
| The linker returns an error | Read the bounded linker diagnostic and repeat with `--keep-temps` when object and response-file inspection is needed |

## Related documentation

- [Command Line Interface](/docs/runtime/getting-started/cli)
- [Projects](/docs/runtime/getting-started/projects)
- [Compilation Pipeline](/docs/runtime/compiler)
- [Module Composition](/docs/runtime/compiler/modules)
- [Mid-Level IR (MIR)](/docs/runtime/compiler/mir)
- [JIT Compiler](/docs/runtime/jit)
- [Supported Architectures](/docs/runtime/development/supported-architectures)
- [Build System](/docs/runtime/development/build-system)
