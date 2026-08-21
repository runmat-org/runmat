# MATLAB Language Compatibility

RunMat is a high-performance runtime designed for MATLAB-syntax code. It targets the core language grammar and semantics of the language, enabling engineers to execute `.m` scripts, functions, and complex object-oriented systems.

## Compatibility Target

RunMat's current compatibility version pin is **MATLAB R2026a**. Compatibility-sensitive queries use the pin: `version("-release")` returns the release label, and `verLessThan("matlab", requiredVersion)` compares against its product version, `26.1`.

This page is the documentation source of truth for the current pin.

For supported functionality, RunMat follows the documented behavior of the compatibility version pin. It also preserves older documented call forms when they remain accepted by the target or can coexist without changing target-compatible programs. There is no blanket earliest-supported release: historical behavior is retained per feature when it is documented, useful, and nonconflicting.

A compatibility target does not imply that every toolbox, object type, or function from that release is implemented. Missing functionality is documented as a coverage limitation; it is not treated as an alternative compatibility rule or a RunMat extension.

## Compatibility Modes

RunMat provides three language-policy modes. Configure the mode with the `compat` key in the [project configuration](/docs/runtime/getting-started/config).

RunMat mode extensions are documented in [MATLAB Language Extensions](/docs/runtime/getting-started/language-extensions).

| Mode | Behavior |
| --- | --- |
| `runmat` | Default. Enables supported RunMat language and builtin extensions and uses RunMat error identifiers. |
| `matlab` | Excludes behavior classified as RunMat-only and uses MATLAB-oriented error identifiers where supported. |
| `strict` | Tightens permissive syntax such as command-style calls. Calls must use explicit parenthesized syntax, such as `hold("on")` rather than `hold on`. |

Programs should retain the same numeric values, classes, shapes, and indexing results across modes.

Automatic GPU residency is an internal execution choice. Operation may gather an automatically resident value transparently when host execution is needed. An explicit `gpuArray` input represents user-visible device intent and may therefore have a separately documented support or extension policy. Selecting `matlab` mode does not disable ordinary automatic gathering.

## Language Feature Coverage

RunMat implements the core grammar of the MATLAB language, moving from raw source to a High-Level IR (HIR) that preserves MATLAB's unique scoping and resolution rules. See the [compiler pipeline](/docs/runtime/compiler) for more details.

### Core Syntax & Semantics

| Category | Support |
| :--- | :--- |
| Variables & data types |`double`, `single`, char arrays, string arrays, logicals, integers (`int8` to `uint64`), complex numbers, `global`, `persistent` |
| Operators | Arithmetic, element-wise, relational, logical (element-wise and short-circuit), transpose (`'` and `.'`), colon ranges |
| Control flow | `if/elseif/else`, `for`, `while`, `switch/case/otherwise`, `break`, `continue`, `return`, `try/catch/end`, `rethrow` |
| Functions | Named functions, multiple returns (`[a,b]=f()`), anonymous functions with closures, `varargin`/`varargout`, `nargin`/`nargout` |
| Indexing & slicing | N-D numeric indexing, logical indexing, `end` arithmetic, struct field access, cell content indexing, function/cell expansion into slice targets |
| OOP (`classdef`) | Properties (including `Dependent`), methods (static/instance), events (`addlistener`/`notify`), handle classes, enumerations, operator overloading, metaclass operator `?Class` |
| Packages & imports | `import pkg.*`, `import pkg.name`, MATLAB-parity precedence (locals > user > specific > wildcard > `Class.*`) |
| Scripting & syntax | `.m` scripts, `%` and `%{ %}` comments, line continuation `...`, semicolon suppression, command-form calls |
| Exceptions | `MException` with MATLAB-compatible identifiers and messages across indexing, arity, and OOP error paths |

### Advanced Indexing

RunMat implements a robust indexing subsystem that handles N-D numeric and logical indexing, linear indexing, and `end` arithmetic. For details, see the [indexing subsystem](/docs/runtime/vm/indexing) documentation.

- Expansion: Supports function and cell expansion into slice targets with dynamic packing.
- L-Value Handling: The HIR lowering stage distinguishes between standard assignments, indexed assignments (`A(1)=2`), and cell assignments (`C{1}=3`).

### Functions

RunMat implements a MATLAB-compatible function model, from simple named functions to nested functions that share their parent's scope.

- Inputs & outputs: Multiple return values, optional arguments via `nargin`/`nargout`, and variable arity with `varargin`/`varargout`.
- Handles & closures: Anonymous functions with capture-by-value closures, function handles, and higher-order builtins like `arrayfun` and `cellfun`.
- Validation: `arguments` blocks with size, class, default, and `mustBe*` validators.

For a complete guide to writing functions, including nested functions, closures, function handles, persistent and global state, and argument validation, see [Functions](/docs/runtime/functions).

### Object-Oriented Programming (classdef)

Unlike Octave, RunMat provides full `classdef` support for the MATLAB language:

- Properties & Methods: Supports attributes such as `Constant`, `Dependent`, `Static`, and access levels (`Private`, `Public`).
- Handle Classes: Implements identity semantics, `isvalid`, and `delete` lifecycle management.
- Events: Full `addlistener` and `notify` support integrated with the runtime event registry.

For a complete guide to writing classes, including value vs. handle classes, inheritance, operator overloading, events, enumerations, and packages, see [Classes (classdef)](/docs/runtime/classes).

## Projects and Execution

Language compatibility extends across the project layouts and execution environments supported by RunMat.

| Area | Current capability |
| :--- | :--- |
| Built-in functions | More than 1,200 documented functions across numerical computing, data, plotting, signal processing, file I/O, and engineering workflows. Consult the [generated built-in function reference](/docs/reference/builtins) for supported functions and call forms. |
| Multi-file projects | Source roots, cross-file resolution, MATLAB-style `+pkg` packages, class folders, private functions, local dependencies, test directories, and named entrypoints. |
| Execution | Run the same source from the CLI, Desktop, or web browser, subject to the capabilities of the selected host. |
| Acceleration | Supported operations can use automatic GPU acceleration without requiring separate GPU code. Automatic acceleration preserves the compatibility rules described on this page. |

Use [`runmat.toml`](/docs/runtime/getting-started/config) to define source roots, local dependencies, test directories, compatibility policy, and named entrypoints. See [Projects](/docs/runtime/getting-started/projects) for the supported project layout.

## Check, Run, and Test Existing Code

Validate an existing project in three stages:

```bash
runmat check analysis.m
runmat run analysis.m
runmat test
```

`runmat check` uses the parser, static analysis, source lookup, and compile validation used by RunMat's editor tooling. It does not execute the program. A clean result means that no static blocker was found.

`runmat run` exercises runtime behavior, external data, files, and integrations reached by the selected entrypoint. Use representative inputs when numerical results or code paths depend on input data.

`runmat test` discovers and executes MATLAB-style script, function, and class tests. Existing project tests provide the strongest project-specific evidence of compatibility for the behavior they exercise. See the [Command Line Interface](/docs/runtime/getting-started/cli) for command options and project test configuration.

## Compatibility Testing

RunMat's codebase contains more than 18,000 automated tests across language semantics, built-in call forms, execution engines, project composition, plotting, filesystem behavior, browser and WebAssembly bindings, and CPU/GPU result parity. Compatibility-focused tests cover successful results as well as accepted inputs, output classes and shapes, warnings, and errors.

The implementation and test infrastructure are public:

- Browse the [generated built-in function reference](/docs/reference/builtins).
- Inspect the [RunMat source and tests](https://github.com/runmat-org/runmat).
- Review the [continuous-integration checks](https://github.com/runmat-org/runmat/blob/main/.github/workflows/ci.yml).
- Read the [testing strategy](/docs/runtime/development/testing).

If supported behavior differs from the compatibility target, [report the difference as a GitHub issue](https://github.com/runmat-org/runmat/issues).

## Current Boundaries

Projects may require changes or a separate MATLAB workflow when they depend on:

- a specialized MathWorks toolbox outside RunMat's current built-in coverage;
- Simulink or graphical block-diagram models;
- a MEX extension that RunMat does not support;
- MathWorks-specific applications or proprietary file formats; or

Check the [built-in function reference](/docs/reference/builtins) for function-level coverage. Compatibility can also depend on a particular option, external dependency, data format, or execution host even when the function itself is available.

## Compiler Pipeline

The compatibility layer is primarily enforced during the "Lowering" phase, where the `runmat-parser` AST is converted into `runmat-hir`, and then into a control-flow graph in the [Mid-Level IR (MIR)](/docs/runtime/compiler/mir) stage. These stages resolve identifiers based on MATLAB's complex precedence rules.

See the [compiler pipeline](/docs/runtime/compiler) for a full breakdown of how RunMat resolves and implements MATLAB language semantics.
