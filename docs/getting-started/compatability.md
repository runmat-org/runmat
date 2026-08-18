---
title: "MATLAB Language Compatibility"
category: "Getting Started"
section: "1.6"
last_updated: "August 18, 2026"
---

# MATLAB Language Compatibility

RunMat is a high-performance runtime designed for MATLAB-syntax code. It targets the core language grammar and semantics, enabling engineers to execute `.m` scripts, functions, and complex object-oriented systems. Compatibility focuses on the core language (variables, operators, control flow, N-D indexing, and `classdef` OOP) and a standard library of 400+ built-in functions.

RunMat is an independent runtime for MATLAB-language source code. Compatibility work is based on publicly available documentation and independent engineering.

Compatibility describes observable program behavior: accepted syntax and call forms, values, classes, shapes, indexing and assignment, side effects, warnings, and errors. It does not require RunMat to use the same internal algorithms, storage layouts, execution devices, performance characteristics, or resource limits as another runtime.

## Compatibility Target

RunMat's current compatibility version pin is **MATLAB R2026a**. Compatibility-sensitive queries use the same pin: `version("-release")` returns the release label, and `verLessThan("matlab", requiredVersion)` compares against its product version, `26.1`.

This page is the reader-facing source of truth for the pin. Individual language and builtin pages describe their observable behavior without repeating the release label, so the compatibility target can advance without scattering version-specific text throughout the documentation.

For supported functionality, RunMat follows the documented behavior of the compatibility target. It also preserves older documented call forms when they remain accepted by the target or can coexist without changing target-compatible programs. There is no blanket earliest-supported release: historical behavior is retained per feature when it is documented, useful, and nonconflicting.

A compatibility target does not imply that every toolbox, object type, or function from that release is implemented. Missing functionality is documented as a coverage limitation; it is not treated as an alternative compatibility rule or a RunMat extension.

## Compatibility Modes

RunMat provides three language-policy modes. They do not emulate different historical releases or select different numeric semantics. Configure the mode with the `compat` key in the [project configuration](/docs/runtime/getting-started/config).

RunMat-only syntax and builtin forms are documented in [MATLAB Language Extensions](/docs/runtime/getting-started/matlab-language-extensions). That page also explains why internal optimizations such as automatic GPU residency and transparent gathering are not language extensions.

| Mode | Behavior |
| --- | --- |
| `runmat` | Default. Enables supported RunMat language and builtin extensions and uses RunMat error identifiers. |
| `matlab` | Excludes behavior classified as RunMat-only and uses MATLAB-oriented error identifiers where supported. |
| `strict` | Tightens permissive syntax such as command-style calls. Calls must use explicit parenthesized syntax, such as `hold("on")` rather than `hold on`. |

Supported MATLAB-language programs should retain the same numeric values, classes, shapes, and indexing results across modes. The modes change admission policy and diagnostics, not the compatibility target.

Automatic GPU residency is an internal execution choice. A compatible operation may gather an automatically resident value transparently when host execution is needed. An explicit `gpuArray` input represents user-visible device intent and may therefore have a separately documented support or extension policy. Selecting `matlab` mode does not disable ordinary automatic gathering.

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

Unlike many alternative MATLAB syntax-based runtimes such as Octave, RunMat provides full `classdef` support.

- Properties & Methods: Supports attributes such as `Constant`, `Dependent`, `Static`, and access levels (`Private`, `Public`).
- Handle Classes: Implements identity semantics, `isvalid`, and `delete` lifecycle management.
- Events: Full `addlistener` and `notify` support integrated with the runtime event registry.

For a complete guide to writing classes, including value vs. handle classes, inheritance, operator overloading, events, enumerations, and packages, see [Classes (classdef)](/docs/runtime/classes).

## Compiler Pipeline

The compatibility layer is primarily enforced during the "Lowering" phase, where the `runmat-parser` AST is converted into `runmat-hir`, and then into a control-flow graph in the [Mid-Level IR (MIR)](/docs/runtime/compiler/mir) stage. These stages resolve identifiers based on MATLAB's complex precedence rules.

See the [compiler pipeline](/docs/runtime/compiler) for a full breakdown of how RunMat resolves and implements MATLAB language semantics.
