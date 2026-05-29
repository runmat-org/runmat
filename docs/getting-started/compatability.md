---
title: "MATLAB Language Compatability"
category: "Getting Started"
section: "1.3"
last_updated: "May 28, 2026"
---

# MATLAB Language Compatability

RunMat is a high-performance runtime designed for MATLAB-syntax code. It targets the core language grammar and semantics, enabling engineers to execute `.m` scripts, functions, and complex object-oriented systems without a license. Compatibility focuses on the core language (variables, operators, control flow, N-D indexing, and `classdef` OOP) and a standard library of 400+ built-in functions.

## Compatibility Modes

RunMat provides three distinct compatibility modes to balance parity with MATLAB's legacy behaviors and modern execution strictness. These are configured via the `compat` key in the [project configuration](/docs/runtime/getting-started/config).


| Mode   | Behavior                                                                                                                                |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| runmat | Default. Accepts MATLAB command syntax (e.g., hold on) but uses RunMat-specific error namespaces (e.g. `RunMat:UndefinedFunction`)                |
| matlab | Overrides error identifiers to use the MATLAB: prefix for exact parity in try/catch blocks (e.g. `MATLAB:UndefinedFunction`), and disables extended features like `async` and `isolated` function modifiers |
| strict | Disables command-style implicit calls. All function calls must use explicit parenthesized syntax f(x) (e.g. `hold("on")` rather than `hold on`) |


## Language Feature Coverage

RunMat implements the core grammar of the MATLAB language, moving from raw source to a High-Level IR (HIR) that preserves MATLAB's unique scoping and resolution rules. See the [compiler pipeline](/docs/runtime/compiler) for more details.

### Core Syntax & Semantics

| Category | Support |
| :--- | :--- |
| Variables & data types |`double`, `single`, char arrays, string arrays, logicals, integers (`int8`…`uint64`), complex numbers, `global`, `persistent` |
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

### Object-Oriented Programming (classdef)

Unlike many alternative MATLAB syntax-based runtimes, RunMat provides full `classdef` support.

- Properties & Methods: Supports attributes such as `Constant`, `Dependent`, `Static`, and access levels (`Private`, `Public`).
- Handle Classes: Implements identity semantics, `isvalid`, and `delete` lifecycle management.
- Events: Full `addlistener` and `notify` support integrated with the runtime event registry.

## Compiler Pipeline

The compatibility layer is primarily enforced during the "Lowering" phase, where the `runmat-parser` AST is converted into `runmat-hir`. This stage resolves identifiers based on MATLAB's complex precedence rules.

See the [compiler pipeline](/docs/runtime/compiler) for a full breakdown of the compiler pipeline.
