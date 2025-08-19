# RunMat Parser

This crate parses MATLAB/Octave source text into a structured AST used by HIR, the interpreter, and the JIT. It consumes tokens from `runmat-lexer` and implements a precedence-based expression parser plus statement/control-flow, functions, indexing, and object-oriented constructs. The parser aims to accept full MATLAB language syntax with accurate surface grammar while deferring semantic enforcement (e.g., name resolution, access checks, `end` arithmetic) to later phases.

## AST overview

Expressions (`Expr`):
- Numbers, strings, identifiers, `end` sentinel
- Unary ops: `+`, `-`, transpose `'`, non-conjugate transpose `.'`, logical not `~`
- Binary ops: arithmetic, element-wise, relational, logical (`&&`, `||`, `&`, `|`), ranges `a:b[:c]`
- Matrix literals `[ ... ]`, cell literals `{ ... }`
- Indexing: `A(...)`, `A[...]`, `A{...}`
- Member and method access: `obj.field`, `obj.method(args...)`
- Function calls, anonymous functions `@(x,y) expr`, function handles `@name`
- Metaclass query: `?Qualified.Name` (see Postfix section for chaining semantics)

Statements (`Stmt`):
- Expression and assignment (including multi-assign `[a,b]=f()`)
- Control flow: `if/elseif/else/end`, `for/end`, `while/end`, `switch/case/otherwise/end`, `try/catch/end`
- Declarations: `function ... end`, `global`, `persistent`
- Break/continue/return
- Class definitions: `classdef Name [< Super] ... end` with `properties`, `methods`, `events`, `enumeration`, `arguments` blocks
- Imports: `import pkg.*` and `import pkg.sub.Class`

## Grammar highlights

- Precedence order (high → low):
  1. Postfix (`()`, `[]`, `{}`, member `.`/method, transpose `'`/`.'`)
  2. Power `^`/`.^` (right-associative)
  3. Unary `+ - ~`
  4. Multiplicative `* / \ .*/./.\`
  5. Additive `+ -` (also handles tokenized `.+ .-` as `.` + `+/-`)
  6. Comparisons `== ~= < <= > >=`
  7. Bitwise `& |`
  8. Short-circuit `&& ||`
  9. Range `a:b[:c]` (binds after comparisons and logical ops inside endpoints)

- `end` can be used as an expression sentinel (e.g., `A(5:end)`), represented as `Expr::EndKeyword`. In command-form, a bare `end` token is accepted as a literal argument and surfaced as `Expr::Ident("end")` for compatibility.
- Parentheses after identifiers are parsed as function calls when the callee is a bare identifier; otherwise as indexing (to support `obj.method()(...)` chaining, and array indexing on expressions).
- Dotted access supports both member reads and method invocation; dynamic member `s.(expr)` is parsed where syntactically valid.
- Command/function duality at statement start: `name arg1 arg2` is parsed as `FuncCall(name, [args])` when unambiguous. See “Command-form hardening” below for disambiguation rules.

## Language features supported

- Variables & data types: numbers, logicals (`true`/`false` as idents), strings (double-quoted string scalars) and char arrays (single-quoted)
- Matrix/array literals, empty `[]`, cell arrays `{}`
- Operators (all arithmetic, element-wise, relational, logical, transpose, colon)
- Statements & control flow (all listed above)
- Functions: definitions, multiple return values, anonymous functions, handles
  - Varargs: `varargin` (inputs) and `varargout` (outputs) are supported with
    language placement rules enforced: each may appear at most once and must be
    the last parameter in its respective list
- Multi-output placeholders: `[a, ~, c] = f(...)`
- Indexing & data access: (), [], {}, slicing, `end` in indexing, struct and method access
- Object-oriented programming: `classdef`, `properties`, `methods`, `events`, `enumeration`, optional super `< handle`
- OOP attributes tolerated syntactically in blocks: e.g., `properties(Access=private)`, `methods(Static)`
- Scripting & syntax: line comments, block comments, line continuation, semicolon, comma separation

## Error handling

Produces `ParseError` with message, position, found token, and expected token hints.

## Tests

Tests live under `crates/runmat-parser/tests/` and are organized by feature:
- Core: `cells_and_indexing.rs`, `lvalue_assign.rs`, `operators_extended.rs`, `logical_precedence.rs`
- Functions & outputs: `functions_handles.rs`, `multi_assign.rs`, `multi_output.rs`
- Command-form & ambiguity: `command_syntax.rs`, `ambiguous_command_and_metaclass.rs`, `fuzz_command_dynamic.rs`, `fuzz_command_edges.rs`
- OOP & classdef: `classdef.rs`, `classdef_minimal.rs`
- Imports & namespaces: `imports_namespaces.rs`

## Metaclass (`?Class`) and postfix

- `?Qualified.Name` parses to `Expr::MetaClass("Qualified.Name")`.
- Postfix after metaclass is enabled: `?Class.prop` → `Expr::Member(MetaClass("Class"), "prop")`; `?Class.method(args...)` → `Expr::MethodCall(MetaClass("Class"), "method", [args])`.
- Heuristic for dotted consumption before postfix: consume package segments (lowercase leading) and the first class segment (uppercase leading) into the metaclass; subsequent dotted segments are treated as postfix (member/method). Examples:
  - `?pkg.sub.Class.size` → `Member(MetaClass("pkg.sub.Class"), "size")`.
  - `?Class.size` → `Member(MetaClass("Class"), "size")`.

### Lowering and runtime dispatch

- Static property/method access from metaclass is handled in later phases:
  - The compiler lowers `Member(MetaClass(c), field)` to `LoadStaticProperty(c, field)` and `MethodCall(MetaClass(c), m, args)` to `CallStaticMethod(c, m, argc)`.
  - The VM enforces access/static checks via the class registry and invokes implementations via `runmat-runtime`.

## Command-form hardening

The MATLAB language allows “command-form” calls at statement start (`name arg1 arg2`). We implement:
- Command-form triggers only when the first token is an identifier and the following run contains only simple arguments (`Ident`, numeric, string, or `end`), and is not immediately followed by `(`, `.`, `[`, `{`, or a transpose token.
- Complex LValues take precedence at statement start: `A(1)=v`, `A{1}=v`, `s.f=v`, `s.(n)=v` are captured as `AssignLValue` before considering command-form.
- Ambiguity guard: sequences like `foo b(1)` are rejected with a targeted error; users should write `foo(b(1))` or quote `b(1)`.
- Ellipsis `...` is supported across command-form lines.

### Additional adjacency cases covered by tests

- Quoted args with doubled escapes (e.g., `"he said ""hi"""`).
- `end` as an argument alongside quoted args and across ellipsis.
- Command-form rejected when a dynamic member `s.(expr)` appears in the argument run (before/after tokens, or across ellipsis).

## Imports & name resolution (semantic overview)

- Parsing accepts `import pkg.*` and `import top.mid.Class` statements.
- Precedence (enforced post-parse): locals > user functions in scope > specific imports > wildcard imports > `Class.*` statics.
- Ambiguities (between specifics, or between multiple wildcards, or between static members) are reported with clear diagnostics in compiler/HIR phases.

## Dynamic member access `s.(expr)`

- Supported within standard expression/assignment contexts and chaining; not accepted as a command-form argument (fuzz tests enforce error surfacing for such cases).

## Outstanding/edge items (tracked)

- Extend fuzz coverage for rare command-form adjacencies (deeply nested quotes, mixes with `end` and punctuation).
- Optional: support indexing postfix after metaclass if language semantics require it (e.g., class arrays of metaobjects) — currently unsupported by design.
- Keep import/namespace ambiguity matrices growing (user/builtin/Class.* statics) to prevent regressions.
- Function-level `arguments ... end`: names are accepted today; adding type/default/range validation hooks is planned in HIR/runtime.
- Classdef `enumeration`: explicit value forms are parsed structurally; richer validations (conflicts/range) can be added.

## Where semantics are enforced (beyond parsing)

- OOP attribute validation (e.g., `Static+Dependent` invalid, `Constant+Dependent` invalid; `Abstract+Sealed` invalid; access values) occurs in HIR.
- Import normalization/ambiguity detection in HIR; unqualified name resolution precedence and final static/property resolution in the compiler/VM.
- `end` arithmetic, slice semantics, and column-major broadcasting are performed by the compiler/VM and runtime.

## Implementation notes

- Precedence and associativity follow language behavior; `.+`/`.-` are handled by token lookahead (`.` then `+/-`).
- Matrix row/column separators support comma or whitespace; column-major layout is preserved in downstream representations.
- String handling: double-quoted string scalars support doubled `""` escapes; single-quoted char arrays handled lexically via contextual apostrophe logic (in the lexer) to disambiguate transpose.
- The parser stays permissive where the MATLAB language is permissive; semantic validation (e.g., OOP access checks, import resolution) occurs in HIR and compiler phases.
- Some MATLAB language semantics (e.g., `varargin/varargout`, command/function duality, private functions, packages) are parsed syntactically as identifiers or standard constructs. Semantic resolution happens in HIR/type phases.
- The parser aims to accept and represent full MATLAB language syntax; evaluation semantics (like short-circuit behavior) are enforced at later stages.