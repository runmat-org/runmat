
# RunMat Parser

This crate parses MATLAB/Octave source text into a structured AST used by HIR,
the interpreter, and the JIT. It consumes tokens from `runmat-lexer` and
implements a precedence-based expression parser plus statement/control-flow,
functions, indexing, and object-oriented constructs.

## AST overview

Expressions (`Expr`):
- Numbers, strings, identifiers, `end` sentinel
- Unary ops: `+`, `-`, transpose `'`, non-conjugate transpose `.'`, logical not `~`
- Binary ops: arithmetic, element-wise, relational, logical (`&&`, `||`, `&`, `|`), ranges `a:b[:c]`
- Matrix literals `[ ... ]`, cell literals `{ ... }`
- Indexing: `A(...)`, `A[...]`, `A{...}`
- Member and method access: `obj.field`, `obj.method(args...)`
- Function calls, anonymous functions `@(x,y) expr`, function handles `@name`
- Metaclass query: `?Qualified.Name`

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

- `end` can be used as an expression sentinel (e.g., `A(5:end)`), represented as `Expr::EndKeyword`.
- Parentheses after identifiers are parsed as function calls; otherwise as indexing (to support `obj.method()(...)` chaining).
- Dotted access supports both member reads and method invocation.
- Command/function duality at statement start: `name arg1 arg2` is parsed as `FuncCall(name, [args])` when unambiguous.

## MATLAB features supported

- Variables & data types: numbers, logicals (`true`/`false` as idents), strings (single and double quoted)
- Matrix/array literals, empty `[]`, cell arrays `{}`
- Operators (all arithmetic, element-wise, relational, logical, transpose, colon)
- Statements & control flow (all listed above)
- Functions: definitions, multiple return values, anonymous functions, handles
  - Varargs: `varargin` (inputs) and `varargout` (outputs) are supported with
    MATLAB placement rules enforced: each may appear at most once and must be
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
- `cells_and_indexing.rs`, `lvalue_assign.rs`, `operators_extended.rs`, `logical_precedence.rs`
- `functions_handles.rs`, `multi_assign.rs`, `multi_output.rs`, `command_syntax.rs`
- `classdef.rs`, `classdef_minimal.rs`
- `imports_namespaces.rs`

## Notes

- Some MATLAB semantics (e.g., `varargin/varargout`, command/function duality,
  private functions, packages) are parsed syntactically as identifiers or
  standard constructs. Semantic resolution happens in HIR/type phases.
- The parser aims to accept and represent full MATLAB syntax; evaluation
  semantics (like short-circuit behavior) are enforced at later stages.

