# RunMat MATLAB Language Coverage Report

RunMat's goal is to provide a high-performance, modern alternative to MATLAB. A core part of this mission is achieving 100% compatibility with the MATLAB language syntax and semantics. This document provides a comprehensive, up-to-date report on our progress toward that goal.

This table is intended to be an exhaustive reference for users and developers. It details which features of the MATLAB language are fully implemented, partially implemented, or not yet available in RunMat. This is a living document that will be updated as the project evolves.

**Note:** This report focuses on the **language syntax and features**, not the library of built-in functions (e.g., `plot`, `fft`, `svd`). Coverage for built-in functions is documented separately in our library reference.

## Language Feature Compatibility

| Feature Category | Language Feature | RunMat Status | Notes |
| :--- | :--- | :---: | :--- |
| **Variables & Data Types** | `double` precision (default numeric type) | ✅ | Implemented as `f64`. |
| | Integer types (`int8`, `uint64`, etc.) | 🟡 | A generic `Value::Int` exists but the full range of specific integer types is not supported. |
| | `complex` numbers | ❌ | Not implemented. |
| | `logical` type (`true`, `false`) | 🟡 | Handled as `1.0` and `0.0` for control flow. A `Type::Bool` exists in HIR, but full logical array support is pending. |
| | Character arrays (`'hello'`) | ✅ | Implemented. |
| | String arrays (`"world"`) | ❌ | Double-quoted strings are not yet supported. |
| | `ans` default variable | 🟡 | Handled by the REPL, but not a core language concept in the execution engine. |
| | `global` variables | ❌ | `global` keyword is not implemented. All function variables are local. |
| | `persistent` variables | ❌ | `persistent` keyword is not implemented. |
| | **Matrices & Arrays** | | |
| | Matrix/Array literals `[ ... ]` | 🟡 | Implemented. However, horizontal concatenation `[A, B]` requires matrices to have the same number of rows. Growing arrays from an empty matrix (e.g., `v = []; v = [v, new_element]`) is not yet supported as it is in MATLAB. |
| | Empty matrix `[]` | ✅ | Implemented. |
| | **Advanced Data Types** | | |
| | Cell Arrays `{ ... }` | ❌ | Lexer recognizes `{` and `}`, but they are not supported by the parser or execution engine. |
| | Structs `s.field` | ❌ | Dot-notation for data structures is not implemented. |
| | Tables | ❌ | Not implemented. |
| | Function Handles `@(x) x^2` | ❌ | The `@` token is not recognized. |
| **Operators** | **Arithmetic** | | |
| | `+`, `-`, `*`, `/`, `\`, `^` | ✅ | All standard arithmetic operators are implemented. |
| | **Element-wise** | | |
| | `.*`, `./`, `.\`, `.^` | ✅ | All element-wise operators are implemented. |
| | `.+`, `.-` | ✅ | These are functionally identical to `+` and `-` in MATLAB and are supported. |
| | **Relational** | | |
| | `==`, `~=`, `<`, `<=`, `>`, `>=` | ✅ | All relational operators are implemented. |
| | **Logical** | | |
| | Short-circuit `&&`, `\|\|` | ❌ | Lexer recognizes tokens, but they are not implemented in the parser or engine. |
| | Element-wise `&`, `\|`, `~` (NOT) | ❌ | Lexer recognizes tokens, but they are not implemented in the parser or engine. |
| | **Other** | | |
| | Transpose `'` | 🟡 | Implemented as `UnOp::Transpose`. No distinction between complex-conjugate (`'`) and non-conjugate (`.'`) transpose. |
| | Colon operator `:` | ✅ | Fully supported for creating ranges (`1:10`) and for indexing (`A(:, 1)`). |
| **Statements & Control Flow** | `if`-`elseif`-`else`-`end` | ✅ | Fully implemented. |
| | `for`-`end` loops | 🟡 | Implemented for iterating over a range expression (e.g., `for i = 1:10`). Iterating over array columns is not yet supported. |
| | `while`-`end` loops | ✅ | Fully implemented. |
| | `switch`-`case`-`otherwise`-`end` | ❌ | Not implemented. |
| | `break` | ✅ | Fully implemented. |
| | `continue` | ✅ | Fully implemented. |
| | `return` | ✅ | Fully implemented for returning from functions. |
| | `try`-`catch`-`end` | ❌ | Not implemented. |
| **Functions** | Function definitions (`function ... end`) | ✅ | Fully implemented, including named inputs and outputs. |
| | Multiple return values `[a,b] = f()` | 🟡 | Syntax is parsed correctly, but the interpreter currently only returns the first output value. |
| | Nested functions | ❌ | Not implemented. Functions cannot currently be defined inside other functions. |
| | Anonymous functions `@(...)` | ❌ | Not implemented. |
| | `varargin` and `varargout` | ❌ | Not implemented. Functions require a fixed number of arguments. |
| | Private functions | ❌ | Module path resolution does not yet support `private` directories. |
| **Indexing & Data Access** | Array indexing with parentheses `A(...)` | ✅ | Fully supported for both vector and matrix indexing. |
| | Slicing with colon operator `A(:, 1:3)` | ✅ | Fully supported. |
| | Logical indexing `A(A > 5)` | ❌ | Not implemented. Comparison operators do not yet produce logical arrays for indexing. |
| | Linear indexing `A(idx)` | ✅ | Fully supported. |
| | Indexing to end `A(5:end)` | ❌ | `end` keyword in indexing context is not yet supported. |
| | Struct field access `data.field` | ❌ | Not implemented. |
| | Cell array content access `C{...}` | ❌ | Not implemented. |
| **Object-Oriented Programming** | `classdef` | ❌ | The entire MATLAB object-oriented system is not yet implemented. |
| | Properties, Methods, Events | ❌ | Not implemented. |
| | Handle classes `< handle` | ❌ | Not implemented. |
| | Dot-notation `obj.method()` | ❌ | Not implemented. |
| **Scripting & Syntax** | Scripts (`.m` files) | ✅ | The primary mode of execution. |
| | Single-line comments `% ...` | ✅ | Fully implemented. |
| | Block comments `%{ ... %}` | ❌ | Not implemented. |
| | Line Continuation `...` | ✅ | Fully implemented. |
| | Semicolon to suppress output | ✅ | Fully implemented. |
| | Comma to separate statements | ❌ | Commas are only supported as separators in matrix literals and function arguments. |
| | Command/function syntax duality | ❌ | Only standard function call syntax `func('arg')` is supported, not command syntax `func arg`. |
