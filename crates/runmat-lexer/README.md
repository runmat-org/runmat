# RunMat Lexer

This crate tokenizes MATLAB/Octave source code into a stream of tokens for the parser.
It uses the `logos` library to define a fast, zero-copy DFA with a small amount of
context via `LexerExtras` to handle MATLAB-specific ambiguities.

## Design goals

- Correct tokenization for the full MATLAB language surface
- Minimal, explicit state for disambiguation (apostrophe transpose vs string, section markers, etc.)
- Compatibility with the rest of the toolchain (parser, HIR, interpreter, JIT)
- Predictable tokens: avoid over-encoding semantics at the lexing stage

## Context-aware lexing

We track two pieces of context in `LexerExtras`:

- `last_was_value: bool` — true if the previous emitted token forms a value.
  Used to disambiguate `'` as transpose vs string start.
- `line_start: bool` — true if we are at the beginning of a logical line.
  Used for `%%` section markers.

## Tokens overview

- Keywords: `function if elseif else for while break continue return end`
- Additional keywords: `switch case otherwise try catch global persistent true false`
- OOP keywords: `classdef properties methods events enumeration arguments`
- Import: `import`
- Identifiers: `[A-Za-z_][A-Za-z0-9_]*`
- Numbers: integers and floats with optional exponents
- Strings:
  - Single-quoted character arrays: `'...'` with doubled quotes `''` inside
  - Double-quoted string scalars: `"..."` with doubled quotes `""` inside
- Operators and punctuation:
  - Arithmetic: `+ - * / \ ^`
  - Element-wise: `.* ./ .\ .^`
  - Relational: `== ~= < <= > >=`
  - Logical: `&& || & | ~`
  - Transpose: `'` (contextual)
  - Colon: `:`
  - Dotted member access: `.`
  - Function handle/anonymous: `@`
  - Meta-class query: `?` (e.g., `?MyClass`)
  - Assignment and separators: `= , ;`
  - Grouping and containers: `() [] {}`
- Comments & layout:
  - Line comment: `%` to end of line
  - Section marker: `%%` at start of line
  - Block comment: `%{ ... %}` (non-nesting)
  - Line continuation: `...` (skips remainder of physical line)
  - Newlines reset `line_start`

## Notable disambiguations

- Apostrophe `'`:
  - If previous token was a value (identifier, number, `) ] }`), emit `Transpose`
  - Otherwise, let the string regex capture a full single-quoted character array
- Section `%%`:
  - Only emitted when `line_start == true`; otherwise `%` starts a normal line comment
- Line continuation `...`:
  - Emits `Ellipsis` and consumes the remainder of the physical line, including any `%` comment following it

## Non-goals at lexing time

The lexer purposefully does not encode high-level semantics:
- Integer class names like `int8`/`uint64` are identifiers
- Special variables like `varargin`/`varargout`/`ans` are identifiers
- OOP features (`handle` inheritance, method attributes) are parsed/handled later
- Command/function syntax duality is resolved in parsing/semantic phases

## Tests

See `tests/` for comprehensive coverage, organized by topic:
- `lexer.rs`: core tokens, operators, single-quoted strings, comments, ellipsis
- `transpose.rs`: detailed diagnostics and assertions for apostrophe (`'`) transpose cases
- `comments_continuation.rs`: `%` line comments, `%{...%}` block comments, `%%` section markers, `...` continuation
- `operators.rs`: logical and element-wise operators (e.g., `.* ./ .\ .^ && || & | ~`)
- `namespaces.rs`: `import` paths (including wildcard) and metaclass `?ClassName`
- `oop_tokens.rs`: OOP keywords (`classdef`, `properties`, `methods`, `events`, `enumeration`, `arguments`) and function handles `@`
- `strings_chars.rs`: double-quoted string scalars and apostrophe disambiguation exercises
- `tokens_basic.rs`: identifiers, numbers, separators (`; ,`), and simple keyword smoke tests

All lexer tests pass when running the crate tests on their own.

## Guidelines for extending the lexer

- Prefer adding new tokens only when lexical distinctions are required.
- When in doubt, keep ambiguous terms as identifiers and resolve in the parser.
- If you need context to disambiguate, add a boolean/flag in `LexerExtras` and
  use a Logos callback to `Emit` or `Skip` appropriately.
- Keep regular expressions simple (no look-around) and rely on token priority
  and callbacks for precedence and control.

## Known compatibility notes

- Non-conjugate transpose `.'` is tokenized as `Dot` then `Transpose`.
  The parser should interpret this pair as the non-conjugating transpose.
- Block comments `%{...%}` are treated as non-nesting by design.
- Error-recovery is implemented to keep producing useful tokens after invalid input; in recovery mode
  double-quoted strings are recognized as a single `Str` token, while malformed single-quoted sequences may
  be split to allow downstream error reporting.

## Crate integration

- This crate only produces tokens; it does not attempt to validate grammar.
- Downstream crates (`runmat-parser`, `runmat-hir`, `runmat-ignition`, `runmat-turbine`) are responsible for structure and semantics.
