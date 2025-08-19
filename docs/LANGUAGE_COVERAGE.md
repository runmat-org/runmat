# RunMat MATLAB Language Coverage Report

RunMat’s goal is to provide a high-performance, modern alternative to MATLAB, with complete language grammar and semantics. This document tracks language coverage in RunMat and compares it to GNU Octave where helpful. It focuses on core language features (syntax and semantics), not breadth of numeric libraries or toolboxes.

Legend: ✅ = fully supported, 🟡 = partial, ❌ = not supported

Note: Octave status is approximate and refers to current mainstream Octave behavior; some areas are evolving and/or implemented differently than MATLAB.

## Language Feature Compatibility

| Feature Category | Language Feature | RunMat Status | Octave Status | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Matrices & Arrays** | Literals `[ … ]` | ✅ | ✅ | Rectangular and ragged cell literals supported; numeric tensors support rectangular construction. |
| | Empty `[]` | ✅ | ✅ | Shape‑consistent behavior. |
| **Advanced Data Types** | Cell arrays `{ … }` | ✅ | ✅ | `{}` indexing, expansion, assignment; expansion into slice targets. |
| | Structs `s.field` | ✅ | ✅ | Dot access, nested assignment, field scatter over cells. |
| | Function handles `@(x)x^2`, closures | ✅ | ✅ | Anonymous functions, free‑var capture, `feval`, nested functions. |
| **Operators** | Arithmetic `+ - * / \ ^` | ✅ | ✅ | Left division and element‑wise left division supported. |
| | Element‑wise `.* ./ .\ .^` | ✅ | ✅ | Broadcasting via slice rules; BLAS/LAPACK in runtime where applicable. |
| | Relational `== ~= < <= > >=` | ✅ | ✅ | Element‑wise on arrays; scalar fallbacks. |
| | Logical element‑wise (&, &#124;, ~) | ✅ | ✅ | Element‑wise logicals on numeric/logical masks. |
| | Logical short‑circuit (&&, &#124;&#124;) | ✅ | ✅ | Semantics match MATLAB short‑circuit rules. |
| | Transpose `'`, non‑conjugate `.'` | ✅ | ✅ | Distinction modeled; `'` is conjugate transpose for complex and `.'` is non‑conjugate; identical for real inputs. |
| | Colon `:` (ranges) | ✅ | ✅ | Construction and indexing; `start:step:end` with step validation. |
| **Statements & Control Flow** | `if/elseif/else/end` | ✅ | ✅ | Full semantics. |
| | `for` loops | ✅ | ✅ | Range iteration; standard MATLAB semantics. |
| | `while` loops | ✅ | ✅ | Full semantics. |
| | `switch/case/otherwise/end` | ✅ | ✅ | Parser and VM supported. |
| | `break`, `continue`, `return` | ✅ | ✅ | Full semantics. |
| | `try/catch/end`, `rethrow` | ✅ | ✅ | `MException` objects with identifiers/messages. |
| **Functions** | Definitions (`function … end`) | ✅ | ✅ | Named inputs/outputs, nested functions, closures. |
| | Multiple returns `[a,b]=f()` | ✅ | ✅ | Multi‑LHS with placeholders (`~`), shape semantics enforced at runtime. |
| | Anonymous functions `@(...)`, closures | ✅ | ✅ | Free‑var capture, closure creation, handles. |
| | `varargin` / `varargout` | ✅ | ✅ | Cell packing/unpacking; MATLAB‑compatible arity checks with error ids (`TooManyInputs`, `VarargoutMismatch`). |
| | `nargin` / `nargout` | ✅ | ✅ | MATLAB‑parity dynamic per‑call counts, including multi‑output calls. |
| **Indexing & Data Access** | `A(...)` numeric indexing | ✅ | ✅ | N‑D, 1‑D linear, mixed selectors. |
| | Slicing `A(:, 1:3)` | ✅ | ✅ | MATLAB‑parity N‑D gather/scatter with broadcast; 2‑D fast paths. |
| | Logical indexing `A(A>5)` | ✅ | ✅ | MATLAB‑parity masks, mixed selectors, and assignment semantics. |
| | `end` in indexing | ✅ | ✅ | MATLAB‑parity `end`, `end-k`, and N‑D `end` arithmetic across dims. |
| | Struct field access `s.f` | ✅ | ✅ | With field scatter over cells. |
| | Cell content `C{...}` | ✅ | ✅ | Indexing and comma‑list expansion. |
| | Function/cell expansion into slice targets | ✅ | 🟡 | MATLAB‑compatible expansion into arbitrary slices using packing (`PackToRow/Col`); Octave behavior varies by construct. |
| **Object‑Oriented Programming** | `classdef` | ✅ | 🟡 | Full parser + runtime registry (attributes, methods/properties); Octave’s classdef support is partial. |
| | Properties/Methods (static/instance), attributes | ✅ | 🟡 | MATLAB‑parity access control, `Dependent`, static props/methods; Octave coverage is limited. |
| | Enumerations | ✅ | 🟡 | Parser + registration; MATLAB‑parity execution supported. |
| | Events | 🟡 | 🟡 | Basic scaffolding; advanced event semantics TBD. |
| | Handle classes `< handle` | 🟡 | 🟡 | Handle‑like object semantics modeled; full MATLAB handle graph semantics TBD. |
| | Operator overloading | ✅ | 🟡 | `plus`, `mtimes`, relational/logical dispatch with MATLAB‑parity precedence; PICs planned in JIT. |
| | Dot/method `obj.method()` | ✅ | ✅ | Instance and static dispatch; precedence with imports. |
| **Packages, Imports & Name Resolution** | `import pkg.*` / `import pkg.name` | ✅ | 🟡 | MATLAB‑parity precedence and diagnostics; Octave support varies. |
| | Metaclass operator `?pkg.Class` | ✅ | ❌ | Enables static property/method access and participates in `Class.*` import resolution; not available in Octave. |
| | Resolution precedence (locals > user > specific > wildcard > Class.*) | ✅ | 🟡 | Exact MATLAB precedence; Octave differs in several cases. |
| **Scripting & Syntax** | Scripts (`.m`) | ✅ | ✅ | Full support. |
| | `%` comments | ✅ | ✅ | Single‑line comments. |
| | Block comments `%{ … %}` | ✅ | ✅ | Block comment parsing. |
| | Line continuation `...` | ✅ | ✅ | Full support. |
| | Semicolon to suppress output | ✅ | ✅ | Full support. |
| | Comma to separate statements | ✅ | ✅ | Statement sequencing. |
| | Command‑form calls `func arg1 arg2` | ✅ | ✅ | Hardened parser (ambiguous cases resolved) with MATLAB‑compatible rules. |
| **Exceptions & Errors** | `MException`, identifiers/messages | ✅ | 🟡 | MATLAB‑compatible `MException` identifiers/messages across indexing, arity, and OOP; Octave identifiers differ. |
| **Variables & Data Types** | Default numeric `double` | ✅ | ✅ | Column‑major numeric tensors (`f64`). |
| | Character arrays `'...'` | ✅ | ✅ | Char row vectors implemented. |
| | String arrays `"..."` | ✅ | 🟡 | RunMat `StringArray` with MATLAB‑parity indexing/comparison; Octave’s string type coverage varies by version. |
| | `ans` default variable (REPL) | ✅ | ✅ | Handled by REPL. |
| | `global` variables | ✅ | ✅ | Name‑based, cross‑function binding; write‑through semantics. |
| | `persistent` variables | ✅ | ✅ | Per‑function lifetime, name‑ and slot‑based restore. |
| | Logical scalars/arrays | ✅ | ✅ | First‑class logical type: scalar `bool` and N‑D `LogicalArray`. `islogical`, `class`, `isa`, `size`, `numel`, `isempty` follow MATLAB; predicates (`isnan`, `isfinite`, `isinf`, comparisons) produce logical arrays; masks are preserved in indexing and assignment. |
| | Integer scalars (`int8`…`uint64`) | ✅ | 🟡 | Per‑width scalar types (`int8`…`uint64`) (not a single platform `int`) with `class`/`isa` parity and numeric ops routed through MATLAB‑compatible paths; integer arrays are planned. |
| | Complex numbers | ✅ | ✅ | Complex scalars and arrays with arithmetic, comparisons, transpose (`'` conjugate, `.'` non‑conjugate), element‑wise power, and matrix power (integer exponents). |
 
### Totals (language features)

- RunMat: ✅ 49, 🟡 5, ❌ 0 (complex numbers complete)
- Octave: ✅ 39, 🟡 9, ❌ 6 (notably: metaclass `?Class`, full classdef features, some precedence cases)

## Notes on semantics parity
 
- N‑D indexing/slicing: RunMat implements gather/scatter with broadcast rules, logical masks, colon, and `end` arithmetic across dimensions; 2‑D fast paths for entire rows/columns are included. Error identifiers match MATLAB (`MATLAB:SliceNonTensor`, `MATLAB:IndexStepZero`, etc.).
- Multi‑LHS and expansion: `[a,b]=f()` and `[~,b]=f()` are supported; function and cell expansions into slice targets are handled with dynamic packing (`PackToRow/PackToCol`).
- Varargs and counts: `varargin`/`varargout` with strict count checks and `nargin`/`nargout` report per‑call counts consistently across single and multi‑output call sites.
- OOP: Static/instance members, operator overloading (`plus`, `mtimes`, relational/logical) and `subsref`/`subsasgn` dispatch ordering with standardized negative errors (`MATLAB:MissingSubsref`, `MATLAB:MissingSubsasgn`).
- Imports and precedence: Specific imports shadow wildcard; locals and user functions take precedence; `Class.*` static resolution participates last; ambiguities are surfaced with clear diagnostics.
- Metaclass: `?pkg.Class` produces a class reference enabling static property/method access and works with import resolution.

## Nuanced examples (advanced features)

### 1) Varargin/varargout with precise output arity
```matlab
function [a, b, varargout] = head_tail(varargin)
  if nargin < 1
    error('MATLAB:NotEnoughInputs', 'Need at least one input');
  end
  a = varargin{1};
  if nargin >= 2, b = varargin{2}; else, b = 0; end
  varargout = varargin(3:end);
end

% Calls
[x, y] = head_tail(10, 20);           % x=10, y=20
[x, y, c, d] = head_tail(1, 2, 3, 4); % varargout -> c=3, d=4
```

### 2) Expansion into slice targets (functions and cells)
```matlab
function [r1, r2, r3] = triple(x)
  r1 = x; r2 = 2*x; r3 = 3*x;
end

A = zeros(5,3);
J = [1,3,2];
A(:,J) = triple(7);  % expand into full columns in order J

C = {11, 22, 33};
A(2,:) = C{:};      % expand cell contents across a row slice
```

### 3) Classdef with static/instance, dependent props, and operator overloads
```matlab
classdef Point
  properties(Dependent)
    r
  end
  properties
    x; y;
  end
  methods
    function obj = Point(x,y), obj.x=x; obj.y=y; end
    function v = get.r(obj), v = hypot(obj.x, obj.y); end
    function z = plus(a,b), z = Point(a.x+b.x, a.y+b.y); end
  end
  methods(Static)
    function o = origin(), o = Point(0,0); end
  end
end

p = Point(3,4);     % p.r == 5
q = Point.origin(); % static method
z = p + q;          % operator overloading via plus
```

### 4) Metaclass operator and imports precedence
```matlab
import mypkg.Point.*   % wildcard
import mypkg.utils.norm2  % specific import takes precedence

mc = ?mypkg.Point;     % metaclass for static access
v = mc.origin();       % static call via metaclass
```

### 5) Try/catch and standardized MException
```matlab
try
  A = zeros(2,2);
  A(:, 0) = 1;  % invalid subscript
catch e
  assert(strcmp(e.identifier, 'MATLAB:IndexOutOfBounds'));
  rethrow(e);   % rethrow preserves identifier/message
end
```

### 6) Globals and persistents (name binding + lifetime)
```matlab
function setg(v)
  global G; G = v;   % write-through, visible to other functions
end

function y = counter()
  persistent k;
  if isempty(k), k = 0; end
  k = k + 1; y = k;
end
```

## Where RunMat intentionally goes beyond MATLAB

- GPU‑native tensor execution: When a device provider is registered, element‑wise ops and matrix multiply execute on the GPU and return opaque `GpuTensorHandle`s, keeping data resident on device across chained ops. CPU paths remain available with identical MATLAB‑compatible semantics.
- Automatic device planning (in progress): A planner is being integrated to pick device placement based on tensor sizes/shapes and fuse compatible operations to minimize host↔device transfers, with room for overlapping compute/transfer via streams and memory pools.
- Zero‑temporary slice expansion: Expansion of function outputs and cell contents into slice targets writes directly to the destination using dynamic packing, avoiding intermediate temporaries and reducing peak memory for large assignments.
- HIR‑powered developer tooling: The HIR/type system tracks value kinds and tensor shapes across control flow and imports, enabling richer LSP features (hover types, shape/arity hints, precise go‑to‑definition across packages/classes, property completions from the class registry).
- Deterministic import diagnostics: MATLAB‑parity import precedence with explicit, readable diagnostics for shadows/ambiguities (including participation of `Class.*` statics) improves debugging and code navigation.
- Stable, testable exceptions: A uniform `MException` model with stable identifiers/messages across indexing, arity, expansion, and OOP paths makes failures easy to assert in tests and consistent across releases.
- Memory safety by construction: Rust implementation eliminates entire classes of memory bugs while sustaining high performance under heavy workloads.

## Where RunMat intentionally goes beyond Octave

- Metaclass operator `?Class` and static access through `Class.*` imports.
- Consistent precedence for specific vs wildcard imports, including `Class.*` statics.
- N‑D `end` arithmetic across dimensions in both gather and scatter with broadcast‑correct semantics and 2‑D fast paths.
- Function/cell expansion directly into slice targets with dynamic packing without intermediate temporaries.
- Uniform `MException` identifier/message model across indexing, arity, expansion, and OOP error paths.

If you notice any discrepancy with MATLAB semantics, please open an issue with a minimal reproducer so we can add a conformance test.
