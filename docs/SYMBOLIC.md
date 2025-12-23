# RunMat Symbolic Mathematics Engine

This document describes the symbolic mathematics engine integrated into RunMat, which enables MATLAB-compatible symbolic computation as a first-class value type.

## Overview

The symbolic engine allows RunMat to handle symbolic expressions as native `Value` types, enabling seamless mixing of symbolic and numeric computation. Unlike toolbox add-ons, symbolic expressions are a first-class citizen in RunMat's type system, with full integration into the runtime, built-ins, and REPL.

Symbolic expressions can be manipulated algebraically, differentiated, integrated, solved, and compiled to efficient numeric functions.

## Architecture

### Core Crate: `runmat-symbolic`

The symbolic engine is implemented in the `runmat-symbolic` crate, which provides:

- **`SymExpr`**: The core symbolic expression type, using reference-counted `Arc<SymExprKind>` for efficient cloning and structural sharing
- **`Coefficient`**: Exact rational arithmetic with fallback to floating-point for large numbers
- **`Symbol`**: Interned symbolic variables with optional attributes (real, positive, integer, nonnegative)
- **`StagedNormalizer`**: Configurable normalization pipeline for expression simplification
- **`BytecodeCompiler`**: Compiles symbolic expressions to stack-based bytecode for efficient numeric evaluation
- **`CompiledExpr`**: Represents a compiled symbolic expression with an interpreter-ready bytecode sequence

### Expression Representation

Symbolic expressions use a tree-based representation:

```rust
pub enum SymExprKind {
    Num(Coefficient),                    // Numeric constant (rational or float)
    Var(Symbol),                         // Symbolic variable
    Add(Vec<SymExpr>),                   // Sum of terms
    Mul(Vec<SymExpr>),                   // Product of factors
    Pow(Box<SymExpr>, Box<SymExpr>),    // Power expression
    Neg(Box<SymExpr>),                   // Negation
    Func(String, Vec<SymExpr>),         // Function application
}
```

### Value Integration

The `Value::Symbolic` variant integrates symbolic expressions into RunMat's type system:

```rust
pub enum Value {
    // ... other variants ...
    Symbolic(runmat_symbolic::SymExpr),
}
```

This first-class integration ensures that:
- Proper error messages when symbolic values hit numeric-only operations
- Type introspection (`class(x)` returns `"sym"`)
- Display formatting for REPL output
- Size reporting in `whos`

## Builtin Functions

### Creating Symbolic Variables

| Function | Description | Example |
|----------|-------------|---------|
| `sym(name)` | Create a symbolic variable | `x = sym('x')` |
| `sym(name, 'real')` | Create with assumption | `x = sym('x', 'real')` |
| `sym(name, 'positive')` | Create with positive assumption | `y = sym('y', 'positive')` |
| `sym(value)` | Convert numeric to symbolic | `two = sym(2)` |
| `syms('x', 'y', 'z')` | Create multiple variables | `vars = syms('x', 'y', 'z')` |

### Calculus Operations

| Function | Description | Example |
|----------|-------------|---------|
| `diff(expr, var)` | Differentiate expression | `diff(x^2, x)` → `2*x` |
| `diff(expr, var, n)` | n-th derivative | `diff(x^3, x, 2)` → `6*x` |
| `int(expr, var)` | Indefinite integral | `int(x^2, x)` → `x^3/3` |

#### Differentiation Capabilities

- **Power rule**: d/dx(x^n) = n·x^(n-1)
- **Product rule**: d/dx(u·v) = u'·v + u·v'
- **Chain rule**: Applied for transcendental functions
- **Higher-order derivatives**: Recursive application of differentiation rules

#### Integration Capabilities

- **Polynomials**: ∫x^n dx = x^(n+1)/(n+1)
- **Exponentials**: ∫e^x dx = e^x
- **Trigonometric**: ∫sin(x) dx = -cos(x), ∫cos(x) dx = sin(x)
- **Linear substitution**: ∫f(ax+b) dx = F(ax+b)/a
- **Products and basic compositions**

### Algebraic Manipulation

| Function | Description | Example |
|----------|-------------|---------|
| `simplify(expr)` | Simplify expression | `simplify(x + x)` → `2*x` |
| `expand(expr)` | Expand products/powers | `expand((x+1)^2)` → `x^2 + 2*x + 1` |
| `factor(expr)` | Factor expression | `factor(x^2 - 1)` → `(x+1)*(x-1)` |
| `collect(expr, var)` | Collect terms by power | `collect(a*x + b*x, x)` → `(a+b)*x` |

#### Factorization Methods

- **Difference of squares**: a² - b² = (a+b)(a-b)
- **Common factor extraction**: ax + ay = a(x+y)
- **Quadratic factoring**: (x-r₁)(x-r₂) where r₁, r₂ have integer coefficients

#### Collection

Collects coefficients by powers of a variable, combining like terms:
- `a*x + b*x` → `(a+b)*x`
- `x^2*a + x^2*b` → `(a+b)*x^2`

### Substitution and Solving

| Function | Description | Example |
|----------|-------------|---------|
| `subs(expr, var, val)` | Substitute value | `subs(x^2, x, 3)` → `9` |
| `solve(expr, var)` | Solve equation = 0 | `solve(x^2 - 4, x)` → `2` or `-2` |

#### Equation Solving

- **Linear equations**: ax + b = 0
- **Quadratic equations**: ax² + bx + c = 0 (via quadratic formula)

Returns the first solution found. Higher-degree polynomial solving is not yet supported.

### Compilation to Numeric Functions

| Function | Description | Example |
|----------|-------------|---------|
| `matlabFunction(expr)` | Compile to function handle | `f = matlabFunction(x^2 + 1)` |

The compiled function can be evaluated on scalar or array arguments. This enables fast numeric evaluation without interpreter overhead.

### Supported Mathematical Functions

The following functions are natively supported in symbolic expressions:

- **Trigonometric**: sin, cos, tan
- **Inverse trigonometric**: asin, acos, atan, atan2
- **Hyperbolic**: sinh, cosh, tanh
- **Exponential/Logarithmic**: exp, log (natural), log10, log2
- **Other**: sqrt, abs, sign, floor, ceil, round

These functions propagate symbolically through expressions and are differentiated using the chain rule.

## Normalization Pipeline

The `StagedNormalizer` applies a sequence of transformation passes to simplify expressions:

1. **SimplifyNeg**: Apply double negation rule, fold negative constants into numbers
2. **Flatten**: Flatten nested Add/Mul operations into flat term/factor lists
3. **MergeConstants**: Combine numeric constants (e.g., 2 + 3 → 5)
4. **RemoveIdentity**: Remove additive (x+0) and multiplicative (x*1) identities
5. **SimplifyPowers**: Apply x^0=1, x^1=x rules
6. **CollectLikeTerms**: Combine like terms (x + x → 2*x)
7. **Sort**: Sort terms in canonical order for consistency

Three built-in pipelines are available:

- `default_pipeline()`: Standard simplification (recommended for most use cases)
- `minimal()`: Just flatten and sort (lightweight)
- `aggressive()`: Includes expansion and full normalization (comprehensive)

## Compilation and Evaluation

Symbolic expressions are compiled to efficient bytecode for numeric evaluation via the standalone `BytecodeCompiler` in the symbolic engine.

### Compilation Process

When `matlabFunction(expr)` is called:

```
SymExpr → BytecodeCompiler → BytecodeOp bytecode → Symbolic VM interpreter
```

The compilation process:
1. Extracts free variables from the expression
2. Compiles the expression tree to a stack-based bytecode using `BytecodeOp` instructions
3. Stores the compiled expression in a registry
4. Returns a function handle that can be called with numeric arguments

**Key advantages:**
- Direct numeric evaluation without interpreter overhead
- Small bytecode size (12 operations vs 168+ for general Ignition)
- Efficient for expressions with limited scope (arithmetic, functions, power)
- No dependencies on HIR/Ignition compilation pipeline

### Evaluation Path

When a compiled symbolic function is called:

1. **Lookup**: Function handle is used to retrieve compiled bytecode from registry
2. **Execution**: Arguments are pushed onto a stack machine
3. **Result**: Bytecode interpreter evaluates and returns numeric result

This provides fast numeric evaluation for algebraic expressions without the overhead of general-purpose bytecode compilation.

## Error Handling

Symbolic operations return descriptive error messages for unsupported operations:

```rust
pub enum SymbolicError {
    UndefinedSymbol(String),      // Variable not defined
    InvalidOperation(String),      // Unsupported operation
    DivisionByZero,               // Detected division by zero
    NumericOverflow,              // Rational arithmetic overflow
}
```

## Symbol Attributes and Assumptions

Symbols can be created with attributes to enable smarter simplification:

```matlab
x = sym('x', 'real');           % real-valued assumption
y = sym('y', 'positive');       % positive assumption
z = sym('z', 'integer');        % integer assumption
```

Supported attributes:

- `real`: Variable is real-valued
- `positive`: Variable is strictly positive (implies real)
- `integer`: Variable is an integer
- `nonnegative`: Variable is non-negative (implies real)

Assumptions enable simplifications like:
- `sqrt(x^2)` → `x` when `x` is positive
- `abs(x)` → `x` when `x` is nonnegative

## Design Principles

1. **Self-contained**: The symbolic engine is architecturally independent, with no dependencies on RunMat's compilation pipeline (Ignition/Turbine), avoiding circular dependencies and enabling reuse in diverse contexts
2. **MATLAB-compatible**: Matches MATLAB Symbolic Toolbox behavior where possible, with clear documentation of differences
3. **Performance-focused**: Structural sharing via Arc, symbol interning, and specialized stack-based bytecode (12 operations) for fast numeric evaluation without compilation overhead
4. **Extensible**: Pluggable simplification rules and composable normalization passes

### Architectural Notes

The symbolic engine deliberately **avoids** integration with Ignition/Turbine because:

- **`runmat-runtime` is reusable**: Adding ignition/hir dependencies would force all dependents (embedded scenarios, tools, other runtimes) to pull in compilation infrastructure
- **No circular dependencies**: The expression type `Value::Symbolic` must remain decoupled from the bytecode compiler
- **Optimization strategy**: Instead of JIT, we optimize the interpreter itself (constant folding, dead code elimination, batch vectorization) for comparable performance with less complexity

## Arithmetic Operators

All arithmetic operators dispatch correctly to symbolic expressions:

| Operator | Function |
|----------|----------|
| `+` | plus |
| `-` | minus |
| `.*` | times (element-wise) |
| `*` | mtimes (matrix multiply) |
| `./` | rdivide (right divide) |
| `.\` | ldivide (left divide) |
| `.^` | power (element-wise) |
| `^` | mpower (matrix power) |
| `-x` | uminus (unary minus) |

Mixed symbolic and numeric expressions are automatically promoted to symbolic.

## Current Limitations

- **Polynomial solving**: Only linear and quadratic equations are solved analytically; cubic and higher-degree polynomials not yet supported
- **Integration**: Limited to polynomials, basic transcendental functions, and simple products with linear substitution
- **Matrix symbolic**: Not yet supported; only scalar symbolic expressions
- **Assumptions**: Basic assumptions supported but not fully propagated through all operations
- **Series expansion**: `taylor`, `series`, and limit computation not yet implemented
- **Differential equations**: `dsolve` not yet implemented

## Implementation Status

### Completed (Phase 1-5)

- ✅ Core expression representation and evaluation
- ✅ Differentiation (with power rule, product rule, chain rule)
- ✅ Substitution and simplification
- ✅ Bytecode compilation to numeric functions
- ✅ MATLAB-compatible API (`sym`, `diff`, `subs`, `simplify`)
- ✅ Algebraic operations (`expand`, `factor`, `collect`)
- ✅ Equation solving (linear and quadratic)
- ✅ Symbolic integration (polynomials and transcendentals)
- ✅ First-class `Value::Symbolic` integration
- ✅ Arithmetic operator support
- ✅ Transcendental function support

### In Progress / Future Work

- Symbolic matrix support
- Extended equation solving (cubic, quartic, systems)
- More integration techniques (substitution, parts)
- Series expansion and limits
- Differential equation solving (`dsolve`)
- Pretty printing with mathematical notation
- Automatic differentiation mode

## Testing

The symbolic engine includes:

- **24 unit tests** in `runmat-symbolic` covering expression operations, normalization, and compilation
- **40+ unit tests** in `runmat-runtime` covering all builtin symbolic functions
- Comprehensive error handling and edge case coverage

Run tests with:

```bash
cargo test -p runmat-symbolic
cargo test -p runmat-runtime symbolic
```

## Examples

### Basic Symbolic Operations

```matlab
% Create symbolic variables
x = sym('x');
y = sym('y');

% Arithmetic
expr = x^2 + 3*x + 2;
expanded = expand((x + 1) * (x + 2));  % x^2 + 3*x + 2

% Differentiation
deriv = diff(expr, x);     % 2*x + 3
deriv2 = diff(expr, x, 2); % 2
```

### Calculus

```matlab
% Integration
integral = int(x^2, x);    % x^3/3
integral_trig = int(sin(x), x);  % -cos(x)

% Solve equations
solutions = solve(x^2 - 5*x + 6, x);  % Returns 2 or 3
```

### Factorization and Simplification

```matlab
% Factor polynomial
factored = factor(x^2 - 1);           % (x+1)*(x-1)
factored2 = factor(x^2 - 4*x + 4);    % (x-2)*(x-2) or (x-2)^2

% Collect terms
collected = collect(a*x + b*x, x);    % (a+b)*x
```

### Compilation to Numeric Functions

```matlab
% Create function handle from symbolic expression
f = matlabFunction(x^2 + 3*x + 1);

% Evaluate on numeric values
result = f(5);              % 41

% Works with arrays
results = arrayfun(f, [1, 2, 3, 4, 5]);
```

### Mixed Symbolic and Numeric

```matlab
% Numeric and symbolic mix seamlessly
x = sym('x');
result = 2 * x + 5;    % Symbolic: 2*x + 5
result2 = subs(result, x, 3);  % Symbolic: 11
```

---

# Improvement Opportunities

## Current Architecture Status

The symbolic engine is **correctly architected** as a self-contained bytecode compiler and interpreter:
- ✅ No circular dependencies
- ✅ Reusable across different contexts (runtime, tools, embedded scenarios)
- ✅ 20+ passing tests
- ✅ Functional implementation of matlabFunction via standalone compiler

## Opportunities for Enhancement

### 1. Bytecode Interpreter Optimization (Medium Effort, High Impact)

**Problem**: The `eval()` and `eval_batch()` methods in `compiler.rs` duplicate the entire bytecode execution logic.

**Improvement**:
- Extract core interpreter loop into a shared function `execute_bytecode(ops, constants, variables, stack)`
- `eval()` wraps it for single evaluation
- `eval_batch()` reuses it for batch processing

**Expected benefit**: Cleaner code, easier to optimize, potential for SIMD batch operations in future

### 2. Constant Folding at Compilation Time (Medium Effort, Medium Impact)

**Problem**: Expressions like `2 * 3 + x` still evaluate `2 * 3 = 6` at runtime on every call.

**Improvement**:
- During bytecode generation, detect constant subexpressions
- Pre-compute and replace with single `PushConst(6)` instruction

**Expected benefit**: Faster evaluation, smaller bytecode, matches MATLAB Symbolic behavior

### 3. Dead Code Elimination (Low Effort, Low-Medium Impact)

**Problem**: Intermediate stack operations (`Dup`, `Swap`, `Pop`) might be dead code.

**Improvement**:
- Post-compilation pass to eliminate unreachable stack operations
- Verify stack consistency and remove redundant operations

### 4. Instruction Cache Strategy (Low Effort, Medium Impact)

**Problem**: Compiled functions are stored in a `HashMap<String, CompiledFunctionEntry>` with hash lookups on each call.

**Improvement**:
- Add optional `Arc<CompiledExpr>` field to `Value::FunctionHandle` for symbolic functions
- Direct pointer dereference instead of HashMap lookup

### 5. Profiling Instrumentation (Medium Effort, High Impact for HPC)

**Problem**: No visibility into symbolic function performance characteristics.

**Opportunity**:
- Add optional call counters to `CompiledFunctionEntry`
- Track evaluation times per function
- Hook into RunMat's profiling infrastructure

### 6. Bytecode Caching for Repeated Expressions (Medium Effort, Medium Impact)

**Problem**: Same expression compiled multiple times creates redundant bytecode.

**Improvement**:
- Hash symbolic expressions for deduplication
- Check cache before compilation

### 7. Extended Function Library (Low Effort, Medium Impact)

**Problem**: `call_function()` is a large manual match statement (60+ lines).

**Improvement**:
- Use a function registry pattern (similar to RunMat's builtin inventory)
- Allow dynamic function registration

### 8. SIMD-Ready Batch Evaluation (High Effort, Very High Impact for arrays)

**Problem**: `eval_batch()` processes arrays element-by-element in serial.

**Opportunity**:
- Restructure bytecode to support vectorized operations
- Use SIMD operations for arithmetic on f64 arrays

**Expected benefit**: 4-8x speedup for array evaluation

## Recommended Priority

1. **High Impact, Low Effort**:
   - Bytecode interpreter refactoring (remove duplication)
   - Instruction cache optimization (#4)
   
2. **High Impact, Medium Effort**:
   - Constant folding (#2)
   - Profiling instrumentation (#5)
   - Bytecode caching (#6)

3. **Medium Impact, Medium Effort**:
   - Extended function library (#7)
   - Dead code elimination (#3)

4. **Very High Impact, High Effort** (Future, post-milestone):
   - SIMD-ready batch evaluation (#8)

## Architecture Decision: Why Not Ignition Integration?

Integration with RunMat's Ignition pipeline for JIT compilation **cannot work** due to:

1. **Circular Dependency**: runmat-runtime is used by many crates; adding ignition/hir would force all dependents to pull compilation infrastructure
2. **Reusability**: Symbolic expressions need to work in contexts without Ignition (embedded use, tools, etc.)
3. **Scope Mismatch**: Ignition is designed for full MATLAB code; symbolic bytecode (12 ops) is highly specialized

**Better approach**: Optimize the standalone compiler directly (#1-8 above) for performance comparable to JIT, without architectural compromise.

## Future Architecture: Optional JIT Bridge Crate

If profiling data eventually shows JIT is warranted, the correct pattern is an **optional bridge crate** that lives *above* the runtime layer:

### Design: `runmat-symbolic-jit`

```
┌─────────────────────────────────────────────────────────┐
│  runmat-symbolic-jit (optional, top-level)              │
│  ├── depends on: runmat-symbolic                        │
│  ├── depends on: runmat-hir / Ignition / Turbine        │
│  └── provides: JIT-backed NumericEvaluator              │
└─────────────────────────────────────────────────────────┘
                         ▲
                         │ (optional dependency)
┌────────────────────────┴────────────────────────────────┐
│  runmat-cli / runmat-gui / HPC driver                   │
│  (full runtime builds that already have Ignition)       │
└─────────────────────────────────────────────────────────┘
```

### Key APIs

```rust
// In runmat-symbolic-jit (bridge crate)
pub fn jit_compile(expr: &SymExpr, vars: &[&str]) -> JittedFnHandle;
pub fn maybe_jit(compiled: &CompiledExpr, profile: &SymbolicProfile) -> Box<dyn NumericEvaluator>;

// Shared trait (could live in runmat-symbolic)
pub trait NumericEvaluator: Send + Sync {
    fn eval(&self, vars: &[f64]) -> Result<f64, String>;
    fn eval_batch(&self, batch: &[&[f64]]) -> Result<Vec<f64>, String>;
}

// Two implementations
enum SymbolicBackend {
    Bytecode(Arc<CompiledExpr>),      // always available
    Jitted(Arc<dyn NumericEvaluator>), // only in bridge crate
}
```

### Why This Works

1. **No circular dependency**: Bridge crate is a leaf node, depends on both symbolic and HIR
2. **Runtime stays minimal**: `runmat-runtime` never pulls Ignition; embedded/tool contexts unaffected
3. **Hotness-based upgrading**: Full runtime builds can:
   - Start with `Bytecode(CompiledExpr)` for all symbolic functions
   - Use profiling hooks to detect hot paths
   - Upgrade hot functions to `Jitted` via the bridge crate

### When to Implement

Only pursue this if **all** conditions are met:

1. **Profiling shows >30-40% of runtime** in symbolic function evaluation (after optimizations #1-4)
2. **Typical expressions are >50-100 ops** (interpreter dispatch overhead becomes significant)
3. **Target environment has Turbine available** (not minimal/embedded builds)
4. **Team accepts increased complexity** in build/deployment matrix

Until these thresholds are crossed, the 12-op interpreter with optimizations (#1-8) is the correct path.

---

## Related Documentation

- **ARCHITECTURE.md**: Overview of RunMat's runtime and compilation model
- **DESIGN_PHILOSOPHY.md**: RunMat's design principles and extensibility model
- **DEVELOPING.md**: Contributing and development workflow
