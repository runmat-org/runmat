# RunMat Symbolic Mathematics Engine

This document describes the first-class symbolic mathematics engine in RunMat, which enables MATLAB-compatible symbolic computation capabilities.

## Overview

The symbolic engine allows RunMat to handle symbolic expressions as native `Value` types, enabling seamless mixing of symbolic and numeric computation. Symbolic expressions can be manipulated algebraically, differentiated, integrated, solved, and compiled to efficient numeric functions.

## Architecture

### Core Crate: `runmat-symbolic`

The symbolic engine is implemented in the `runmat-symbolic` crate, which provides:

- **`SymExpr`**: The core symbolic expression type, using reference-counted `Arc<SymExprKind>` for efficient cloning
- **`Coefficient`**: Exact rational arithmetic with fallback to floating-point for large numbers
- **`Symbol`**: Interned symbolic variables with optional attributes (real, positive, integer, nonnegative)
- **`StagedNormalizer`**: Configurable normalization pipeline for expression simplification
- **`BytecodeCompiler`**: Stack-based bytecode compiler for efficient numeric evaluation

### Expression Representation

Symbolic expressions use a tree-based representation:

```rust
pub enum SymExprKind {
    Num(Coefficient),           // Numeric constant (rational or float)
    Var(Symbol),                // Symbolic variable
    Add(Vec<SymExpr>),          // Sum of terms
    Mul(Vec<SymExpr>),          // Product of factors
    Pow(Box<SymExpr>, Box<SymExpr>), // Power expression
    Neg(Box<SymExpr>),          // Negation
    Func(String, Vec<SymExpr>), // Function application
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

## Builtin Functions

### Creating Symbolic Variables

| Function | Description | Example |
|----------|-------------|---------|
| `sym(name)` | Create a symbolic variable | `x = sym('x')` |
| `sym(name, 'real')` | Create with assumption | `x = sym('x', 'real')` |
| `sym(value)` | Convert numeric to symbolic | `two = sym(2)` |
| `syms('x', 'y', 'z')` | Create multiple variables | `vars = syms('x', 'y', 'z')` |

### Calculus Operations

| Function | Description | Example |
|----------|-------------|---------|
| `diff(expr, var)` | Differentiate expression | `diff(x^2, x)` → `2*x` |
| `diff(expr, var, n)` | n-th derivative | `diff(x^3, x, 2)` → `6*x` |
| `int(expr, var)` | Indefinite integral | `int(x^2, x)` → `x^3/3` |

### Algebraic Manipulation

| Function | Description | Example |
|----------|-------------|---------|
| `simplify(expr)` | Simplify expression | `simplify(x + x)` → `2*x` |
| `expand(expr)` | Expand products/powers | `expand((x+1)^2)` → `x^2 + 2*x + 1` |
| `factor(expr)` | Factor expression | `factor(x^2 - 1)` → `(x+1)*(x-1)` |
| `collect(expr, var)` | Collect terms by power | `collect(a*x + b*x, x)` → `(a+b)*x` |

### Substitution and Solving

| Function | Description | Example |
|----------|-------------|---------|
| `subs(expr, var, val)` | Substitute value | `subs(x^2, x, 3)` → `9` |
| `solve(expr, var)` | Solve equation = 0 | `solve(x^2 - 4, x)` → `2` or `-2` |

### Compilation to Numeric Functions

| Function | Description | Example |
|----------|-------------|---------|
| `matlabFunction(expr)` | Compile to function handle | `f = matlabFunction(x^2 + 1)` |

## Normalization Pipeline

The `StagedNormalizer` applies a sequence of transformation passes:

1. **SimplifyNeg**: Apply double negation rule, fold negative constants
2. **Flatten**: Flatten nested Add/Mul operations
3. **MergeConstants**: Combine numeric constants
4. **RemoveIdentity**: Remove x+0, x*1
5. **SimplifyPowers**: Apply x^0=1, x^1=x
6. **CollectLikeTerms**: Combine x + x → 2*x
7. **Sort**: Sort terms in canonical order

Three built-in pipelines are available:
- `default_pipeline()`: Standard simplification
- `minimal()`: Just flatten and sort
- `aggressive()`: Includes expansion and full normalization

## Bytecode Compilation

The bytecode compiler converts symbolic expressions to stack-based instructions for efficient numeric evaluation:

```rust
pub enum BytecodeOp {
    PushConst(usize),    // Push constant by index
    LoadVar(usize),      // Load variable by index
    Add, Sub, Mul, Div, Pow,  // Binary operations
    Neg,                 // Unary negation
    Call(String, usize), // Function call with argument count
}
```

The `CompiledExpr` struct supports:
- `eval(var_values)`: Single evaluation
- `eval_batch(var_values_batch)`: Vectorized evaluation
- `eval_range(values)`: Evaluate over a range (single-variable)

## Supported Functions

The following mathematical functions are supported in symbolic expressions:

- **Trigonometric**: sin, cos, tan
- **Inverse trig**: asin, acos, atan, atan2
- **Hyperbolic**: sinh, cosh, tanh
- **Exponential/Log**: exp, log (natural), log10, log2
- **Other**: sqrt, abs, sign, floor, ceil, round

## Error Handling

Symbolic operations return descriptive error messages for unsupported operations:

```rust
pub enum SymbolicError {
    UndefinedSymbol(String),
    InvalidOperation(String),
    DivisionByZero,
    NumericOverflow,
}
```

## Design Principles

1. **RunMat-native**: Integrates with RunMat's IR and type system
2. **MATLAB-compatible**: Matches MATLAB Symbolic Toolbox behavior where possible
3. **Performance**: Structural sharing via Arc, symbol interning, bytecode compilation
4. **Extensible**: Pluggable simplification rules and normalization passes

## Current Limitations

- **Polynomial solving**: Only linear and quadratic equations are currently solved analytically
- **Integration**: Limited to polynomials, basic transcendental functions, and simple products
- **Matrix symbolic**: Not yet supported (scalar symbolic only)
- **Assumptions**: Basic assumptions supported but not fully propagated through all operations

## Future Work

- Extend equation solving to higher-degree polynomials
- Add symbolic matrix support
- Implement more integration techniques (substitution, parts)
- Add symbolic ODE solving
- JIT compilation of symbolic expressions via Cranelift
