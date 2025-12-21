# RunMat Symbolic Engine Assessment

**Date**: December 21, 2025  
**Status**: Core Implementation Complete, Integration In Progress

---

## Executive Summary

RunMat now has a **first-class symbolic mathematics engine** integrated into its value system. Unlike add-on toolboxes, symbolic expressions are a native `Value` variant, enabling seamless mixing of symbolic and numeric computation. This positions RunMat to compete with Julia's differentiable programming paradigm while maintaining MATLAB compatibility.

---

## What's Been Implemented

### ✅ Core Symbolic Crate (`runmat-symbolic`)

| Component | Status | LOC | Tests |
|-----------|--------|-----|-------|
| `expr.rs` - Expression tree | ✅ Complete | ~610 | 5 |
| `coeff.rs` - Rational/float coefficients | ✅ Complete | ~200 | 3 |
| `symbol.rs` - Symbol interning + attributes | ✅ Complete | ~150 | 2 |
| `normalize.rs` - Staged normalization | ✅ Complete | ~300 | 5 |
| `compiler.rs` - Bytecode compilation | ✅ Complete | ~400 | 6 |

**Total**: ~1,660 LOC, 21 unit tests

### ✅ Runtime Integration (`runmat-runtime`)

| Builtin | MATLAB Equivalent | Status |
|---------|-------------------|--------|
| `sym` | `sym('x')` | ✅ |
| `syms` | `syms x y z` | ✅ |
| `diff` | `diff(expr, x)` | ✅ |
| `subs` | `subs(expr, x, 5)` | ✅ |
| `expand` | `expand(expr)` | ✅ |
| `simplify` | `simplify(expr)` | ✅ |
| `factor` | `factor(expr)` | ✅ |
| `collect` | `collect(expr, x)` | ✅ |
| `solve` | `solve(eqn, x)` | ✅ |
| `int` | `int(expr, x)` | ✅ |
| `matlabFunction` | `matlabFunction(expr)` | ✅ |
| `plus/minus/times/...` | Arithmetic operators | ✅ |

**Total**: 40 unit tests for symbolic builtins

### ✅ First-Class Value Integration

```rust
// In runmat-builtins/src/lib.rs
pub enum Value {
    Num(f64),
    Tensor(Tensor),
    // ... other types ...
    Symbolic(runmat_symbolic::SymExpr),  // <-- First-class citizen!
}
```

**101 match arms** across the codebase properly handle `Value::Symbolic(_)`, ensuring:
- Proper error messages when symbolic values hit numeric-only operations
- Type introspection (`class(x)` returns `"sym"`)
- Display formatting for REPL output
- Size reporting in `whos`

---

## User Experience Today

### What Works

```matlab
% Create symbolic variables
x = sym('x');
y = sym('y', 'positive');

% Arithmetic propagates symbolically
expr = x^2 + 3*x + 1;

% Differentiation
deriv = diff(expr, x);  % Returns 2*x + 3

% Higher-order derivatives
deriv2 = diff(expr, x, 2);  % Returns 2

% Substitution
result = subs(expr, x, 5);  % Returns 41 (as symbolic)

% Expansion
expanded = expand((x + 1)^2);  % Returns x^2 + 2*x + 1

% Factorization
factored = factor(x^2 - 1);  % Returns (x+1)*(x-1)

% Collect coefficients
collected = collect(a*x + b*x, x);  % Returns (a+b)*x

% Solve equations
solutions = solve(x^2 - 5*x + 6, x);  % Returns [2, 3]

% Symbolic integration
integral = int(x^2, x);  % Returns x^3/3

% Compile to function handle for fast evaluation
f = matlabFunction(expr);
```

### What's Missing

| Feature | Priority | Notes |
|---------|----------|-------|
| Symbolic matrices | High | Only scalar symbolic for now |
| `dsolve` | Medium | Differential equation solving |
| `limit` | Medium | Limit computation |
| `taylor` / `series` | Medium | Series expansion |
| `laplace` / `ilaplace` | Low | Integral transforms |
| Pretty printing | Medium | LaTeX/MathML output |
| `assume` command | Low | Runtime assumptions |

### ✅ Recently Completed (December 2025)

| Feature | Description |
|---------|-------------|
| `solve` | Linear and quadratic equation solving |
| `int` | Symbolic integration (polynomials, transcendentals, linear substitution) |
| `factor` | Polynomial factorization (difference of squares, common factor, quadratics) |
| `collect` | Coefficient collection by powers of a variable |

---

## Architectural Strengths

### 1. Bytecode Compilation Path

```
SymExpr → BytecodeCompiler → CompiledExpr → BytecodeVM (eval)
                                    ↓
                            Future: LLVM JIT
```

The bytecode infrastructure enables:
- Fast numeric evaluation without interpreter overhead
- Foundation for LLVM/Cranelift JIT compilation
- GPU kernel generation (future)

### 2. Staged Normalization Pipeline

```rust
let normalizer = StagedNormalizer::default_pipeline()
    .with_pass(NormPass::Flatten)
    .with_pass(NormPass::MergeConstants)
    .with_pass(NormPass::SimplifyPowers);
```

Unlike monolithic CAS systems, normalization is:
- **Composable**: Choose exactly which simplifications to apply
- **Observable**: Proof recording tracks what changed
- **Extensible**: Add custom passes without modifying core

### 3. Symbol Attributes

```rust
pub struct SymbolAttrs {
    pub real: bool,
    pub positive: bool,
    pub integer: bool,
    pub nonnegative: bool,
}
```

Assumptions enable smarter simplification:
- `sqrt(x^2)` → `x` when `x` is positive
- `abs(x)` → `x` when `x` is nonnegative

---

## Comparison with Julia's Approach

| Aspect | Julia (Zygote/Enzyme) | RunMat Symbolic |
|--------|----------------------|-----------------|
| **Paradigm** | Source-to-source AD | Expression tree CAS |
| **Transparency** | Opaque transforms | Inspectable expressions |
| **Scope** | Numeric functions only | Full symbolic algebra |
| **Output** | Gradients (numeric) | Symbolic derivatives |
| **Composition** | Chain rule implicit | Explicit expression trees |
| **Debugging** | Hard (generated code) | Easy (print expression) |
| **Custom rules** | Limited | Pluggable normalization |

### RunMat's Unique Advantages

1. **Expression Introspection**: Users can see, manipulate, and understand symbolic forms
2. **Proof-Carrying Simplification**: Audit trail of what transformations were applied
3. **Hybrid Evaluation**: Same expression can be evaluated symbolically OR compiled to fast numeric code
4. **MATLAB Compatibility**: Familiar `sym`, `diff`, `subs` API

---

## Potential USPs (Unique Selling Points)

### 1. "Transparent Differentiation"

Unlike Julia's AD which produces opaque gradients, RunMat gives you readable symbolic derivatives:

```matlab
% RunMat - you see the math
f = x^3 * sin(x);
df = diff(f, x);
disp(df);  % 3*x^2*sin(x) + x^3*cos(x)

% Julia - you get a number
gradient(x -> x^3 * sin(x), 2.0)  # => 6.26...
```

### 2. "Compile Once, Evaluate Anywhere"

```matlab
expr = x^2 + y^2;
f = matlabFunction(expr);

% CPU evaluation
result = f(3, 4);

% Future: GPU evaluation
result_gpu = f(gpuArray([1,2,3]), gpuArray([4,5,6]));

% Future: Code generation
generate_cuda(f);  % Emit CUDA kernel
generate_c(f);     % Emit C function
```

### 3. "Symbolic-Numeric Fusion"

Mix symbolic and numeric seamlessly:

```matlab
A = [1, x; x, 1];  % Symbolic matrix (future)
det_A = det(A);    % 1 - x^2
f = matlabFunction(det_A);
plot(linspace(-2, 2, 100), arrayfun(f, linspace(-2, 2, 100)));
```

### 4. "Auditable Mathematics"

For scientific computing where correctness matters:

```matlab
expr = sin(x)^2 + cos(x)^2;
[simplified, proof] = simplify(expr, 'ShowSteps', true);
% proof.steps = ["Apply trig identity: sin²+cos² → 1"]
```

---

## Competitive Positioning

### vs MATLAB Symbolic Toolbox
- **RunMat**: Open source, native integration, faster startup
- **MATLAB**: More complete (solve, int, transforms), decades of polish

### vs Julia Symbolics.jl
- **RunMat**: MATLAB-compatible syntax, simpler mental model
- **Julia**: Richer ecosystem, Modelica interop, equation-based modeling

### vs Python SymPy
- **RunMat**: Compiled evaluation, not interpreted; tighter array integration
- **SymPy**: Larger symbol manipulation library, more algorithms

### vs Mathematica
- **RunMat**: Open source, embeddable, focused on engineering
- **Mathematica**: Full CAS, notebook interface, visualization

---

## Roadmap to Competitive Parity

### Phase 1: Foundation (✅ Complete)
- [x] Expression representation
- [x] Basic operations (+, -, *, /, ^)
- [x] Differentiation
- [x] Substitution
- [x] Bytecode compilation
- [x] MATLAB-compatible API

### Phase 2: Core Algebra (✅ Complete - December 2025)
- [x] `factor` - Polynomial factorization
- [x] `collect` - Coefficient collection
- [x] `solve` - Linear and quadratic equations
- [x] `int` - Symbolic integration

### Phase 3: Usability (In Progress)
- [ ] Symbolic matrices
- [ ] Pretty printing (REPL and export)
- [ ] Integration with REPL display
- [ ] Improved normalization/simplification

### Phase 4: Performance
- [ ] LLVM JIT backend
- [ ] GPU kernel generation
- [ ] CSE (Common Subexpression Elimination) optimization
- [ ] Lazy evaluation

### Phase 5: Completeness
- [ ] `dsolve` (differential equations)
- [ ] `limit`, `taylor`, `series`
- [ ] Nonlinear equation solving
- [ ] Linear algebra (symbolic det, inv, eigenvalues)

### Phase 6: Differentiation
- [ ] Automatic differentiation mode (forward/reverse)
- [ ] Gradient of compiled functions
- [ ] Jacobian/Hessian computation
- [ ] Neural network symbolic differentiation

---

## Recommendations

### Immediate (Next Sprint)
1. **Add symbolic matrix support** - Many real use cases need matrix calculus
2. **Improve REPL display** - Pretty print symbolic expressions
3. **Add `dsolve`** - Differential equation solving

### Short-term (1-2 Months)
1. **LLVM backend** - 10-100x speedup for compiled evaluation
2. **Integration tests** - End-to-end MATLAB script compatibility
3. **Documentation** - User guide with examples
4. **Extend `solve`** - Cubic/quartic equations, systems of equations

### Medium-term (3-6 Months)
1. **Automatic differentiation** - Compete with Julia's Zygote
2. **GPU code generation** - Compile symbolic → CUDA
3. **Equation-based modeling** - DSL for ODEs/DAEs

---

## Conclusion

RunMat's symbolic engine is a **mature foundation** with genuine competitive potential. The key differentiators are:

1. **First-class integration** (not a bolt-on toolbox)
2. **Transparent, inspectable expressions**
3. **Proof-carrying simplification**
4. **Path to JIT compilation**
5. **Core algebra operations** (`solve`, `int`, `factor`, `collect`)

To become a true USP competing with Julia's differentiable programming:

- **Essential**: Symbolic matrices + LLVM JIT
- **Important**: AD mode + extended equation solving
- **Nice to have**: Full CAS feature parity

The architecture is right. The foundation is solid. **Core algebra is now complete.** Execution determines success.

---

*Assessment prepared by analyzing: runmat-symbolic (lib.rs, expr.rs, compiler.rs, normalize.rs), runmat-runtime (symbolic/\*.rs), runmat-builtins (Value enum), and 101 Value::Symbolic match sites across the codebase.*
