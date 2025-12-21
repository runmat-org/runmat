# RunMat Symbolic Engine – Implementation Plan

This document describes a concrete, buildable plan to add a **MATLAB Symbolic Toolbox–compatible** symbolic engine to **RunMat**, using **Symbolica** as a backend while respecting licensing constraints. The focus is on **performance**, **maintainability**, and **tight integration with the RunMat JIT and runtime**.

---

## 1. Goals

1. **MATLAB compatibility first**
   - Match MATLAB Symbolic Toolbox behavior and API semantics, not internal implementation.
   - Prioritize correctness of results, shapes, and evaluation rules.

2. **Symbolica-based backend**
   - Use Symbolica as a symbolic algebra engine.
   - Do **not** copy or redistribute Symbolica source code without permission.

3. **High performance**
   - Structural interning, caching, and pure-function JIT optimizations.
   - Avoid string-based symbolic workflows.

4. **Maintainable and extensible**
   - Clean backend abstraction.
   - Multiple backends possible in the future.

5. **RunMat-native**
   - Symbolic values are first-class runtime values.
   - Symbolic-to-numeric code generation targets RunMat IR/JIT directly.

---

## 2. Legal & Distribution Strategy (Phase 0)

### 2.1 Distribution model (decision required upfront)

**Option A – Optional backend (recommended initially)**
- RunMat symbolic package ships independently.
- Symbolica is loaded dynamically or via feature-flag build.
- Users install Symbolica separately.
- No redistribution of Symbolica code or binaries by RunMat.

**Option B – Redistributable backend**
- Requires explicit permission or commercial agreement with Symbolica author.
- Simplifies user installation.

### 2.2 Compatibility contract

Define what “fully compatible” means:
- Exact behavior vs mathematically equivalent behavior
- Handling of assumptions, exact vs approximate arithmetic
- Error conditions and edge cases

Deliverable:
```
docs/symbolic/compatibility.md
```

---

## 3. High-level Architecture

```
MATLAB-compatible API layer (@sym, syms, diff, subs, ...)
                |
                v
      Symbolic Compatibility Core (RunMat package)
                |
        Backend Trait (abstract interface)
                |
                v
        Symbolica Backend (external dependency)
```

### Key principle
**MATLAB compatibility logic never depends on Symbolica internals.**
Symbolica is just one backend that satisfies a symbolic-algebra trait.

---

## 4. Phase 1 – Scaffolding & Core Types (Week 1)

### 4.1 New RunMat package

```
runmat/
  packages/
    symbolic/
      README.md
      m/
        @sym/
          sym.m
          plus.m
          mtimes.m
          diff.m
          subs.m
        syms.m
      src/
        lib.rs
        value.rs          # Value::Symbolic
        backend/
          mod.rs
          trait.rs
          symbolica.rs
```

### 4.2 Runtime value integration

Add a new runtime value kind:
- `Value::Symbolic`
- Stores:
  - opaque expression handle (`ExprId`)
  - backend vtable pointer
  - optional assumptions context

Properties:
- Immutable
- Hashable
- Cheap to clone (handle-based)

---

## 5. Phase 2 – Backend Abstraction (Week 2)

### 5.1 Backend trait

Define a minimal but extensible interface:

- Construction:
  - symbols
  - integers, rationals, floats
  - function application

- Core operations:
  - add, mul, pow
  - simplify, expand, factor
  - substitute
  - differentiate

- Queries:
  - free symbols
  - polynomial detection

- Output:
  - ASCII string
  - pretty string
  - LaTeX

- Capabilities:
  - supports_solve
  - supports_series

### 5.2 Symbolica backend implementation

- Thin wrapper around Symbolica API
- No MATLAB semantics here
- Pure algebra only

---

## 6. Phase 3 – MATLAB Compatibility Layer (Weeks 3–4)

### 6.1 Core symbolic API

Implement MATLAB-compatible behavior for:

- `sym`, `syms`
- Arithmetic operators
- Matrix construction and scalar expansion
- `subs`
- `simplify`, `expand`, `factor`
- `diff`

Key focus areas:
- Shape rules
- Broadcasting
- Empty and scalar edge cases

### 6.2 Assumptions

- Introduce an assumptions context object
- Attach to symbolic values
- Pass through backend where supported

---

## 7. Phase 4 – Performance Engineering (Weeks 4–5)

### 7.1 Structural interning

- Global symbol table
- Hash-cons expression nodes
- Structural equality via pointer comparison

### 7.2 Caching

- Simplification cache: `(expr, assumptions) -> simplified expr`
- Substitution cache for repeated `subs` calls

### 7.3 JIT integration

Mark symbolic builtins as:
- Pure
- Side-effect free

Enable:
- Constant folding
- Common subexpression elimination

---

## 8. Phase 5 – Solvers & Series (Weeks 6–7)

### 8.1 Solvers

- `solve` (linear systems first)
- Polynomial roots (where backend supports)

### 8.2 Series & polynomials

- `series`, `taylor`
- `coeffs`, `collect`, `numden`

---

## 9. Phase 6 – Code Generation (Week 8)

### 9.1 `matlabFunction`

- Convert symbolic expressions directly into RunMat IR
- Avoid string code generation

### 9.2 Numeric fusion

- Symbolic preprocessing feeds numeric JIT
- Enables aggressive fusion and vectorization

---

## 10. Testing Strategy

### 10.1 Compatibility tests

- Run identical `.m` files in MATLAB and RunMat
- Compare:
  - `isequal`
  - `char()` output
  - numeric sampling

### 10.2 Performance benchmarks

- Large substitution workloads
- Repeated simplification loops
- Symbolic-to-numeric pipelines

---

## 11. Deliverables Summary

- Symbolic RunMat package
- Backend abstraction + Symbolica backend
- MATLAB-compatible symbolic API (core coverage)
- JIT-integrated symbolic execution
- Compatibility + performance test suite

---

## 12. Future Extensions

- Alternative backends (e.g., numeric-only symbolic subset)
- GPU-targeted symbolic-to-numeric lowering
- Partial evaluation and automatic differentiation integration

---

**Outcome:**
A fast, clean, legally safe symbolic engine that feels like MATLAB, but is architecturally aligned with RunMat and outperforms traditional string-based symbolic workflows.

