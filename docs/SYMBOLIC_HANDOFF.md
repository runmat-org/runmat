# Symbolic Engine Handoff - December 21, 2025

## Session Summary

This session implemented major Phase 4-5 features of the symbolic engine.

---

## What Was Implemented

### Phase 3 (Completed)
- Fixed `power` operator bug in `elementwise.rs`
- Added symbolic support to: `mtimes`, `rdivide`, `ldivide`
- Added symbolic support to: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`

### Phase 4 (Completed)
- **`factor`** - polynomial factorization (`factor.rs`)
  - Difference of squares: a² - b² = (a+b)(a-b)
  - Common factor extraction: ax + ay = a(x+y)
  - Quadratic factoring with integer roots
  
- **`collect`** - polynomial term collection (`collect.rs`)
  - Collects coefficients by powers of a variable
  - ax + bx = (a+b)x

### Phase 5 (Completed)
- **`solve`** - equation solving (`solve.rs`)
  - Linear equations: ax + b = 0
  - Quadratic equations: ax² + bx + c = 0 (quadratic formula)
  
- **`int`** - symbolic integration (`integrate.rs`)
  - Polynomials: ∫x^n dx = x^(n+1)/(n+1)
  - Transcendentals: ∫sin(x), ∫cos(x), ∫exp(x)
  - Linear substitution: ∫f(ax+b) dx = F(ax+b)/a

---

## Files Created

| File | Purpose |
|------|---------|
| `crates/runmat-runtime/src/builtins/symbolic/factor.rs` | Polynomial factorization |
| `crates/runmat-runtime/src/builtins/symbolic/collect.rs` | Polynomial collection |
| `crates/runmat-runtime/src/builtins/symbolic/solve.rs` | Equation solving |
| `crates/runmat-runtime/src/builtins/symbolic/integrate.rs` | Symbolic integration |

## Files Modified

| File | Change |
|------|--------|
| `elementwise.rs` | Fixed symbolic power pattern |
| `mtimes.rs`, `rdivide.rs`, `ldivide.rs` | Added symbolic support |
| `sin.rs`, `cos.rs`, `tan.rs`, `exp.rs`, `log.rs`, `sqrt.rs` | Added symbolic support |
| `symbolic/mod.rs` | Registered new modules |

---

## Test Results

- **40 symbolic tests pass** (up from 26)
- **clippy**: Clean
- **cargo fmt**: Clean

---

## FIXED: Builtins Not Found in REPL

### Original Symptom
```
runmat> int(x^2, x)
Error: Execution failed: MATLAB:UndefinedFunction: Undefined function: int
```

### Root Cause
Cross-crate inventory registration was being stripped by the linker because no symbols
from `runmat-runtime` were directly referenced in the REPL binary.

### Solution Applied
Added `ensure_builtins_linked()` function to `runmat-runtime/src/lib.rs` that uses
`std::hint::black_box` to prevent dead-code elimination of inventory registrations.
Called from `runmat-repl/src/main.rs` at startup.

### Files Changed
- `crates/runmat-runtime/src/lib.rs` - Added `ensure_builtins_linked()` function
- `crates/runmat-repl/src/main.rs` - Called `ensure_builtins_linked()` at startup

---

## Correctness Fixes Applied

### integrate.rs
- **Fixed**: `∫sqrt(x) dx` formula was incorrect (had wrong order of factors)
- **Updated**: Comment now correctly states `∫sqrt(x) dx = (2/3)*x^(3/2)`

### factor.rs
- **Fixed**: `extract_term_power` was overwriting power instead of accumulating
  - `x * x * x` is now correctly identified as power 3 (was: power 1)
- **Fixed**: `try_quadratic_factor` was overwriting coefficients instead of summing
  - `x^2 + 2x^2` now correctly extracts `a = 3` (was: `a = 2`)

### Known Limitations (documented, not bugs)
- `solve` returns only the first solution for quadratics (TODO: return all)
- `factor` only handles simple quadratics with integer roots
- `polynomial_degree` doesn't expand `(x+1)^2` before checking degree

---

## Architecture Alignment

Per `docs/RunMat_curated_confederation_model.md`:

### Aligned ✅
- Staged normalization pipeline = transformation insertion points
- MATLAB-compatible core builtins
- Extensible architecture

### Gaps to Address
1. **Transformation Registry** - Not yet implemented
2. **Tier Labels** - New builtins not labeled (core vs extension)
3. **Symbolic IR Stability Policy** - Not published
4. **Extension Guide** - Not written

---

## Documentation Updates Needed

1. **SYMBOLIC_ENGINE_ASSESSMENT.md** - Update to show `solve` and `int` are implemented
2. **Create transformation registry doc** per confederation recommendations
3. **Add tier labels** to new builtins

---

## Commands for Next Session

```cmd
# Set up environment
set PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%

# Build and test
cd c:\GitHub\runmat
cargo fmt
cargo build -p runmat-repl
cargo test -p runmat-runtime symbolic
cargo clippy -p runmat-runtime --all-targets -- -D warnings

# Run REPL
target\debug\runmat-repl.exe
```

---

## Key Files to Review

- `crates/runmat-macros/src/lib.rs` - The `runtime_builtin` macro
- `crates/runmat-builtins/src/lib.rs` - Inventory collection at line 1015
- `crates/runmat-runtime/src/dispatcher.rs` - `call_builtin` function
- `crates/runmat-repl/Cargo.toml` - Dependencies on runtime
