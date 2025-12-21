# RunMat Symbolic Engine - Implementation Status

## Completed

### 1. New Crate: `runmat-symbolic` (crates/runmat-symbolic/)
- **lib.rs** - Main exports and error types
- **symbol.rs** - Symbol interning with SymbolId, Symbol, SymbolAttrs
- **coeff.rs** - Exact rational + float coefficient representation
- **expr.rs** - SymExpr tree with Add, Mul, Pow, Neg, Func, diff(), substitute(), eval()
- **normalize.rs** - Staged normalization pipeline (Flatten, Sort, MergeConstants, etc.)
- **compiler.rs** - Bytecode compiler and VM for numeric evaluation

### 2. Updated: `runmat-builtins`
- Added `Value::Symbolic(runmat_symbolic::SymExpr)` variant
- Added Display handling for Symbolic
- Added Type::from_value handling for Symbolic

### 3. Updated: `runmat-gc`
- Added Symbolic handling in collector.rs mark_object

### 4. New Builtins: `runmat-runtime/src/builtins/symbolic/`
- **mod.rs** - Module structure
- **sym.rs** - `sym('x')` - create symbolic variable
- **syms.rs** - `syms('x','y')` - create multiple variables
- **diff.rs** - `diff(expr, x)` - differentiation
- **subs.rs** - `subs(expr, x, 2)` - substitution
- **simplify.rs** - `simplify(expr)` - normalization
- **expand.rs** - `expand(expr)` - polynomial expansion
- **arithmetic.rs** - plus, minus, times, rdivide, power for symbolic
- **matlabfunction.rs** - `matlabFunction(expr)` - compile to numeric

### 5. Updated: Cargo.toml files
- Added runmat-symbolic to workspace members
- Added runmat-symbolic as dependency in runmat-builtins, runmat-runtime

## Remaining Work

### Must Fix: Non-exhaustive pattern matches
Many files in runmat-runtime match on `Value` but don't handle `Symbolic`:
- `crates/runmat-runtime/src/builtins/common/format.rs` ~line 941
- `crates/runmat-runtime/src/builtins/acceleration/gpu/pagefun.rs` ~line 950
- `crates/runmat-runtime/src/builtins/array/creation/meshgrid.rs` ~line 487
- `crates/runmat-runtime/src/builtins/array/shape/circshift.rs` ~line 263
- Plus potentially more files

**Pattern to fix:** Add `Value::Symbolic(_) => ...` case to each match.
For most cases: Return error like "symbolic not supported for this operation"
For format.rs: Use `write!(f, "{}", expr)` since SymExpr implements Display

### After fixing patterns:
1. Run `cargo fmt --all`
2. Run `cargo clippy -p runmat-symbolic -p runmat-builtins -p runmat-runtime -- -D warnings`
3. Run `cargo test -p runmat-symbolic`

## Build Environment (Windows MinGW)
```cmd
set PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%
set CPATH=C:\msys64\ucrt64\include
set LIBRARY_PATH=C:\msys64\ucrt64\lib
cargo build
```

## Key Design Decisions
- **No Symbolica dependency**: All code is original, following standard CAS patterns
- **RunMat naming**: Uses RunMat idioms (SymExpr, not Atom)
- **Immutable expressions**: SymExpr uses Arc<SymExprKind> for efficient sharing
- **Staged normalization**: Composable passes, not monolithic normalize()
- **Bytecode compilation**: Enables fast numeric evaluation without tree-walking
