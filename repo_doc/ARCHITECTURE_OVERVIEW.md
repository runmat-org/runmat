# RunMat Architecture Overview

This document provides a high-level view of RunMat's architecture for contributors who want to understand the codebase or extend its capabilities.

---

## Crate Structure

RunMat is organized as a Cargo workspace with specialized crates:

```
runmat/
├── crates/
│   ├── runmat-lexer        # Tokenizer for MATLAB/Octave syntax
│   ├── runmat-parser       # AST generation from tokens
│   ├── runmat-hir          # High-level IR (HIR) lowering
│   ├── runmat-ignition     # Bytecode compiler and interpreter
│   ├── runmat-turbine      # JIT compiler (Cranelift-based)
│   ├── runmat-runtime      # Builtins, operators, and runtime APIs
│   ├── runmat-builtins     # Value types and builtin registry
│   ├── runmat-macros       # Procedural macros (#[runtime_builtin])
│   ├── runmat-symbolic     # Symbolic expression engine
│   ├── runmat-accelerate   # GPU acceleration (WGPU backend)
│   ├── runmat-gc           # Garbage collector
│   ├── runmat-plot         # Plotting subsystem
│   ├── runmat-snapshot     # Snapshot serialization
│   └── runmat-repl         # Interactive REPL binary
```

---

## Execution Pipeline

```
Source Code
    │
    ▼
┌─────────────┐
│   Lexer     │  runmat-lexer
└─────────────┘
    │ Tokens
    ▼
┌─────────────┐
│   Parser    │  runmat-parser
└─────────────┘
    │ AST
    ▼
┌─────────────┐
│ HIR Lowering│  runmat-hir
└─────────────┘
    │ HIR
    ▼
┌─────────────┐
│  Compiler   │  runmat-ignition
└─────────────┘
    │ Bytecode
    ▼
┌──────────────────────────────┐
│  Execution Engine            │
│  ┌────────────┬────────────┐ │
│  │ Interpreter│    JIT     │ │
│  │ (Ignition) │ (Turbine)  │ │
│  └────────────┴────────────┘ │
└──────────────────────────────┘
    │
    ▼
┌─────────────┐
│  Runtime    │  runmat-runtime (builtins, operators)
└─────────────┘
```

---

## Key Abstractions

### Value Type

All runtime values are represented by the `Value` enum in `runmat-builtins`:

```rust
pub enum Value {
    Num(f64),
    Int(IntValue),
    Bool(bool),
    String(String),
    Tensor(Tensor),
    Cell(CellArray),
    Struct(StructValue),
    Symbolic(SymExpr),
    // ... and more
}
```

Contributors adding new operations must handle relevant `Value` variants.

### Builtin Registration

Builtins are registered via the `#[runtime_builtin]` macro:

```rust
#[runtime_builtin(name = "myfunction", category = "math")]
fn my_function(x: Value) -> Result<Value, String> {
    // implementation
}
```

The macro generates wrapper code and registers the function with the `inventory` crate for runtime discovery.

### HIR (High-level IR)

The HIR represents parsed code in a form suitable for analysis and compilation:

- `HirStmt` - Statements (assignments, control flow, function definitions)
- `HirExpr` - Expressions (literals, operations, function calls)
- `VarId` - Variable identifiers (indices into variable arrays)

---

## Extension Points

### 1. Adding Builtins

The primary extension point. See [CONTRIBUTING_EXTENSIONS.md](CONTRIBUTING_EXTENSIONS.md) for details.

**Location**: `crates/runmat-runtime/src/builtins/`

### 2. Normalization Passes

For symbolic computation, normalization passes transform expressions:

**Location**: `crates/runmat-symbolic/src/normalize.rs`

Passes are composable via `StagedNormalizer`:

```rust
let normalizer = StagedNormalizer::default_pipeline()
    .with_pass(NormPass::Flatten)
    .with_pass(NormPass::MergeConstants);
```

### 3. Acceleration Backends

GPU acceleration is abstracted via provider traits:

**Location**: `crates/runmat-accelerate/src/`

### 4. Plot Backends

Plotting can use different renderers (GUI, Jupyter, headless):

**Location**: `crates/runmat-plot/src/`

---

## Design Principles

### MATLAB Compatibility First

RunMat aims for behavioral compatibility with MATLAB. When in doubt:
- Match MATLAB semantics for core operations
- Use MATLAB function names and signatures
- Document deviations explicitly

### Two-Tier Ecosystem

| Tier | Characteristics |
|------|-----------------|
| **Core** | MATLAB-compatible, stable, always-on |
| **Extension** | Opt-in, experimental, may require assumptions |

New features should be classified. Core features require compatibility review.

### Correctness Declarations

Operations should declare their correctness properties:

| Level | Meaning |
|-------|---------|
| **Semantics-preserving** | Always produces mathematically equivalent results |
| **Assumption-dependent** | Correct under stated assumptions (e.g., nonzero denominator) |
| **Heuristic** | Best-effort, may not always succeed |

---

## Testing

### Running Tests

```bash
# Format check
cargo fmt --check

# Lint check
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
cargo test --all
```

### Test Organization

- Unit tests: Adjacent to implementation (`#[cfg(test)]` modules)
- Integration tests: `crates/*/tests/`
- End-to-end tests: `tests/`

---

## Further Reading

- [CONTRIBUTING_EXTENSIONS.md](CONTRIBUTING_EXTENSIONS.md) - How to add new functionality
- [AGENTS.md](../AGENTS.md) - Guidelines for AI coding agents
- [README.md](../README.md) - Project overview
