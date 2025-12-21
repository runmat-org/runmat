# Contributing Extensions to RunMat

This guide explains how to extend RunMat with new builtins, operators, or external packages.

---

## Table of Contents

1. [Adding a Builtin Function](#adding-a-builtin-function)
2. [Builtin Metadata](#builtin-metadata)
3. [Handling Value Types](#handling-value-types)
4. [Testing Your Builtin](#testing-your-builtin)
5. [Creating External Extension Crates](#creating-external-extension-crates)
6. [Classification and Review](#classification-and-review)

---

## Adding a Builtin Function

### Step 1: Choose the Right Location

Builtins are organized by category in `crates/runmat-runtime/src/builtins/`:

```
builtins/
├── math/           # Mathematical functions (sin, cos, exp, ...)
├── array/          # Array operations (zeros, ones, reshape, ...)
├── strings/        # String manipulation
├── io/             # File I/O
├── stats/          # Statistical functions
├── logical/        # Logical operations
└── ...
```

Create a new file or add to an existing one based on the function's category.

### Step 2: Implement the Function

Use the `#[runtime_builtin]` macro:

```rust
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

/// Compute the square of a number.
#[runtime_builtin(
    name = "square",
    category = "math",
    summary = "Compute the square of each element."
)]
fn square_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::Num(n) => Ok(Value::Num(n * n)),
        Value::Tensor(t) => {
            let data: Vec<f64> = t.data.iter().map(|v| v * v).collect();
            Ok(Value::Tensor(Tensor::new(data, t.shape.clone())?))
        }
        _ => Err("square: unsupported input type".to_string()),
    }
}
```

### Step 3: Register the Module

Add your module to the parent `mod.rs`:

```rust
// In builtins/math/mod.rs
mod square;
```

The `#[runtime_builtin]` macro automatically registers the function via `inventory`.

### Step 4: Rebuild and Test

```bash
cargo build -p runmat-runtime
cargo test -p runmat-runtime square
```

---

## Builtin Metadata

The `#[runtime_builtin]` macro accepts these attributes:

| Attribute | Required | Description |
|-----------|----------|-------------|
| `name` | Yes | MATLAB-visible function name |
| `category` | No | Category for documentation grouping |
| `summary` | No | One-line description |
| `keywords` | No | Comma-separated search keywords |
| `related` | No | Related functions |
| `examples` | No | Usage examples |
| `status` | No | `stable`, `experimental`, or `deprecated` |

Example with full metadata:

```rust
#[runtime_builtin(
    name = "myfunc",
    category = "math",
    summary = "Compute something useful.",
    keywords = "compute,calculate,math",
    related = "otherfunc,anotherfunc",
    status = "stable"
)]
fn myfunc_builtin(x: Value) -> Result<Value, String> {
    // ...
}
```

---

## Handling Value Types

### Common Patterns

**Numeric scalar or tensor:**

```rust
fn process(x: Value) -> Result<Value, String> {
    match x {
        Value::Num(n) => {
            // Handle scalar
            Ok(Value::Num(f(n)))
        }
        Value::Tensor(t) => {
            // Handle tensor element-wise
            let data: Vec<f64> = t.data.iter().map(|v| f(*v)).collect();
            Ok(Value::Tensor(Tensor::new(data, t.shape.clone())?))
        }
        _ => Err("unsupported type".to_string()),
    }
}
```

**Multiple arguments with type coercion:**

```rust
fn add_values(a: Value, b: Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x + y)),
        // Handle other combinations...
        _ => Err("incompatible types".to_string()),
    }
}
```

### Type Conversion Helpers

Use `TryInto` for automatic conversion:

```rust
let n: f64 = (&value).try_into()?;
let s: String = (&value).try_into()?;
```

---

## Testing Your Builtin

### Unit Tests

Add tests in the same file:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_scalar() {
        let result = square_builtin(Value::Num(3.0)).unwrap();
        assert_eq!(result, Value::Num(9.0));
    }

    #[test]
    fn test_square_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = square_builtin(Value::Tensor(t)).unwrap();
        // verify result...
    }
}
```

### Integration Tests

For REPL-level testing, add to `tests/`:

```rust
#[test]
fn test_square_in_repl() {
    let mut engine = ReplEngine::new().unwrap();
    let result = engine.execute("square(4)").unwrap();
    assert_eq!(result.value, Some(Value::Num(16.0)));
}
```

---

## Creating External Extension Crates

For larger extensions, create a separate crate:

### Naming Convention

```
runmat-ext-<feature>    # General extensions
runmat-pass-<name>      # Normalization/transformation passes
runmat-solver-<name>    # Specialized solvers
```

### Crate Structure

```
runmat-ext-myfeature/
├── Cargo.toml
├── src/
│   └── lib.rs
└── README.md
```

### Cargo.toml

```toml
[package]
name = "runmat-ext-myfeature"
version = "0.1.0"
edition = "2021"

[dependencies]
runmat-builtins = { version = "0.2" }
runmat-macros = { version = "0.2" }
inventory = "0.3"
```

### Registration

External crates register builtins the same way. The main binary must depend on the extension crate to link the inventory registrations.

---

## Classification and Review

### Tier Classification

Before submitting, classify your extension:

| Tier | Criteria |
|------|----------|
| **Core** | MATLAB-compatible, well-tested, stable API |
| **Extension** | Useful but experimental, may have assumptions |
| **Research** | Academic/niche, no stability guarantee |

### PR Checklist

- [ ] `cargo fmt` passes
- [ ] `cargo clippy --all-targets -- -D warnings` passes
- [ ] `cargo test` passes
- [ ] Unit tests cover main functionality
- [ ] Error messages are clear and actionable
- [ ] Function name matches MATLAB convention (if applicable)
- [ ] Docstring describes behavior
- [ ] Tier classification noted in PR description

### MATLAB Compatibility

If your builtin mirrors a MATLAB function:

1. Match the function signature
2. Match edge-case behavior where documented
3. Note any intentional deviations
4. Add comparison tests if possible

---

## Examples

### Example 1: Simple Math Builtin

See `crates/runmat-runtime/src/builtins/math/` for examples like `sin`, `cos`, `exp`.

### Example 2: Array Operation

See `crates/runmat-runtime/src/builtins/array/` for examples like `zeros`, `ones`, `reshape`.

### Example 3: String Function

See `crates/runmat-runtime/src/builtins/strings/` for examples like `strcat`, `upper`, `lower`.

---

## Getting Help

- Open an issue for design questions before implementing large features
- Reference existing builtins as templates
- Check [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) for system context
