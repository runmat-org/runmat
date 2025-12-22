//! `ffi_call` builtin - Call a function in a native library.

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::registry::{get_function_signature, global_registry, load_library};
use crate::types::FfiSignature;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "ffi_call"
category: "ffi"
keywords: ["ffi", "call", "native", "c", "library", "function", "signature"]
summary: "Call a function in a native shared library."
---

# ffi_call

Call a function in a native shared library.

## Syntax

```matlab
result = ffi_call("libname", "funcname", arg1, arg2, ...)
```

## Description

`ffi_call` invokes a C function from a previously loaded native library.

### Type-Safe Calls

When a library is loaded with a signature file (`ffi_load("lib", "lib.ffi")`),
function calls are validated against the declared signatures. This provides:

- Argument count validation
- Type checking (where possible)
- Better error messages

### Supported Function Signatures

Currently supports these native function types:

1. **Scalar functions**: `double func(double, double, ...)`
2. **Unary functions**: `double func(double)`
3. **Array functions**: Functions with pointer arguments

## Examples

```matlab
% Load library (auto-loaded on first call if not already loaded)
result = ffi_call("mymath", "add", 1.0, 2.0);  % Returns 3.0

% Call a unary function
y = ffi_call("mymath", "square", 4.0);  % Returns 16.0
```

## Notes

- Libraries are automatically loaded on first use if not already loaded.
- If no signature file is loaded, types are inferred from argument types.
- Arrays are passed as pointers with row/column counts.
"#;

/// Call a function in a native library.
///
/// Syntax: ffi_call("libname", "funcname", arg1, arg2, ...)
#[runtime_builtin(
    name = "ffi_call",
    category = "ffi",
    summary = "Call a function in a native shared library.",
    keywords = "ffi,call,native,c,library,signature"
)]
pub fn ffi_call_builtin(lib_name: Value, func_name: Value, rest: Vec<Value>) -> Result<Value, String> {
    // Extract library name
    let lib_name_str = extract_string(&lib_name, "library name")?;

    // Extract function name
    let func_name_str = extract_string(&func_name, "function name")?;

    // Ensure library is loaded
    load_library(&lib_name_str)?;

    // Check if we have an explicit signature for this function
    let signature = get_function_signature(&lib_name_str, &func_name_str)?;

    // Get the library from registry
    let registry = global_registry()
        .lock()
        .map_err(|_| "Failed to acquire library registry lock")?;

    let library = registry
        .get(&lib_name_str)
        .ok_or_else(|| format!("Library '{}' not found in registry", lib_name_str))?;

    // If we have a signature, validate and use it
    if let Some(ref sig) = signature {
        return call_with_signature(library, sig, &rest);
    }

    // Otherwise, infer from argument count/types (legacy behavior)
    match rest.len() {
        0 => call_nullary(library, &func_name_str),
        1 => call_unary(library, &func_name_str, &rest[0]),
        2 => call_binary(library, &func_name_str, &rest[0], &rest[1]),
        n => call_variadic(library, &func_name_str, &rest, n),
    }
}

/// Call a function using an explicit signature.
fn call_with_signature(
    library: &crate::library::NativeLibrary,
    sig: &FfiSignature,
    args: &[Value],
) -> Result<Value, String> {
    // Validate argument count
    let expected_scalar_args = sig.args.iter().filter(|t| !t.is_pointer()).count();

    // For all-scalar signatures, validate exact count
    if sig.is_all_f64_scalar() {
        if args.len() != sig.args.len() {
            return Err(format!(
                "ffi_call: function '{}' expects {} arguments, got {}",
                sig.name,
                sig.args.len(),
                args.len()
            ));
        }
        // Use existing scalar dispatch
        return match args.len() {
            0 => call_nullary(library, &sig.name),
            1 => call_scalar_unary(library, &sig.name, &args[0]),
            2 => call_binary(library, &sig.name, &args[0], &args[1]),
            _ => call_variadic(library, &sig.name, args, args.len()),
        };
    }

    // For signatures with pointers, we need more sophisticated dispatch
    // Check if this looks like an array function pattern
    if has_array_pattern(sig) {
        return call_array_with_signature(library, sig, args);
    }

    // Fall back to scalar dispatch if we can extract all f64
    if args.len() == expected_scalar_args {
        return match args.len() {
            0 => call_nullary(library, &sig.name),
            1 => call_scalar_unary(library, &sig.name, &args[0]),
            2 => call_binary(library, &sig.name, &args[0], &args[1]),
            _ => call_variadic(library, &sig.name, args, args.len()),
        };
    }

    Err(format!(
        "ffi_call: cannot dispatch function '{}' with signature {} and {} arguments",
        sig.name, sig, args.len()
    ))
}

/// Check if signature matches common array function patterns.
fn has_array_pattern(sig: &FfiSignature) -> bool {
    sig.args.iter().any(|t| t.is_pointer())
}

/// Call an array function using an explicit signature.
fn call_array_with_signature(
    library: &crate::library::NativeLibrary,
    sig: &FfiSignature,
    args: &[Value],
) -> Result<Value, String> {
    // Common pattern: (ptr<f64>, usize, usize, ..., ptr_mut<f64>) -> i32
    // For now, support unary array: one input matrix, optional scalars, one output

    if args.len() != 1 {
        return Err(format!(
            "ffi_call: array function '{}' currently only supports single matrix argument",
            sig.name
        ));
    }

    let input = match &args[0] {
        Value::Tensor(t) => t,
        _ => return Err(format!("ffi_call: expected matrix argument for '{}'", sig.name)),
    };

    // Use the existing array call logic
    call_unary_array(library, &sig.name, input)
}

/// Call a unary scalar function (ensures scalar dispatch even for tensors).
fn call_scalar_unary(
    library: &crate::library::NativeLibrary,
    func_name: &str,
    arg: &Value,
) -> Result<Value, String> {
    type UnaryFn = unsafe extern "C" fn(f64) -> f64;
    let x = extract_f64(arg)?;
    let func: libloading::Symbol<UnaryFn> = unsafe { library.get_function(func_name)? };
    let result = unsafe { func(x) };
    Ok(Value::Num(result))
}

/// Extract a string from a Value.
fn extract_string(value: &Value, context: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) => Ok(ca.data.iter().collect()),
        _ => Err(format!("ffi_call: {} must be a string", context)),
    }
}

/// Extract f64 from a Value.
fn extract_f64(value: &Value) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) if t.rows == 1 && t.cols == 1 => Ok(t.data[0]),
        _ => Err("ffi_call: expected scalar numeric argument".to_string()),
    }
}

/// Call a nullary function: () -> f64
fn call_nullary(
    library: &crate::library::NativeLibrary,
    func_name: &str,
) -> Result<Value, String> {
    type NullaryFn = unsafe extern "C" fn() -> f64;

    let func: libloading::Symbol<NullaryFn> = unsafe { library.get_function(func_name)? };
    let result = unsafe { func() };
    Ok(Value::Num(result))
}

/// Call a unary function: f64 -> f64
fn call_unary(
    library: &crate::library::NativeLibrary,
    func_name: &str,
    arg: &Value,
) -> Result<Value, String> {
    // Check if argument is a scalar or array
    match arg {
        Value::Tensor(t) if t.rows > 1 || t.cols > 1 => {
            // Array argument - try array function signature
            call_unary_array(library, func_name, t)
        }
        _ => {
            // Scalar argument
            type UnaryFn = unsafe extern "C" fn(f64) -> f64;
            let x = extract_f64(arg)?;
            let func: libloading::Symbol<UnaryFn> = unsafe { library.get_function(func_name)? };
            let result = unsafe { func(x) };
            Ok(Value::Num(result))
        }
    }
}

/// Call a unary array function: (ptr, rows, cols) -> status, with output array
fn call_unary_array(
    library: &crate::library::NativeLibrary,
    func_name: &str,
    input: &Tensor,
) -> Result<Value, String> {
    // Try the array signature: int func(const double* in, size_t rows, size_t cols,
    //                                   double* out, size_t* out_rows, size_t* out_cols)
    type ArrayUnaryFn = unsafe extern "C" fn(
        *const f64,
        usize,
        usize,
        *mut f64,
        *mut usize,
        *mut usize,
    ) -> i32;

    let func: libloading::Symbol<ArrayUnaryFn> =
        unsafe { library.get_function(func_name)? };

    // Allocate output buffer (same size as input initially)
    let mut output = vec![0.0f64; input.data.len()];
    let mut out_rows = input.rows;
    let mut out_cols = input.cols;

    let status = unsafe {
        func(
            input.data.as_ptr(),
            input.rows,
            input.cols,
            output.as_mut_ptr(),
            &mut out_rows,
            &mut out_cols,
        )
    };

    if status != 0 {
        return Err(format!(
            "ffi_call: native function '{}' returned error code {}",
            func_name, status
        ));
    }

    // Truncate output if needed
    output.truncate(out_rows * out_cols);

    let tensor = Tensor::new_2d(output, out_rows, out_cols)
        .map_err(|e| format!("ffi_call: failed to create output tensor: {}", e))?;

    Ok(Value::Tensor(tensor))
}

/// Call a binary function: (f64, f64) -> f64
fn call_binary(
    library: &crate::library::NativeLibrary,
    func_name: &str,
    arg1: &Value,
    arg2: &Value,
) -> Result<Value, String> {
    type BinaryFn = unsafe extern "C" fn(f64, f64) -> f64;

    let x = extract_f64(arg1)?;
    let y = extract_f64(arg2)?;

    let func: libloading::Symbol<BinaryFn> = unsafe { library.get_function(func_name)? };
    let result = unsafe { func(x, y) };
    Ok(Value::Num(result))
}

/// Call a variadic scalar function.
fn call_variadic(
    library: &crate::library::NativeLibrary,
    func_name: &str,
    args: &[Value],
    _n: usize,
) -> Result<Value, String> {
    // For now, support up to 8 scalar arguments
    let scalars: Result<Vec<f64>, String> = args.iter().map(extract_f64).collect();
    let scalars = scalars?;

    match scalars.len() {
        3 => {
            type Fn3 = unsafe extern "C" fn(f64, f64, f64) -> f64;
            let func: libloading::Symbol<Fn3> = unsafe { library.get_function(func_name)? };
            let result = unsafe { func(scalars[0], scalars[1], scalars[2]) };
            Ok(Value::Num(result))
        }
        4 => {
            type Fn4 = unsafe extern "C" fn(f64, f64, f64, f64) -> f64;
            let func: libloading::Symbol<Fn4> = unsafe { library.get_function(func_name)? };
            let result = unsafe { func(scalars[0], scalars[1], scalars[2], scalars[3]) };
            Ok(Value::Num(result))
        }
        5 => {
            type Fn5 = unsafe extern "C" fn(f64, f64, f64, f64, f64) -> f64;
            let func: libloading::Symbol<Fn5> = unsafe { library.get_function(func_name)? };
            let result =
                unsafe { func(scalars[0], scalars[1], scalars[2], scalars[3], scalars[4]) };
            Ok(Value::Num(result))
        }
        _ => Err(format!(
            "ffi_call: unsupported number of arguments: {}",
            scalars.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_string() {
        let s = Value::String("test".to_string());
        assert_eq!(extract_string(&s, "test").unwrap(), "test");
    }

    #[test]
    fn test_extract_f64() {
        assert_eq!(extract_f64(&Value::Num(3.14)).unwrap(), 3.14);
    }
}
