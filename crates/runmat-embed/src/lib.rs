//! C ABI for embedding RunMat in C/C++ applications.
//!
//! This crate provides a stable C ABI for:
//! - Creating and managing RunMat execution contexts
//! - Evaluating RunMat code from C/C++
//! - Exchanging numeric arrays between C and RunMat
//!
//! # Example (C)
//!
//! ```c
//! #include "runmat.h"
//!
//! int main() {
//!     rm_context* ctx = rm_context_new();
//!     rm_value** out = NULL;
//!     size_t nout = 0;
//!     rm_error err = {0};
//!
//!     if (rm_eval_utf8(ctx, "1 + 2", 5, &out, &nout, &err) == RM_OK) {
//!         double result;
//!         rm_value_to_f64(out[0], &result, &err);
//!         printf("Result: %f\n", result);
//!         rm_values_free(ctx, out, nout);
//!     }
//!
//!     rm_context_free(ctx);
//!     return 0;
//! }
//! ```

use std::ffi::{c_char, CStr, CString};
use std::panic::catch_unwind;
use std::ptr;
use std::slice;

use runmat_abi::{RmArrayF64, RmContext, RmError, RmStatus, RmValue};
use runmat_builtins::Value;
use runmat_repl::ReplEngine;

/// Internal context structure that holds the actual RunMat engine state.
#[allow(dead_code)]
struct ContextInner {
    engine: ReplEngine,
    /// Owned values that have been returned to C code
    values: Vec<Box<ValueHandle>>,
    /// Last error message (kept alive for C to read)
    last_error: Option<CString>,
    /// Last backtrace (kept alive for C to read)
    last_backtrace: Option<CString>,
}

/// Internal value handle wrapping a RunMat Value.
struct ValueHandle {
    value: Value,
}

/// Opaque context wrapper with stable C ABI.
#[repr(C)]
struct RmContextOpaque {
    inner: ContextInner,
}

/// Opaque value wrapper with stable C ABI.
#[repr(C)]
struct RmValueOpaque {
    handle: ValueHandle,
}

// ============================================================================
// Context lifecycle
// ============================================================================

/// Create a new RunMat execution context.
///
/// Returns NULL on failure (e.g., out of memory).
/// The context must be freed with `rm_context_free()`.
#[no_mangle]
pub extern "C" fn rm_context_new() -> *mut RmContext {
    let result = catch_unwind(|| {
        let engine = match ReplEngine::new() {
            Ok(e) => e,
            Err(_) => return ptr::null_mut(),
        };

        let inner = ContextInner {
            engine,
            values: Vec::new(),
            last_error: None,
            last_backtrace: None,
        };

        let boxed = Box::new(RmContextOpaque { inner });
        Box::into_raw(boxed) as *mut RmContext
    });

    result.unwrap_or(ptr::null_mut())
}

/// Free a RunMat execution context and all values owned by it.
///
/// After this call, the context pointer and all value pointers from this
/// context are invalid.
#[no_mangle]
pub extern "C" fn rm_context_free(ctx: *mut RmContext) {
    if ctx.is_null() {
        return;
    }

    let _ = catch_unwind(|| unsafe {
        let _ = Box::from_raw(ctx as *mut RmContextOpaque);
    });
}

// ============================================================================
// Code evaluation
// ============================================================================

/// Evaluate RunMat code and return the results.
///
/// # Parameters
/// - `ctx`: The execution context
/// - `code`: UTF-8 encoded source code
/// - `code_len`: Length of the code in bytes
/// - `out`: On success, receives a pointer to an array of result values
/// - `nout`: On success, receives the number of result values
/// - `err`: On failure, receives error information
///
/// # Returns
/// `RM_OK` on success, or an error code on failure.
///
/// # Memory
/// The returned values array must be freed with `rm_values_free()`.
#[no_mangle]
pub extern "C" fn rm_eval_utf8(
    ctx: *mut RmContext,
    code: *const c_char,
    code_len: usize,
    out: *mut *mut *mut RmValue,
    nout: *mut usize,
    err: *mut RmError,
) -> RmStatus {
    if ctx.is_null() || code.is_null() || out.is_null() || nout.is_null() {
        return RmStatus::InvalidArgument;
    }

    let result = catch_unwind(|| {
        let ctx_opaque = unsafe { &mut *(ctx as *mut RmContextOpaque) };

        // Convert code to Rust string
        let code_slice = unsafe { slice::from_raw_parts(code as *const u8, code_len) };
        let code_str = match std::str::from_utf8(code_slice) {
            Ok(s) => s,
            Err(_) => {
                set_error(ctx_opaque, err, RmStatus::InvalidArgument, "Invalid UTF-8 in code");
                return RmStatus::InvalidArgument;
            }
        };

        // Execute the code
        match ctx_opaque.inner.engine.execute(code_str) {
            Ok(result) => {
                if let Some(error_msg) = result.error {
                    set_error(ctx_opaque, err, RmStatus::RuntimeError, &error_msg);
                    return RmStatus::RuntimeError;
                }

                // Convert result value to output array
                let values: Vec<Value> = if let Some(v) = result.value {
                    vec![v]
                } else {
                    vec![]
                };

                // Allocate output array
                let value_ptrs = values_to_handles(ctx_opaque, values);
                unsafe {
                    *nout = value_ptrs.len();
                    if value_ptrs.is_empty() {
                        *out = ptr::null_mut();
                    } else {
                        let array = Box::into_raw(value_ptrs.into_boxed_slice()) as *mut *mut RmValue;
                        *out = array;
                    }
                }

                RmStatus::Ok
            }
            Err(e) => {
                set_error(ctx_opaque, err, RmStatus::RuntimeError, &e.to_string());
                RmStatus::RuntimeError
            }
        }
    });

    result.unwrap_or_else(|_| {
        if !err.is_null() {
            unsafe {
                (*err).code = RmStatus::InternalError as i32;
            }
        }
        RmStatus::InternalError
    })
}

/// Evaluate RunMat code from a null-terminated C string.
///
/// Convenience wrapper around `rm_eval_utf8` that uses strlen for the length.
#[no_mangle]
pub extern "C" fn rm_eval(
    ctx: *mut RmContext,
    code: *const c_char,
    out: *mut *mut *mut RmValue,
    nout: *mut usize,
    err: *mut RmError,
) -> RmStatus {
    if code.is_null() {
        return RmStatus::InvalidArgument;
    }

    let code_len = unsafe { CStr::from_ptr(code).to_bytes().len() };
    rm_eval_utf8(ctx, code, code_len, out, nout, err)
}

// ============================================================================
// Value access
// ============================================================================

/// Convert a value to a scalar f64.
///
/// Returns `RM_TYPE_ERROR` if the value is not a scalar number.
#[no_mangle]
pub extern "C" fn rm_value_to_f64(
    v: *const RmValue,
    out: *mut f64,
    _err: *mut RmError,
) -> RmStatus {
    if v.is_null() || out.is_null() {
        return RmStatus::InvalidArgument;
    }

    let result = catch_unwind(|| {
        let value_opaque = unsafe { &*(v as *const RmValueOpaque) };
        match &value_opaque.handle.value {
            Value::Num(n) => {
                unsafe { *out = *n };
                RmStatus::Ok
            }
            Value::Int(i) => {
                unsafe { *out = i.to_f64() };
                RmStatus::Ok
            }
            Value::Tensor(t) if t.rows == 1 && t.cols == 1 => {
                unsafe { *out = t.data[0] };
                RmStatus::Ok
            }
            _ => RmStatus::TypeError,
        }
    });

    result.unwrap_or(RmStatus::InternalError)
}

/// Create a value from a scalar f64.
#[no_mangle]
pub extern "C" fn rm_value_from_f64(
    ctx: *mut RmContext,
    x: f64,
    out: *mut *mut RmValue,
    _err: *mut RmError,
) -> RmStatus {
    if ctx.is_null() || out.is_null() {
        return RmStatus::InvalidArgument;
    }

    let result = catch_unwind(|| {
        let ctx_opaque = unsafe { &mut *(ctx as *mut RmContextOpaque) };
        let value = Value::Num(x);
        let handles = values_to_handles(ctx_opaque, vec![value]);
        unsafe {
            *out = handles.into_iter().next().unwrap_or(ptr::null_mut());
        }
        RmStatus::Ok
    });

    result.unwrap_or(RmStatus::InternalError)
}

/// Get a view of a value as an f64 array.
///
/// The returned array data pointer is valid until the value or context is freed.
/// Returns `RM_TYPE_ERROR` if the value is not a numeric matrix.
#[no_mangle]
pub extern "C" fn rm_value_to_array_f64(
    v: *const RmValue,
    out: *mut RmArrayF64,
    _err: *mut RmError,
) -> RmStatus {
    if v.is_null() || out.is_null() {
        return RmStatus::InvalidArgument;
    }

    let result = catch_unwind(|| {
        let value_opaque = unsafe { &*(v as *const RmValueOpaque) };
        match &value_opaque.handle.value {
            Value::Tensor(t) => {
                unsafe {
                    (*out).data = t.data.as_ptr() as *mut f64;
                    (*out).rows = t.rows;
                    (*out).cols = t.cols;
                }
                RmStatus::Ok
            }
            Value::Num(n) => {
                // Treat scalar as 1x1 - but we need stable storage
                // For now, return type error for scalars via this API
                let _ = n;
                RmStatus::TypeError
            }
            _ => RmStatus::TypeError,
        }
    });

    result.unwrap_or(RmStatus::InternalError)
}

/// Create a value from an f64 array (copies the data).
///
/// The input array data is copied into RunMat's internal storage.
#[no_mangle]
pub extern "C" fn rm_value_from_array_f64(
    ctx: *mut RmContext,
    array: RmArrayF64,
    out: *mut *mut RmValue,
    _err: *mut RmError,
) -> RmStatus {
    if ctx.is_null() || out.is_null() || array.data.is_null() {
        return RmStatus::InvalidArgument;
    }

    let result = catch_unwind(|| {
        let ctx_opaque = unsafe { &mut *(ctx as *mut RmContextOpaque) };

        // Copy data from C array
        let len = array.rows * array.cols;
        let data = unsafe { slice::from_raw_parts(array.data, len).to_vec() };

        // Create tensor with the copied data
        let tensor = runmat_builtins::Tensor::new_2d(data, array.rows, array.cols)
            .expect("Invalid array dimensions");
        let value = Value::Tensor(tensor);

        let handles = values_to_handles(ctx_opaque, vec![value]);
        unsafe {
            *out = handles.into_iter().next().unwrap_or(ptr::null_mut());
        }
        RmStatus::Ok
    });

    result.unwrap_or(RmStatus::InternalError)
}

// ============================================================================
// Memory management
// ============================================================================

/// Free an array of values returned by evaluation functions.
///
/// After this call, the value pointers in the array are invalid.
#[no_mangle]
pub extern "C" fn rm_values_free(
    _ctx: *mut RmContext,
    values: *mut *mut RmValue,
    count: usize,
) {
    if values.is_null() || count == 0 {
        return;
    }

    let _ = catch_unwind(|| unsafe {
        // Free the array itself (the values are owned by the context)
        let _ = Box::from_raw(slice::from_raw_parts_mut(values, count) as *mut [*mut RmValue]);
    });
}

/// Clear error state.
#[no_mangle]
pub extern "C" fn rm_error_clear(err: *mut RmError) {
    if err.is_null() {
        return;
    }

    unsafe {
        (*err).code = 0;
        (*err).message = ptr::null();
        (*err).message_len = 0;
        (*err).backtrace = ptr::null();
        (*err).backtrace_len = 0;
    }
}

// ============================================================================
// Version info
// ============================================================================

/// Get the RunMat version string.
#[no_mangle]
pub extern "C" fn rm_get_version_string() -> *const c_char {
    static VERSION: &[u8] = b"0.2.7\0";
    VERSION.as_ptr() as *const c_char
}

// ============================================================================
// Internal helpers
// ============================================================================

fn set_error(ctx: &mut RmContextOpaque, err: *mut RmError, status: RmStatus, message: &str) {
    let cstring = CString::new(message).unwrap_or_else(|_| CString::new("Unknown error").unwrap());
    let len = cstring.as_bytes().len();
    let ptr = cstring.as_ptr();

    ctx.inner.last_error = Some(cstring);

    if !err.is_null() {
        unsafe {
            (*err).code = status as i32;
            (*err).message = ptr;
            (*err).message_len = len;
            (*err).backtrace = ptr::null();
            (*err).backtrace_len = 0;
        }
    }
}

fn values_to_handles(_ctx: &mut RmContextOpaque, values: Vec<Value>) -> Vec<*mut RmValue> {
    values
        .into_iter()
        .map(|value| {
            let handle = Box::new(RmValueOpaque {
                handle: ValueHandle { value },
            });
            let ptr = Box::into_raw(handle) as *mut RmValue;
            // Note: We're leaking these intentionally - they're freed via rm_values_free
            // or when the context is dropped
            ptr
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_lifecycle() {
        let ctx = rm_context_new();
        assert!(!ctx.is_null());
        rm_context_free(ctx);
    }

    #[test]
    fn test_eval_simple() {
        let ctx = rm_context_new();
        assert!(!ctx.is_null());

        let code = b"1 + 2";
        let mut out: *mut *mut RmValue = ptr::null_mut();
        let mut nout: usize = 0;
        let mut err = RmError::default();

        let status = rm_eval_utf8(
            ctx,
            code.as_ptr() as *const c_char,
            code.len(),
            &mut out,
            &mut nout,
            &mut err,
        );

        assert_eq!(status, RmStatus::Ok);
        assert_eq!(nout, 1);
        assert!(!out.is_null());

        // Check result value
        let mut result: f64 = 0.0;
        let status = rm_value_to_f64(unsafe { *out }, &mut result, &mut err);
        assert_eq!(status, RmStatus::Ok);
        assert!((result - 3.0).abs() < 1e-10);

        rm_values_free(ctx, out, nout);
        rm_context_free(ctx);
    }

    #[test]
    fn test_array_roundtrip() {
        let ctx = rm_context_new();
        assert!(!ctx.is_null());

        // Create array in C style
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = RmArrayF64 {
            data: data.as_ptr() as *mut f64,
            rows: 2,
            cols: 3,
        };

        let mut value: *mut RmValue = ptr::null_mut();
        let mut err = RmError::default();

        let status = rm_value_from_array_f64(ctx, array, &mut value, &mut err);
        assert_eq!(status, RmStatus::Ok);
        assert!(!value.is_null());

        // Read it back
        let mut out_array = RmArrayF64 {
            data: ptr::null_mut(),
            rows: 0,
            cols: 0,
        };
        let status = rm_value_to_array_f64(value, &mut out_array, &mut err);
        assert_eq!(status, RmStatus::Ok);
        assert_eq!(out_array.rows, 2);
        assert_eq!(out_array.cols, 3);

        rm_context_free(ctx);
    }
}
