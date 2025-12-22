//! C ABI types for RunMat embedding and FFI.
//!
//! This crate defines the stable C ABI types shared between `runmat-embed`
//! (for embedding RunMat from C/C++) and `runmat-ffi` (for calling native
//! code from RunMat).

use std::ffi::c_char;

/// Status codes returned by all C ABI functions.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RmStatus {
    /// Success
    Ok = 0,
    /// Runtime error (parse failure, execution error, etc.)
    RuntimeError = 1,
    /// Type mismatch error
    TypeError = 2,
    /// Internal error (panic, unexpected state)
    InternalError = 3,
    /// Invalid argument passed to function
    InvalidArgument = 4,
}

/// Error information returned by C ABI functions.
///
/// The message and backtrace strings are UTF-8 encoded and null-terminated.
/// They are owned by the context and valid until the next call or context free.
#[repr(C)]
pub struct RmError {
    /// Error code (mirrors RmStatus or finer-grained codes)
    pub code: i32,
    /// UTF-8 error message (null-terminated, owned by context)
    pub message: *const c_char,
    /// Length of message in bytes (excluding null terminator)
    pub message_len: usize,
    /// UTF-8 backtrace (null-terminated, may be null)
    pub backtrace: *const c_char,
    /// Length of backtrace in bytes (excluding null terminator)
    pub backtrace_len: usize,
}

impl Default for RmError {
    fn default() -> Self {
        Self {
            code: 0,
            message: std::ptr::null(),
            message_len: 0,
            backtrace: std::ptr::null(),
            backtrace_len: 0,
        }
    }
}

/// Base numeric types supported by the ABI.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RmBaseType {
    F64,
    F32,
    I32,
    I64,
    U8,
    Bool,
    ComplexF32,
    ComplexF64,
}

/// Dense 2D column-major array of f64 values.
///
/// Memory layout: column-major (Fortran order), i.e., element at (row, col)
/// is at offset `row + col * rows`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RmArrayF64 {
    /// Pointer to the first element
    pub data: *mut f64,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
}

/// Dense 2D column-major array of f32 values.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RmArrayF32 {
    pub data: *mut f32,
    pub rows: usize,
    pub cols: usize,
}

/// Dense 2D column-major array of i32 values.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RmArrayI32 {
    pub data: *mut i32,
    pub rows: usize,
    pub cols: usize,
}

/// Dense 2D column-major array of i64 values.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RmArrayI64 {
    pub data: *mut i64,
    pub rows: usize,
    pub cols: usize,
}

/// Opaque context handle.
///
/// Created via `rm_context_new()`, freed via `rm_context_free()`.
/// All values and error strings are owned by the context.
#[repr(C)]
pub struct RmContext {
    _private: [u8; 0],
}

/// Opaque value handle.
///
/// Represents a RunMat value (scalar, matrix, string, etc.).
/// Owned by the context that created it.
#[repr(C)]
pub struct RmValue {
    _private: [u8; 0],
}

// ABI version for compatibility checking
pub const RUNMAT_ABI_VERSION: u32 = 1;

/// Returns the ABI version.
#[no_mangle]
pub extern "C" fn rm_get_abi_version() -> u32 {
    RUNMAT_ABI_VERSION
}
