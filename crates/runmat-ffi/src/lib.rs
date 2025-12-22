//! Foreign Function Interface for calling native C libraries from RunMat.
//!
//! This crate provides the `ffi.call` builtin function that allows RunMat code
//! to call functions in native shared libraries (.dll/.so/.dylib).
//!
//! # Example (RunMat)
//!
//! ```matlab
//! % Load a library and call a function
//! result = ffi_call("mylib", "add_vectors", x, y);
//! ```
//!
//! # Signature Files
//!
//! Type-safe FFI calls can be made using `.ffi` signature files:
//!
//! ```text
//! # mymath.ffi
//! add: (f64, f64) -> f64
//! square: (f64) -> f64
//! scale_array: (ptr<f64>, usize, usize, f64, ptr_mut<f64>) -> i32
//! ```
//!
//! Load with: `ffi_load("mymath", "mymath.ffi")`
//!
//! # Supported Function Signatures
//!
//! Native functions must follow a specific ABI:
//!
//! ```c
//! // Simple scalar function
//! double my_func(double x, double y);
//!
//! // Array function (column-major, returns status)
//! int process_array(
//!     const double* input, size_t rows, size_t cols,
//!     double* output,
//!     size_t* out_rows, size_t* out_cols
//! );
//! ```

mod library;
mod parser;
mod registry;
mod types;

pub mod builtins;

pub use library::NativeLibrary;
pub use parser::{ParseError, SignatureFile};
pub use registry::LibraryRegistry;
pub use types::{FfiSignature, FfiType};
