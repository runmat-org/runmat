//! MEX Compatibility Layer implementation.
//!
//! This module provides MATLAB MEX-like C API functions for working with
//! numeric arrays. It wraps RunMat's internal Tensor type in an mxArray
//! structure that can be used from C code.

use std::ffi::{c_char, c_int, CStr};
use std::panic::catch_unwind;
use std::ptr;

use runmat_builtins::Tensor;

/// Complexity flag for matrix creation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MxComplexity {
    Real = 0,
    Complex = 1,
}

/// Internal mxArray structure.
///
/// This wraps a RunMat Tensor and provides MEX-compatible access.
#[repr(C)]
pub struct MxArray {
    /// The underlying tensor data
    tensor: Tensor,
}

// ============================================================================
// Matrix Creation
// ============================================================================

/// Create a 2D double matrix initialized to zero.
#[no_mangle]
pub extern "C" fn mxCreateDoubleMatrix(
    m: usize,
    n: usize,
    complexity: MxComplexity,
) -> *mut MxArray {
    if complexity != MxComplexity::Real {
        return ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let data = vec![0.0f64; m * n];
        let tensor = Tensor::new_2d(data, m, n).ok()?;
        let array = Box::new(MxArray { tensor });
        Some(Box::into_raw(array))
    });

    result.unwrap_or(None).unwrap_or(ptr::null_mut())
}

/// Create a double scalar (1x1 matrix).
#[no_mangle]
pub extern "C" fn mxCreateDoubleScalar(value: f64) -> *mut MxArray {
    let result = catch_unwind(|| {
        let data = vec![value];
        let tensor = Tensor::new_2d(data, 1, 1).ok()?;
        let array = Box::new(MxArray { tensor });
        Some(Box::into_raw(array))
    });

    result.unwrap_or(None).unwrap_or(ptr::null_mut())
}

// ============================================================================
// Matrix Information
// ============================================================================

/// Get number of rows.
#[no_mangle]
pub extern "C" fn mxGetM(pa: *const MxArray) -> usize {
    if pa.is_null() {
        return 0;
    }

    let result = catch_unwind(|| {
        let array = unsafe { &*pa };
        array.tensor.rows
    });

    result.unwrap_or(0)
}

/// Get number of columns.
#[no_mangle]
pub extern "C" fn mxGetN(pa: *const MxArray) -> usize {
    if pa.is_null() {
        return 0;
    }

    let result = catch_unwind(|| {
        let array = unsafe { &*pa };
        array.tensor.cols
    });

    result.unwrap_or(0)
}

/// Get total number of elements.
#[no_mangle]
pub extern "C" fn mxGetNumberOfElements(pa: *const MxArray) -> usize {
    if pa.is_null() {
        return 0;
    }

    let result = catch_unwind(|| {
        let array = unsafe { &*pa };
        array.tensor.rows * array.tensor.cols
    });

    result.unwrap_or(0)
}

/// Check if array is empty.
#[no_mangle]
pub extern "C" fn mxIsEmpty(pa: *const MxArray) -> bool {
    mxGetNumberOfElements(pa) == 0
}

/// Check if array is a scalar (1x1).
#[no_mangle]
pub extern "C" fn mxIsScalar(pa: *const MxArray) -> bool {
    if pa.is_null() {
        return false;
    }

    let result = catch_unwind(|| {
        let array = unsafe { &*pa };
        array.tensor.rows == 1 && array.tensor.cols == 1
    });

    result.unwrap_or(false)
}

/// Check if array is a double array (always true for our implementation).
#[no_mangle]
pub extern "C" fn mxIsDouble(pa: *const MxArray) -> bool {
    !pa.is_null()
}

// ============================================================================
// Data Access
// ============================================================================

/// Get pointer to real data (column-major order).
#[no_mangle]
pub extern "C" fn mxGetPr(pa: *const MxArray) -> *mut f64 {
    if pa.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let array = unsafe { &*pa };
        array.tensor.data.as_ptr() as *mut f64
    });

    result.unwrap_or(ptr::null_mut())
}

/// Get scalar value from a 1x1 array.
#[no_mangle]
pub extern "C" fn mxGetScalar(pa: *const MxArray) -> f64 {
    if pa.is_null() {
        return f64::NAN;
    }

    let result = catch_unwind(|| {
        let array = unsafe { &*pa };
        if array.tensor.rows == 1 && array.tensor.cols == 1 {
            array.tensor.data[0]
        } else if !array.tensor.data.is_empty() {
            array.tensor.data[0]
        } else {
            f64::NAN
        }
    });

    result.unwrap_or(f64::NAN)
}

// ============================================================================
// Memory Management
// ============================================================================

/// Destroy an mxArray and free its memory.
#[no_mangle]
pub extern "C" fn mxDestroyArray(pa: *mut MxArray) {
    if pa.is_null() {
        return;
    }

    let _ = catch_unwind(|| unsafe {
        let _ = Box::from_raw(pa);
    });
}

/// Duplicate an mxArray.
#[no_mangle]
pub extern "C" fn mxDuplicateArray(pa: *const MxArray) -> *mut MxArray {
    if pa.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let array = unsafe { &*pa };
        let new_tensor = array.tensor.clone();
        let new_array = Box::new(MxArray { tensor: new_tensor });
        Box::into_raw(new_array)
    });

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// MEX Helper Functions
// ============================================================================

/// Print an error message.
#[no_mangle]
pub extern "C" fn mexErrMsgTxt(msg: *const c_char) {
    if msg.is_null() {
        return;
    }

    let _ = catch_unwind(|| {
        let cstr = unsafe { CStr::from_ptr(msg) };
        if let Ok(s) = cstr.to_str() {
            eprintln!("MEX Error: {}", s);
        }
    });
}

/// Print a warning message.
#[no_mangle]
pub extern "C" fn mexWarnMsgTxt(msg: *const c_char) {
    if msg.is_null() {
        return;
    }

    let _ = catch_unwind(|| {
        let cstr = unsafe { CStr::from_ptr(msg) };
        if let Ok(s) = cstr.to_str() {
            eprintln!("MEX Warning: {}", s);
        }
    });
}

/// Print to console (simplified - no format args support).
///
/// Note: This is a simplified implementation that just prints the format string.
/// For full printf support, pass pre-formatted strings.
#[no_mangle]
pub extern "C" fn mexPrintf(fmt: *const c_char) -> c_int {
    if fmt.is_null() {
        return 0;
    }

    let result = catch_unwind(|| {
        let cstr = unsafe { CStr::from_ptr(fmt) };
        if let Ok(s) = cstr.to_str() {
            print!("{}", s);
            s.len() as c_int
        } else {
            0
        }
    });

    result.unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_destroy() {
        let arr = mxCreateDoubleMatrix(3, 4, MxComplexity::Real);
        assert!(!arr.is_null());
        assert_eq!(mxGetM(arr), 3);
        assert_eq!(mxGetN(arr), 4);
        assert_eq!(mxGetNumberOfElements(arr), 12);
        mxDestroyArray(arr);
    }

    #[test]
    fn test_create_scalar() {
        let arr = mxCreateDoubleScalar(3.14);
        assert!(!arr.is_null());
        assert!(mxIsScalar(arr));
        assert!((mxGetScalar(arr) - 3.14).abs() < 1e-10);
        mxDestroyArray(arr);
    }

    #[test]
    fn test_data_access() {
        let arr = mxCreateDoubleMatrix(2, 3, MxComplexity::Real);
        assert!(!arr.is_null());

        let data = mxGetPr(arr);
        assert!(!data.is_null());

        // Write some data
        unsafe {
            for i in 0..6 {
                *data.add(i) = (i + 1) as f64;
            }
        }

        // Read it back
        unsafe {
            assert!(((*data.add(0)) - 1.0).abs() < 1e-10);
            assert!(((*data.add(5)) - 6.0).abs() < 1e-10);
        }

        mxDestroyArray(arr);
    }

    #[test]
    fn test_duplicate() {
        let arr1 = mxCreateDoubleScalar(42.0);
        let arr2 = mxDuplicateArray(arr1);

        assert!(!arr2.is_null());
        assert!((mxGetScalar(arr2) - 42.0).abs() < 1e-10);

        // Modify original shouldn't affect duplicate
        unsafe {
            *mxGetPr(arr1) = 100.0;
        }
        assert!((mxGetScalar(arr2) - 42.0).abs() < 1e-10);

        mxDestroyArray(arr1);
        mxDestroyArray(arr2);
    }

    #[test]
    fn test_null_safety() {
        assert_eq!(mxGetM(ptr::null()), 0);
        assert_eq!(mxGetN(ptr::null()), 0);
        assert!(mxGetPr(ptr::null()).is_null());
        assert!(mxGetScalar(ptr::null()).is_nan());
        mxDestroyArray(ptr::null_mut()); // Should not crash
    }
}
