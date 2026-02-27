//! MATLAB-compatible `isequal` builtin for RunMat.
//!
//! Tests whether all input arrays have the same size, class, and content.

use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "isequal";
const IDENT_NOT_ENOUGH_INPUTS: &str = "MATLAB:isequal:NotEnoughInputs";

fn isequal_error(message: impl Into<String>, identifier: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(identifier)
        .build()
}

/// Compares all input values for equality.
///
/// Returns `true` if all inputs have the same size, class, and content.
/// Returns `false` otherwise. NaN values are NOT considered equal.
#[runtime_builtin(
    name = "isequal",
    category = "logical/rel",
    summary = "Test arrays for equality in size, class, and content.",
    keywords = "isequal,equality,comparison,logical",
    accel = "cpu",
    builtin_path = "crate::builtins::logical::rel::isequal"
)]
async fn isequal_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(isequal_error(
            "isequal: requires at least two input arguments",
            IDENT_NOT_ENOUGH_INPUTS,
        ));
    }

    // Gather all values to host if needed
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gather_value(arg).await?);
    }

    // Compare first value against all others
    let first = &gathered[0];
    for other in gathered.iter().skip(1) {
        if !values_equal(first, other) {
            return Ok(Value::Bool(false));
        }
    }

    Ok(Value::Bool(true))
}

async fn gather_value(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| {
                    isequal_error(format!("{BUILTIN_NAME}: {err}"), IDENT_NOT_ENOUGH_INPUTS)
                })?;
            Ok(Value::Tensor(tensor))
        }
        other => Ok(other),
    }
}

/// Compare two values for equality (same size, class, and content).
/// NaN values are NOT considered equal.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        // Numeric scalars
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Int(x), Value::Int(y)) => x == y,

        // Promote scalar to match types
        (Value::Num(x), Value::Bool(y)) => *x == if *y { 1.0 } else { 0.0 },
        (Value::Bool(x), Value::Num(y)) => (if *x { 1.0 } else { 0.0 }) == *y,
        (Value::Num(x), Value::Int(y)) => *x == y.to_f64(),
        (Value::Int(x), Value::Num(y)) => x.to_f64() == *y,
        (Value::Bool(x), Value::Int(y)) => (if *x { 1 } else { 0 }) == y.to_i64(),
        (Value::Int(x), Value::Bool(y)) => x.to_i64() == if *y { 1 } else { 0 },

        // Complex scalars
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => ar == br && ai == bi,
        (Value::Num(x), Value::Complex(br, bi)) => *x == *br && *bi == 0.0,
        (Value::Complex(ar, ai), Value::Num(y)) => *ar == *y && *ai == 0.0,

        // Tensors
        (Value::Tensor(a), Value::Tensor(b)) => tensors_equal(a, b),
        (Value::Tensor(t), Value::Num(n)) => scalar_tensor_equal(t, *n),
        (Value::Num(n), Value::Tensor(t)) => scalar_tensor_equal(t, *n),

        // Complex tensors
        (Value::ComplexTensor(a), Value::ComplexTensor(b)) => complex_tensors_equal(a, b),

        // Logical arrays
        (Value::LogicalArray(a), Value::LogicalArray(b)) => logical_arrays_equal(a, b),
        (Value::Bool(x), Value::LogicalArray(a)) => scalar_logical_equal(a, *x),
        (Value::LogicalArray(a), Value::Bool(x)) => scalar_logical_equal(a, *x),

        // Character arrays
        (Value::CharArray(a), Value::CharArray(b)) => char_arrays_equal(a, b),

        // Strings
        (Value::String(a), Value::String(b)) => a == b,
        (Value::StringArray(a), Value::StringArray(b)) => string_arrays_equal(a, b),
        (Value::String(a), Value::StringArray(b)) => {
            b.shape == vec![1, 1] && b.data.len() == 1 && b.data[0] == *a
        }
        (Value::StringArray(a), Value::String(b)) => {
            a.shape == vec![1, 1] && a.data.len() == 1 && a.data[0] == *b
        }

        // Cells
        (Value::Cell(a), Value::Cell(b)) => a.shape == b.shape && cells_equal(a, b),

        // Structs
        (Value::Struct(a), Value::Struct(b)) => structs_equal(a, b),

        // Different types are not equal
        _ => false,
    }
}

fn tensors_equal(a: &Tensor, b: &Tensor) -> bool {
    if a.shape != b.shape {
        return false;
    }
    if a.data.len() != b.data.len() {
        return false;
    }
    // NaN != NaN in isequal (use isequaln for NaN equality)
    a.data.iter().zip(b.data.iter()).all(|(x, y)| x == y)
}

fn scalar_tensor_equal(t: &Tensor, n: f64) -> bool {
    if t.data.len() != 1 {
        return false;
    }
    t.data[0] == n
}

fn complex_tensors_equal(a: &ComplexTensor, b: &ComplexTensor) -> bool {
    if a.shape != b.shape {
        return false;
    }
    if a.data.len() != b.data.len() {
        return false;
    }
    a.data
        .iter()
        .zip(b.data.iter())
        .all(|((ar, ai), (br, bi))| ar == br && ai == bi)
}

fn logical_arrays_equal(a: &LogicalArray, b: &LogicalArray) -> bool {
    if a.shape != b.shape {
        return false;
    }
    a.data == b.data
}

fn scalar_logical_equal(a: &LogicalArray, x: bool) -> bool {
    if a.data.len() != 1 {
        return false;
    }
    (a.data[0] != 0) == x
}

fn char_arrays_equal(a: &CharArray, b: &CharArray) -> bool {
    a.rows == b.rows && a.cols == b.cols && a.data == b.data
}

fn string_arrays_equal(a: &StringArray, b: &StringArray) -> bool {
    if a.shape != b.shape {
        return false;
    }
    a.data == b.data
}

fn cells_equal(a: &CellArray, b: &CellArray) -> bool {
    if a.data.len() != b.data.len() {
        return false;
    }
    a.data
        .iter()
        .zip(b.data.iter())
        .all(|(x, y)| values_equal(x, y))
}

fn structs_equal(a: &runmat_builtins::StructValue, b: &runmat_builtins::StructValue) -> bool {
    if a.fields.len() != b.fields.len() {
        return false;
    }
    a.fields
        .iter()
        .zip(b.fields.iter())
        .all(|((key_a, val_a), (key_b, val_b))| key_a == key_b && values_equal(val_a, val_b))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::CellArray;

    fn run_isequal(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(isequal_builtin(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_two_scalars_equal() {
        let result = run_isequal(vec![Value::Num(5.0), Value::Num(5.0)]).expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_two_scalars_not_equal() {
        let result = run_isequal(vec![Value::Num(5.0), Value::Num(4.0)]).expect("isequal");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_three_args_all_equal() {
        let result =
            run_isequal(vec![Value::Num(3.0), Value::Num(3.0), Value::Num(3.0)]).expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_three_args_one_different() {
        let result =
            run_isequal(vec![Value::Num(3.0), Value::Num(3.0), Value::Num(4.0)]).expect("isequal");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_tensors_equal() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let t2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = run_isequal(vec![Value::Tensor(t1), Value::Tensor(t2)]).expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_tensors_different_shape() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let t2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = run_isequal(vec![Value::Tensor(t1), Value::Tensor(t2)]).expect("isequal");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_tensors_different_values() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let t2 = Tensor::new(vec![1.0, 2.0, 4.0], vec![1, 3]).unwrap();
        let result = run_isequal(vec![Value::Tensor(t1), Value::Tensor(t2)]).expect("isequal");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_empty_arrays() {
        // Test that empty arrays are equal
        let empty_a = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let empty_b = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result =
            run_isequal(vec![Value::Tensor(empty_a), Value::Tensor(empty_b)]).expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_empty_cell_arrays() {
        // Test the cell example from the failing test: cell(2,2) elements should be []
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let c1 = CellArray::new(vec![Value::Tensor(empty.clone()); 4], 2, 2).unwrap();
        let c2 = CellArray::new(vec![Value::Tensor(empty); 4], 2, 2).unwrap();
        let result = run_isequal(vec![Value::Cell(c1), Value::Cell(c2)]).expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_cell_element_with_empty() {
        // Test isequal(C{1,1}, [], C{2,2}, []) pattern
        let empty_a = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let empty_b = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let empty_c = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let empty_d = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = run_isequal(vec![
            Value::Tensor(empty_a),
            Value::Tensor(empty_b),
            Value::Tensor(empty_c),
            Value::Tensor(empty_d),
        ])
        .expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_nan_not_equal() {
        // In isequal, NaN != NaN (use isequaln for NaN equality)
        let result =
            run_isequal(vec![Value::Num(f64::NAN), Value::Num(f64::NAN)]).expect("isequal");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_strings() {
        let result = run_isequal(vec![
            Value::String("hello".into()),
            Value::String("hello".into()),
        ])
        .expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_different_types() {
        let result =
            run_isequal(vec![Value::Num(5.0), Value::String("5".into())]).expect("isequal");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_not_enough_args() {
        let err = run_isequal(vec![Value::Num(5.0)]).unwrap_err();
        assert!(err.message().contains("at least two"));
        assert_eq!(err.identifier(), Some(IDENT_NOT_ENOUGH_INPUTS));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_bool_and_num() {
        let result = run_isequal(vec![Value::Bool(true), Value::Num(1.0)]).expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_complex() {
        let result =
            run_isequal(vec![Value::Complex(1.0, 2.0), Value::Complex(1.0, 2.0)]).expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }
}
