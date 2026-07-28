//! MATLAB-compatible `isequal` builtin for RunMat.
//!
//! Tests whether all input arrays have the same size, class, and content.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "isequal";
const ISEQUALN_BUILTIN_NAME: &str = "isequaln";

const ISEQUAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when all inputs are equal in size, class, and content.",
}];

const ISEQUAL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Values to compare (at least two).",
}];

const ISEQUAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isequal(A, B, ...)",
    inputs: &ISEQUAL_INPUTS,
    outputs: &ISEQUAL_OUTPUT,
}];

const ISEQUALN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isequaln(A, B, ...)",
    inputs: &ISEQUAL_INPUTS,
    outputs: &ISEQUAL_OUTPUT,
}];

const ISEQUAL_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISEQUAL.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:isequal:NotEnoughInputs"),
    when: "Fewer than two arguments are supplied.",
    message: "isequal: requires at least two input arguments",
};

const ISEQUAL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISEQUAL.INTERNAL",
    identifier: Some("RunMat:isequal:Internal"),
    when: "Internal gather/host normalization fails.",
    message: "isequal: internal error",
};

const ISEQUAL_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ISEQUAL_ERROR_NOT_ENOUGH_INPUTS, ISEQUAL_ERROR_INTERNAL];

pub const ISEQUAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISEQUAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISEQUAL_ERRORS,
};

pub const ISEQUALN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISEQUALN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISEQUAL_ERRORS,
};

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
    descriptor(crate::builtins::logical::rel::isequal::ISEQUAL_DESCRIPTOR),
    builtin_path = "crate::builtins::logical::rel::isequal"
)]
async fn isequal_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    equality_builtin(args, false, BUILTIN_NAME).await
}

/// Compares all input values for equality, treating NaN values as equal.
#[runtime_builtin(
    name = "isequaln",
    category = "logical/rel",
    summary = "Test arrays for equality, treating NaN values as equal.",
    keywords = "isequaln,equality,comparison,nan,logical",
    accel = "cpu",
    descriptor(crate::builtins::logical::rel::isequal::ISEQUALN_DESCRIPTOR),
    builtin_path = "crate::builtins::logical::rel::isequal"
)]
async fn isequaln_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    equality_builtin(args, true, ISEQUALN_BUILTIN_NAME).await
}

async fn equality_builtin(
    args: Vec<Value>,
    nan_equal: bool,
    builtin_name: &'static str,
) -> crate::BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(equality_error(
            builtin_name,
            &ISEQUAL_ERROR_NOT_ENOUGH_INPUTS,
        ));
    }

    // Gather all values to host if needed
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gather_value(arg, builtin_name).await?);
    }

    // Compare first value against all others
    let first = &gathered[0];
    for other in gathered.iter().skip(1) {
        if !values_equal(first, other, nan_equal) {
            return Ok(Value::Bool(false));
        }
    }

    Ok(Value::Bool(true))
}

async fn gather_value(value: Value, builtin_name: &'static str) -> crate::BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| {
                    equality_error_with_detail(
                        builtin_name,
                        &ISEQUAL_ERROR_INTERNAL,
                        err.to_string(),
                    )
                })?;
            Ok(Value::Tensor(tensor))
        }
        other => Ok(other),
    }
}

/// Compare two values for equality (same size, class, and content).
/// NaN values are NOT considered equal.
fn values_equal(a: &Value, b: &Value, nan_equal: bool) -> bool {
    match (a, b) {
        // Numeric scalars
        (Value::Num(x), Value::Num(y)) => floats_equal(*x, *y, nan_equal),
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Int(x), Value::Int(y)) => x == y,

        // Complex scalars
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            floats_equal(*ar, *br, nan_equal) && floats_equal(*ai, *bi, nan_equal)
        }
        (Value::Num(x), Value::Complex(br, bi)) => {
            floats_equal(*x, *br, nan_equal) && floats_equal(*bi, 0.0, nan_equal)
        }
        (Value::Complex(ar, ai), Value::Num(y)) => {
            floats_equal(*ar, *y, nan_equal) && floats_equal(*ai, 0.0, nan_equal)
        }

        // Tensors
        (Value::Tensor(a), Value::Tensor(b)) => tensors_equal(a, b, nan_equal),
        (Value::Tensor(t), Value::Num(n)) => scalar_tensor_equal(t, *n, nan_equal),
        (Value::Num(n), Value::Tensor(t)) => scalar_tensor_equal(t, *n, nan_equal),

        // Complex tensors
        (Value::ComplexTensor(a), Value::ComplexTensor(b)) => {
            complex_tensors_equal(a, b, nan_equal)
        }

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
        (Value::Cell(a), Value::Cell(b)) => a.shape == b.shape && cells_equal(a, b, nan_equal),

        // Structs
        (Value::Struct(a), Value::Struct(b)) => structs_equal(a, b, nan_equal),

        // Different types are not equal
        _ => false,
    }
}

fn floats_equal(a: f64, b: f64, nan_equal: bool) -> bool {
    a == b || (nan_equal && a.is_nan() && b.is_nan())
}

fn tensors_equal(a: &Tensor, b: &Tensor, nan_equal: bool) -> bool {
    if a.dtype != b.dtype || a.shape != b.shape {
        return false;
    }
    if a.data.len() != b.data.len() {
        return false;
    }
    // NaN != NaN in isequal (use isequaln for NaN equality)
    a.data
        .iter()
        .zip(b.data.iter())
        .all(|(x, y)| floats_equal(*x, *y, nan_equal))
}

fn scalar_tensor_equal(t: &Tensor, n: f64, nan_equal: bool) -> bool {
    if t.dtype != runmat_builtins::NumericDType::F64 || t.data.len() != 1 {
        return false;
    }
    floats_equal(t.data[0], n, nan_equal)
}

fn complex_tensors_equal(a: &ComplexTensor, b: &ComplexTensor, nan_equal: bool) -> bool {
    if a.shape != b.shape {
        return false;
    }
    if a.data.len() != b.data.len() {
        return false;
    }
    a.data
        .iter()
        .zip(b.data.iter())
        .all(|((ar, ai), (br, bi))| {
            floats_equal(*ar, *br, nan_equal) && floats_equal(*ai, *bi, nan_equal)
        })
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

fn cells_equal(a: &CellArray, b: &CellArray, nan_equal: bool) -> bool {
    if a.data.len() != b.data.len() {
        return false;
    }
    a.data
        .iter()
        .zip(b.data.iter())
        .all(|(x, y)| values_equal(x, y, nan_equal))
}

fn structs_equal(
    a: &runmat_builtins::StructValue,
    b: &runmat_builtins::StructValue,
    nan_equal: bool,
) -> bool {
    if a.fields.len() != b.fields.len() {
        return false;
    }
    a.fields
        .iter()
        .zip(b.fields.iter())
        .all(|((key_a, val_a), (key_b, val_b))| {
            key_a == key_b && values_equal(val_a, val_b, nan_equal)
        })
}

fn equality_error(
    builtin_name: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(builtin_name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn equality_error_with_detail(
    builtin_name: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let message = format!("{}: {}", error.message, detail.as_ref());
    let mut builder = build_runtime_error(message).with_builtin(builtin_name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::CellArray;

    fn run_isequal(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(isequal_builtin(args))
    }

    fn run_isequaln(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(isequaln_builtin(args))
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
    fn isequaln_nan_values_are_equal_recursively() {
        let t1 = Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).unwrap();
        let t2 = Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).unwrap();
        let cells = CellArray::new(vec![Value::Tensor(t1), Value::Num(f64::NAN)], 1, 2).unwrap();
        let other = CellArray::new(vec![Value::Tensor(t2), Value::Num(f64::NAN)], 1, 2).unwrap();
        let result = run_isequaln(vec![Value::Cell(cells), Value::Cell(other)]).expect("isequaln");
        assert_eq!(result, Value::Bool(true));
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
        assert_eq!(err.identifier(), ISEQUAL_ERROR_NOT_ENOUGH_INPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_bool_and_num_have_different_classes() {
        let result = run_isequal(vec![Value::Bool(true), Value::Num(1.0)]).expect("isequal");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_tensor_dtype_must_match() {
        let double_tensor =
            Tensor::new_with_dtype(vec![1.0], vec![1, 1], runmat_builtins::NumericDType::F64)
                .unwrap();
        let uint32_tensor =
            Tensor::new_with_dtype(vec![1.0], vec![1, 1], runmat_builtins::NumericDType::U32)
                .unwrap();
        let result = run_isequal(vec![
            Value::Tensor(double_tensor),
            Value::Tensor(uint32_tensor),
        ])
        .expect("isequal");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_complex() {
        let result =
            run_isequal(vec![Value::Complex(1.0, 2.0), Value::Complex(1.0, 2.0)]).expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }
}
