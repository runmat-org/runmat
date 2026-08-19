//! MATLAB-compatible `isequal` builtin for RunMat.
//!
//! Tests whether all input arrays have the same size, class, and content.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, LogicalArray, NumericScalar, StringArray, Tensor, Value,
};
use runmat_value::{IntValue, SparseTensor};

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
const ISEQUAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "A", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Variadic numeric inputs compare by value across classes; exact integer storage is never preconverted to floating point." }];
pub const ISEQUAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "tf = isequal(integer_A, B, ...)", inputs: &ISEQUAL_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::Logical, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Same-shape numeric values compare class-insensitively using exact signed/unsigned/integer/floating ordering; resident values gather exactly and return a host logical scalar." }];
pub const ISEQUALN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "tf = isequaln(integer_A, B, ...)", inputs: &ISEQUAL_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::Logical, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Shares exact class-insensitive numeric equality with isequal while treating corresponding floating missing values as equal." }];

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
    integer_capabilities(crate::builtins::logical::rel::isequal::ISEQUAL_INTEGER_CAPABILITIES),
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
    integer_capabilities(crate::builtins::logical::rel::isequal::ISEQUALN_INTEGER_CAPABILITIES),
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
            let owner = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
                equality_error_with_detail(
                    builtin_name,
                    &ISEQUAL_ERROR_INTERNAL,
                    "no acceleration provider owns the resident input",
                )
            })?;
            gpu_helpers::download_value_preserving_residency_async(owner, &handle)
                .await
                .map_err(|err| {
                    equality_error_with_detail(
                        builtin_name,
                        &ISEQUAL_ERROR_INTERNAL,
                        err.to_string(),
                    )
                })
        }
        other => Ok(other),
    }
}

/// Compare two values for equality (same size, class, and content).
/// NaN values are NOT considered equal.
fn values_equal(a: &Value, b: &Value, nan_equal: bool) -> bool {
    if let (Some(a), Some(b)) = (real_numeric_view(a), real_numeric_view(b)) {
        return real_numeric_views_equal(&a, &b, nan_equal);
    }
    if let (Some(real), Value::ComplexTensor(complex)) = (real_numeric_view(a), b) {
        return real_complex_equal(&real, complex, nan_equal);
    }
    if let (Value::ComplexTensor(complex), Some(real)) = (a, real_numeric_view(b)) {
        return real_complex_equal(&real, complex, nan_equal);
    }
    match (a, b) {
        // Complex scalars
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            floats_equal(*ar, *br, nan_equal) && floats_equal(*ai, *bi, nan_equal)
        }
        (a, Value::Complex(br, bi)) if real_numeric_view(a).is_some() => {
            let real = real_numeric_view(a).expect("guarded real numeric value");
            real.shape() == vec![1, 1]
                && numeric_scalars_equal(real.value_at(0), NumericScalar::F64(*br), nan_equal)
                && numeric_scalars_equal(
                    NumericScalar::F64(*bi),
                    NumericScalar::F64(0.0),
                    nan_equal,
                )
        }
        (Value::Complex(ar, ai), b) if real_numeric_view(b).is_some() => {
            let real = real_numeric_view(b).expect("guarded real numeric value");
            real.shape() == vec![1, 1]
                && numeric_scalars_equal(NumericScalar::F64(*ar), real.value_at(0), nan_equal)
                && numeric_scalars_equal(
                    NumericScalar::F64(*ai),
                    NumericScalar::F64(0.0),
                    nan_equal,
                )
        }

        // Complex tensors
        (Value::ComplexTensor(a), Value::ComplexTensor(b)) => {
            complex_tensors_equal(a, b, nan_equal)
        }

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

enum RealNumericView<'a> {
    Scalar(NumericScalar),
    Tensor(&'a Tensor),
    Logical(&'a LogicalArray),
    Char(&'a CharArray),
    Sparse(&'a SparseTensor),
}

impl RealNumericView<'_> {
    fn shape(&self) -> Vec<usize> {
        match self {
            Self::Scalar(_) => vec![1, 1],
            Self::Tensor(value) => value.shape.clone(),
            Self::Logical(value) => value.shape.clone(),
            Self::Char(value) => vec![value.rows, value.cols],
            Self::Sparse(value) => vec![value.rows, value.cols],
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(value) => value.len(),
            Self::Logical(value) => value.data.len(),
            Self::Char(value) => value.data.len(),
            Self::Sparse(value) => value.rows.saturating_mul(value.cols),
        }
    }

    fn value_at(&self, index: usize) -> NumericScalar {
        match self {
            Self::Scalar(value) => *value,
            Self::Tensor(value) => value
                .numeric_value_at(index)
                .expect("numeric view index validated by length"),
            Self::Logical(value) => NumericScalar::U8(u8::from(value.data[index] != 0)),
            Self::Char(value) => NumericScalar::U32(value.data[index] as u32),
            Self::Sparse(value) => sparse_numeric_value_at(value, index),
        }
    }
}

fn real_numeric_view(value: &Value) -> Option<RealNumericView<'_>> {
    match value {
        Value::Num(value) => Some(RealNumericView::Scalar(NumericScalar::F64(*value))),
        Value::Int(value) => Some(RealNumericView::Scalar(NumericScalar::from(value.clone()))),
        Value::Bool(value) => Some(RealNumericView::Scalar(NumericScalar::U8(u8::from(*value)))),
        Value::Tensor(value) => Some(RealNumericView::Tensor(value)),
        Value::LogicalArray(value) => Some(RealNumericView::Logical(value)),
        Value::CharArray(value) => Some(RealNumericView::Char(value)),
        Value::SparseTensor(value) => Some(RealNumericView::Sparse(value)),
        _ => None,
    }
}

fn sparse_numeric_value_at(value: &SparseTensor, index: usize) -> NumericScalar {
    let row = index % value.rows.max(1);
    let col = index / value.rows.max(1);
    let start = value.col_ptrs.get(col).copied().unwrap_or(0);
    let end = value.col_ptrs.get(col + 1).copied().unwrap_or(start);
    match value.row_indices[start..end].binary_search(&row) {
        Ok(offset) => value
            .numeric_value_at(start + offset)
            .expect("validated sparse stored index"),
        Err(_) => NumericScalar::F64(0.0),
    }
}

fn real_complex_equal(
    real: &RealNumericView<'_>,
    complex: &ComplexTensor,
    nan_equal: bool,
) -> bool {
    if real.shape() != complex.shape || real.len() != complex.len() {
        return false;
    }
    (0..real.len()).all(|index| {
        let (complex_real, complex_imag) = complex
            .numeric_value_at(index)
            .expect("complex tensor index validated by length");
        numeric_scalars_equal(real.value_at(index), complex_real, nan_equal)
            && numeric_scalars_equal(complex_imag, NumericScalar::F64(0.0), nan_equal)
    })
}

fn real_numeric_views_equal(
    a: &RealNumericView<'_>,
    b: &RealNumericView<'_>,
    nan_equal: bool,
) -> bool {
    if a.shape() != b.shape() || a.len() != b.len() {
        return false;
    }
    (0..a.len()).all(|index| numeric_scalars_equal(a.value_at(index), b.value_at(index), nan_equal))
}

fn floats_equal(a: f64, b: f64, nan_equal: bool) -> bool {
    a == b || (nan_equal && a.is_nan() && b.is_nan())
}

fn numeric_scalars_equal(a: NumericScalar, b: NumericScalar, nan_equal: bool) -> bool {
    match (a.into_int_value(), b.into_int_value()) {
        (Some(a), Some(b)) => {
            return crate::builtins::logical::rel::integer_comparison::compare_integer_values(a, b)
                == std::cmp::Ordering::Equal;
        }
        (Some(a), None) => return integer_equals_float(a, b),
        (None, Some(b)) => return integer_equals_float(b, a),
        (None, None) => {}
    }
    match (a, b) {
        (NumericScalar::F64(a), NumericScalar::F64(b)) => floats_equal(a, b, nan_equal),
        (NumericScalar::F32(a), NumericScalar::F32(b)) => {
            a == b || (nan_equal && a.is_nan() && b.is_nan())
        }
        (NumericScalar::F64(a), NumericScalar::F32(b)) => floats_equal(a, f64::from(b), nan_equal),
        (NumericScalar::F32(a), NumericScalar::F64(b)) => floats_equal(f64::from(a), b, nan_equal),
        _ => unreachable!("integer cases returned above"),
    }
}

fn integer_equals_float(integer: IntValue, float: NumericScalar) -> bool {
    let float = match float {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        _ => unreachable!("integer argument separated before mixed comparison"),
    };
    crate::builtins::logical::rel::integer_comparison::integer_f64_order(integer, float)
        == Some(std::cmp::Ordering::Equal)
}

fn complex_tensors_equal(a: &ComplexTensor, b: &ComplexTensor, nan_equal: bool) -> bool {
    if a.shape != b.shape || a.len() != b.len() {
        return false;
    }
    (0..a.len()).all(|index| {
        let (ar, ai) = a
            .numeric_value_at(index)
            .expect("complex tensor index validated by length");
        let (br, bi) = b
            .numeric_value_at(index)
            .expect("complex tensor index validated by length");
        numeric_scalars_equal(ar, br, nan_equal) && numeric_scalars_equal(ai, bi, nan_equal)
    })
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
    a: &runmat_value::StructValue,
    b: &runmat_value::StructValue,
    nan_equal: bool,
) -> bool {
    if a.fields.len() != b.fields.len() {
        return false;
    }
    a.fields.iter().all(|(key, value)| {
        b.fields
            .get(key)
            .is_some_and(|other| values_equal(value, other, nan_equal))
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
    use runmat_value::{CellArray, ComplexTensor, IntegerComplexStorage, IntegerStorage};

    fn run_isequal(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(isequal_builtin(args))
    }

    fn run_isequaln(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(isequaln_builtin(args))
    }

    fn typed_complex_u64(real: u64, imag: u64) -> ComplexTensor {
        ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![real]),
                IntegerStorage::U64(vec![imag]),
            )
            .expect("matching components"),
            vec![1, 1],
        )
        .expect("typed complex")
    }

    #[test]
    fn numeric_equality_is_class_insensitive_without_losing_wide_integers() {
        let signed = Tensor::new_integer(IntegerStorage::I64(vec![255]), vec![1, 1]).unwrap();
        let unsigned = Tensor::new_integer(IntegerStorage::U8(vec![255]), vec![1, 1]).unwrap();
        assert_eq!(
            run_isequal(vec![Value::Tensor(signed), Value::Tensor(unsigned)]).unwrap(),
            Value::Bool(true)
        );

        let wide = (1_u64 << 53) + 1;
        let exact = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        let rounded = Tensor::new(vec![wide as f64], vec![1, 1]).unwrap();
        assert_eq!(
            run_isequal(vec![Value::Tensor(exact), Value::Tensor(rounded)]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_isequal(vec![Value::Bool(true), Value::Num(1.0)]).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn real_complex_and_sparse_numeric_values_compare_across_classes() {
        let wide = u64::MAX;
        let real = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        let complex = typed_complex_u64(wide, 0);
        assert_eq!(
            run_isequal(vec![Value::Tensor(real), Value::ComplexTensor(complex)]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            run_isequal(vec![Value::Int(IntValue::I32(1)), Value::Complex(1.0, 0.0)]).unwrap(),
            Value::Bool(true)
        );
        let sparse =
            SparseTensor::new_integer(2, 1, vec![0, 1], vec![1], IntegerStorage::U64(vec![wide]))
                .unwrap();
        let dense = Tensor::new_integer(IntegerStorage::U64(vec![0, wide]), vec![2, 1]).unwrap();
        assert_eq!(
            run_isequal(vec![Value::SparseTensor(sparse), Value::Tensor(dense)]).unwrap(),
            Value::Bool(true)
        );
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

    #[test]
    fn isequal_and_isequaln_compare_single_storage_without_widening_class() {
        let left = Tensor::from_f32(vec![0.1, f32::NAN], vec![1, 2]).unwrap();
        let right = Tensor::from_f32(vec![0.1, f32::NAN], vec![1, 2]).unwrap();

        assert_eq!(
            run_isequal(vec![
                Value::Tensor(left.clone()),
                Value::Tensor(right.clone())
            ])
            .expect("isequal"),
            Value::Bool(false)
        );
        assert_eq!(
            run_isequaln(vec![Value::Tensor(left), Value::Tensor(right)]).expect("isequaln"),
            Value::Bool(true)
        );
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
    fn isequal_bool_and_num_compare_by_numeric_value() {
        let result = run_isequal(vec![Value::Bool(true), Value::Num(1.0)]).expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_tensor_numeric_classes_compare_by_value() {
        let double_tensor =
            Tensor::new_with_dtype(vec![1.0], vec![1, 1], runmat_value::NumericDType::F64).unwrap();
        let uint32_tensor =
            Tensor::new_with_dtype(vec![1.0], vec![1, 1], runmat_value::NumericDType::U32).unwrap();
        let result = run_isequal(vec![
            Value::Tensor(double_tensor),
            Value::Tensor(uint32_tensor),
        ])
        .expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn isequal_integer_tensor_values_are_compared_exactly() {
        let same_left = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            vec![1, 2],
        )
        .expect("left integer tensor");
        let same_right = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            vec![1, 2],
        )
        .expect("right integer tensor");
        let different = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 53), u64::MAX]),
            vec![1, 2],
        )
        .expect("different integer tensor");

        assert_eq!(
            run_isequal(vec![Value::Tensor(same_left), Value::Tensor(same_right)])
                .expect("isequal same"),
            Value::Bool(true)
        );
        assert_eq!(
            run_isequal(vec![
                Value::Tensor(different),
                Value::Tensor(
                    Tensor::new_integer(
                        IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
                        vec![1, 2],
                    )
                    .expect("comparison tensor"),
                ),
            ])
            .expect("isequal different"),
            Value::Bool(false)
        );
    }

    #[test]
    fn isequal_integer_tensor_classes_compare_exact_values() {
        let unsigned = Tensor::new_integer(IntegerStorage::U64(vec![255]), vec![1, 1])
            .expect("unsigned integer tensor");
        let signed = Tensor::new_integer(IntegerStorage::I64(vec![255]), vec![1, 1])
            .expect("signed integer tensor");

        assert_eq!(
            run_isequal(vec![Value::Tensor(unsigned), Value::Tensor(signed)]).expect("isequal"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isequal_complex() {
        let result =
            run_isequal(vec![Value::Complex(1.0, 2.0), Value::Complex(1.0, 2.0)]).expect("isequal");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn isequal_typed_complex_integer_compares_exact_components() {
        let left = typed_complex_u64(u64::MAX, 1_u64 << 63);
        let same = left.clone();
        let different = typed_complex_u64(u64::MAX - 1, 1_u64 << 63);

        assert_eq!(
            run_isequal(vec![Value::ComplexTensor(left), Value::ComplexTensor(same)])
                .expect("isequal"),
            Value::Bool(true)
        );
        assert_eq!(
            run_isequal(vec![
                Value::ComplexTensor(different),
                Value::ComplexTensor(typed_complex_u64(u64::MAX, 1_u64 << 63)),
            ])
            .expect("isequal"),
            Value::Bool(false)
        );
    }
}
