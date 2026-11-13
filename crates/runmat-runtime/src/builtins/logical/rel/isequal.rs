//! MATLAB-compatible `isequal` builtin for RunMat.
//!
//! Tests whether all input arrays have the same size and numeric values or recursive content.

use runmat_builtins::{
    BuiltinErrorDescriptor, ISEQUALN_ERROR_INTERNAL, ISEQUALN_ERROR_NOT_ENOUGH_INPUTS,
    ISEQUALN_ERROR_TOO_MANY_OUTPUTS, ISEQUAL_ERROR_INTERNAL, ISEQUAL_ERROR_NOT_ENOUGH_INPUTS,
    ISEQUAL_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, LogicalArray, NumericScalar, StringArray, Tensor, Value,
};
use runmat_value::{IntValue, SparseTensor};

use crate::builtins::common::shape::array_shapes_equal;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "isequal";
const ISEQUALN_BUILTIN_NAME: &str = "isequaln";

/// Compares all input values for equality.
///
/// Returns `true` if all inputs have the same size and numeric values or recursive content.
/// Returns `false` otherwise. NaN values are NOT considered equal.
#[runtime_builtin(
    name = "isequal",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::rel::isequal"
)]
async fn isequal_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    equality_builtin(args, false, BUILTIN_NAME).await
}

/// Compares all input values for equality, treating NaN values as equal.
#[runtime_builtin(
    name = "isequaln",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::rel::isequal"
)]
async fn isequaln_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    equality_builtin(args, true, ISEQUALN_BUILTIN_NAME).await
}

async fn equality_builtin(
    args: crate::call::BuiltinArguments,
    nan_equal: bool,
    builtin_name: &'static str,
) -> crate::BuiltinResult<Value> {
    let (arity_error, output_error, internal_error) = if nan_equal {
        (
            &ISEQUALN_ERROR_NOT_ENOUGH_INPUTS,
            &ISEQUALN_ERROR_TOO_MANY_OUTPUTS,
            &ISEQUALN_ERROR_INTERNAL,
        )
    } else {
        (
            &ISEQUAL_ERROR_NOT_ENOUGH_INPUTS,
            &ISEQUAL_ERROR_TOO_MANY_OUTPUTS,
            &ISEQUAL_ERROR_INTERNAL,
        )
    };
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(equality_error(builtin_name, output_error));
    }
    if args.len() < 2 {
        return Err(equality_error(builtin_name, arity_error));
    }

    // Gather all values to host if needed
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gather_value(arg, builtin_name, internal_error).await?);
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

async fn gather_value(
    value: Value,
    builtin_name: &'static str,
    internal_error: &'static BuiltinErrorDescriptor,
) -> crate::BuiltinResult<Value> {
    match value {
        Value::Cell(_) | Value::Struct(_) | Value::GpuTensor(_) => {
            crate::dispatcher::gather_if_needed_owned_async(value, None)
                .await
                .map_err(|error| {
                    equality_error_with_detail(builtin_name, internal_error, error.to_string())
                })
        }
        Value::StructArray(records) => {
            crate::dispatcher::gather_if_needed_owned_async(Value::StructArray(records), None).await
        }
        other => Ok(other),
    }
}

/// Compare two values for equality (same size and numeric values or recursive content).
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
            array_shapes_equal(&real.shape(), &[1, 1])
                && numeric_scalars_equal(real.value_at(0), NumericScalar::F64(*br), nan_equal)
                && numeric_scalars_equal(
                    NumericScalar::F64(*bi),
                    NumericScalar::F64(0.0),
                    nan_equal,
                )
        }
        (Value::Complex(ar, ai), b) if real_numeric_view(b).is_some() => {
            let real = real_numeric_view(b).expect("guarded real numeric value");
            array_shapes_equal(&real.shape(), &[1, 1])
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
        (Value::Complex(real, imag), Value::ComplexTensor(tensor))
        | (Value::ComplexTensor(tensor), Value::Complex(real, imag)) => {
            complex_scalar_tensor_equal(*real, *imag, tensor, nan_equal)
        }

        // Strings
        (Value::String(a), Value::String(b)) => string_elements_equal(a, b, nan_equal),
        (Value::StringArray(a), Value::StringArray(b)) => string_arrays_equal(a, b, nan_equal),
        (Value::String(a), Value::StringArray(b)) => {
            array_shapes_equal(&b.shape, &[1, 1])
                && b.data().len() == 1
                && string_elements_equal(&b.data()[0], a, nan_equal)
        }
        (Value::StringArray(a), Value::String(b)) => {
            array_shapes_equal(&a.shape, &[1, 1])
                && a.data().len() == 1
                && string_elements_equal(&a.data()[0], b, nan_equal)
        }

        // Cells
        (Value::Cell(a), Value::Cell(b)) => {
            array_shapes_equal(&a.shape, &b.shape) && cells_equal(a, b, nan_equal)
        }

        // Structs
        (Value::Struct(a), Value::Struct(b)) => structs_equal(a, b, nan_equal),
        (Value::StructArray(a), Value::StructArray(b)) => {
            array_shapes_equal(a.shape(), b.shape())
                && a.field_names().count() == b.field_names().count()
                && a.field_names().all(|name| {
                    a.field_values(name)
                        .zip(b.field_values(name))
                        .is_some_and(|(left, right)| {
                            left.len() == right.len()
                                && left
                                    .iter()
                                    .zip(right)
                                    .all(|(left, right)| values_equal(left, right, nan_equal))
                        })
                })
        }

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
            Self::Tensor(value) => value.shape().to_vec(),
            Self::Logical(value) => value.shape.to_vec(),
            Self::Char(value) => value.shape.as_slice().to_vec(),
            Self::Sparse(value) => vec![value.rows, value.cols],
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(value) => value.len(),
            Self::Logical(value) => value.data.len(),
            Self::Char(value) => value.data().len(),
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
            Self::Char(value) => NumericScalar::U32(u32::from(value.data()[index].raw())),
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
    if !array_shapes_equal(&real.shape(), complex.shape()) || real.len() != complex.len() {
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
    if !array_shapes_equal(&a.shape(), &b.shape()) || a.len() != b.len() {
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
    if !array_shapes_equal(a.shape(), b.shape()) || a.len() != b.len() {
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

fn complex_scalar_tensor_equal(
    real: f64,
    imag: f64,
    tensor: &ComplexTensor,
    nan_equal: bool,
) -> bool {
    if !array_shapes_equal(tensor.shape(), &[1, 1]) || tensor.len() != 1 {
        return false;
    }
    let (tensor_real, tensor_imag) = tensor
        .numeric_value_at(0)
        .expect("scalar complex tensor index validated by length");
    numeric_scalars_equal(NumericScalar::F64(real), tensor_real, nan_equal)
        && numeric_scalars_equal(NumericScalar::F64(imag), tensor_imag, nan_equal)
}

fn string_elements_equal(
    a: &runmat_value::StringElement,
    b: &runmat_value::StringElement,
    nan_equal: bool,
) -> bool {
    match (a.as_str(), b.as_str()) {
        (Some(a), Some(b)) => a == b,
        (None, None) => nan_equal,
        _ => false,
    }
}

fn string_arrays_equal(a: &StringArray, b: &StringArray, nan_equal: bool) -> bool {
    if !array_shapes_equal(&a.shape, &b.shape) {
        return false;
    }
    a.data().len() == b.data().len()
        && a.data()
            .iter()
            .zip(b.data())
            .all(|(a, b)| string_elements_equal(a, b, nan_equal))
}

fn cells_equal(a: &CellArray, b: &CellArray, nan_equal: bool) -> bool {
    if a.data().len() != b.data().len() {
        return false;
    }
    a.data()
        .iter()
        .zip(b.data().iter())
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
    a.fields.iter().all(|(name, left)| {
        b.fields
            .get(name)
            .is_some_and(|right| values_equal(left, right, nan_equal))
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
#[path = "isequal/tests.rs"]
pub(crate) mod tests;
