use std::{borrow::Cow, convert::TryFrom};

use num_complex::Complex64;
use runmat_builtins::{
    ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType, Tensor, Value,
};

use crate::dispatcher::gather_if_needed_async;

/// Return the total number of elements for a given shape.
pub fn element_count(shape: &[usize]) -> usize {
    let mut acc: u128 = 1;
    for &dim in shape {
        let dim128 = dim as u128;
        acc = acc
            .checked_mul(dim128)
            .expect("tensor::element_count: overflow computing element count");
    }
    usize::try_from(acc).expect("tensor::element_count: overflow converting to usize")
}

/// Construct a zero-filled tensor with the provided shape.
pub fn zeros(shape: &[usize]) -> Result<Tensor, String> {
    Tensor::new(vec![0.0; element_count(shape)], shape.to_vec())
        .map_err(|e| format!("tensor zeros: {e}"))
}

/// Construct an one-filled tensor with the provided shape.
pub fn ones(shape: &[usize]) -> Result<Tensor, String> {
    Tensor::new(vec![1.0; element_count(shape)], shape.to_vec())
        .map_err(|e| format!("tensor ones: {e}"))
}

/// Construct a zero-filled tensor with an explicit dtype flag.
pub fn zeros_with_dtype(shape: &[usize], dtype: NumericDType) -> Result<Tensor, String> {
    integer_tensor_with_value(shape, dtype, false)
        .unwrap_or_else(|| {
            Tensor::new_with_dtype(vec![0.0; element_count(shape)], shape.to_vec(), dtype)
        })
        .map_err(|e| format!("tensor zeros: {e}"))
}

/// Construct a one-filled tensor with an explicit dtype flag.
pub fn ones_with_dtype(shape: &[usize], dtype: NumericDType) -> Result<Tensor, String> {
    integer_tensor_with_value(shape, dtype, true)
        .unwrap_or_else(|| {
            Tensor::new_with_dtype(vec![1.0; element_count(shape)], shape.to_vec(), dtype)
        })
        .map_err(|e| format!("tensor ones: {e}"))
}

fn integer_tensor_with_value(
    shape: &[usize],
    dtype: NumericDType,
    ones: bool,
) -> Option<Result<Tensor, String>> {
    let len = element_count(shape);
    let storage = match dtype {
        NumericDType::I8 => IntegerStorage::I8(vec![if ones { 1 } else { 0 }; len]),
        NumericDType::I16 => IntegerStorage::I16(vec![if ones { 1 } else { 0 }; len]),
        NumericDType::I32 => IntegerStorage::I32(vec![if ones { 1 } else { 0 }; len]),
        NumericDType::I64 => IntegerStorage::I64(vec![if ones { 1 } else { 0 }; len]),
        NumericDType::U8 => IntegerStorage::U8(vec![if ones { 1 } else { 0 }; len]),
        NumericDType::U16 => IntegerStorage::U16(vec![if ones { 1 } else { 0 }; len]),
        NumericDType::U32 => IntegerStorage::U32(vec![if ones { 1 } else { 0 }; len]),
        NumericDType::U64 => IntegerStorage::U64(vec![if ones { 1 } else { 0 }; len]),
        NumericDType::F32 | NumericDType::F64 => return None,
    };
    Some(Tensor::new_integer(storage, shape.to_vec()))
}

/// Converts floating-point values to an exact integer tensor using the
/// prototype class's MATLAB assignment semantics.
pub fn integer_tensor_from_f64_like(
    prototype: &IntegerStorage,
    values: Vec<f64>,
    shape: &[usize],
) -> Result<Tensor, String> {
    let storage = prototype
        .from_same_class_values(
            values
                .into_iter()
                .map(|value| prototype.cast_f64_assignment(value))
                .collect(),
        )
        .map_err(|e| format!("integer tensor conversion: {e}"))?;
    Tensor::new_integer(storage, shape.to_vec())
        .map_err(|e| format!("integer tensor conversion: {e}"))
}

/// Return a tensor's numeric values as f64, reading typed integer storage
/// exactly instead of using the compatibility backing buffer.
pub fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor
        .integer_storage()
        .map(|storage| {
            storage
                .exact_values()
                .into_iter()
                .map(|value| value.to_f64())
                .collect()
        })
        .unwrap_or_else(|| tensor.data.clone())
}

/// Return a borrowed view for ordinary double tensors and materialize exact
/// f64 values only for typed integer storage.
pub fn tensor_values_f64_cow(tensor: &Tensor) -> Cow<'_, [f64]> {
    if tensor.integer_storage().is_some() {
        Cow::Owned(tensor_values_f64(tensor))
    } else {
        Cow::Borrowed(&tensor.data)
    }
}

/// Return one numeric tensor value as f64, reading typed integer storage exactly
/// instead of using the compatibility backing buffer.
pub fn tensor_value_f64(tensor: &Tensor, index: usize) -> f64 {
    match tensor.integer_storage() {
        Some(storage) => storage
            .value_at(index)
            .expect("tensor_value_f64: integer storage index is in bounds")
            .to_f64(),
        None => tensor.data[index],
    }
}

/// Return an exact integer scalar from a scalar integer value or scalar typed
/// integer tensor.
pub fn scalar_integer_value(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

/// Consume a tensor and return f64 values, preserving the fast path for
/// ordinary double tensors while reading typed integer storage exactly.
pub fn tensor_into_values_f64(tensor: Tensor) -> Vec<f64> {
    if tensor.integer_storage().is_some() {
        tensor_values_f64(&tensor)
    } else {
        tensor.data
    }
}

/// Return a complex tensor's numeric values as Complex64, reading typed integer
/// real/imaginary storage exactly instead of using the compatibility buffer.
pub fn complex_tensor_values_complex64(tensor: &ComplexTensor) -> Vec<Complex64> {
    tensor
        .integer_data
        .as_ref()
        .map(|storage| {
            let real = storage.real.exact_values();
            let imag = storage.imag.exact_values();
            real.into_iter()
                .zip(imag)
                .map(|(re, im)| Complex64::new(re.to_f64(), im.to_f64()))
                .collect()
        })
        .unwrap_or_else(|| {
            tensor
                .data
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect()
        })
}

/// Return a complex tensor's element count, reading exact integer-complex
/// storage instead of the floating compatibility buffer when present.
pub fn complex_tensor_element_len(tensor: &ComplexTensor) -> usize {
    tensor
        .integer_data
        .as_ref()
        .map_or(tensor.data.len(), |storage| storage.len())
}

/// Return true when a complex tensor contains exactly one scalar element.
pub fn is_scalar_complex_tensor(tensor: &ComplexTensor) -> bool {
    complex_tensor_element_len(tensor) == 1
}

/// Return one complex tensor value as Complex64, reading typed integer-complex
/// storage exactly instead of using the compatibility backing buffer.
pub fn complex_tensor_value_complex64(tensor: &ComplexTensor, index: usize) -> Complex64 {
    match tensor.integer_data.as_ref() {
        Some(storage) => {
            let real = storage
                .real
                .value_at(index)
                .expect("complex_tensor_value_complex64: real storage index is in bounds")
                .to_f64();
            let imag = storage
                .imag
                .value_at(index)
                .expect("complex_tensor_value_complex64: imaginary storage index is in bounds")
                .to_f64();
            Complex64::new(real, imag)
        }
        None => {
            let (real, imag) = tensor.data[index];
            Complex64::new(real, imag)
        }
    }
}

/// Consume a complex tensor and return Complex64 values, preserving the fast
/// path for ordinary complex double tensors.
pub fn complex_tensor_into_values_complex64(tensor: ComplexTensor) -> Vec<Complex64> {
    if tensor.integer_data.is_some() {
        complex_tensor_values_complex64(&tensor)
    } else {
        tensor
            .data
            .into_iter()
            .map(|(re, im)| Complex64::new(re, im))
            .collect()
    }
}

/// Normalize typed integer tensors to f64 at builtin boundaries that
/// intentionally compute double-valued numeric/statistical results.
pub fn integer_tensor_to_f64(tensor: Tensor) -> Result<Tensor, String> {
    if tensor.integer_storage().is_none() {
        return Ok(tensor);
    }
    Tensor::new(tensor_values_f64(&tensor), tensor.shape.clone())
        .map_err(|e| format!("integer tensor conversion: {e}"))
}

/// Convert a logical array (0/1 bytes) into a numeric tensor.
pub fn logical_to_tensor(logical: &LogicalArray) -> Result<Tensor, String> {
    let data: Vec<f64> = logical
        .data
        .iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, logical.shape.clone()).map_err(|e| format!("logical->tensor: {e}"))
}

fn value_into_tensor_impl(name: &str, value: Value) -> Result<Tensor, String> {
    match value {
        Value::Tensor(t) => Ok(t),
        Value::LogicalArray(logical) => logical_to_tensor(&logical),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("tensor: {e}")),
        Value::Int(i) => Tensor::new_integer(IntegerStorage::from_scalar(i), vec![1, 1])
            .map_err(|e| format!("tensor: {e}")),
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|e| format!("tensor: {e}")),
        other => Err(format!(
            "{name}: unsupported input type {:?}; expected numeric or logical values",
            other
        )),
    }
}

/// Convert a `Value` into an owned `Tensor`, defaulting error messages to `"sum"`.
pub fn value_into_tensor(value: Value) -> Result<Tensor, String> {
    value_into_tensor_impl("sum", value)
}

/// Convert a `Value` into a tensor while customising the builtin name in error messages.
pub fn value_into_tensor_for(name: &str, value: Value) -> Result<Tensor, String> {
    value_into_tensor_impl(name, value)
}

/// Clone a `Value` and coerce it into a tensor.
pub fn value_to_tensor(value: &Value) -> Result<Tensor, String> {
    value_into_tensor(value.clone())
}

/// Convert a `Tensor` back into a runtime value.
///
/// Scalars (exactly one element) become their exact scalar representation;
/// all other tensors remain as dense tensor variants.
pub fn tensor_into_value(tensor: Tensor) -> Value {
    if is_scalar_tensor(&tensor) {
        if let Some(storage) = tensor.integer_storage() {
            return Value::Int(storage.value_at(0).expect("one-element integer storage"));
        }
        Value::Num(tensor_value_f64(&tensor, 0))
    } else {
        Value::Tensor(tensor)
    }
}

/// Return true when a tensor contains exactly one scalar element.
pub fn is_scalar_tensor(tensor: &Tensor) -> bool {
    tensor_element_len(tensor) == 1
}

pub fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor
        .integer_storage()
        .map_or(tensor.data.len(), |storage| storage.len())
}

fn scalar_f64_from_host_value(value: &Value) -> Result<Option<f64>, String> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(t) => {
            if is_scalar_tensor(t) {
                if let Some(storage) = t.integer_storage() {
                    return Ok(Some(
                        storage
                            .value_at(0)
                            .expect("one-element integer storage")
                            .to_f64(),
                    ));
                }
                Ok(Some(tensor_value_f64(t, 0)))
            } else {
                Err(format!(
                    "expected scalar tensor, got tensor of size {}",
                    tensor_element_len(t)
                ))
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() == 1 {
                Ok(Some(if la.data[0] != 0 { 1.0 } else { 0.0 }))
            } else {
                Err(format!(
                    "expected scalar logical array, got array of size {}",
                    la.data.len()
                ))
            }
        }
        _ => Ok(None),
    }
}

/// Attempt to extract a scalar f64 from a runtime value asynchronously.
pub async fn scalar_f64_from_value_async(value: &Value) -> Result<Option<f64>, String> {
    match value {
        Value::GpuTensor(handle) => {
            if !handle.shape.is_empty() {
                let len = element_count(&handle.shape);
                if len != 1 {
                    return Err(format!("expected scalar gpuArray, got array of size {len}"));
                }
            }
            let gathered = gather_if_needed_async(&Value::GpuTensor(handle.clone()))
                .await
                .map_err(|e| format!("scalar: {e}"))?;
            scalar_f64_from_host_value(&gathered)
        }
        _ => scalar_f64_from_host_value(value),
    }
}

/// Attempt to parse a dimension index from a scalar-like runtime value.
pub async fn dimension_from_value_async(
    value: &Value,
    name: &str,
    allow_zero: bool,
) -> Result<Option<usize>, String> {
    match value {
        Value::Int(value) => return parse_integer_dimension(value, name, allow_zero).map(Some),
        Value::Tensor(tensor) if is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                return parse_integer_dimension(&value, name, allow_zero).map(Some);
            }
        }
        _ => {}
    }
    let Some(raw) = scalar_f64_from_value_async(value).await? else {
        return Ok(None);
    };
    parse_numeric_dimension_value(raw, name, allow_zero).map(Some)
}

fn parse_integer_dimension(
    value: &IntValue,
    name: &str,
    allow_zero: bool,
) -> Result<usize, String> {
    let dim = value
        .try_to_usize()
        .ok_or_else(|| format!("{name}: dimension is outside the supported range"))?;
    if !allow_zero && dim == 0 {
        return Err(format!("{name}: dimension must be >= 1"));
    }
    Ok(dim)
}

/// Parse every element of a typed integer tensor as a dimension selector without
/// routing through the tensor's f64 compatibility backing store.
pub fn integer_tensor_dimension_vector(
    tensor: &Tensor,
    name: &str,
    allow_zero: bool,
) -> Option<Result<Vec<usize>, String>> {
    let storage = tensor.integer_storage()?;
    Some(
        (0..storage.len())
            .map(|index| {
                let value = storage
                    .value_at(index)
                    .expect("integer tensor storage length matches element count");
                parse_integer_dimension(&value, name, allow_zero)
            })
            .collect(),
    )
}

fn parse_integer_shape_dimension(value: &IntValue) -> Result<usize, String> {
    value
        .try_to_usize()
        .ok_or_else(|| "dimensions must be non-negative platform integers".to_string())
}

fn parse_numeric_dimension(value: f64) -> Result<usize, String> {
    if !value.is_finite() {
        return Err("dimensions must be finite".to_string());
    }
    if value < 0.0 {
        return Err("matrix dimensions must be non-negative".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("dimensions must be integers".to_string());
    }
    if !fits_platform_usize(rounded) {
        return Err("dimensions are outside the supported platform range".to_string());
    }
    Ok(rounded as usize)
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}

fn dims_from_tensor_values(values: &[f64], shape: &[usize]) -> Result<Option<Vec<usize>>, String> {
    let len = values.len();
    if len == 0 {
        return Ok(Some(Vec::new()));
    }
    let is_scalar = len == 1;
    let is_row = shape.len() >= 2 && shape[0] == 1;
    let is_column = shape.len() >= 2 && shape[1] == 1;
    if !(is_row || is_column || is_scalar || shape.len() == 1) {
        return Ok(None);
    }
    let mut dims = Vec::with_capacity(len);
    for &value in values {
        dims.push(parse_numeric_dimension(value)?);
    }
    Ok(Some(dims))
}

fn dims_from_integer_tensor_values(
    storage: &IntegerStorage,
    shape: &[usize],
) -> Result<Option<Vec<usize>>, String> {
    let len = storage.len();
    if len == 0 {
        return Ok(Some(Vec::new()));
    }
    let is_scalar = len == 1;
    let is_row = shape.len() >= 2 && shape[0] == 1;
    let is_column = shape.len() >= 2 && shape[1] == 1;
    if !(is_row || is_column || is_scalar || shape.len() == 1) {
        return Ok(None);
    }
    let mut dims = Vec::with_capacity(len);
    for index in 0..len {
        dims.push(parse_integer_shape_dimension(
            &storage.value_at(index).expect("integer storage index"),
        )?);
    }
    Ok(Some(dims))
}

/// Attempt to parse a dimension vector from a runtime value asynchronously.
pub async fn dims_from_value_async(value: &Value) -> Result<Option<Vec<usize>>, String> {
    match value {
        Value::Num(n) => parse_numeric_dimension(*n).map(|dim| Some(vec![dim])),
        Value::Int(i) => parse_integer_shape_dimension(i).map(|dim| Some(vec![dim])),
        Value::Tensor(t) => match t.integer_storage() {
            Some(storage) => dims_from_integer_tensor_values(storage, &t.shape),
            None => dims_from_tensor_values(&t.data, &t.shape),
        },
        Value::LogicalArray(la) => {
            let values: Vec<f64> = la
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            dims_from_tensor_values(&values, &la.shape)
        }
        Value::GpuTensor(handle) => {
            let gathered = gather_if_needed_async(&Value::GpuTensor(handle.clone()))
                .await
                .map_err(|e| format!("dimensions: {e}"))?;
            match gathered {
                Value::Tensor(t) => {
                    if t.data.is_empty() {
                        tracing::warn!(
                            gpu_shape = ?handle.shape,
                            "dims_from_value_async: gathered GPU tensor has no data"
                        );
                    }
                    tracing::trace!(
                        "dims_from_value_async: GPU tensor values gpu_shape={:?} host_shape={:?} values={:?}",
                        handle.shape,
                        t.shape,
                        t.data
                    );
                    let dims = match t.integer_storage() {
                        Some(storage) => dims_from_integer_tensor_values(storage, &t.shape)?,
                        None => dims_from_tensor_values(&t.data, &t.shape)?,
                    };
                    if dims.is_none() {
                        tracing::debug!(
                            gpu_shape = ?handle.shape,
                            host_shape = ?t.shape,
                            "dims_from_value_async: GPU tensor not interpretable as dims"
                        );
                    }
                    Ok(dims)
                }
                Value::LogicalArray(la) => {
                    let values: Vec<f64> = la
                        .data
                        .iter()
                        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                        .collect();
                    let dims = dims_from_tensor_values(&values, &la.shape)?;
                    if dims.is_none() {
                        tracing::debug!(
                            gpu_shape = ?handle.shape,
                            host_shape = ?la.shape,
                            "dims_from_value_async: GPU logical not interpretable as dims"
                        );
                    }
                    Ok(dims)
                }
                Value::Num(n) => parse_numeric_dimension(n).map(|dim| Some(vec![dim])),
                Value::Int(i) => parse_integer_shape_dimension(&i).map(|dim| Some(vec![dim])),
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

/// Convert an argument into a dimension index (1-based) if possible.
pub fn parse_dimension(value: &Value, name: &str) -> Result<usize, String> {
    match value {
        Value::Int(i) => parse_integer_dimension(i, name, false),
        Value::Tensor(tensor) if is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let value = storage.value_at(0).expect("one-element integer storage");
                return parse_integer_dimension(&value, name, false);
            }
            parse_numeric_dimension_value(tensor_value_f64(tensor, 0), name, false)
        }
        Value::Num(n) => parse_numeric_dimension_value(*n, name, false),
        other => Err(format!(
            "{name}: dimension must be numeric, got {:?}",
            other
        )),
    }
}

fn parse_numeric_dimension_value(
    value: f64,
    name: &str,
    allow_zero: bool,
) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(format!("{name}: dimension must be finite"));
    }
    let rounded = value.round();
    // Allow small floating error tolerance when users pass float-typed dims
    if (rounded - value).abs() > 1e-6 {
        return Err(format!("{name}: dimension must be an integer"));
    }
    let min = if allow_zero { 0.0 } else { 1.0 };
    if rounded < min {
        let bound = if allow_zero { 0 } else { 1 };
        return Err(format!("{name}: dimension must be >= {bound}"));
    }
    if !fits_platform_usize(rounded) {
        return Err(format!("{name}: dimension is outside the supported range"));
    }
    Ok(rounded as usize)
}

/// Attempt to extract a string from a runtime value.
pub fn value_to_string(value: &Value) -> Option<String> {
    String::try_from(value).ok()
}

/// Return a canonical 2-D shape for a tensor given its shape slice and element count.
///
/// * Empty data (`len == 0`) → `[0, 1]` (MATLAB convention for empty arrays).
/// * No shape info (`shape.is_empty()`) → `[1, 1]` (scalar).
/// * Otherwise → the tensor's own shape.
pub fn default_shape_for(shape: &[usize], len: usize) -> Vec<usize> {
    if len == 0 {
        vec![0, 1]
    } else if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

/// Clamp a scalar f64 to the uint8 range [0, 255], rounding to the nearest integer.
pub fn clamp_u8(value: f64) -> f64 {
    value.round().clamp(0.0, u8::MAX as f64)
}

/// Clamp a scalar f64 to the uint16 range [0, 65535], rounding to the nearest integer.
pub fn clamp_u16(value: f64) -> f64 {
    value.round().clamp(0.0, u16::MAX as f64)
}

/// Clamp a scalar f64 to the uint32 range [0, 4294967295], rounding to the nearest integer.
pub fn clamp_u32(value: f64) -> f64 {
    value.round().clamp(0.0, u32::MAX as f64)
}

/// Cast all elements of a tensor to the target dtype in-place, preserving the f64 backing store.
pub fn coerce_tensor_dtype(mut tensor: Tensor, dtype: NumericDType) -> Tensor {
    match dtype {
        NumericDType::F64 => {
            tensor.integer_data = None;
            tensor.dtype = NumericDType::F64;
        }
        NumericDType::F32 => {
            tensor.integer_data = None;
            for value in &mut tensor.data {
                *value = (*value as f32) as f64;
            }
            tensor.dtype = NumericDType::F32;
        }
        integer_dtype => {
            let shape = tensor.shape.clone();
            let prototype = match integer_dtype {
                NumericDType::I8 => IntegerStorage::I8(Vec::new()),
                NumericDType::I16 => IntegerStorage::I16(Vec::new()),
                NumericDType::I32 => IntegerStorage::I32(Vec::new()),
                NumericDType::I64 => IntegerStorage::I64(Vec::new()),
                NumericDType::U8 => IntegerStorage::U8(Vec::new()),
                NumericDType::U16 => IntegerStorage::U16(Vec::new()),
                NumericDType::U32 => IntegerStorage::U32(Vec::new()),
                NumericDType::U64 => IntegerStorage::U64(Vec::new()),
                NumericDType::F32 | NumericDType::F64 => unreachable!(),
            };
            if let Some(storage) = tensor.integer_storage() {
                let values = storage
                    .exact_values()
                    .into_iter()
                    .map(|value| prototype.cast_exact_assignment(&value))
                    .collect();
                return Tensor::new_integer(
                    prototype
                        .from_same_class_values(values)
                        .expect("integer coercion produces target-class values"),
                    shape,
                )
                .expect("dtype coercion preserves the tensor element count");
            }
            let values = std::mem::take(&mut tensor.data);
            return integer_tensor_from_f64_like(&prototype, values, &shape)
                .expect("dtype coercion preserves the tensor element count");
        }
    }
    tensor
}

#[cfg(test)]
mod dtype_tests {
    use super::{coerce_tensor_dtype, ones_with_dtype, zeros_with_dtype};
    use runmat_builtins::{IntegerStorage, NumericDType, Tensor};

    #[test]
    fn dtype_directed_constructors_materialize_all_integer_classes() {
        let cases = [
            (NumericDType::I8, IntegerStorage::I8(vec![0, 0])),
            (NumericDType::I16, IntegerStorage::I16(vec![0, 0])),
            (NumericDType::I32, IntegerStorage::I32(vec![0, 0])),
            (NumericDType::I64, IntegerStorage::I64(vec![0, 0])),
            (NumericDType::U8, IntegerStorage::U8(vec![0, 0])),
            (NumericDType::U16, IntegerStorage::U16(vec![0, 0])),
            (NumericDType::U32, IntegerStorage::U32(vec![0, 0])),
            (NumericDType::U64, IntegerStorage::U64(vec![0, 0])),
        ];

        for (dtype, expected_zeros) in cases {
            let zeros = zeros_with_dtype(&[1, 2], dtype).expect("zeros");
            assert_eq!(zeros.dtype, dtype);
            assert_eq!(zeros.integer_storage(), Some(&expected_zeros));

            let ones = ones_with_dtype(&[1, 2], dtype).expect("ones");
            assert_eq!(ones.dtype, dtype);
            assert_eq!(ones.integer_storage(), Some(&expected_zeros.ones_like(2)));
        }
    }

    #[test]
    fn coercion_creates_exact_storage_and_float_conversion_clears_it() {
        let input = Tensor::new(vec![-2.4, 2.6], vec![1, 2]).expect("input");
        let typed = coerce_tensor_dtype(input, NumericDType::I16);
        assert_eq!(typed.dtype, NumericDType::I16);
        assert_eq!(
            typed.integer_storage(),
            Some(&IntegerStorage::I16(vec![-2, 3]))
        );

        let float = coerce_tensor_dtype(typed, NumericDType::F64);
        assert_eq!(float.dtype, NumericDType::F64);
        assert!(float.integer_storage().is_none());
    }

    #[test]
    fn integer_to_integer_coercion_reads_exact_storage_not_f64_mirror() {
        let wide = 9_007_199_254_740_993_u64;
        let mut input = Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2])
            .expect("input");
        input.data.clear();

        let same_class = coerce_tensor_dtype(input.clone(), NumericDType::U64);
        assert_eq!(same_class.dtype, NumericDType::U64);
        assert_eq!(
            same_class.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide, u64::MAX]))
        );

        let signed = coerce_tensor_dtype(input, NumericDType::I64);
        assert_eq!(signed.dtype, NumericDType::I64);
        assert_eq!(
            signed.integer_storage(),
            Some(&IntegerStorage::I64(vec![
                i64::try_from(wide).expect("wide value fits int64"),
                i64::MAX,
            ]))
        );
    }

    #[test]
    fn integer_to_integer_coercion_preserves_every_integer_class_exactly() {
        let cases = [
            (IntegerStorage::I8(vec![i8::MIN, i8::MAX]), NumericDType::I8),
            (
                IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
                NumericDType::I16,
            ),
            (
                IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
                NumericDType::I32,
            ),
            (
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                NumericDType::I64,
            ),
            (IntegerStorage::U8(vec![0, u8::MAX]), NumericDType::U8),
            (IntegerStorage::U16(vec![0, u16::MAX]), NumericDType::U16),
            (IntegerStorage::U32(vec![0, u32::MAX]), NumericDType::U32),
            (IntegerStorage::U64(vec![0, u64::MAX]), NumericDType::U64),
        ];

        for (storage, dtype) in cases {
            let mut input =
                Tensor::new_integer(storage.clone(), vec![1, 2]).expect("integer input");
            input.data.clear();
            let output = coerce_tensor_dtype(input, dtype);
            assert_eq!(output.dtype, dtype);
            assert_eq!(output.integer_storage(), Some(&storage));
        }
    }

    #[test]
    fn integer_to_integer_coercion_preserves_empty_shape_and_storage_class() {
        let input =
            Tensor::new_integer(IntegerStorage::I64(Vec::new()), vec![0, 3]).expect("empty input");
        let output = coerce_tensor_dtype(input, NumericDType::U64);

        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(output.dtype, NumericDType::U64);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(Vec::new()))
        );
    }
}

#[cfg(test)]
mod dimension_tests {
    use super::{
        dimension_from_value_async, dims_from_value_async, integer_tensor_to_f64, parse_dimension,
        scalar_f64_from_value_async, tensor_into_value, tensor_into_values_f64, tensor_values_f64,
        tensor_values_f64_cow,
    };
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor, Value};

    #[test]
    fn typed_dimension_parsers_preserve_representable_uint64_values() {
        assert_eq!(
            parse_dimension(&Value::Int(IntValue::U64(3)), "size"),
            Ok(3)
        );
        match usize::try_from(u64::MAX) {
            Ok(value) => assert_eq!(
                parse_dimension(&Value::Int(IntValue::U64(u64::MAX)), "size"),
                Ok(value)
            ),
            Err(_) => {
                assert!(parse_dimension(&Value::Int(IntValue::U64(u64::MAX)), "size").is_err())
            }
        }
        assert_eq!(
            block_on(dims_from_value_async(&Value::Int(IntValue::U64(3)))),
            Ok(Some(vec![3]))
        );
        assert_eq!(
            block_on(dimension_from_value_async(
                &Value::Int(IntValue::U64(3)),
                "size",
                false
            )),
            Ok(Some(3))
        );
        assert!(block_on(dims_from_value_async(&Value::Int(IntValue::I64(-1)))).is_err());
    }

    #[test]
    fn typed_integer_tensor_dimension_parsers_use_exact_storage() {
        let dims =
            Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2]).expect("integer dims");
        assert_eq!(
            block_on(dims_from_value_async(&Value::Tensor(dims))),
            Ok(Some(vec![2, 3]))
        );

        let mut scalar_dim = Tensor::new_integer(IntegerStorage::U64(vec![3]), vec![1, 1])
            .expect("integer scalar dim");
        scalar_dim.data.clear();
        assert_eq!(
            block_on(dimension_from_value_async(
                &Value::Tensor(scalar_dim),
                "size",
                false,
            )),
            Ok(Some(3))
        );
    }

    #[test]
    fn typed_integer_tensor_dimension_parsers_preserve_large_values_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let mut scalar = Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1])
            .expect("large integer dim");
        scalar.data.clear();
        assert_eq!(
            parse_dimension(&Value::Tensor(scalar.clone()), "size"),
            Ok(large as usize)
        );
        assert_eq!(
            block_on(dimension_from_value_async(
                &Value::Tensor(scalar),
                "size",
                false,
            )),
            Ok(Some(large as usize))
        );

        let dims = Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1])
            .expect("large integer dims");
        assert_eq!(
            block_on(dims_from_value_async(&Value::Tensor(dims))),
            Ok(Some(vec![large as usize]))
        );
    }

    #[test]
    fn typed_integer_tensor_dimension_parsers_reject_negative_values() {
        let mut negative =
            Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("negative dim");
        negative.data.clear();
        assert!(parse_dimension(&Value::Tensor(negative.clone()), "size").is_err());
        assert!(block_on(dimension_from_value_async(
            &Value::Tensor(negative.clone()),
            "size",
            false,
        ))
        .is_err());
        assert!(block_on(dims_from_value_async(&Value::Tensor(negative))).is_err());
    }

    #[test]
    fn typed_integer_tensor_f64_boundary_reads_exact_storage() {
        let wide = 9_007_199_254_740_993_u64;
        let mut scalar =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("scalar");
        scalar.data.clear();
        assert_eq!(
            block_on(scalar_f64_from_value_async(&Value::Tensor(scalar))),
            Ok(Some(IntValue::U64(wide).to_f64()))
        );

        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide, wide - 1]), vec![1, 2])
            .expect("integer tensor");
        assert_eq!(
            tensor_values_f64(&tensor),
            vec![
                IntValue::U64(wide).to_f64(),
                IntValue::U64(wide - 1).to_f64()
            ]
        );

        let normalized = integer_tensor_to_f64(tensor).expect("normalize");
        assert!(normalized.integer_storage().is_none());
        assert_eq!(normalized.shape, vec![1, 2]);
        assert_eq!(
            normalized.data,
            vec![
                IntValue::U64(wide).to_f64(),
                IntValue::U64(wide - 1).to_f64()
            ]
        );
    }

    #[test]
    fn tensor_values_f64_cow_borrows_double_storage() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
        match tensor_values_f64_cow(&tensor) {
            std::borrow::Cow::Borrowed(values) => assert_eq!(values, &[1.0, 2.0]),
            std::borrow::Cow::Owned(values) => panic!("expected borrowed values, got {values:?}"),
        }
    }

    #[test]
    fn tensor_into_values_f64_reads_typed_integer_storage_exactly() {
        let wide = 9_007_199_254_740_993_u64;
        let mut tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
            .expect("integer tensor");
        tensor.data.fill(0.0);

        assert_eq!(
            tensor_into_values_f64(tensor),
            vec![IntValue::U64(wide).to_f64()]
        );
    }

    #[test]
    fn tensor_into_value_reads_typed_integer_scalar_storage_exactly() {
        let wide = 9_007_199_254_740_993_u64;
        let mut tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
            .expect("integer tensor");
        tensor.data.clear();

        assert_eq!(tensor_into_value(tensor), Value::Int(IntValue::U64(wide)));
    }

    #[test]
    fn binary_numeric_tensors_reads_typed_integer_scalar_storage_exactly() {
        let wide = 9_007_199_254_740_993_u64;
        let mut lhs = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
            .expect("integer tensor");
        lhs.data.fill(0.0);
        let rhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("double tensor");

        let (lhs_values, rhs_values, shape) =
            super::binary_numeric_tensors(&lhs, &rhs, "plus", "plus").expect("align");
        assert_eq!(lhs_values, vec![IntValue::U64(wide).to_f64(); 2]);
        assert_eq!(rhs_values, vec![1.0, 2.0]);
        assert_eq!(shape, vec![1, 2]);
    }

    #[test]
    fn binary_numeric_tensors_reads_typed_integer_array_storage_exactly() {
        let mut lhs =
            Tensor::new_integer(IntegerStorage::I16(vec![-2, 3]), vec![1, 2]).expect("lhs");
        lhs.data.fill(0.0);
        let mut rhs =
            Tensor::new_integer(IntegerStorage::U64(vec![4, 5]), vec![1, 2]).expect("rhs");
        rhs.data.fill(9.0);

        let (lhs_values, rhs_values, shape) =
            super::binary_numeric_tensors(&lhs, &rhs, "times", "times").expect("align");
        assert_eq!(lhs_values, vec![-2.0, 3.0]);
        assert_eq!(rhs_values, vec![4.0, 5.0]);
        assert_eq!(shape, vec![1, 2]);
    }

    #[test]
    fn binary_numeric_tensors_reads_typed_integer_rhs_scalar_storage_exactly() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("double tensor");
        let mut rhs =
            Tensor::new_integer(IntegerStorage::I64(vec![-7]), vec![1, 1]).expect("integer tensor");
        rhs.data.fill(0.0);

        let (lhs_values, rhs_values, shape) =
            super::binary_numeric_tensors(&lhs, &rhs, "minus", "minus").expect("align");
        assert_eq!(lhs_values, vec![1.0, 2.0]);
        assert_eq!(rhs_values, vec![-7.0, -7.0]);
        assert_eq!(shape, vec![1, 2]);
    }

    #[test]
    fn binary_numeric_tensors_preserves_shape_mismatch_error() {
        let lhs = Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2]).expect("lhs");
        let rhs = Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).expect("rhs");

        let err = super::binary_numeric_tensors(&lhs, &rhs, "plus", "plus").unwrap_err();
        assert!(err.message().contains("matching sizes"));
        assert_eq!(err.context.builtin.as_deref(), Some("plus"));
    }

    #[test]
    fn floating_dimension_parsers_reject_values_outside_platform_range() {
        let out_of_range = usize::MAX as f64;
        assert!(parse_dimension(&Value::Num(out_of_range), "size").is_err());
        assert!(block_on(dimension_from_value_async(
            &Value::Num(out_of_range),
            "size",
            false
        ))
        .is_err());
        assert!(block_on(dims_from_value_async(&Value::Num(out_of_range))).is_err());
    }
}

/// Align two numeric tensors for a binary element-wise operation with scalar broadcasting.
///
/// Returns `(lhs_data, rhs_data, output_shape)`.  If either operand is a
/// single element it is broadcast to the other's length.  `builtin` names the
/// calling builtin and is embedded in the error message when the shapes are
/// incompatible.
pub fn binary_numeric_tensors(
    lhs: &Tensor,
    rhs: &Tensor,
    context: &str,
    builtin: &str,
) -> crate::BuiltinResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let lhs_values = tensor_values_f64_cow(lhs);
    let rhs_values = tensor_values_f64_cow(rhs);
    let lhs_shape = default_shape_for(&lhs.shape, lhs_values.len());
    let rhs_shape = default_shape_for(&rhs.shape, rhs_values.len());
    match (lhs_values.len(), rhs_values.len()) {
        (1, 1) => Ok((vec![lhs_values[0]], vec![rhs_values[0]], vec![1, 1])),
        (1, len) => Ok((vec![lhs_values[0]; len], rhs_values.into_owned(), rhs_shape)),
        (len, 1) => Ok((lhs_values.into_owned(), vec![rhs_values[0]; len], lhs_shape)),
        (left, right) if left == right && lhs_shape == rhs_shape => {
            Ok((lhs_values.into_owned(), rhs_values.into_owned(), lhs_shape))
        }
        _ => Err(crate::build_runtime_error(format!(
            "{context}: operands must be scalar or have matching sizes"
        ))
        .with_builtin(builtin)
        .build()),
    }
}
