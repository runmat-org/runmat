pub(crate) mod character_numeric;

use std::{borrow::Cow, convert::TryFrom};

use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorStorage, HostTensorOwned};
use runmat_value::{
    ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType, NumericStorage, Tensor,
    Value,
};

use crate::dispatcher::gather_if_needed_async;
use crate::RuntimeError;

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

/// Materialize a tensor's authoritative numeric values in the f64 computation domain.
/// Callers that admit integers are responsible for any required exactness check.
pub fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

/// Return a borrowed view for native double tensors and explicitly materialize
/// every other numeric class in the f64 computation domain.
pub fn tensor_values_f64_cow(tensor: &Tensor) -> Cow<'_, [f64]> {
    match tensor.as_f64_slice() {
        Some(values) => Cow::Borrowed(values),
        None => Cow::Owned(tensor.materialize_f64()),
    }
}

/// Read one authoritative numeric tensor value into the f64 computation domain.
/// Callers that admit integers are responsible for any required exactness check.
pub fn tensor_value_f64(tensor: &Tensor, index: usize) -> f64 {
    match tensor
        .numeric_value_at(index)
        .expect("tensor_value_f64: numeric storage index is in bounds")
    {
        runmat_value::NumericScalar::F64(value) => value,
        runmat_value::NumericScalar::F32(value) => f64::from(value),
        scalar => scalar
            .into_int_value()
            .expect("non-floating numeric scalar is integer")
            .to_f64(),
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

/// Consume a tensor and explicitly materialize its authoritative storage in
/// the f64 computation domain.
pub fn tensor_into_values_f64(tensor: Tensor) -> Vec<f64> {
    tensor
        .into_numeric_storage()
        .expect("validated tensor storage")
        .materialize_f64()
}

/// Consume a tensor at a provider API boundary that currently accepts only owned f64 storage.
pub fn tensor_into_host_f64_owned(tensor: Tensor) -> HostTensorOwned {
    let shape = tensor.shape().to_vec();
    HostTensorOwned {
        data: tensor_into_values_f64(tensor),
        shape,
        storage: GpuTensorStorage::Real,
    }
}

/// Return a complex tensor's numeric values as Complex64, reading typed integer
/// real/imaginary storage exactly instead of using the compatibility buffer.
pub fn complex_tensor_values_complex64(tensor: &ComplexTensor) -> Vec<Complex64> {
    tensor
        .integer_storage()
        .as_ref()
        .map(|storage| {
            let real = storage.real().exact_values();
            let imag = storage.imag().exact_values();
            real.into_iter()
                .zip(imag)
                .map(|(re, im)| Complex64::new(re.to_f64(), im.to_f64()))
                .collect()
        })
        .unwrap_or_else(|| {
            tensor
                .materialize_f64()
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect()
        })
}

/// Return a complex tensor's element count, reading exact integer-complex
/// storage instead of the floating compatibility buffer when present.
pub fn complex_tensor_element_len(tensor: &ComplexTensor) -> usize {
    tensor
        .integer_storage()
        .as_ref()
        .map_or(tensor.materialize_f64().len(), |storage| storage.len())
}

/// Return true when a complex tensor contains exactly one scalar element.
pub fn is_scalar_complex_tensor(tensor: &ComplexTensor) -> bool {
    complex_tensor_element_len(tensor) == 1
}

/// Return one complex tensor value as Complex64, reading typed integer-complex
/// storage exactly instead of using the compatibility backing buffer.
pub fn complex_tensor_value_complex64(tensor: &ComplexTensor, index: usize) -> Complex64 {
    match tensor.integer_storage() {
        Some(storage) => {
            let real = storage
                .real()
                .value_at(index)
                .expect("complex_tensor_value_complex64: real storage index is in bounds")
                .to_f64();
            let imag = storage
                .imag()
                .value_at(index)
                .expect("complex_tensor_value_complex64: imaginary storage index is in bounds")
                .to_f64();
            Complex64::new(real, imag)
        }
        None => {
            let (real, imag) = tensor.materialize_f64()[index];
            Complex64::new(real, imag)
        }
    }
}

/// Consume a complex tensor and return Complex64 values, preserving the fast
/// path for ordinary complex double tensors.
pub fn complex_tensor_into_values_complex64(tensor: ComplexTensor) -> Vec<Complex64> {
    if tensor.integer_storage().is_some() {
        complex_tensor_values_complex64(&tensor)
    } else {
        tensor
            .materialize_f64()
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
    Tensor::new(tensor_values_f64(&tensor), tensor.shape().to_vec())
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
        if tensor.numeric_dtype() == runmat_value::NumericDType::F64 {
            Value::Num(tensor_value_f64(&tensor, 0))
        } else {
            Value::Tensor(tensor)
        }
    } else {
        Value::Tensor(tensor)
    }
}

/// Return true when a tensor contains exactly one scalar element.
pub fn is_scalar_tensor(tensor: &Tensor) -> bool {
    tensor_element_len(tensor) == 1
}

pub fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

pub(crate) fn scalar_f64_from_host_value(value: &Value) -> Result<Option<f64>, String> {
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
            if !handle.shape().is_empty() {
                let len = element_count(handle.shape());
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

/// Decode an integer or floating scalar dimension without provider access.
/// Other value kinds are left to the existing asynchronous scalar admission.
pub fn scalar_dimension_from_value_typed(
    value: &Value,
    allow_zero: bool,
) -> Result<Option<usize>, DimensionDecodeError> {
    match value {
        Value::Int(value) => parse_integer_dimension_typed(value, allow_zero).map(Some),
        Value::Num(value) => parse_numeric_dimension_value_typed(*value, allow_zero).map(Some),
        _ => Ok(None),
    }
}

fn parse_integer_dimension_typed(
    value: &IntValue,
    allow_zero: bool,
) -> Result<usize, DimensionDecodeError> {
    if value.to_i128() < 0 {
        return Err(DimensionDecodeError::IntegerNegative);
    }
    let dimension = value
        .try_to_usize()
        .ok_or(DimensionDecodeError::IntegerOutOfPlatformRange)?;
    if !allow_zero && dimension == 0 {
        return Err(DimensionDecodeError::BelowMinimum);
    }
    Ok(dimension)
}

fn parse_numeric_dimension_value_typed(
    value: f64,
    allow_zero: bool,
) -> Result<usize, DimensionDecodeError> {
    use runmat_types::dimension::ScalarDimensionValueError as Error;
    runmat_types::dimension::decode_scalar_dimension(value, allow_zero).map_err(|error| match error
    {
        Error::NonFinite => DimensionDecodeError::NonFinite,
        Error::Fractional => DimensionDecodeError::Fractional,
        Error::BelowMinimum => DimensionDecodeError::BelowMinimum,
        Error::OutsidePlatformRange => DimensionDecodeError::OutOfPlatformRange,
    })
}

fn scalar_dimension_legacy_message(
    error: DimensionDecodeError,
    name: &str,
    allow_zero: bool,
) -> String {
    match error {
        DimensionDecodeError::NonFinite => format!("{name}: dimension must be finite"),
        DimensionDecodeError::Fractional => format!("{name}: dimension must be an integer"),
        DimensionDecodeError::BelowMinimum | DimensionDecodeError::Negative => {
            format!(
                "{name}: dimension must be >= {}",
                if allow_zero { 0 } else { 1 }
            )
        }
        DimensionDecodeError::OutOfPlatformRange
        | DimensionDecodeError::IntegerNegative
        | DimensionDecodeError::IntegerOutOfPlatformRange => {
            format!("{name}: dimension is outside the supported range")
        }
        DimensionDecodeError::Gather(source) => format!("{name}: {source}"),
    }
}

fn parse_integer_dimension(
    value: &IntValue,
    name: &str,
    allow_zero: bool,
) -> Result<usize, String> {
    parse_integer_dimension_typed(value, allow_zero)
        .map_err(|error| scalar_dimension_legacy_message(error, name, allow_zero))
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

/// Why a shape/dimension vector cannot be decoded. Public builtins retain
/// their own diagnostic wording while dispatching on these semantic causes.
#[derive(Debug, thiserror::Error)]
pub enum DimensionDecodeError {
    #[error("dimensions must be finite")]
    NonFinite,
    #[error("matrix dimensions must be non-negative")]
    Negative,
    #[error("dimensions must be integers")]
    Fractional,
    #[error("dimensions are outside the supported platform range")]
    OutOfPlatformRange,
    #[error("dimensions must be non-negative platform integers")]
    IntegerNegative,
    #[error("dimensions must be non-negative platform integers")]
    IntegerOutOfPlatformRange,
    #[error("dimension is below the allowed minimum")]
    BelowMinimum,
    #[error("dimensions: {0}")]
    Gather(#[source] Box<RuntimeError>),
}

impl DimensionDecodeError {
    /// Preserve provider provenance and retry policy beneath an owning
    /// builtin's public diagnostic rather than formatting away the source.
    pub fn attach_cause(self, mut owner: RuntimeError) -> RuntimeError {
        if let Self::Gather(cause) = self {
            owner.gpu_gather_retry = cause.gpu_gather_retry;
            if owner.span.is_none() {
                owner.span = cause.span;
            }
            owner.source = Some(cause);
        }
        owner
    }
}

fn parse_integer_shape_dimension(value: &IntValue) -> Result<usize, DimensionDecodeError> {
    if value.to_i128() < 0 {
        return Err(DimensionDecodeError::IntegerNegative);
    }
    value
        .try_to_usize()
        .ok_or(DimensionDecodeError::IntegerOutOfPlatformRange)
}

fn parse_numeric_dimension(value: f64) -> Result<usize, DimensionDecodeError> {
    runmat_types::decode_shape_dimension(value).map_err(|cause| match cause {
        runmat_types::DimensionValueError::NonFinite => DimensionDecodeError::NonFinite,
        runmat_types::DimensionValueError::Negative => DimensionDecodeError::Negative,
        runmat_types::DimensionValueError::Fractional => DimensionDecodeError::Fractional,
        runmat_types::DimensionValueError::OutsidePlatformRange => {
            DimensionDecodeError::OutOfPlatformRange
        }
    })
}

/// A borrowed, allocation-free view of the canonical dimension-vector
/// grammar. Callers with an admitted collection owner can materialize only
/// their final dimensions instead of first allocating a temporary decoder Vec.
pub(crate) enum BorrowedHostShapeDimensions<'a> {
    Floating(&'a Tensor, std::ops::Range<usize>),
    Integer(&'a IntegerStorage, std::ops::Range<usize>),
    Logical(&'a [u8], std::ops::Range<usize>),
    Number(Option<f64>),
    IntegerScalar(Option<&'a IntValue>),
}

impl BorrowedHostShapeDimensions<'_> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Floating(_, indices) | Self::Integer(_, indices) | Self::Logical(_, indices) => {
                indices.len()
            }
            Self::Number(value) => usize::from(value.is_some()),
            Self::IntegerScalar(value) => usize::from(value.is_some()),
        }
    }
}

impl Iterator for BorrowedHostShapeDimensions<'_> {
    type Item = Result<usize, DimensionDecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Floating(tensor, indices) => indices
                .next()
                .map(|index| parse_numeric_dimension(tensor_value_f64(tensor, index))),
            Self::Integer(storage, indices) => indices.next().map(|index| {
                parse_integer_shape_dimension(
                    &storage.value_at(index).expect("integer storage index"),
                )
            }),
            Self::Logical(values, indices) => indices
                .next()
                .map(|index| parse_numeric_dimension(if values[index] != 0 { 1.0 } else { 0.0 })),
            Self::Number(value) => value.take().map(parse_numeric_dimension),
            Self::IntegerScalar(value) => value.take().map(parse_integer_shape_dimension),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for BorrowedHostShapeDimensions<'_> {}

pub(crate) fn borrowed_host_shape_dimensions(
    value: &Value,
) -> Option<BorrowedHostShapeDimensions<'_>> {
    let is_vector = |len: usize, shape: &[usize]| {
        len == 0
            || len == 1
            || shape.len() == 1
            || (shape.len() >= 2 && (shape[0] == 1 || shape[1] == 1))
    };
    match value {
        Value::Num(value) => Some(BorrowedHostShapeDimensions::Number(Some(*value))),
        Value::Int(value) => Some(BorrowedHostShapeDimensions::IntegerScalar(Some(value))),
        Value::Tensor(tensor) if is_vector(tensor.len(), tensor.shape()) => {
            match tensor.integer_storage() {
                Some(storage) => Some(BorrowedHostShapeDimensions::Integer(
                    storage,
                    0..storage.len(),
                )),
                None => Some(BorrowedHostShapeDimensions::Floating(
                    tensor,
                    0..tensor.len(),
                )),
            }
        }
        Value::LogicalArray(logical) if is_vector(logical.data.len(), &logical.shape) => Some(
            BorrowedHostShapeDimensions::Logical(&logical.data, 0..logical.data.len()),
        ),
        _ => None,
    }
}

fn dims_from_tensor_values(
    values: impl ExactSizeIterator<Item = f64>,
    shape: &[usize],
) -> Result<Option<Vec<usize>>, DimensionDecodeError> {
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
    for value in values {
        dims.push(parse_numeric_dimension(value)?);
    }
    Ok(Some(dims))
}

fn dims_from_integer_tensor_values(
    storage: &IntegerStorage,
    shape: &[usize],
) -> Result<Option<Vec<usize>>, DimensionDecodeError> {
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
    dims_from_value_async_typed(value)
        .await
        .map_err(|error| error.to_string())
}

/// Typed dimension-vector admission for owners that need semantic error
/// identity and a retained provider cause. The string-returning wrapper above
/// remains for existing callers until they migrate to this boundary.
pub async fn dims_from_value_async_typed(
    value: &Value,
) -> Result<Option<Vec<usize>>, DimensionDecodeError> {
    match value {
        Value::Num(n) => parse_numeric_dimension(*n).map(|dim| Some(vec![dim])),
        Value::Int(i) => parse_integer_shape_dimension(i).map(|dim| Some(vec![dim])),
        Value::Tensor(t) => match t.integer_storage() {
            Some(storage) => dims_from_integer_tensor_values(storage, t.shape()),
            None => dims_from_tensor_values(
                (0..t.len()).map(|index| tensor_value_f64(t, index)),
                t.shape(),
            ),
        },
        Value::LogicalArray(la) => {
            let values = la.data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 });
            dims_from_tensor_values(values, &la.shape)
        }
        Value::GpuTensor(handle) => {
            let gathered = gather_if_needed_async(&Value::GpuTensor(handle.clone()))
                .await
                .map_err(|cause| DimensionDecodeError::Gather(Box::new(cause)))?;
            match gathered {
                Value::Tensor(t) => {
                    if t.is_empty() {
                        tracing::warn!(
                            gpu_shape = ?handle.shape(),
                            "dims_from_value_async: gathered GPU tensor has no data"
                        );
                    }
                    tracing::trace!(
                        "dims_from_value_async: GPU tensor values gpu_shape={:?} host_shape={:?} class={} elements={}",
                        handle.shape(),
                        t.shape(),
                        t.numeric_dtype().class_name(),
                        t.len()
                    );
                    let dims = match t.integer_storage() {
                        Some(storage) => dims_from_integer_tensor_values(storage, t.shape())?,
                        None => dims_from_tensor_values(
                            (0..t.len()).map(|index| tensor_value_f64(&t, index)),
                            t.shape(),
                        )?,
                    };
                    if dims.is_none() {
                        tracing::debug!(
                            gpu_shape = ?handle.shape(),
                            host_shape = ?t.shape(),
                            "dims_from_value_async: GPU tensor not interpretable as dims"
                        );
                    }
                    Ok(dims)
                }
                Value::LogicalArray(la) => {
                    let values = la.data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 });
                    let dims = dims_from_tensor_values(values, &la.shape)?;
                    if dims.is_none() {
                        tracing::debug!(
                            gpu_shape = ?handle.shape(),
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
    parse_numeric_dimension_value_typed(value, allow_zero)
        .map_err(|error| scalar_dimension_legacy_message(error, name, allow_zero))
}

/// Attempt to extract a string from a runtime value.
pub fn value_to_string(value: &Value) -> Option<String> {
    String::try_from(value).ok()
}

/// Preserve the tensor's shape, including explicit empty dimensions.
///
/// Only absent shape metadata receives a default: `[0, 1]` for empty data,
/// or `[1, 1]` otherwise.
pub fn default_shape_for(shape: &[usize], len: usize) -> Vec<usize> {
    if !shape.is_empty() {
        shape.to_vec()
    } else if len == 0 {
        vec![0, 1]
    } else {
        vec![1, 1]
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

/// Cast all elements of a tensor to the target dtype through typed constructors.
pub fn coerce_tensor_dtype(tensor: Tensor, dtype: NumericDType) -> Tensor {
    // A same-class tensor already has the requested representation. Moving it
    // preserves its payload, shape allocation, and any exact host reservation
    // carried by that payload; rebuilding it would silently detach the claim.
    if tensor.numeric_dtype() == dtype {
        return tensor;
    }
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .expect("validated tensor storage");
    match dtype {
        NumericDType::F64 => {
            Tensor::from_numeric_storage(NumericStorage::F64(storage.materialize_f64()), shape)
                .expect("dtype coercion preserves the tensor element count")
        }
        NumericDType::F32 => {
            Tensor::from_numeric_storage(NumericStorage::F32(storage.materialize_f32()), shape)
                .expect("dtype coercion preserves the tensor element count")
        }
        integer_dtype => {
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
            let floating_storage = match storage.into_integer_storage() {
                Ok(storage) => {
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
                Err(storage) => storage,
            };
            integer_tensor_from_f64_like(&prototype, floating_storage.materialize_f64(), &shape)
                .expect("dtype coercion preserves the tensor element count")
        }
    }
}

#[cfg(test)]
mod dtype_tests {
    use super::{coerce_tensor_dtype, ones_with_dtype, zeros_with_dtype};
    use runmat_value::{IntegerStorage, NumericDType, NumericStorage, Tensor};

    #[test]
    fn same_dtype_coercion_moves_all_ten_numeric_classes_and_shape() {
        let cases = [
            NumericStorage::F64(vec![-0.0, f64::INFINITY]),
            NumericStorage::F32(vec![-0.0, f32::INFINITY]),
            NumericStorage::I8(vec![i8::MIN, i8::MAX]),
            NumericStorage::I16(vec![i16::MIN, i16::MAX]),
            NumericStorage::I32(vec![i32::MIN, i32::MAX]),
            NumericStorage::I64(vec![i64::MIN, i64::MAX]),
            NumericStorage::U8(vec![u8::MIN, u8::MAX]),
            NumericStorage::U16(vec![u16::MIN, u16::MAX]),
            NumericStorage::U32(vec![u32::MIN, u32::MAX]),
            NumericStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];
        for storage in cases {
            let input = Tensor::from_numeric_storage(storage, vec![1, 2]).expect("typed input");
            let dtype = input.numeric_dtype();
            let shape_pointer = input.shape().as_ptr();
            let payload_pointer = unsafe { input.host_buffer().foreign_data_pointer() };
            let output = coerce_tensor_dtype(input, dtype);
            assert_eq!(output.numeric_dtype(), dtype);
            assert_eq!(output.shape(), [1, 2]);
            assert_eq!(output.shape().as_ptr(), shape_pointer);
            assert_eq!(
                unsafe { output.host_buffer().foreign_data_pointer() },
                payload_pointer
            );
            match dtype {
                NumericDType::F64 => assert_eq!(
                    output.as_f64_slice().expect("double output")[0].to_bits(),
                    (-0.0_f64).to_bits()
                ),
                NumericDType::F32 => assert_eq!(
                    output.as_f32_slice().expect("single output")[0].to_bits(),
                    (-0.0_f32).to_bits()
                ),
                NumericDType::U64 => assert_eq!(
                    output.integer_storage(),
                    Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
                ),
                _ => {}
            }
        }
    }

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
            assert_eq!(zeros.numeric_dtype(), dtype);
            assert_eq!(zeros.integer_storage(), Some(&expected_zeros));

            let ones = ones_with_dtype(&[1, 2], dtype).expect("ones");
            assert_eq!(ones.numeric_dtype(), dtype);
            assert_eq!(ones.integer_storage(), Some(&expected_zeros.ones_like(2)));
        }
    }

    #[test]
    fn coercion_creates_exact_storage_and_float_conversion_clears_it() {
        let input = Tensor::new(vec![-2.4, 2.6], vec![1, 2]).expect("input");
        let typed = coerce_tensor_dtype(input, NumericDType::I16);
        assert_eq!(typed.numeric_dtype(), NumericDType::I16);
        assert_eq!(
            typed.integer_storage(),
            Some(&IntegerStorage::I16(vec![-2, 3]))
        );

        let float = coerce_tensor_dtype(typed, NumericDType::F64);
        assert_eq!(float.numeric_dtype(), NumericDType::F64);
        assert!(float.integer_storage().is_none());
    }

    #[test]
    fn integer_to_integer_coercion_reads_exact_storage_not_f64_mirror() {
        let wide = 9_007_199_254_740_993_u64;
        let input = Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2])
            .expect("input");

        let same_class = coerce_tensor_dtype(input.clone(), NumericDType::U64);
        assert_eq!(same_class.numeric_dtype(), NumericDType::U64);
        assert_eq!(
            same_class.integer_storage(),
            Some(&IntegerStorage::U64(vec![wide, u64::MAX]))
        );

        let signed = coerce_tensor_dtype(input, NumericDType::I64);
        assert_eq!(signed.numeric_dtype(), NumericDType::I64);
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
            let input = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("integer input");
            let output = coerce_tensor_dtype(input, dtype);
            assert_eq!(output.numeric_dtype(), dtype);
            assert_eq!(output.integer_storage(), Some(&storage));
        }
    }

    #[test]
    fn integer_to_integer_coercion_preserves_empty_shape_and_storage_class() {
        let input =
            Tensor::new_integer(IntegerStorage::I64(Vec::new()), vec![0, 3]).expect("empty input");
        let output = coerce_tensor_dtype(input, NumericDType::U64);

        assert_eq!(output.shape(), [0, 3]);
        assert_eq!(output.numeric_dtype(), NumericDType::U64);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(Vec::new()))
        );
    }
}

#[cfg(test)]
mod dimension_tests {
    use super::{
        default_shape_for, dimension_from_value_async, dims_from_value_async,
        dims_from_value_async_typed, integer_tensor_to_f64, parse_dimension,
        scalar_dimension_from_value_typed, scalar_f64_from_value_async, tensor_into_value,
        tensor_into_values_f64, tensor_values_f64, tensor_values_f64_cow, DimensionDecodeError,
    };
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, LogicalArray, NumericStorage, Tensor, Value};

    #[test]
    fn shape_defaults_preserve_explicit_empty_geometry() {
        for (shape, len, expected) in [
            (vec![0, 3], 0, vec![0, 3]),
            (vec![3, 0], 0, vec![3, 0]),
            (vec![2, 0, 4], 0, vec![2, 0, 4]),
            (vec![2, 3], 6, vec![2, 3]),
            (vec![], 0, vec![0, 1]),
            (vec![], 1, vec![1, 1]),
        ] {
            assert_eq!(default_shape_for(&shape, len), expected);
        }
    }

    #[test]
    fn typed_dimension_decoder_retains_semantic_error_identity() {
        // Single and logical controls decode directly into the final usize
        // vector; neither needs a temporary floating-point payload.
        for (value, expected) in [
            (
                Value::Tensor(Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).unwrap()),
                vec![2, 3],
            ),
            (
                Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap()),
                vec![1, 0],
            ),
        ] {
            assert_eq!(
                block_on(dims_from_value_async_typed(&value)).unwrap(),
                Some(expected)
            );
        }
        assert!(matches!(
            block_on(dims_from_value_async_typed(&Value::Tensor(
                Tensor::from_f32(vec![1.25], vec![1, 1]).unwrap()
            ))),
            Err(DimensionDecodeError::Fractional)
        ));
        assert!(matches!(
            block_on(dims_from_value_async_typed(&Value::Num(f64::NAN))),
            Err(DimensionDecodeError::NonFinite)
        ));
        assert!(matches!(
            block_on(dims_from_value_async_typed(&Value::Num(-1.0))),
            Err(DimensionDecodeError::Negative)
        ));
        assert!(matches!(
            block_on(dims_from_value_async_typed(&Value::Num(1.25))),
            Err(DimensionDecodeError::Fractional)
        ));
        assert!(matches!(
            block_on(dims_from_value_async_typed(&Value::Int(IntValue::I64(-1)))),
            Err(DimensionDecodeError::IntegerNegative)
        ));
        assert!(matches!(
            scalar_dimension_from_value_typed(&Value::Num(0.0), false),
            Err(DimensionDecodeError::BelowMinimum)
        ));
        assert!(matches!(
            scalar_dimension_from_value_typed(&Value::Num(1.25), false),
            Err(DimensionDecodeError::Fractional)
        ));
        assert_eq!(
            scalar_dimension_from_value_typed(&Value::Num(1.0 + 5.0e-7), false)
                .expect("legacy scalar tolerance retained"),
            Some(1)
        );
    }

    #[test]
    fn typed_dimension_error_preserves_nested_provider_cause_and_retry() {
        let mut cause = crate::RuntimeError::new("provider transfer failed");
        cause.identifier = Some("RunMat:test:ProviderTransfer".into());
        cause.gpu_gather_retry = crate::GpuGatherRetry::Never;
        let owner = DimensionDecodeError::Gather(Box::new(cause))
            .attach_cause(crate::RuntimeError::new("mean: invalid dimension"));
        assert_eq!(owner.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let source = owner.source.expect("provider cause retained");
        assert_eq!(source.to_string(), "provider transfer failed");
    }

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

        let scalar_dim = Tensor::new_integer(IntegerStorage::U64(vec![3]), vec![1, 1])
            .expect("integer scalar dim");
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
    fn typed_integer_dimension_parsers_ignore_poisoned_f64_mirrors_for_all_classes() {
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];

        for storage in storages {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer dim");
            assert_eq!(
                block_on(dims_from_value_async(&Value::Tensor(tensor))),
                Ok(Some(vec![2]))
            );
        }
    }

    #[test]
    fn typed_integer_tensor_dimension_parsers_preserve_large_values_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let scalar = Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1])
            .expect("large integer dim");
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
        let negative =
            Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("negative dim");
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
        let scalar =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("scalar");
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
        assert_eq!(normalized.shape(), [1, 2]);
        assert_eq!(
            normalized.materialize_f64(),
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
    fn tensor_values_f64_cow_materializes_native_single_storage() {
        let tensor = Tensor::from_f32(vec![0.1, f32::MAX], vec![1, 2]).expect("single tensor");
        match tensor_values_f64_cow(&tensor) {
            std::borrow::Cow::Owned(values) => {
                assert_eq!(values, vec![f64::from(0.1_f32), f64::from(f32::MAX)])
            }
            std::borrow::Cow::Borrowed(values) => {
                panic!("expected explicit single materialization, got {values:?}")
            }
        }
    }

    #[test]
    fn tensor_into_values_f64_reads_typed_integer_storage_exactly() {
        let wide = 9_007_199_254_740_993_u64;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
            .expect("integer tensor");

        assert_eq!(
            tensor_into_values_f64(tensor),
            vec![IntValue::U64(wide).to_f64()]
        );
    }

    #[test]
    fn tensor_into_value_reads_typed_integer_scalar_storage_exactly() {
        let wide = 9_007_199_254_740_993_u64;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
            .expect("integer tensor");

        assert_eq!(tensor_into_value(tensor), Value::Int(IntValue::U64(wide)));
    }

    #[test]
    fn tensor_into_value_preserves_native_single_scalar_storage() {
        let tensor = Tensor::from_f32(vec![0.25], vec![1, 1]).expect("single tensor");
        let Value::Tensor(output) = tensor_into_value(tensor) else {
            panic!("single scalar must retain tensor class");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.25])
        );
    }

    #[test]
    fn binary_numeric_tensors_reads_typed_integer_scalar_storage_exactly() {
        let wide = 9_007_199_254_740_993_u64;
        let lhs = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
            .expect("integer tensor");
        let rhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("double tensor");

        let (lhs_values, rhs_values, shape) =
            super::binary_numeric_tensors(&lhs, &rhs, "plus", "plus").expect("align");
        assert_eq!(lhs_values, vec![IntValue::U64(wide).to_f64(); 2]);
        assert_eq!(rhs_values, vec![1.0, 2.0]);
        assert_eq!(shape, vec![1, 2]);
    }

    #[test]
    fn binary_numeric_tensors_reads_typed_integer_array_storage_exactly() {
        let lhs = Tensor::new_integer(IntegerStorage::I16(vec![-2, 3]), vec![1, 2]).expect("lhs");
        let rhs = Tensor::new_integer(IntegerStorage::U64(vec![4, 5]), vec![1, 2]).expect("rhs");

        let (lhs_values, rhs_values, shape) =
            super::binary_numeric_tensors(&lhs, &rhs, "times", "times").expect("align");
        assert_eq!(lhs_values, vec![-2.0, 3.0]);
        assert_eq!(rhs_values, vec![4.0, 5.0]);
        assert_eq!(shape, vec![1, 2]);
    }

    #[test]
    fn binary_numeric_tensors_reads_typed_integer_rhs_scalar_storage_exactly() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("double tensor");
        let rhs =
            Tensor::new_integer(IntegerStorage::I64(vec![-7]), vec![1, 1]).expect("integer tensor");

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
    let lhs_shape = default_shape_for(lhs.shape(), lhs_values.len());
    let rhs_shape = default_shape_for(rhs.shape(), rhs_values.len());
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
