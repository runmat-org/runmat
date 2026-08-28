use std::sync::Arc;

use runmat_python::{PythonArray, PythonBufferOwner, PythonDType, PythonValue};
use runmat_value::{
    CellArray, ComplexStorage, ComplexTensor, HostComplexBuffer, HostLogicalBuffer,
    HostNumericBuffer, IntValue, IntegerStorage, LogicalArray, NumericDType, NumericStorage,
    StructValue, Tensor, Value,
};

use super::super::{foreign_error, ForeignErrorKind};
use crate::RuntimeError;

#[derive(Debug)]
struct NumericOwner(HostNumericBuffer);

impl PythonBufferOwner for NumericOwner {
    fn address(&self) -> usize {
        // SAFETY: this owner clone keeps the COW allocation stable and Python
        // receives a read-only view for no longer than the owner capsule lives.
        unsafe { self.0.foreign_data_pointer() as usize }
    }

    fn byte_length(&self) -> usize {
        self.0.checked_byte_len().unwrap_or(0)
    }

    fn copy_bytes(&self) -> Vec<u8> {
        copy_pointer_bytes(self.address(), self.byte_length())
    }
}

#[derive(Debug)]
struct LogicalOwner(HostLogicalBuffer);

impl PythonBufferOwner for LogicalOwner {
    fn address(&self) -> usize {
        // SAFETY: the cloned COW owner pins this read-only allocation.
        unsafe { self.0.foreign_data_pointer() as usize }
    }

    fn byte_length(&self) -> usize {
        self.0.len()
    }

    fn copy_bytes(&self) -> Vec<u8> {
        self.0.as_slice().to_vec()
    }
}

#[derive(Debug)]
struct ComplexF64Owner(HostComplexBuffer<f64>);

impl PythonBufferOwner for ComplexF64Owner {
    fn address(&self) -> usize {
        // SAFETY: the cloned COW owner pins this read-only allocation.
        unsafe { self.0.foreign_data_pointer() as usize }
    }

    fn byte_length(&self) -> usize {
        self.0.len() * std::mem::size_of::<runmat_value::ComplexElement<f64>>()
    }

    fn copy_bytes(&self) -> Vec<u8> {
        copy_pointer_bytes(self.address(), self.byte_length())
    }
}

#[derive(Debug)]
struct ComplexF32Owner(HostComplexBuffer<f32>);

impl PythonBufferOwner for ComplexF32Owner {
    fn address(&self) -> usize {
        // SAFETY: the cloned COW owner pins this read-only allocation.
        unsafe { self.0.foreign_data_pointer() as usize }
    }

    fn byte_length(&self) -> usize {
        self.0.len() * std::mem::size_of::<runmat_value::ComplexElement<f32>>()
    }

    fn copy_bytes(&self) -> Vec<u8> {
        copy_pointer_bytes(self.address(), self.byte_length())
    }
}

fn copy_pointer_bytes(address: usize, length: usize) -> Vec<u8> {
    if length == 0 {
        return Vec::new();
    }
    // SAFETY: each caller owns a live immutable host buffer proving at least
    // length bytes at address for the duration of this copy.
    unsafe { std::slice::from_raw_parts(address as *const u8, length) }.to_vec()
}

pub(super) fn value_to_python(value: Value) -> Result<PythonValue, RuntimeError> {
    match value {
        Value::Bool(value) => Ok(PythonValue::Bool(value)),
        Value::Num(value) => Ok(PythonValue::Float(value)),
        Value::Complex(real, imaginary) => Ok(PythonValue::Complex { real, imaginary }),
        Value::Int(value) => Ok(match value {
            IntValue::I8(value) => PythonValue::Signed(i64::from(value)),
            IntValue::I16(value) => PythonValue::Signed(i64::from(value)),
            IntValue::I32(value) => PythonValue::Signed(i64::from(value)),
            IntValue::I64(value) => PythonValue::Signed(value),
            IntValue::U8(value) => PythonValue::Unsigned(u64::from(value)),
            IntValue::U16(value) => PythonValue::Unsigned(u64::from(value)),
            IntValue::U32(value) => PythonValue::Unsigned(u64::from(value)),
            IntValue::U64(value) => PythonValue::Unsigned(value),
        }),
        Value::String(value) => Ok(PythonValue::String(value)),
        Value::CharArray(value) => value
            .row_string()
            .map(PythonValue::String)
            .ok_or_else(|| invalid_conversion("character conversion requires a row vector")),
        Value::StringArray(value) => Ok(PythonValue::List(
            value.data.into_iter().map(PythonValue::String).collect(),
        )),
        Value::Tensor(value) => {
            let dtype = python_dtype(value.numeric_dtype());
            let shape = value.shape.clone();
            let owner = Arc::new(NumericOwner(value.host_buffer().clone()));
            Ok(PythonValue::Array(PythonArray {
                dtype,
                shape,
                column_major: true,
                read_only: true,
                owner,
            }))
        }
        Value::LogicalArray(value) => {
            let shape = value.shape.clone();
            let owner = Arc::new(LogicalOwner(value.data.clone()));
            Ok(PythonValue::Array(PythonArray {
                dtype: PythonDType::Bool,
                shape,
                column_major: true,
                read_only: true,
                owner,
            }))
        }
        Value::ComplexTensor(value) => {
            let shape = value.shape.clone();
            let (dtype, owner): (PythonDType, Arc<dyn PythonBufferOwner>) =
                match value.complex_storage() {
                    ComplexStorage::F64(owner) => (
                        PythonDType::Complex128,
                        Arc::new(ComplexF64Owner(owner.clone())),
                    ),
                    ComplexStorage::F32(owner) => (
                        PythonDType::Complex64,
                        Arc::new(ComplexF32Owner(owner.clone())),
                    ),
                    ComplexStorage::Integer(_) => {
                        return Err(invalid_conversion(
                            "integer-complex arrays require an explicit Python conversion",
                        ));
                    }
                };
            Ok(PythonValue::Array(PythonArray {
                dtype,
                shape,
                column_major: true,
                read_only: true,
                owner,
            }))
        }
        Value::Cell(value) => Ok(PythonValue::Tuple(
            value
                .data
                .into_iter()
                .map(value_to_python)
                .collect::<Result<_, _>>()?,
        )),
        Value::Struct(value) => Ok(PythonValue::Dict(
            value
                .fields
                .into_iter()
                .map(|(name, value)| Ok((PythonValue::String(name), value_to_python(value)?)))
                .collect::<Result<_, RuntimeError>>()?,
        )),
        Value::Object(value) if value.class_name == "RunMat.PythonArguments" => {
            keyword_bundle(value)
        }
        Value::Foreign(_) => Err(invalid_conversion(
            "foreign objects must be resolved by their owning adapter",
        )),
        other => Err(invalid_conversion(format!(
            "RunMat value {} is not convertible to Python",
            value_kind(&other)
        ))),
    }
}

fn keyword_bundle(value: runmat_value::ObjectInstance) -> Result<PythonValue, RuntimeError> {
    let names = value
        .properties
        .get("Names")
        .and_then(|value| match value {
            Value::Cell(value) => Some(&value.data),
            _ => None,
        })
        .ok_or_else(|| invalid_conversion("invalid pyargs name storage"))?;
    let values = value
        .properties
        .get("Values")
        .and_then(|value| match value {
            Value::Cell(value) => Some(&value.data),
            _ => None,
        })
        .ok_or_else(|| invalid_conversion("invalid pyargs value storage"))?;
    if names.len() != values.len() {
        return Err(invalid_conversion("invalid pyargs bundle length"));
    }
    Ok(PythonValue::Keywords(
        names
            .iter()
            .zip(values)
            .map(|(name, value)| {
                let name = String::try_from(name)
                    .map_err(|_| invalid_conversion("pyargs names must be text"))?;
                Ok((name, value_to_python(value.clone())?))
            })
            .collect::<Result<_, RuntimeError>>()?,
    ))
}

pub(super) fn value_from_python(value: PythonValue) -> Result<Value, RuntimeError> {
    match value {
        PythonValue::None => Tensor::new(Vec::new(), vec![0, 0])
            .map(Value::Tensor)
            .map_err(invalid_conversion),
        PythonValue::Bool(value) => Ok(Value::Bool(value)),
        PythonValue::Signed(value) => Ok(Value::Int(IntValue::I64(value))),
        PythonValue::Unsigned(value) => Ok(Value::Int(IntValue::U64(value))),
        PythonValue::Float(value) => Ok(Value::Num(value)),
        PythonValue::Complex { real, imaginary } => Ok(Value::Complex(real, imaginary)),
        PythonValue::String(value) => Ok(Value::String(value)),
        PythonValue::Bytes(value) => {
            let length = value.len();
            Tensor::new_integer(IntegerStorage::U8(value), vec![1, length])
                .map(Value::Tensor)
                .map_err(invalid_conversion)
        }
        PythonValue::List(values) | PythonValue::Tuple(values) => {
            let length = values.len();
            CellArray::new(
                values
                    .into_iter()
                    .map(value_from_python)
                    .collect::<Result<_, _>>()?,
                length,
                1,
            )
            .map(Value::Cell)
            .map_err(invalid_conversion)
        }
        PythonValue::Dict(values) => dictionary_from_python(values),
        PythonValue::Array(value) => array_from_python(value),
        PythonValue::Object(_) => Err(invalid_conversion(
            "Python object identity must be registered before value conversion",
        )),
        PythonValue::Callback(_) => Err(invalid_conversion(
            "Python callback tokens are valid only while entering Python",
        )),
        PythonValue::Keywords(_) => Err(invalid_conversion(
            "Python keyword bundles are valid only as call arguments",
        )),
    }
}

fn dictionary_from_python(values: Vec<(PythonValue, PythonValue)>) -> Result<Value, RuntimeError> {
    if values
        .iter()
        .all(|(key, _)| matches!(key, PythonValue::String(_)))
    {
        let mut structure = StructValue::new();
        for (key, value) in values {
            let PythonValue::String(key) = key else {
                unreachable!();
            };
            structure.insert(key, value_from_python(value)?);
        }
        return Ok(Value::Struct(structure));
    }
    let rows = values.len();
    let mut cells = Vec::with_capacity(rows * 2);
    for (key, value) in values {
        cells.push(value_from_python(key)?);
        cells.push(value_from_python(value)?);
    }
    CellArray::new(cells, rows, 2)
        .map(Value::Cell)
        .map_err(invalid_conversion)
}

fn array_from_python(array: PythonArray) -> Result<Value, RuntimeError> {
    let bytes = array.owner.copy_bytes();
    let expected = array
        .shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| count.checked_mul(array.dtype.byte_width()));
    if expected != Some(bytes.len()) {
        return Err(invalid_conversion(
            "Python array payload length does not match its dtype and shape",
        ));
    }
    match array.dtype {
        PythonDType::Bool => LogicalArray::new(bytes, array.shape)
            .map(Value::LogicalArray)
            .map_err(invalid_conversion),
        PythonDType::Complex64 => ComplexTensor::from_f32(decode_complex_f32(&bytes)?, array.shape)
            .map(Value::ComplexTensor)
            .map_err(invalid_conversion),
        PythonDType::Complex128 => ComplexTensor::new(decode_complex_f64(&bytes)?, array.shape)
            .map(Value::ComplexTensor)
            .map_err(invalid_conversion),
        dtype => Tensor::from_numeric_storage(decode_numeric(dtype, &bytes)?, array.shape)
            .map(Value::Tensor)
            .map_err(invalid_conversion),
    }
}

fn decode_numeric(dtype: PythonDType, bytes: &[u8]) -> Result<NumericStorage, RuntimeError> {
    macro_rules! decode {
        ($type:ty, $variant:ident) => {{
            let width = std::mem::size_of::<$type>();
            let values = bytes
                .chunks_exact(width)
                .map(|chunk| <$type>::from_ne_bytes(chunk.try_into().expect("exact chunk")))
                .collect();
            NumericStorage::$variant(values)
        }};
    }
    Ok(match dtype {
        PythonDType::Float64 => decode!(f64, F64),
        PythonDType::Float32 => decode!(f32, F32),
        PythonDType::Int8 => NumericStorage::I8(bytes.iter().map(|byte| *byte as i8).collect()),
        PythonDType::Int16 => decode!(i16, I16),
        PythonDType::Int32 => decode!(i32, I32),
        PythonDType::Int64 => decode!(i64, I64),
        PythonDType::Uint8 => NumericStorage::U8(bytes.to_vec()),
        PythonDType::Uint16 => decode!(u16, U16),
        PythonDType::Uint32 => decode!(u32, U32),
        PythonDType::Uint64 => decode!(u64, U64),
        PythonDType::Bool | PythonDType::Complex64 | PythonDType::Complex128 => {
            return Err(invalid_conversion("invalid real numeric Python dtype"));
        }
    })
}

fn decode_complex_f32(bytes: &[u8]) -> Result<Vec<(f32, f32)>, RuntimeError> {
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            (
                f32::from_ne_bytes(chunk[..4].try_into().expect("exact chunk")),
                f32::from_ne_bytes(chunk[4..].try_into().expect("exact chunk")),
            )
        })
        .collect())
}

fn decode_complex_f64(bytes: &[u8]) -> Result<Vec<(f64, f64)>, RuntimeError> {
    Ok(bytes
        .chunks_exact(16)
        .map(|chunk| {
            (
                f64::from_ne_bytes(chunk[..8].try_into().expect("exact chunk")),
                f64::from_ne_bytes(chunk[8..].try_into().expect("exact chunk")),
            )
        })
        .collect())
}

fn python_dtype(dtype: NumericDType) -> PythonDType {
    match dtype {
        NumericDType::F64 => PythonDType::Float64,
        NumericDType::F32 => PythonDType::Float32,
        NumericDType::I8 => PythonDType::Int8,
        NumericDType::I16 => PythonDType::Int16,
        NumericDType::I32 => PythonDType::Int32,
        NumericDType::I64 => PythonDType::Int64,
        NumericDType::U8 => PythonDType::Uint8,
        NumericDType::U16 => PythonDType::Uint16,
        NumericDType::U32 => PythonDType::Uint32,
        NumericDType::U64 => PythonDType::Uint64,
    }
}

fn invalid_conversion(message: impl Into<String>) -> RuntimeError {
    foreign_error(ForeignErrorKind::ConversionFailed, message)
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::SparseTensor(_) => "sparse",
        Value::Symbolic(_) | Value::SymbolicArray(_) => "symbolic",
        Value::GpuTensor(_) => "GPU-resident",
        Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_) => "object",
        Value::Listener(_) => "listener",
        Value::OutputList(_) => "output-list",
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => "callable",
        Value::ClassRef(_) => "class-reference",
        Value::MException(_) => "exception",
        Value::Future(_) | Value::Task(_) | Value::Pool(_) | Value::Job(_) => "execution-handle",
        _ => "value",
    }
}
