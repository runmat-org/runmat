use std::sync::Arc;

use chrono::{Datelike, NaiveDate, Timelike, Utc};
use runmat_python::{
    PythonArray, PythonBufferOwner, PythonDType, PythonDateTime, PythonTimeDelta, PythonValue,
};
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
        Value::StructArray(_) => Err(invalid_conversion(
            "structure arrays require a shape-preserving Python conversion",
        )),
        Value::Object(value) if value.is_class(runmat_types::standard::DATETIME) => {
            datetime_to_python(&value)
        }
        Value::Object(value) if value.is_class(runmat_types::standard::DURATION) => {
            duration_to_python(Value::Object(value))
        }
        Value::Object(value)
            if value
                .class_name
                .is(runmat_types::standard::PYTHON_ARGUMENTS) =>
        {
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
        PythonValue::DateTime(value) => datetime_from_python(value),
        PythonValue::TimeDelta(value) => timedelta_from_python(value),
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
        PythonDType::DateTime64Micros => datetime_array_from_python(&bytes, array.shape),
        PythonDType::TimeDelta64Micros => duration_array_from_python(&bytes, array.shape),
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
        PythonDType::Bool
        | PythonDType::Complex64
        | PythonDType::Complex128
        | PythonDType::DateTime64Micros
        | PythonDType::TimeDelta64Micros => {
            return Err(invalid_conversion("invalid real numeric Python dtype"));
        }
    })
}

const MICROS_PER_SECOND: i64 = 1_000_000;
const SECONDS_PER_DAY: i64 = 86_400;
const MICROS_PER_DAY: i64 = SECONDS_PER_DAY * MICROS_PER_SECOND;

fn datetime_to_python(value: &runmat_value::ObjectInstance) -> Result<PythonValue, RuntimeError> {
    let tensor = crate::builtins::datetime::serial_tensor_for_object(value)?;
    let mut micros = Vec::with_capacity(tensor.len());
    let mut scalar = None;
    for serial in tensor.materialize_f64() {
        let date = crate::builtins::datetime::naive_from_datenum(serial)?;
        let value = PythonDateTime {
            year: date.year(),
            month: date.month() as u8,
            day: date.day() as u8,
            hour: date.hour() as u8,
            minute: date.minute() as u8,
            second: date.second() as u8,
            microsecond: date.and_utc().timestamp_subsec_micros(),
        };
        if tensor.len() == 1 {
            scalar = Some(value);
        } else {
            micros.push(date.and_utc().timestamp_micros());
        }
    }
    if let Some(value) = scalar {
        return Ok(PythonValue::DateTime(value));
    }
    Ok(temporal_array(
        PythonDType::DateTime64Micros,
        tensor.shape,
        micros,
    ))
}

fn duration_to_python(value: Value) -> Result<PythonValue, RuntimeError> {
    let tensor = crate::builtins::duration::duration_tensor_from_duration_value(&value)?;
    let micros = tensor
        .materialize_f64()
        .into_iter()
        .map(|days| days_to_micros(days, "duration"))
        .collect::<Result<Vec<_>, _>>()?;
    if let [total] = micros.as_slice() {
        let days = total.div_euclid(MICROS_PER_DAY);
        let remainder = total.rem_euclid(MICROS_PER_DAY);
        return Ok(PythonValue::TimeDelta(PythonTimeDelta {
            days,
            seconds: (remainder / MICROS_PER_SECOND) as u32,
            microseconds: (remainder % MICROS_PER_SECOND) as u32,
        }));
    }
    Ok(temporal_array(
        PythonDType::TimeDelta64Micros,
        tensor.shape,
        micros,
    ))
}

fn temporal_array(dtype: PythonDType, shape: Vec<usize>, values: Vec<i64>) -> PythonValue {
    let bytes = values
        .into_iter()
        .flat_map(i64::to_ne_bytes)
        .collect::<Vec<_>>();
    PythonValue::Array(PythonArray::from_owned_bytes(
        dtype, shape, true, true, bytes,
    ))
}

fn datetime_from_python(value: PythonDateTime) -> Result<Value, RuntimeError> {
    let date = NaiveDate::from_ymd_opt(value.year, u32::from(value.month), u32::from(value.day))
        .and_then(|date| {
            date.and_hms_micro_opt(
                u32::from(value.hour),
                u32::from(value.minute),
                u32::from(value.second),
                value.microsecond,
            )
        })
        .ok_or_else(|| invalid_conversion("Python datetime contains invalid components"))?;
    let serial = crate::builtins::datetime::datenum_from_naive(date);
    let tensor = Tensor::new(vec![serial], vec![1, 1]).map_err(invalid_conversion)?;
    crate::builtins::datetime::datetime_object_from_serial_tensor(tensor, "dd-MMM-yyyy HH:mm:ss")
}

fn timedelta_from_python(value: PythonTimeDelta) -> Result<Value, RuntimeError> {
    let days = value.days;
    let seconds = value.seconds;
    let microseconds = value.microseconds;
    let total = i128::from(days)
        .checked_mul(i128::from(MICROS_PER_DAY))
        .and_then(|total| total.checked_add(i128::from(seconds) * i128::from(MICROS_PER_SECOND)))
        .and_then(|total| total.checked_add(i128::from(microseconds)))
        .ok_or_else(|| invalid_conversion("Python timedelta is outside RunMat's duration range"))?;
    let total = matlab_duration_micros(total)?;
    duration_from_micros(vec![total], vec![1, 1])
}

fn datetime_array_from_python(bytes: &[u8], shape: Vec<usize>) -> Result<Value, RuntimeError> {
    let serials = decode_i64(bytes)
        .into_iter()
        .map(|micros| {
            if micros == i64::MIN {
                return Err(invalid_conversion(
                    "NumPy NaT cannot be converted to datetime",
                ));
            }
            let date = chrono::DateTime::<Utc>::from_timestamp_micros(micros)
                .ok_or_else(|| invalid_conversion("NumPy datetime64 is outside RunMat's range"))?
                .naive_utc();
            Ok(crate::builtins::datetime::datenum_from_naive(date))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tensor = Tensor::new(serials, shape).map_err(invalid_conversion)?;
    crate::builtins::datetime::datetime_object_from_serial_tensor(tensor, "dd-MMM-yyyy HH:mm:ss")
}

fn duration_array_from_python(bytes: &[u8], shape: Vec<usize>) -> Result<Value, RuntimeError> {
    let values = decode_i64(bytes)
        .into_iter()
        .map(|micros| {
            if micros == i64::MIN {
                return Err(invalid_conversion(
                    "NumPy NaT cannot be converted to duration",
                ));
            }
            matlab_duration_micros(i128::from(micros))
        })
        .collect::<Result<Vec<_>, _>>()?;
    duration_from_micros(values, shape)
}

fn duration_from_micros(values: Vec<i64>, shape: Vec<usize>) -> Result<Value, RuntimeError> {
    let days = values
        .into_iter()
        .map(|value| value as f64 / MICROS_PER_DAY as f64)
        .collect();
    let tensor = Tensor::new(days, shape).map_err(invalid_conversion)?;
    crate::builtins::duration::duration_object_from_days_tensor(tensor, "hh:mm:ss")
}

fn matlab_duration_micros(value: i128) -> Result<i64, RuntimeError> {
    let milliseconds = value / 1_000;
    i64::try_from(milliseconds * 1_000)
        .map_err(|_| invalid_conversion("Python duration is outside RunMat's duration range"))
}

fn days_to_micros(days: f64, kind: &str) -> Result<i64, RuntimeError> {
    let micros = (days * MICROS_PER_DAY as f64).trunc();
    if !micros.is_finite() || micros < i64::MIN as f64 || micros > i64::MAX as f64 {
        return Err(invalid_conversion(format!(
            "{kind} is outside Python's temporal range"
        )));
    }
    Ok(micros as i64)
}

fn decode_i64(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_ne_bytes(chunk.try_into().expect("exact chunk")))
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_conversion_rejects_typed_structure_arrays_without_reclassifying_cells() {
        let mut first = runmat_value::StructValue::new();
        first.insert("payload", Value::Num(1.0));
        let mut second = runmat_value::StructValue::new();
        second.insert("payload", Value::Num(2.0));
        let array = runmat_value::StructArray::with_fields(
            vec!["payload".into()],
            vec![first.clone(), second.clone()],
            vec![1, 2],
        )
        .expect("structure array");
        let error = value_to_python(Value::StructArray(array))
            .expect_err("unshaped Python tuples must not represent structure arrays");
        assert!(error
            .message()
            .contains("shape-preserving Python conversion"));

        let cell =
            runmat_value::CellArray::new(vec![Value::Struct(first), Value::Struct(second)], 1, 2)
                .expect("cell of structures");
        let PythonValue::Tuple(values) = value_to_python(Value::Cell(cell)).expect("cell") else {
            panic!("expected Python tuple");
        };
        assert_eq!(values.len(), 2);
        assert!(values
            .iter()
            .all(|value| matches!(value, PythonValue::Dict(_))));
    }

    #[test]
    fn datetime_scalar_uses_python_datetime_components() {
        let date = NaiveDate::from_ymd_opt(2026, 8, 27)
            .unwrap()
            .and_hms_micro_opt(12, 34, 56, 654_321)
            .unwrap();
        let tensor = Tensor::new(
            vec![crate::builtins::datetime::datenum_from_naive(date)],
            vec![1, 1],
        )
        .unwrap();
        let value = crate::builtins::datetime::datetime_object_from_serial_tensor(
            tensor,
            "dd-MMM-yyyy HH:mm:ss",
        )
        .unwrap();
        let PythonValue::DateTime(converted) = value_to_python(value).unwrap() else {
            panic!("expected Python datetime");
        };
        assert_eq!(
            (converted.year, converted.month, converted.day),
            (2026, 8, 27)
        );
        assert_eq!(
            (converted.hour, converted.minute, converted.second),
            (12, 34, 56)
        );
        assert!(converted.microsecond.abs_diff(654_321) <= 10);
    }

    #[test]
    fn python_temporal_values_convert_to_runmat_objects_with_documented_precision() {
        let value = value_from_python(PythonValue::DateTime(PythonDateTime {
            year: 1969,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 59,
            microsecond: 123_456,
        }))
        .unwrap();
        let Value::Object(object) = value else {
            panic!("expected datetime object");
        };
        let serial = crate::builtins::datetime::serial_tensor_for_object(&object).unwrap();
        let round_trip =
            crate::builtins::datetime::naive_from_datenum(serial.materialize_f64()[0]).unwrap();
        assert!(round_trip.and_utc().timestamp_micros().abs_diff(-876_544) <= 10);

        let duration = value_from_python(PythonValue::TimeDelta(PythonTimeDelta {
            days: -1,
            seconds: 86_399,
            microseconds: 998_999,
        }))
        .unwrap();
        let tensor =
            crate::builtins::duration::duration_tensor_from_duration_value(&duration).unwrap();
        let micros = (tensor.materialize_f64()[0] * MICROS_PER_DAY as f64).round() as i64;
        assert_eq!(micros, -1_000, "Python duration truncates to milliseconds");
    }

    #[test]
    fn numpy_temporal_arrays_keep_shape_and_reject_nat() {
        let dates = PythonArray::from_owned_bytes(
            PythonDType::DateTime64Micros,
            vec![2, 1],
            true,
            false,
            [-876_544_i64, 1_787_834_096_654_321_i64]
                .into_iter()
                .flat_map(i64::to_ne_bytes)
                .collect(),
        );
        let value = value_from_python(PythonValue::Array(dates)).unwrap();
        let Value::Object(object) = value else {
            panic!("expected datetime object");
        };
        let tensor = crate::builtins::datetime::serial_tensor_for_object(&object).unwrap();
        assert_eq!(tensor.shape, vec![2, 1]);
        assert!(
            crate::builtins::datetime::naive_from_datenum(tensor.materialize_f64()[0])
                .unwrap()
                .and_utc()
                .timestamp_micros()
                .abs_diff(-876_544)
                <= 10
        );

        let nat = PythonArray::from_owned_bytes(
            PythonDType::TimeDelta64Micros,
            vec![1, 1],
            true,
            false,
            i64::MIN.to_ne_bytes().to_vec(),
        );
        let error = value_from_python(PythonValue::Array(nat)).unwrap_err();
        assert!(error.message().contains("NaT"));
    }
}
