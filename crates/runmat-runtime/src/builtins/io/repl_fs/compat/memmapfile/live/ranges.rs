//! Selective mapped Data access using the ordinary object-index planner.
//! A selected element or field never materializes or rewrites unrelated bytes.

use super::super::{format_error, MemmapDataType, MemmapFormat, MemmapFormatPresentation};
use crate::execution::CpuCooperationBudget;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::MEMMAPFILE_CLASS;
use runmat_value::{
    CharArray, IntValue, IntegerStorage, MemmapReceiverState, NumericDType, Tensor, Value,
};

mod plan;
mod effects;
pub(super) use effects::{read, write};

struct Selected {
    offsets: Vec<u64>,
    unit_bytes: usize,
    shape: Vec<usize>,
    format: MemmapFormat,
    records: bool,
}

fn receiver(value: &Value) -> BuiltinResult<&MemmapReceiverState> {
    let Value::Object(object) = value else {
        return Err(format_error("memmapfile: receiver is not an object"));
    };
    if !object.is_class(MEMMAPFILE_CLASS) {
        return Err(format_error("memmapfile: receiver has the wrong class"));
    }
    object
        .memmap_receiver()
        .ok_or_else(|| format_error("memmapfile: receiver has no sealed state"))
}

fn mapped_error(
    descriptor: &'static runmat_builtins::BuiltinErrorDescriptor,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
    prior_committed: usize,
) -> RuntimeError {
    build_runtime_error(format!(
        "memmapfile: mapped range operation failed after {prior_committed} prior bytes committed"
    ))
    .with_builtin("memmapfile")
    .with_identifier(descriptor.identifier.expect("typed mapped range error"))
    .with_source(source)
    .build()
}

async fn checkpoint(budget: &mut Option<CpuCooperationBudget>, bytes: usize) -> BuiltinResult<()> {
    if let Some(budget) = budget {
        budget
            .checkpoint(bytes as u64)
            .await
            .map_err(|source| source.into_runtime_error())?;
    }
    Ok(())
}

fn field_width(dtype: MemmapDataType, shape: &[usize]) -> BuiltinResult<usize> {
    let count = shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| format_error("memmapfile: Format shape is too large"))?;
    count
        .checked_mul(dtype.byte_size())
        .ok_or_else(|| format_error("memmapfile: Format field is too large"))
}

fn scalar_format(dtype: MemmapDataType, shape: Vec<usize>) -> MemmapFormat {
    MemmapFormat {
        dtype,
        shape,
        field: None,
        additional_fields: Vec::new(),
        presentation: MemmapFormatPresentation::TypeName,
    }
}

fn scalar_bytes(dtype: MemmapDataType, value: &Value) -> Option<Vec<u8>> {
    match (dtype, value) {
        (MemmapDataType::Numeric(runmat_value::NumericDType::F64), Value::Num(value)) => {
            Some(value.to_le_bytes().to_vec())
        }
        (
            MemmapDataType::Numeric(runmat_value::NumericDType::I8),
            Value::Int(IntValue::I8(value)),
        ) => Some(vec![*value as u8]),
        (
            MemmapDataType::Numeric(runmat_value::NumericDType::U8),
            Value::Int(IntValue::U8(value)),
        ) => Some(vec![*value]),
        (
            MemmapDataType::Numeric(runmat_value::NumericDType::I16),
            Value::Int(IntValue::I16(value)),
        ) => Some(value.to_le_bytes().to_vec()),
        (
            MemmapDataType::Numeric(runmat_value::NumericDType::U16),
            Value::Int(IntValue::U16(value)),
        ) => Some(value.to_le_bytes().to_vec()),
        (
            MemmapDataType::Numeric(runmat_value::NumericDType::I32),
            Value::Int(IntValue::I32(value)),
        ) => Some(value.to_le_bytes().to_vec()),
        (
            MemmapDataType::Numeric(runmat_value::NumericDType::U32),
            Value::Int(IntValue::U32(value)),
        ) => Some(value.to_le_bytes().to_vec()),
        (
            MemmapDataType::Numeric(runmat_value::NumericDType::I64),
            Value::Int(IntValue::I64(value)),
        ) => Some(value.to_le_bytes().to_vec()),
        (
            MemmapDataType::Numeric(runmat_value::NumericDType::U64),
            Value::Int(IntValue::U64(value)),
        ) => Some(value.to_le_bytes().to_vec()),
        _ => None,
    }
}

fn empty_value(dtype: MemmapDataType, shape: Vec<usize>) -> BuiltinResult<Value> {
    match dtype {
        MemmapDataType::Character => Ok(Value::CharArray(
            CharArray::from_column_major(Vec::new(), shape).map_err(format_error)?,
        )),
        MemmapDataType::Numeric(dtype) => {
            let tensor = match dtype {
                NumericDType::F64 => Tensor::new(Vec::new(), shape),
                NumericDType::F32 => Tensor::from_f32(Vec::new(), shape),
                NumericDType::I8 => Tensor::new_integer(IntegerStorage::I8(Vec::new()), shape),
                NumericDType::U8 => Tensor::new_integer(IntegerStorage::U8(Vec::new()), shape),
                NumericDType::I16 => Tensor::new_integer(IntegerStorage::I16(Vec::new()), shape),
                NumericDType::U16 => Tensor::new_integer(IntegerStorage::U16(Vec::new()), shape),
                NumericDType::I32 => Tensor::new_integer(IntegerStorage::I32(Vec::new()), shape),
                NumericDType::U32 => Tensor::new_integer(IntegerStorage::U32(Vec::new()), shape),
                NumericDType::I64 => Tensor::new_integer(IntegerStorage::I64(Vec::new()), shape),
                NumericDType::U64 => Tensor::new_integer(IntegerStorage::U64(Vec::new()), shape),
            }
            .map_err(format_error)?;
            Ok(Value::Tensor(tensor))
        }
    }
}

fn base_shape(
    format: &MemmapFormat,
    records: usize,
    repeat: Option<usize>,
) -> BuiltinResult<Vec<usize>> {
    let record_values = format
        .shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| format_error("memmapfile: Format shape is too large"))?;
    let total = records
        .checked_mul(record_values)
        .ok_or_else(|| format_error("memmapfile: mapped Data is too large"))?;
    let mut shape = format.shape.clone();
    if repeat.is_some() && records > 1 {
        shape.push(records);
    } else if record_values != total {
        shape = vec![total, 1];
    }
    Ok(shape)
}
