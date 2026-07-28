use crate::indexing::plan::IndexPlan;
use crate::interpreter::errors::mex;
use runmat_builtins::{IntValue, IntegerStorage};
use runmat_runtime::RuntimeError;

#[derive(Clone)]
pub(crate) enum IntegerAssignmentValue {
    Exact(IntValue),
    Float(f64),
}

pub(crate) fn values(storage: &IntegerStorage) -> Vec<IntValue> {
    match storage {
        IntegerStorage::I8(values) => values.iter().copied().map(IntValue::I8).collect(),
        IntegerStorage::I16(values) => values.iter().copied().map(IntValue::I16).collect(),
        IntegerStorage::I32(values) => values.iter().copied().map(IntValue::I32).collect(),
        IntegerStorage::I64(values) => values.iter().copied().map(IntValue::I64).collect(),
        IntegerStorage::U8(values) => values.iter().copied().map(IntValue::U8).collect(),
        IntegerStorage::U16(values) => values.iter().copied().map(IntValue::U16).collect(),
        IntegerStorage::U32(values) => values.iter().copied().map(IntValue::U32).collect(),
        IntegerStorage::U64(values) => values.iter().copied().map(IntValue::U64).collect(),
    }
}

fn cast_signed(value: &IntegerAssignmentValue, min: i64, max: i64) -> i64 {
    match value {
        IntegerAssignmentValue::Exact(value) => value.to_i64().clamp(min, max),
        IntegerAssignmentValue::Float(value) if value.is_nan() => 0,
        IntegerAssignmentValue::Float(value) if value.is_infinite() => {
            if value.is_sign_negative() {
                min
            } else {
                max
            }
        }
        IntegerAssignmentValue::Float(value) => value.round().clamp(min as f64, max as f64) as i64,
    }
}

fn cast_unsigned(value: &IntegerAssignmentValue, max: u64) -> u64 {
    match value {
        IntegerAssignmentValue::Exact(IntValue::U64(value)) => (*value).min(max),
        IntegerAssignmentValue::Exact(value) => (value.to_i64().max(0) as u64).min(max),
        IntegerAssignmentValue::Float(value) if value.is_nan() || value.is_sign_negative() => 0,
        IntegerAssignmentValue::Float(value) if value.is_infinite() => max,
        IntegerAssignmentValue::Float(value) => value.round().clamp(0.0, max as f64) as u64,
    }
}

pub(crate) fn scatter(
    storage: &mut IntegerStorage,
    plan: &IndexPlan,
    rhs_values: &[IntegerAssignmentValue],
) -> Result<(), RuntimeError> {
    if rhs_values.len() != plan.indices.len() {
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }
    macro_rules! write_values {
        ($values:expr, $convert:expr) => {{
            for (&dst, value) in plan.indices.iter().zip(rhs_values.iter()) {
                $values[dst as usize] = $convert(value);
            }
        }};
    }
    match storage {
        IntegerStorage::I8(values) => write_values!(values, |value| cast_signed(
            value,
            i8::MIN as i64,
            i8::MAX as i64
        ) as i8),
        IntegerStorage::I16(values) => write_values!(values, |value| cast_signed(
            value,
            i16::MIN as i64,
            i16::MAX as i64
        ) as i16),
        IntegerStorage::I32(values) => write_values!(values, |value| cast_signed(
            value,
            i32::MIN as i64,
            i32::MAX as i64
        ) as i32),
        IntegerStorage::I64(values) => {
            write_values!(values, |value| cast_signed(value, i64::MIN, i64::MAX))
        }
        IntegerStorage::U8(values) => {
            write_values!(values, |value| cast_unsigned(value, u8::MAX as u64) as u8)
        }
        IntegerStorage::U16(values) => {
            write_values!(values, |value| cast_unsigned(value, u16::MAX as u64) as u16)
        }
        IntegerStorage::U32(values) => {
            write_values!(values, |value| cast_unsigned(value, u32::MAX as u64) as u32)
        }
        IntegerStorage::U64(values) => {
            write_values!(values, |value| cast_unsigned(value, u64::MAX))
        }
    }
    Ok(())
}
