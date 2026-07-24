//! Exact relational comparisons for native MATLAB integer storage.

use std::cmp::Ordering;

use runmat_builtins::{IntValue, IntegerStorage, LogicalArray, Value};

use crate::builtins::common::broadcast::BroadcastPlan;

#[derive(Clone, Copy)]
pub(crate) enum IntegerComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug)]
pub(crate) enum IntegerComparisonError {
    SizeMismatch,
    Internal,
}

/// Performs a comparison when native integer storage is compared against other
/// native integer storage or a scalar numeric value. Other numeric domains
/// deliberately stay on the established builtin paths.
pub(crate) fn try_integer_comparison(
    lhs: &Value,
    rhs: &Value,
    operation: IntegerComparisonOp,
) -> Result<Option<Value>, IntegerComparisonError> {
    let lhs_integer = integer_operand(lhs);
    let rhs_integer = integer_operand(rhs);
    let result = match (lhs_integer, rhs_integer) {
        (None, None) => return Ok(None),
        (Some(lhs), Some(rhs)) => compare_integer_operands(&lhs, &rhs, operation)?,
        (Some(lhs), None) => {
            let Some(rhs) = numeric_scalar(rhs) else {
                return Ok(None);
            };
            compare_integer_scalar(&lhs, rhs, true, operation)?
        }
        (None, Some(rhs)) => {
            let Some(lhs) = numeric_scalar(lhs) else {
                return Ok(None);
            };
            compare_integer_scalar(&rhs, lhs, false, operation)?
        }
    };
    Ok(Some(result))
}

fn compare_integer_operands(
    lhs: &IntegerOperand<'_>,
    rhs: &IntegerOperand<'_>,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|_| IntegerComparisonError::SizeMismatch)?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, lhs_index, rhs_index) in plan.iter() {
        let ordering = compare(lhs.value_at(lhs_index), rhs.value_at(rhs_index));
        data.push(matches_relation(ordering, operation) as u8);
    }
    logical_result(data, plan.output_shape().to_vec())
}

fn compare_integer_scalar(
    integer: &IntegerOperand<'_>,
    scalar: f64,
    integer_is_left: bool,
    operation: IntegerComparisonOp,
) -> Result<Value, IntegerComparisonError> {
    let mut data = Vec::with_capacity(integer.shape.iter().product());
    for index in 0..integer.shape.iter().product() {
        let ordering = integer_f64_order(integer.value_at(index), scalar);
        let ordering = if integer_is_left {
            ordering
        } else {
            ordering.map(Ordering::reverse)
        };
        data.push(matches_optional_relation(ordering, operation) as u8);
    }
    logical_result(data, integer.shape.clone())
}

fn logical_result(data: Vec<u8>, shape: Vec<usize>) -> Result<Value, IntegerComparisonError> {
    if data.len() == 1 {
        return Ok(Value::Bool(data[0] != 0));
    }
    Ok(Value::LogicalArray(
        LogicalArray::new(data, shape).map_err(|_| IntegerComparisonError::Internal)?,
    ))
}

struct IntegerOperand<'a> {
    storage: IntegerStorageRef<'a>,
    shape: Vec<usize>,
}

impl IntegerOperand<'_> {
    fn value_at(&self, index: usize) -> IntValue {
        self.storage.value_at(index)
    }
}

enum IntegerStorageRef<'a> {
    Scalar(&'a IntValue),
    Array(&'a IntegerStorage),
}

impl IntegerStorageRef<'_> {
    fn value_at(&self, index: usize) -> IntValue {
        match self {
            Self::Scalar(value) => (*value).clone(),
            Self::Array(storage) => storage_value(storage, index),
        }
    }
}

fn integer_operand(value: &Value) -> Option<IntegerOperand<'_>> {
    match value {
        Value::Int(value) => Some(IntegerOperand {
            storage: IntegerStorageRef::Scalar(value),
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => tensor.integer_storage().map(|storage| IntegerOperand {
            storage: IntegerStorageRef::Array(storage),
            shape: tensor.shape.clone(),
        }),
        _ => None,
    }
}

fn compare(lhs: IntValue, rhs: IntValue) -> Ordering {
    match (signed_value(&lhs), signed_value(&rhs)) {
        (Some(lhs), Some(rhs)) => lhs.cmp(&rhs),
        (None, None) => unsigned_value(&lhs).cmp(&unsigned_value(&rhs)),
        (Some(lhs), None) => {
            if lhs < 0 {
                Ordering::Less
            } else {
                (lhs as u64).cmp(&unsigned_value(&rhs))
            }
        }
        (None, Some(rhs)) => {
            if rhs < 0 {
                Ordering::Greater
            } else {
                unsigned_value(&lhs).cmp(&(rhs as u64))
            }
        }
    }
}

fn integer_f64_order(integer: IntValue, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float == f64::INFINITY {
        return Some(Ordering::Less);
    }
    if float == f64::NEG_INFINITY {
        return Some(Ordering::Greater);
    }

    const MIN_I64: f64 = -9_223_372_036_854_775_808.0;
    const U64_EXCLUSIVE_UPPER: f64 = 18_446_744_073_709_551_616.0;
    if float < MIN_I64 {
        return Some(Ordering::Greater);
    }
    if float >= U64_EXCLUSIVE_UPPER {
        return Some(Ordering::Less);
    }

    let integer = integer_as_i128(&integer);
    let truncated = float as i128;
    let ordering = integer.cmp(&truncated);
    if float.fract() == 0.0 {
        return Some(ordering);
    }
    Some(if float.is_sign_positive() {
        if ordering == Ordering::Greater {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    } else if ordering == Ordering::Less {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}

fn integer_as_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn numeric_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.integer_storage().is_none() && tensor.data.len() == 1 => {
            tensor.data.first().copied()
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Some(if array.data[0] == 0 { 0.0 } else { 1.0 })
        }
        _ => None,
    }
}

fn signed_value(value: &IntValue) -> Option<i64> {
    match value {
        IntValue::I8(value) => Some(*value as i64),
        IntValue::I16(value) => Some(*value as i64),
        IntValue::I32(value) => Some(*value as i64),
        IntValue::I64(value) => Some(*value),
        IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => None,
    }
}

fn unsigned_value(value: &IntValue) -> u64 {
    match value {
        IntValue::U8(value) => *value as u64,
        IntValue::U16(value) => *value as u64,
        IntValue::U32(value) => *value as u64,
        IntValue::U64(value) => *value,
        IntValue::I8(_) | IntValue::I16(_) | IntValue::I32(_) | IntValue::I64(_) => {
            unreachable!("unsigned conversion is only used for unsigned integer values")
        }
    }
}

fn storage_value(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
    }
}

fn matches_relation(ordering: Ordering, operation: IntegerComparisonOp) -> bool {
    match operation {
        IntegerComparisonOp::Eq => ordering == Ordering::Equal,
        IntegerComparisonOp::Ne => ordering != Ordering::Equal,
        IntegerComparisonOp::Lt => ordering == Ordering::Less,
        IntegerComparisonOp::Le => ordering != Ordering::Greater,
        IntegerComparisonOp::Gt => ordering == Ordering::Greater,
        IntegerComparisonOp::Ge => ordering != Ordering::Less,
    }
}

fn matches_optional_relation(ordering: Option<Ordering>, operation: IntegerComparisonOp) -> bool {
    match ordering {
        Some(ordering) => matches_relation(ordering, operation),
        None => matches!(operation, IntegerComparisonOp::Ne),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(runmat_builtins::Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    #[test]
    fn compares_signed_unsigned_and_uint64_exactly() {
        let lhs = Value::Int(IntValue::U64(u64::MAX));
        let rhs = Value::Int(IntValue::I64(i64::MAX));
        assert_eq!(
            try_integer_comparison(&lhs, &rhs, IntegerComparisonOp::Gt).expect("comparison"),
            Some(Value::Bool(true))
        );
        assert_eq!(
            try_integer_comparison(
                &Value::Int(IntValue::I8(-1)),
                &Value::Int(IntValue::U8(0)),
                IntegerComparisonOp::Lt,
            )
            .expect("comparison"),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn broadcasts_exact_integer_arrays_for_all_relations() {
        let lhs = array(IntegerStorage::U64(vec![0, u64::MAX]), vec![2, 1]);
        let rhs = array(IntegerStorage::I64(vec![0, 1, i64::MAX]), vec![1, 3]);
        let result = try_integer_comparison(&lhs, &rhs, IntegerComparisonOp::Ge)
            .expect("comparison")
            .expect("integer path");
        assert_eq!(
            result,
            Value::LogicalArray(
                LogicalArray::new(vec![1, 1, 0, 1, 0, 1], vec![2, 3]).expect("logical result")
            )
        );
    }

    #[test]
    fn compares_integer_storage_to_scalar_double_without_64_bit_loss() {
        let exact = Value::Int(IntValue::U64((1_u64 << 53) + 1));
        let rounded = Value::Num((1_u64 << 53) as f64);
        assert_eq!(
            try_integer_comparison(&exact, &rounded, IntegerComparisonOp::Eq).expect("comparison"),
            Some(Value::Bool(false))
        );
        assert_eq!(
            try_integer_comparison(&exact, &rounded, IntegerComparisonOp::Gt).expect("comparison"),
            Some(Value::Bool(true))
        );

        let tensor = array(IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]), vec![1, 2]);
        assert_eq!(
            try_integer_comparison(&tensor, &rounded, IntegerComparisonOp::Ne).expect("comparison"),
            Some(Value::LogicalArray(
                LogicalArray::new(vec![1, 1], vec![1, 2]).expect("logical result")
            ))
        );
    }
}
