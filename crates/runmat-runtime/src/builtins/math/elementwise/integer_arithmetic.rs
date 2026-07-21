//! Exact MATLAB integer arithmetic shared by elementwise binary builtins.

use runmat_builtins::{IntValue, IntegerStorage, Tensor, Value};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::math::elementwise::integer_cast::IntegerTarget;

#[derive(Clone, Copy)]
pub(crate) enum IntegerBinaryOp {
    Add,
    Subtract,
    Multiply,
}

/// Applies a MATLAB integer binary operation when either operand is integer
/// storage. `Ok(None)` means neither operand is an integer and the caller
/// should retain its normal floating/complex path.
pub(crate) fn try_integer_binary(
    lhs: &Value,
    rhs: &Value,
    operation: IntegerBinaryOp,
    builtin: &str,
) -> Result<Option<Value>, String> {
    let left = integer_operand(lhs);
    let right = integer_operand(rhs);
    if left.is_none() && right.is_none() {
        return Ok(None);
    }

    let (integer, other, integer_is_left) = match (left, right) {
        (Some(left), Some(right)) => {
            if left.target != right.target {
                return Err(format!(
                    "{builtin}: integer operands must have the same integer class"
                ));
            }
            return apply_exact_integer_pair(&left, &right, operation)
                .map(Some)
                .map_err(|error| format!("{builtin}: {error}"));
        }
        (Some(integer), None) => (integer, rhs, true),
        (None, Some(integer)) => (integer, lhs, false),
        (None, None) => unreachable!("integer presence checked above"),
    };

    let Some(scalar) = real_scalar(other) else {
        return Err(format!(
            "{builtin}: integer arrays can only be combined with scalar double or logical values"
        ));
    };
    apply_integer_scalar(&integer, scalar, integer_is_left, operation)
        .map(Some)
        .map_err(|error| format!("{builtin}: {error}"))
}

struct IntegerOperand<'a> {
    storage: IntegerStorageRef<'a>,
    shape: Vec<usize>,
    target: IntegerTarget,
}

enum IntegerStorageRef<'a> {
    Scalar(&'a IntValue),
    Array(&'a IntegerStorage),
}

impl IntegerStorageRef<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Array(storage) => storage.len(),
        }
    }

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
            target: IntegerTarget::from_int_value(value),
        }),
        Value::Tensor(tensor) => tensor.integer_storage().map(|storage| IntegerOperand {
            storage: IntegerStorageRef::Array(storage),
            shape: tensor.shape.clone(),
            target: IntegerTarget::from_storage(storage),
        }),
        _ => None,
    }
}

fn real_scalar(value: &Value) -> Option<f64> {
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

fn apply_exact_integer_pair(
    lhs: &IntegerOperand<'_>,
    rhs: &IntegerOperand<'_>,
    operation: IntegerBinaryOp,
) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)?;
    let mut values = Vec::with_capacity(plan.len());
    for (_, left_index, right_index) in plan.iter() {
        values.push(apply_exact(
            lhs.storage.value_at(left_index),
            rhs.storage.value_at(right_index),
            operation,
        ));
    }
    integer_values_into_value(lhs.target, values, plan.output_shape().to_vec())
}

fn apply_integer_scalar(
    integer: &IntegerOperand<'_>,
    scalar: f64,
    integer_is_left: bool,
    operation: IntegerBinaryOp,
) -> Result<Value, String> {
    let mut values = Vec::with_capacity(integer.storage.len());
    for index in 0..integer.storage.len() {
        let integer_value = integer.storage.value_at(index).to_f64();
        let result = if integer_is_left {
            apply_float(integer_value, scalar, operation)
        } else {
            apply_float(scalar, integer_value, operation)
        };
        values.push(integer.target.cast_scalar(result));
    }
    integer_values_into_value(integer.target, values, integer.shape.clone())
}

fn apply_float(lhs: f64, rhs: f64, operation: IntegerBinaryOp) -> f64 {
    match operation {
        IntegerBinaryOp::Add => lhs + rhs,
        IntegerBinaryOp::Subtract => lhs - rhs,
        IntegerBinaryOp::Multiply => lhs * rhs,
    }
}

fn apply_exact(lhs: IntValue, rhs: IntValue, operation: IntegerBinaryOp) -> IntValue {
    macro_rules! apply {
        ($lhs:expr, $rhs:expr, $variant:ident) => {
            IntValue::$variant(match operation {
                IntegerBinaryOp::Add => $lhs.saturating_add($rhs),
                IntegerBinaryOp::Subtract => $lhs.saturating_sub($rhs),
                IntegerBinaryOp::Multiply => $lhs.saturating_mul($rhs),
            })
        };
    }
    match (lhs, rhs) {
        (IntValue::I8(lhs), IntValue::I8(rhs)) => apply!(lhs, rhs, I8),
        (IntValue::I16(lhs), IntValue::I16(rhs)) => apply!(lhs, rhs, I16),
        (IntValue::I32(lhs), IntValue::I32(rhs)) => apply!(lhs, rhs, I32),
        (IntValue::I64(lhs), IntValue::I64(rhs)) => apply!(lhs, rhs, I64),
        (IntValue::U8(lhs), IntValue::U8(rhs)) => apply!(lhs, rhs, U8),
        (IntValue::U16(lhs), IntValue::U16(rhs)) => apply!(lhs, rhs, U16),
        (IntValue::U32(lhs), IntValue::U32(rhs)) => apply!(lhs, rhs, U32),
        (IntValue::U64(lhs), IntValue::U64(rhs)) => apply!(lhs, rhs, U64),
        _ => unreachable!("integer class compatibility was checked before applying"),
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

fn integer_values_into_value(
    target: IntegerTarget,
    values: Vec<IntValue>,
    shape: Vec<usize>,
) -> Result<Value, String> {
    if values.len() == 1 {
        return Ok(Value::Int(values.into_iter().next().expect("one value")));
    }
    Ok(Value::Tensor(Tensor::new_integer(
        target.storage(values),
        shape,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    #[test]
    fn same_class_integer_arrays_saturate_and_keep_exact_storage() {
        let result = try_integer_binary(
            &integer(IntegerStorage::U64(vec![u64::MAX, 3]), vec![1, 2]),
            &integer(IntegerStorage::U64(vec![1, 4]), vec![1, 2]),
            IntegerBinaryOp::Add,
            "plus",
        )
        .expect("integer operation")
        .expect("integer path");
        assert_eq!(
            result,
            integer(IntegerStorage::U64(vec![u64::MAX, 7]), vec![1, 2])
        );
    }

    #[test]
    fn same_class_integer_arrays_support_column_major_broadcasting() {
        let result = try_integer_binary(
            &integer(IntegerStorage::I16(vec![100, -100]), vec![2, 1]),
            &integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]),
            IntegerBinaryOp::Subtract,
            "minus",
        )
        .expect("integer operation")
        .expect("integer path");
        assert_eq!(
            result,
            integer(
                IntegerStorage::I16(vec![99, -101, 98, -102, 97, -103]),
                vec![2, 3]
            )
        );
    }

    #[test]
    fn scalar_double_is_rounded_saturated_and_keeps_integer_class() {
        let result = try_integer_binary(
            &integer(IntegerStorage::I8(vec![100, -100]), vec![1, 2]),
            &Value::Num(2.6),
            IntegerBinaryOp::Multiply,
            "times",
        )
        .expect("integer operation")
        .expect("integer path");
        assert_eq!(
            result,
            integer(IntegerStorage::I8(vec![127, -128]), vec![1, 2])
        );
    }

    #[test]
    fn mixed_classes_and_nonscalar_double_are_rejected() {
        let mixed = try_integer_binary(
            &Value::Int(IntValue::I8(1)),
            &Value::Int(IntValue::I16(1)),
            IntegerBinaryOp::Add,
            "plus",
        )
        .expect_err("mixed classes must reject");
        assert!(mixed.contains("same integer class"));

        let nonscalar = try_integer_binary(
            &integer(IntegerStorage::I8(vec![1, 2]), vec![1, 2]),
            &Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("double tensor")),
            IntegerBinaryOp::Add,
            "plus",
        )
        .expect_err("nonscalar double must reject");
        assert!(nonscalar.contains("scalar double"));
    }
}
