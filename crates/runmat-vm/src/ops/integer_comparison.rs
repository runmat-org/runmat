//! Exact comparisons for MATLAB integer storage.

use std::cmp::Ordering;

use runmat_value::{IntValue, Value};

/// Returns an exact ordering whenever either scalar needs integer-aware
/// comparison. `None` leaves ordinary floating-point comparison untouched.
pub(crate) fn scalar_order(left: &Value, right: &Value) -> Option<Ordering> {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => Some(integer_order(left, right)),
        (Value::Int(left), Value::Num(right)) => integer_f64_order(left, *right),
        (Value::Num(left), Value::Int(right)) => {
            integer_f64_order(right, *left).map(Ordering::reverse)
        }
        _ => None,
    }
}

fn integer_order(left: &IntValue, right: &IntValue) -> Ordering {
    integer_as_i128(left).cmp(&integer_as_i128(right))
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

fn integer_f64_order(integer: &IntValue, float: f64) -> Option<Ordering> {
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

    let truncated = float as i128;
    let ordering = integer_as_i128(integer).cmp(&truncated);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_comparisons_preserve_64_bit_integer_precision() {
        assert_eq!(
            scalar_order(
                &Value::Int(IntValue::U64((1_u64 << 53) + 1)),
                &Value::Num((1_u64 << 53) as f64),
            ),
            Some(Ordering::Greater)
        );
        assert_eq!(
            scalar_order(
                &Value::Int(IntValue::U64(u64::MAX)),
                &Value::Num(18_446_744_073_709_551_616.0),
            ),
            Some(Ordering::Less)
        );
        assert_eq!(
            scalar_order(
                &Value::Int(IntValue::I64(i64::MIN)),
                &Value::Num(-9_223_372_036_854_775_808.0),
            ),
            Some(Ordering::Equal)
        );
    }
}
