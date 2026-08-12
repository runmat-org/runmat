//! Exact comparisons for MATLAB integer storage.

use std::cmp::Ordering;

use runmat_value::{IntValue, Tensor, Value};

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

pub(crate) fn tensor_elements_equal(left: &Tensor, right: &Tensor, index: usize) -> bool {
    let left = left
        .numeric_value_at(index)
        .expect("left tensor storage index");
    let right = right
        .numeric_value_at(index)
        .expect("right tensor storage index");
    match (left.into_int_value(), right.into_int_value()) {
        (Some(left), Some(right)) => integer_order(&left, &right) == Ordering::Equal,
        (Some(left), None) => {
            integer_f64_order(&left, right.materialize_f64()) == Some(Ordering::Equal)
        }
        (None, Some(right)) => {
            integer_f64_order(&right, left.materialize_f64()).map(Ordering::reverse)
                == Some(Ordering::Equal)
        }
        (None, None) => left.materialize_f64() == right.materialize_f64(),
    }
}

pub(crate) fn tensor_element_equals_scalar(tensor: &Tensor, index: usize, scalar: &Value) -> bool {
    let value = tensor
        .numeric_value_at(index)
        .expect("tensor storage index");
    match (value, scalar) {
        (value, Value::Int(scalar)) => match value.into_int_value() {
            Some(value) => integer_order(&value, scalar) == Ordering::Equal,
            None => {
                integer_f64_order(scalar, value.materialize_f64()).map(Ordering::reverse)
                    == Some(Ordering::Equal)
            }
        },
        (value, Value::Num(scalar)) => match value.into_int_value() {
            Some(value) => integer_f64_order(&value, *scalar) == Some(Ordering::Equal),
            None => value.materialize_f64() == *scalar,
        },
        _ => false,
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
    use runmat_value::IntegerStorage;

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

    #[test]
    fn tensor_comparisons_read_exact_integer_storage() {
        let integer = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            vec![1, 2],
        )
        .expect("integer tensor");
        let rounded = Tensor::new(
            vec![(1_u64 << 53) as f64, 18_446_744_073_709_551_616.0],
            vec![1, 2],
        )
        .expect("double tensor");

        assert!(!tensor_elements_equal(&integer, &rounded, 0));
        assert!(!tensor_elements_equal(&integer, &rounded, 1));
        assert!(!tensor_element_equals_scalar(
            &integer,
            0,
            &Value::Num((1_u64 << 53) as f64)
        ));
        assert!(tensor_element_equals_scalar(
            &integer,
            1,
            &Value::Int(IntValue::U64(u64::MAX))
        ));
    }

    #[test]
    fn floating_tensor_equality_is_exact_and_native_single_aware() {
        let left = Tensor::new(vec![1.0, f64::MIN_POSITIVE], vec![1, 2]).expect("left");
        let right = Tensor::new(
            vec![1.0 + f64::EPSILON, f64::MIN_POSITIVE + f64::from_bits(1)],
            vec![1, 2],
        )
        .expect("right");
        assert!(!tensor_elements_equal(&left, &right, 0));
        assert!(!tensor_elements_equal(&left, &right, 1));

        let single = Tensor::from_f32(vec![0.1], vec![1, 1]).expect("single");
        let double = Tensor::new(vec![f64::from(0.1_f32)], vec![1, 1]).expect("double");
        assert!(tensor_elements_equal(&single, &double, 0));
        assert!(!tensor_element_equals_scalar(
            &single,
            0,
            &Value::Num(0.1_f64)
        ));
    }
}
