//! Typed equality for the supported `IsEqualTo` constraint form.
//!
//! This does not implement optional comparators or tolerances. It keeps
//! numeric class and exact integer storage intact for the default comparison.

use runmat_value::{IntValue, NumericDType, NumericScalar, Tensor, Value};

enum RealNumeric<'a> {
    Double(f64),
    Integer(&'a IntValue),
    Array(&'a Tensor),
}

impl RealNumeric<'_> {
    fn class(&self) -> NumericDType {
        match self {
            Self::Double(_) => NumericDType::F64,
            Self::Integer(value) => value.numeric_dtype(),
            Self::Array(value) => value.numeric_dtype(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Double(_) | Self::Integer(_) => &[1, 1],
            Self::Array(value) => value.shape(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Double(_) | Self::Integer(_) => 1,
            Self::Array(value) => value.len(),
        }
    }

    fn value_at(&self, index: usize) -> Option<NumericScalar> {
        match self {
            Self::Double(value) if index == 0 => Some(NumericScalar::F64(*value)),
            Self::Integer(value) if index == 0 => Some(NumericScalar::from((**value).clone())),
            Self::Array(value) => value.numeric_value_at(index),
            _ => None,
        }
    }
}

fn real_numeric(value: &Value) -> Option<RealNumeric<'_>> {
    match value {
        Value::Num(value) => Some(RealNumeric::Double(*value)),
        Value::Int(value) => Some(RealNumeric::Integer(value)),
        Value::Tensor(value) => Some(RealNumeric::Array(value)),
        _ => None,
    }
}

fn same_scalar(actual: NumericScalar, expected: NumericScalar) -> bool {
    match (actual, expected) {
        (NumericScalar::F64(actual), NumericScalar::F64(expected)) => {
            actual == expected || (actual.is_nan() && expected.is_nan())
        }
        (NumericScalar::F32(actual), NumericScalar::F32(expected)) => {
            actual == expected || (actual.is_nan() && expected.is_nan())
        }
        (actual, expected) => actual == expected,
    }
}

pub(crate) fn constraint_values_equal(actual: &Value, expected: &Value) -> bool {
    if let (Some(actual), Some(expected)) = (real_numeric(actual), real_numeric(expected)) {
        return actual.class() == expected.class()
            && actual.shape() == expected.shape()
            && actual.len() == expected.len()
            && (0..actual.len()).all(|index| {
                actual
                    .value_at(index)
                    .zip(expected.value_at(index))
                    .is_some_and(|(actual, expected)| same_scalar(actual, expected))
            });
    }
    // Non-real values retain their existing typed Value equality. The wider
    // comparator, tolerance, and user-class contracts remain explicit gaps.
    actual == expected
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntegerStorage;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn numeric_constraint_equality_uses_exact_low_rank_shape() {
        for shape in [Vec::new(), vec![2]] {
            let count = shape.iter().product::<usize>();
            let input = Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX; count]), shape.clone())
                    .unwrap(),
            );
            let row = Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX; count]), vec![1, count])
                    .unwrap(),
            );
            assert!(constraint_values_equal(&input, &input));
            assert!(!constraint_values_equal(&input, &row));
        }
    }

    #[test]
    fn wide_integers_are_compared_without_floating_conversion() {
        let first = Value::Int(IntValue::U64(9_007_199_254_740_992));
        let next = Value::Int(IntValue::U64(9_007_199_254_740_993));
        assert!(!constraint_values_equal(&first, &next));
        assert!(constraint_values_equal(&next, &next));
        let tensor = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap(),
        );
        assert!(constraint_values_equal(&next, &tensor));
        assert!(!constraint_values_equal(&first, &tensor));
    }

    #[test]
    fn numerical_classes_and_shapes_must_match() {
        let single = Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap());
        let double = Value::Num(1.0);
        assert!(!constraint_values_equal(&single, &double));
        assert!(!constraint_values_equal(
            &Value::Int(IntValue::U64(1)),
            &double
        ));
        let row = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let column = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        assert!(!constraint_values_equal(&row, &column));
    }

    #[test]
    fn equal_class_nans_match_in_real_arrays() {
        let scalar = Value::Num(f64::NAN);
        let tensor = Value::Tensor(Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap());
        assert!(constraint_values_equal(&scalar, &tensor));
    }
}
