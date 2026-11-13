use runmat_value::{NumericScalar, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExactLogicalError {
    NotScalarNumericOrLogical,
    NotZeroOrOne,
}

pub(crate) fn decode(value: &Value) -> Result<bool, ExactLogicalError> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Int(value) if value.is_zero() => Ok(false),
        Value::Int(value) if value.try_to_u64() == Some(1) => Ok(true),
        Value::Int(_) => Err(ExactLogicalError::NotZeroOrOne),
        Value::Num(value) => decode_numeric(NumericScalar::F64(*value)),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .numeric_value_at(0)
            .ok_or(ExactLogicalError::NotScalarNumericOrLogical)
            .and_then(decode_numeric),
        Value::LogicalArray(_) | Value::Tensor(_) => {
            Err(ExactLogicalError::NotScalarNumericOrLogical)
        }
        _ => Err(ExactLogicalError::NotScalarNumericOrLogical),
    }
}

fn decode_numeric(value: NumericScalar) -> Result<bool, ExactLogicalError> {
    match value {
        NumericScalar::F64(0.0) | NumericScalar::F32(0.0) => Ok(false),
        NumericScalar::F64(1.0) | NumericScalar::F32(1.0) => Ok(true),
        value if value.into_int_value().is_some_and(|value| value.is_zero()) => Ok(false),
        value
            if value
                .into_int_value()
                .is_some_and(|value| value.try_to_u64() == Some(1)) =>
        {
            Ok(true)
        }
        _ => Err(ExactLogicalError::NotZeroOrOne),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn accepts_only_exact_scalar_zero_and_one() {
        for value in [
            Value::Bool(true),
            Value::Int(IntValue::I8(1)),
            Value::Int(IntValue::U64(1)),
            Value::Num(1.0),
        ] {
            assert_eq!(decode(&value), Ok(true));
        }
        assert_eq!(decode(&Value::Int(IntValue::I64(0))), Ok(false));
        assert_eq!(
            decode(&Value::Int(IntValue::U64(u64::MAX))),
            Err(ExactLogicalError::NotZeroOrOne)
        );
        let array = Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap();
        assert_eq!(
            decode(&Value::Tensor(array)),
            Err(ExactLogicalError::NotScalarNumericOrLogical)
        );
    }
}
