use runmat_builtins::{IntValue, Value};
use runmat_runtime::builtins::common::tensor::{
    complex_tensor_element_len, complex_tensor_value_complex64, is_scalar_tensor, tensor_value_f64,
};

#[derive(Debug, Clone, Copy)]
pub struct ValueToF64Error;

/// Converts an integer only when the resulting double retains its exact value.
/// `end` expressions are subsequently used as indices, so accepting a rounded
/// integer here would select a different element than the source expression.
fn exact_integer_to_f64(value: &IntValue) -> Result<f64, ValueToF64Error> {
    let converted = value.to_f64();
    let exact = match value {
        IntValue::I8(value) => converted as i128 == i128::from(*value),
        IntValue::I16(value) => converted as i128 == i128::from(*value),
        IntValue::I32(value) => converted as i128 == i128::from(*value),
        IntValue::I64(value) => converted as i128 == i128::from(*value),
        IntValue::U8(value) => converted as u128 == u128::from(*value),
        IntValue::U16(value) => converted as u128 == u128::from(*value),
        IntValue::U32(value) => converted as u128 == u128::from(*value),
        IntValue::U64(value) => converted as u128 == u128::from(*value),
    };
    exact.then_some(converted).ok_or(ValueToF64Error)
}

pub fn value_to_f64(v: &Value) -> Result<f64, ValueToF64Error> {
    match v {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => exact_integer_to_f64(i),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if is_scalar_tensor(t) => match t.integer_storage() {
            Some(storage) => exact_integer_to_f64(&storage.value_at(0).ok_or(ValueToF64Error)?),
            None => Ok(tensor_value_f64(t, 0)),
        },
        Value::Complex(re, im) if im.abs() < 1e-12 => Ok(*re),
        Value::ComplexTensor(ct) if complex_tensor_element_len(ct) == 1 => {
            let value = complex_tensor_value_complex64(ct, 0);
            if value.im.abs() < 1e-12 {
                Ok(value.re)
            } else {
                Err(ValueToF64Error)
            }
        }
        _ => Err(ValueToF64Error),
    }
}

#[cfg(test)]
mod tests {
    use super::value_to_f64;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor, Value};

    #[test]
    fn value_to_f64_reads_all_typed_integer_tensor_classes_without_f64_mirrors() {
        macro_rules! assert_typed_scalar {
            ($storage:expr, $expected:expr) => {{
                let mut tensor = Tensor::new_integer($storage, vec![1, 1]).expect("scalar tensor");
                tensor.data.clear();
                assert_eq!(value_to_f64(&Value::Tensor(tensor)).unwrap(), $expected);
            }};
        }

        assert_typed_scalar!(IntegerStorage::I8(vec![-8]), -8.0);
        assert_typed_scalar!(IntegerStorage::I16(vec![-16]), -16.0);
        assert_typed_scalar!(IntegerStorage::I32(vec![-32]), -32.0);
        assert_typed_scalar!(IntegerStorage::I64(vec![-64]), -64.0);
        assert_typed_scalar!(IntegerStorage::U8(vec![8]), 8.0);
        assert_typed_scalar!(IntegerStorage::U16(vec![16]), 16.0);
        assert_typed_scalar!(IntegerStorage::U32(vec![32]), 32.0);
        assert_typed_scalar!(IntegerStorage::U64(vec![64]), 64.0);
    }

    #[test]
    fn value_to_f64_rejects_wide_integer_values_that_would_round_indices() {
        for value in [
            IntValue::I64(i64::MAX),
            IntValue::U64((1_u64 << 53) + 1),
            IntValue::U64(u64::MAX),
        ] {
            assert!(value_to_f64(&Value::Int(value)).is_err());
        }

        let mut tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
                .expect("scalar tensor");
        tensor.data = vec![f64::NAN];
        assert!(value_to_f64(&Value::Tensor(tensor)).is_err());
    }
}
