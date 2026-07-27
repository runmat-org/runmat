use runmat_builtins::Value;
use runmat_runtime::builtins::common::tensor::{
    complex_tensor_element_len, complex_tensor_value_complex64, is_scalar_tensor, tensor_value_f64,
};

#[derive(Debug, Clone, Copy)]
pub struct ValueToF64Error;

pub fn value_to_f64(v: &Value) -> Result<f64, ValueToF64Error> {
    match v {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if is_scalar_tensor(t) => Ok(tensor_value_f64(t, 0)),
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
    use runmat_builtins::{IntegerStorage, Tensor, Value};

    #[test]
    fn value_to_f64_reads_typed_integer_tensor_storage_exactly() {
        let mut tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![42]), vec![1, 1]).expect("scalar tensor");
        tensor.data.clear();

        assert_eq!(value_to_f64(&Value::Tensor(tensor)).unwrap(), 42.0);
    }
}
