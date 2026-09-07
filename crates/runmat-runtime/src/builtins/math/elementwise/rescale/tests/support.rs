use runmat_value::{IntegerStorage, NumericDType, Tensor, Value};

pub(super) fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).expect("tensor"))
}

pub(super) fn single(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_with_dtype(data, shape, NumericDType::F32).expect("single tensor"))
}

pub(super) fn integer(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
}

pub(super) fn values(value: Value) -> (Vec<f64>, Vec<usize>, NumericDType) {
    match value {
        Value::Num(number) => (vec![number], vec![1, 1], NumericDType::F64),
        Value::Tensor(tensor) => {
            let dtype = tensor.numeric_dtype();
            (tensor.materialize_f64(), tensor.shape, dtype)
        }
        other => panic!("expected numeric output, got {other:?}"),
    }
}

pub(super) fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "index {index}: expected NaN, got {actual}");
        } else {
            assert!(
                (actual - expected).abs() < 1e-12,
                "index {index}: expected {expected}, got {actual}"
            );
        }
    }
}
