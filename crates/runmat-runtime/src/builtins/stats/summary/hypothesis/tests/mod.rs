use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

fn output_list(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list"),
        other => panic!("expected output list, got {other:?}"),
    }
}

fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual}"
    );
}

fn poisoned_integer_scalar(storage: IntegerStorage) -> Value {
    let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
    Value::Tensor(tensor)
}

fn poisoned_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

#[path = "ttest2.rs"]
mod ttest2;
#[path = "kstest.rs"]
mod kstest;
