mod basic;
mod errors;
mod integer;
mod providers;
mod shapes;

use futures::executor::block_on;
use runmat_value::{Tensor, Value};

use super::implementation;
use crate::BuiltinResult;

fn run(value: Value, partitions: Vec<Value>) -> BuiltinResult<Value> {
    block_on(implementation::execute(value, partitions))
}

fn row_vector(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("row vector"))
}

fn column_vector(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).expect("column vector"))
}

#[test]
fn requires_a_partition_vector() {
    let error = run(Value::Num(1.0), Vec::new()).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:mat2cell:InvalidInput"));
}
