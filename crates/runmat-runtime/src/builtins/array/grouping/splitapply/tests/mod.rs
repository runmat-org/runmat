mod callbacks;
mod compatibility;
mod groups;
mod provider;

use futures::executor::block_on;
use runmat_value::{Tensor, Value};

use super::splitapply_builtin;

fn call(function: &str, first: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(splitapply_builtin(
        Value::FunctionHandle(function.into()),
        first,
        rest,
    ))
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn numeric(value: Value) -> Vec<f64> {
    let Value::Tensor(value) = value else {
        panic!("expected tensor")
    };
    value.materialize_f64()
}

fn output_list(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values,
        other => panic!("expected output list, got {other:?}"),
    }
}
