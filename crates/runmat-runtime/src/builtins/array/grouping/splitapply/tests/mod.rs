mod callbacks;
mod compatibility;
mod groups;
mod provider;

use futures::executor::block_on;
use runmat_value::{Tensor, Value};

use super::splitapply_builtin;

fn call(
    function: &str,
    first: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    block_on(splitapply_builtin(
        Value::FunctionHandle(function.into()),
        first,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn numeric(value: crate::execution::RootedValue) -> Vec<f64> {
    numeric_value(value.value())
}

fn numeric_value(value: &Value) -> Vec<f64> {
    let Value::Tensor(value) = value else {
        panic!("expected tensor")
    };
    value.materialize_f64()
}

fn output_list(value: &crate::execution::RootedValue) -> &[Value] {
    match value.value() {
        Value::OutputList(values) => values,
        other => panic!("expected output list, got {other:?}"),
    }
}
