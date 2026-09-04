mod edges;
mod labels;
mod provider;

use futures::executor::block_on;
use runmat_value::Value;

use super::discretize_builtin;

fn call(x: Value, edges: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(discretize_builtin(x, edges, rest))
}

fn tensor(value: Value) -> runmat_value::Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}
