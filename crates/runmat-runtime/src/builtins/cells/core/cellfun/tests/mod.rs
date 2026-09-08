use super::cellfun_builtin;
use futures::executor::block_on;
use runmat_value::{CellArray, Tensor, Value};

fn call(function: Value, arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(cellfun_builtin(function, arguments))
}

fn cell(values: Vec<Value>, shape: &[usize]) -> Value {
    Value::Cell(CellArray::new_with_shape(values, shape.to_vec()).expect("valid cell fixture"))
}

fn tensor_values(value: Value) -> Vec<f64> {
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor")
    };
    tensor.materialize_f64()
}

fn tensor(values: Vec<f64>, shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new(values, shape.to_vec()).expect("valid tensor fixture"))
}

mod callback;
mod invocation;
mod options;
mod provider;
mod storage;
