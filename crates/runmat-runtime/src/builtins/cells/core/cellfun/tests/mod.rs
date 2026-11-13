use super::cellfun_builtin;
use futures::executor::block_on;
use runmat_value::{CellArray, Tensor, Value};

fn call(
    function: Value,
    arguments: Vec<Value>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let arguments = crate::call::BuiltinArguments::from_borrowed(&arguments, None)?;
    block_on(cellfun_builtin(None, function, arguments))
}

fn rooted_sequence(
    sequence: runmat_value::ValueSequence,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    crate::execution::RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())
}

fn cell(values: Vec<Value>, shape: &[usize]) -> Value {
    Value::Cell(CellArray::new_with_shape(values, shape.to_vec()).expect("valid cell fixture"))
}

fn tensor_values(value: &Value) -> Vec<f64> {
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
mod linear_order;
mod options;
mod provider;
mod storage;
