use super::*;
use runmat_value::{IntegerComplexStorage, IntegerStorage, NumericStorage};

async fn maxk_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    super::maxk_builtin(value, rest.into()).await
}

async fn mink_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    super::mink_builtin(value, rest.into()).await
}

pub(super) fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

pub(super) fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

pub(super) fn outputs(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list"),
        other => vec![other],
    }
}
mod storage;
mod shape_outputs;
mod controls;
mod integer_classes;
mod ordering;
mod provider;
mod sink;
