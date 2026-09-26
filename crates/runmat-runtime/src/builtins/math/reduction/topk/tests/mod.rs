use super::*;
use runmat_value::{IntegerComplexStorage, IntegerStorage, NumericStorage};

pub(super) fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

pub(super) fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

pub(super) fn outputs(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values,
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
