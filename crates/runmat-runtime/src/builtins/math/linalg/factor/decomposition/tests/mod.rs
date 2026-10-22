use super::*;
use crate::builtins::common::test_support;
use crate::OBJECT_INDEX_MEMBER;
use futures::executor::block_on;
use runmat_value::{IntegerComplexStorage, IntegerStorage, NumericScalar};

async fn decomposition_subsref(
    obj: Value,
    kind: String,
    payload: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    validate_output_count()?;
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::decomposition_subsref(obj, subscript).await
}

pub(super) fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data.to_vec(), vec![rows, cols]).unwrap())
}

pub(super) fn complex_tensor(data: &[(f64, f64)], rows: usize, cols: usize) -> Value {
    Value::ComplexTensor(ComplexTensor::new(data.to_vec(), vec![rows, cols]).unwrap())
}

pub(super) fn typed_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
}

pub(super) fn mirrorless_complex_int_tensor(
    real: IntegerStorage,
    imag: IntegerStorage,
    shape: Vec<usize>,
) -> Value {
    let tensor =
        ComplexTensor::new_integer(IntegerComplexStorage::new(real, imag).unwrap(), shape).unwrap();
    Value::ComplexTensor(tensor)
}

pub(super) fn all_integer_coefficient_storages() -> [IntegerStorage; 8] {
    [
        IntegerStorage::I8(vec![2, 0, 0, 4]),
        IntegerStorage::I16(vec![2, 0, 0, 4]),
        IntegerStorage::I32(vec![2, 0, 0, 4]),
        IntegerStorage::I64(vec![2, 0, 0, 4]),
        IntegerStorage::U8(vec![2, 0, 0, 4]),
        IntegerStorage::U16(vec![2, 0, 0, 4]),
        IntegerStorage::U32(vec![2, 0, 0, 4]),
        IntegerStorage::U64(vec![2, 0, 0, 4]),
    ]
}

pub(super) fn call_constructor(args: Vec<Value>) -> BuiltinResult<Value> {
    block_on(decomposition_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

pub(super) fn call_mldivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(decomposition_mldivide(lhs, rhs))
}

pub(super) fn call_mrdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(decomposition_mrdivide(lhs, rhs))
}

pub(super) fn call_mtimes(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(decomposition_mtimes(lhs, rhs))
}

pub(super) fn call_rdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(decomposition_rdivide(lhs, rhs))
}

pub(super) fn call_ctranspose(value: Value) -> BuiltinResult<Value> {
    block_on(decomposition_ctranspose(value))
}

pub(super) fn approx_vec(value: Value) -> Vec<f64> {
    match value {
        Value::Num(n) => vec![n],
        Value::Tensor(t) => t.materialize_f64(),
        Value::GpuTensor(handle) => test_support::gather(Value::GpuTensor(handle))
            .expect("gather decomposition solve result")
            .materialize_f64(),
        other => panic!("expected real tensor, got {other:?}"),
    }
}

pub(super) fn approx_vec_rooted(value: crate::execution::RootedValue) -> Vec<f64> {
    value.with_owned(approx_vec)
}

mod compatibility;
mod construction;
mod cutover;
mod integer_storage;
mod properties;
#[path = "scale_range.rs"]
mod scale_range;
mod solves;
