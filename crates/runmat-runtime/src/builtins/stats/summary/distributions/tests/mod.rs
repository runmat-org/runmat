use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

mod provider_preflight;

fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "actual={actual} expected={expected}"
    );
}

fn icdf_scalar(name: &str, p: f64, rest: Vec<Value>) -> f64 {
    match block_on(icdf::icdf_builtin(Value::from(name), Value::Num(p), rest)).unwrap() {
        Value::Num(value) => value,
        other => panic!("expected scalar icdf result, got {other:?}"),
    }
}

fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn all_integer_binocdf_storages(value: i8) -> Vec<IntegerStorage> {
    vec![
        IntegerStorage::I8(vec![value]),
        IntegerStorage::I16(vec![i16::from(value)]),
        IntegerStorage::I32(vec![i32::from(value)]),
        IntegerStorage::I64(vec![i64::from(value)]),
        IntegerStorage::U8(vec![value as u8]),
        IntegerStorage::U16(vec![value as u16]),
        IntegerStorage::U32(vec![value as u32]),
        IntegerStorage::U64(vec![value as u64]),
    ]
}

mod normal;
#[path = "icdf.rs"]
mod inverse_distribution;
mod student_t;
mod binomial;
mod chi_square;
mod weibull;
mod shared;
