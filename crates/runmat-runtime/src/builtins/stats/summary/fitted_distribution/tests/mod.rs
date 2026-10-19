use super::super::random::random_builtin_registered;
use super::super::{
    cdf::cdf_builtin, fitdist::fitdist_builtin, pdf::pdf_builtin, random::random_builtin,
};
use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

fn vec_tensor(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
}

fn int_vec_tensor(storage: IntegerStorage, len: usize) -> Value {
    Value::Tensor(Tensor::new_integer(storage, vec![len, 1]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    Value::Tensor(tensor)
}

fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    Value::Tensor(tensor)
}

fn all_cdf_integer_storages(value: i8) -> Vec<IntegerStorage> {
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

fn all_fitdist_integer_storages() -> Vec<IntegerStorage> {
    vec![
        IntegerStorage::I8(vec![1, 2, 3]),
        IntegerStorage::I16(vec![1, 2, 3]),
        IntegerStorage::I32(vec![1, 2, 3]),
        IntegerStorage::I64(vec![1, 2, 3]),
        IntegerStorage::U8(vec![1, 2, 3]),
        IntegerStorage::U16(vec![1, 2, 3]),
        IntegerStorage::U32(vec![1, 2, 3]),
        IntegerStorage::U64(vec![1, 2, 3]),
    ]
}

#[test]
fn fitdist_broadcast_indexing_appends_trailing_singletons() {
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    assert_eq!(
        broadcast_tensor_to(PDF_NAME, &tensor, &[2, 1, 3]).unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

fn object(value: Value) -> ObjectInstance {
    match value {
        Value::Object(object) => object,
        other => panic!("expected object, got {other:?}"),
    }
}

mod cdf;
mod fit;
mod generic;
mod ownership;
mod pdf;
mod random;
#[cfg(feature = "wgpu")]
mod wgpu;
