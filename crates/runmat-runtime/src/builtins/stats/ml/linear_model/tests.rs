use super::*;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, ObjectInstance};

fn tensor_value(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn all_fitlm_integer_storages() -> Vec<IntegerStorage> {
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
fn scalar_option_parsers_read_all_integer_storage_variants() {
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];
    for storage in storages {
        let value = poisoned_int_tensor(storage, 1, 1);
        assert_eq!(numeric_scalar(&value, "Alpha").unwrap(), 1.0);
        assert!(bool_scalar(&value, "RobustOpts").unwrap());
        assert!(
            matches!(exclude_spec(&value), Ok(ExcludeSpec::Indices(indices)) if indices == vec![1])
        );
    }
}

fn object(value: Value) -> ObjectInstance {
    match value {
        Value::Object(object) => object,
        other => panic!("expected object, got {other:?}"),
    }
}

fn tensor(value: &Value) -> &Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

mod admission;
mod model;
