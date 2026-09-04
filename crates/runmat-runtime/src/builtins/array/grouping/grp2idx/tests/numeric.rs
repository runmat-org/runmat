use runmat_value::{IntegerStorage, Tensor, Value};

use super::super::execute;
use super::outputs;

#[tokio::test]
async fn integer_groups_are_sorted_without_losing_wide_values() {
    let low = 9_007_199_254_740_993_u64;
    let high = u64::MAX;
    let input = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![high, low, high]), vec![3, 1]).unwrap(),
    );
    let values = outputs(execute::apply(input).await.unwrap());
    let Value::Tensor(g) = &values[0] else {
        panic!("expected g")
    };
    assert_eq!(g.materialize_f64(), vec![2.0, 1.0, 2.0]);
    let Value::Tensor(levels) = &values[2] else {
        panic!("expected levels")
    };
    assert_eq!(
        levels.integer_storage(),
        Some(&IntegerStorage::U64(vec![low, high]))
    );
}

#[tokio::test]
async fn every_fixed_width_integer_class_preserves_its_level_storage() {
    let storages = vec![
        IntegerStorage::I8(vec![2, -1, 2]),
        IntegerStorage::I16(vec![2, -1, 2]),
        IntegerStorage::I32(vec![2, -1, 2]),
        IntegerStorage::I64(vec![2, -1, 2]),
        IntegerStorage::U8(vec![2, 1, 2]),
        IntegerStorage::U16(vec![2, 1, 2]),
        IntegerStorage::U32(vec![2, 1, 2]),
        IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, u64::MAX]),
    ];
    for storage in storages {
        let class = storage.numeric_dtype();
        let values = outputs(
            execute::apply(Value::Tensor(
                Tensor::new_integer(storage, vec![3, 1]).unwrap(),
            ))
            .await
            .unwrap(),
        );
        let Value::Tensor(levels) = &values[2] else {
            panic!("expected typed levels")
        };
        assert_eq!(levels.numeric_dtype(), class, "{class:?}");
        assert_eq!(levels.shape, vec![2, 1], "{class:?}");
    }
}

#[tokio::test]
async fn logical_groups_are_sorted_false_then_true() {
    let input =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap());
    let values = outputs(execute::apply(input).await.unwrap());
    let Value::Tensor(g) = &values[0] else {
        panic!("expected g")
    };
    assert_eq!(g.materialize_f64(), vec![2.0, 1.0, 2.0]);
    assert!(matches!(&values[2], Value::LogicalArray(levels) if levels.data == vec![0, 1]));
}

#[tokio::test]
async fn missing_numeric_values_remain_ungrouped() {
    let input = Value::Tensor(Tensor::new(vec![3.0, f64::NAN, 1.0, 3.0], vec![4, 1]).unwrap());
    let values = outputs(execute::apply(input).await.unwrap());
    let Value::Tensor(g) = &values[0] else {
        panic!("expected g")
    };
    let values = g.materialize_f64();
    assert_eq!(values[0], 2.0);
    assert!(values[1].is_nan());
    assert_eq!(&values[2..], &[1.0, 2.0]);
}
