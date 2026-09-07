use runmat_value::{IntValue, IntegerStorage, LogicalArray, NumericDType, NumericStorage, Value};

use super::super::rescale_builtin;
use super::support::{assert_close, integer, single, tensor, values};

#[tokio::test]
async fn preserves_empty_shape_and_single_storage() {
    let empty = rescale_builtin(tensor(Vec::new(), vec![0, 3]), vec![])
        .await
        .expect("empty");
    assert_eq!(values(empty), (Vec::new(), vec![0, 3], NumericDType::F64));

    let single = rescale_builtin(single(vec![1.0, 2.0, 3.0], vec![1, 3]), vec![])
        .await
        .expect("single");
    let Value::Tensor(single) = single else {
        panic!("single tensor")
    };
    assert_eq!(single.numeric_dtype(), NumericDType::F32);
    assert_eq!(
        single.into_numeric_storage().expect("storage"),
        NumericStorage::F32(vec![0.0, 0.5, 1.0])
    );
}

#[tokio::test]
async fn logical_and_integer_inputs_return_double() {
    let logical =
        Value::LogicalArray(LogicalArray::new(vec![0, 1, 1], vec![1, 3]).expect("logical"));
    let (data, shape, dtype) = values(
        rescale_builtin(logical, vec![])
            .await
            .expect("logical scale"),
    );
    assert_eq!(shape, vec![1, 3]);
    assert_eq!(dtype, NumericDType::F64);
    assert_close(&data, &[0.0, 1.0, 1.0]);

    let integer_input = integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]);
    let (data, _, dtype) = values(
        rescale_builtin(integer_input, vec![])
            .await
            .expect("integer scale"),
    );
    assert_eq!(dtype, NumericDType::F64);
    assert_close(&data, &[0.0, 0.5, 1.0]);

    let bounded = rescale_builtin(
        integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]),
        vec![
            integer(IntegerStorage::I16(vec![-1]), vec![1, 1]),
            integer(IntegerStorage::I16(vec![1]), vec![1, 1]),
            Value::from("InputMin"),
            integer(IntegerStorage::I16(vec![1]), vec![1, 1]),
            Value::from("InputMax"),
            integer(IntegerStorage::I16(vec![3]), vec![1, 1]),
        ],
    )
    .await
    .expect("integer bounds");
    assert_close(&values(bounded).0, &[-1.0, 0.0, 1.0]);
}

#[tokio::test]
async fn exact_wide_integer_boundary_is_not_signed_saturated() {
    let center = (1_u64 << 63) + 4096;
    let result = rescale_builtin(
        Value::Int(IntValue::U64(center)),
        vec![
            Value::from("InputMin"),
            Value::Int(IntValue::U64(center - 4096)),
            Value::from("InputMax"),
            Value::Int(IntValue::U64(center + 4096)),
        ],
    )
    .await
    .expect("wide boundary");
    assert_close(&values(result).0, &[0.5]);
}
