use super::*;
use runmat_value::{IntegerStorage, NumericStorage, StringArray, Tensor};

#[test]
fn integer_replacements_preserve_class_and_use_zero_for_missing_values() {
    let output = call(
        Value::Tensor(Tensor::new(vec![-1.0, 0.5, 1.5, 3.0], vec![1, 4]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap()),
        vec![Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 7]), vec![1, 2]).unwrap(),
        )],
    )
    .unwrap();
    assert_eq!(
        tensor(output).integer_storage(),
        Some(&IntegerStorage::U64(vec![0, u64::MAX, 7, 0]))
    );
}

#[test]
fn every_fixed_width_replacement_class_is_preserved() {
    let cases = [
        NumericStorage::I8(vec![-8]),
        NumericStorage::I16(vec![-16]),
        NumericStorage::I32(vec![-32]),
        NumericStorage::I64(vec![-64]),
        NumericStorage::U8(vec![8]),
        NumericStorage::U16(vec![16]),
        NumericStorage::U32(vec![32]),
        NumericStorage::U64(vec![u64::MAX]),
    ];
    for labels in cases {
        let expected = labels.clone();
        let output = call(
            Value::Num(0.5),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            vec![Value::Tensor(
                Tensor::from_numeric_storage(labels, vec![1, 1]).unwrap(),
            )],
        )
        .unwrap();
        assert_eq!(tensor(output).into_numeric_storage(), Ok(expected));
    }
}

#[test]
fn text_replacements_preserve_input_shape_and_use_empty_missing_values() {
    let output = call(
        Value::Tensor(Tensor::new(vec![-1.0, 0.5, 1.5], vec![3, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap()),
        vec![Value::StringArray(
            StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
        )],
    )
    .unwrap();
    let Value::StringArray(output) = output else {
        panic!("expected string array");
    };
    assert_eq!(output.shape, vec![3, 1]);
    assert_eq!(output.data, vec!["", "low", "high"]);
}

#[test]
fn floating_replacements_use_nan_for_missing_values() {
    let output = call(
        Value::Tensor(Tensor::new(vec![-1.0, 0.5], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
        vec![Value::Num(42.0)],
    )
    .unwrap();
    let values = tensor(output).materialize_f64();
    assert!(values[0].is_nan());
    assert_eq!(values[1], 42.0);
}
