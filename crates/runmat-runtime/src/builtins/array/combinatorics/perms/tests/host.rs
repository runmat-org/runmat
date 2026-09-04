use super::*;
use runmat_value::{
    CellArray, CharArray, IntValue, IntegerStorage, LogicalArray, NumericDType, NumericStorage,
    StringArray,
};

#[test]
fn numeric_vectors_use_reverse_lexicographic_order() {
    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("input");
    let Value::Tensor(output) = call(Value::Tensor(input)).expect("perms") else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape, vec![6, 3]);
    assert_eq!(output.numeric_dtype(), NumericDType::F64);
    assert_eq!(
        tensor_rows(&output),
        vec![
            vec![3.0, 2.0, 1.0],
            vec![3.0, 1.0, 2.0],
            vec![2.0, 3.0, 1.0],
            vec![2.0, 1.0, 3.0],
            vec![1.0, 3.0, 2.0],
            vec![1.0, 2.0, 3.0],
        ]
    );
}

#[test]
fn columns_duplicates_scalars_and_empty_vectors_keep_compatible_semantics() {
    let column = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).expect("column");
    let Value::Tensor(output) = call(Value::Tensor(column)).expect("perms") else {
        panic!("expected tensor")
    };
    assert_eq!(tensor_rows(&output)[0], vec![30.0, 20.0, 10.0]);

    let duplicates = Tensor::new(vec![1.0, 1.0, 2.0], vec![1, 3]).expect("duplicates");
    let Value::Tensor(output) = call(Value::Tensor(duplicates)).expect("perms") else {
        panic!("expected tensor")
    };
    assert_eq!(output.rows, 6);
    assert_eq!(tensor_rows(&output)[0], tensor_rows(&output)[1]);

    assert_eq!(call(Value::Num(7.0)).expect("scalar"), Value::Num(7.0));
    assert_eq!(
        call(Value::Int(IntValue::U32(7))).expect("integer scalar"),
        Value::Int(IntValue::U32(7))
    );
    for shape in [vec![1, 0], vec![0, 0]] {
        let empty = Tensor::new(Vec::new(), shape).expect("empty");
        let Value::Tensor(output) = call(Value::Tensor(empty)).expect("perms") else {
            panic!("expected tensor")
        };
        assert_eq!(output.shape, vec![1, 0]);
    }
}

#[test]
fn native_numeric_storage_is_preserved_exactly() {
    let single = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).expect("single");
    let Value::Tensor(single) = call(Value::Tensor(single)).expect("perms") else {
        panic!("expected tensor")
    };
    assert_eq!(
        single.into_numeric_storage().expect("storage"),
        NumericStorage::F32(vec![2.0, 1.0, 1.0, 2.0])
    );

    let storages = [
        IntegerStorage::I8(vec![-2, 7]),
        IntegerStorage::I16(vec![-300, 400]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    ];
    for storage in storages {
        let values = storage.exact_values();
        let expected = storage
            .from_exact_values_like(vec![
                values[1].clone(),
                values[0].clone(),
                values[0].clone(),
                values[1].clone(),
            ])
            .expect("expected storage");
        let input = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
        let Value::Tensor(output) = call(Value::Tensor(input)).expect("perms") else {
            panic!("expected tensor")
        };
        assert_eq!(output.integer_storage(), Some(&expected));
    }
}

#[test]
fn complex_and_container_vectors_preserve_representation() {
    let complex =
        ComplexTensor::new(vec![(1.0, 1.0), (2.0, -2.0), (3.0, 0.5)], vec![1, 3]).expect("complex");
    let Value::ComplexTensor(output) = call(Value::ComplexTensor(complex)).expect("perms") else {
        panic!("expected complex tensor")
    };
    assert_eq!(
        complex_rows(&output)[0],
        vec![(3.0, 0.5), (2.0, -2.0), (1.0, 1.0)]
    );
    assert_eq!(
        complex_rows(&output)[5],
        vec![(1.0, 1.0), (2.0, -2.0), (3.0, 0.5)]
    );

    let logical = LogicalArray::new(vec![0, 1, 1], vec![1, 3]).expect("logical");
    let Value::LogicalArray(logical) = call(Value::LogicalArray(logical)).expect("perms") else {
        panic!("expected logical array")
    };
    assert_eq!(logical.shape, vec![6, 3]);
    assert_eq!(
        (0..logical.shape[1])
            .map(|column| logical.data[column * logical.shape[0]])
            .collect::<Vec<_>>(),
        vec![1, 1, 0]
    );
    let characters = CharArray::new_row("abc");
    let Value::CharArray(characters) = call(Value::CharArray(characters)).expect("perms") else {
        panic!("expected characters")
    };
    assert_eq!(characters.data[0..3].iter().collect::<String>(), "cba");
    let strings = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).expect("strings");
    let Value::StringArray(strings) = call(Value::StringArray(strings)).expect("perms") else {
        panic!("expected string array")
    };
    assert_eq!(strings.shape, vec![2, 2]);
    assert_eq!(strings.data, vec!["b", "a", "a", "b"]);
    let cells = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).expect("cells");
    let Value::Cell(cells) = call(Value::Cell(cells)).expect("perms") else {
        panic!("expected cell array")
    };
    assert_eq!(cells.shape, vec![2, 2]);
    assert_eq!(
        cells.data,
        vec![
            Value::Num(2.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(2.0),
        ]
    );
}

#[test]
fn invalid_and_oversized_inputs_return_stable_errors() {
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("matrix");
    let error = call(Value::Tensor(matrix)).expect_err("matrix rejected");
    assert_eq!(error.identifier(), Some("RunMat:perms:InvalidInput"));

    let sparse = Value::SparseTensor(runmat_value::SparseTensor::zeros(1, 1));
    let error = call(sparse).expect_err("sparse rejected");
    assert_eq!(error.identifier(), Some("RunMat:perms:InvalidInput"));

    let long = Tensor::new((1..=11).map(f64::from).collect(), vec![1, 11]).expect("long");
    let error = call(Value::Tensor(long)).expect_err("large output rejected");
    assert_eq!(error.identifier(), Some("RunMat:perms:TooLarge"));
}
