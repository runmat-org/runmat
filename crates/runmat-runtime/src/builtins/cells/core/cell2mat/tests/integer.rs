use super::*;
use runmat_value::Tensor;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn typed_integer_cells_preserve_every_native_class_exactly() {
    let cases = [
        (IntegerStorage::I8(vec![-7]), IntegerStorage::I8(vec![4])),
        (
            IntegerStorage::I16(vec![-700]),
            IntegerStorage::I16(vec![4]),
        ),
        (
            IntegerStorage::I32(vec![-70_000]),
            IntegerStorage::I32(vec![4]),
        ),
        (
            IntegerStorage::I64(vec![i64::MIN + 1]),
            IntegerStorage::I64(vec![4]),
        ),
        (IntegerStorage::U8(vec![7]), IntegerStorage::U8(vec![4])),
        (IntegerStorage::U16(vec![700]), IntegerStorage::U16(vec![4])),
        (
            IntegerStorage::U32(vec![70_000]),
            IntegerStorage::U32(vec![4]),
        ),
        (
            IntegerStorage::U64(vec![u64::MAX]),
            IntegerStorage::U64(vec![1_u64 << 63]),
        ),
    ];

    for (left, right) in cases {
        let cell = crate::make_cell(
            vec![
                Value::Tensor(Tensor::new_integer(left.clone(), vec![1, 1]).expect("left")),
                Value::Tensor(Tensor::new_integer(right.clone(), vec![1, 1]).expect("right")),
            ],
            1,
            2,
        )
        .expect("cell");
        let result = run(cell).expect("cell2mat");
        let Value::Tensor(output) = result else {
            panic!("expected typed integer tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.integer_storage(),
            Some(&append_same_class(&left, &right))
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn typed_integer_cells_preserve_block_shape_and_saturate_mixed_classes() {
    let first =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 7]), vec![2, 1]).expect("first");
    let second =
        Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, 3]), vec![2, 1]).expect("second");
    let cell =
        crate::make_cell(vec![Value::Tensor(first), Value::Tensor(second)], 1, 2).expect("cell");
    let result = run(cell).expect("cell2mat");
    assert!(matches!(
        result,
        Value::Tensor(output)
            if output.shape == vec![2, 2]
                && output.integer_storage()
                    == Some(&IntegerStorage::U64(vec![u64::MAX, 7, 1_u64 << 63, 3]))
    ));

    let scalar_cell = crate::make_cell(
        vec![
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Int(IntValue::U64(1_u64 << 63)),
        ],
        1,
        2,
    )
    .expect("scalar cell");
    let scalar_result = run(scalar_cell).expect("cell2mat");
    assert!(matches!(
        scalar_result,
        Value::Tensor(output)
            if output.integer_storage()
                == Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
    ));

    let empty = Tensor::new_integer(IntegerStorage::U64(Vec::new()), vec![0, 0])
        .expect("empty typed tensor");
    let empty_cell = crate::make_cell(vec![Value::Tensor(empty)], 1, 1).expect("empty cell");
    let empty_result = run(empty_cell).expect("cell2mat");
    assert!(matches!(
        empty_result,
        Value::Tensor(output)
            if output.shape == vec![0, 0]
                && output.integer_storage() == Some(&IntegerStorage::U64(Vec::new()))
    ));

    let mixed = crate::make_cell(
        vec![
            Value::Int(IntValue::I8(-7)),
            Value::Int(IntValue::U64(u64::MAX)),
        ],
        1,
        2,
    )
    .expect("cell");
    let Value::Tensor(output) = run(mixed).expect("mixed integer cell") else {
        panic!("expected integer tensor");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::I8(vec![-7, i8::MAX]))
    );

    let unsigned_left = crate::make_cell(
        vec![Value::Int(IntValue::U8(3)), Value::Int(IntValue::I64(-9))],
        1,
        2,
    )
    .expect("cell");
    let Value::Tensor(output) = run(unsigned_left).expect("mixed integer cell") else {
        panic!("expected integer tensor");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U8(vec![3, 0]))
    );

    let scalar_double =
        crate::make_cell(vec![Value::Num(300.2), Value::Int(IntValue::U8(5))], 1, 2).expect("cell");
    let Value::Tensor(output) = run(scalar_double).expect("scalar double") else {
        panic!("expected integer tensor");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U8(vec![u8::MAX, 5]))
    );

    let wide_with_double =
        crate::make_cell(vec![Value::Int(IntValue::U64(1)), Value::Num(2.0)], 1, 2).expect("cell");
    assert!(run(wide_with_double).is_err());
}
