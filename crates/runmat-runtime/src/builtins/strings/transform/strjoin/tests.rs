use super::*;
use futures::executor::block_on;
use runmat_value::CellArray;

#[test]
fn legacy_scalar_integer_delimiter_preserves_all_integer_classes_exactly() {
    use runmat_value::IntValue;
    let values = [
        IntValue::I8(-2),
        IntValue::I16(-3),
        IntValue::I32(-4),
        IntValue::I64(i64::MIN),
        IntValue::U8(2),
        IntValue::U16(3),
        IntValue::U32(4),
        IntValue::U64(u64::MAX),
    ];
    for value in values {
        let expected = format!("a{}b", value.decimal_string());
        let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = block_on(strjoin_builtin_source(
            strings(&["a", "b"], vec![1, 2]),
            vec![Value::Int(value.clone())],
        ))
        .unwrap();
        assert_eq!(result, strings(&[&expected], vec![1, 1]));
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(strjoin_builtin_source(
            strings(&["a", "b"], vec![1, 2]),
            vec![Value::Int(value)],
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            STRJOIN_ERROR_INVALID_DELIMITER.identifier
        );
    }
}

fn strings(items: &[&str], shape: Vec<usize>) -> Value {
    Value::StringArray(
        StringArray::new(items.iter().map(|s| (*s).into()).collect(), shape).unwrap(),
    )
}
fn cellstr(items: &[&str], shape: Vec<usize>) -> Value {
    Value::Cell(
        CellArray::from_column_major(
            items
                .iter()
                .map(|s| Value::CharArray(CharArray::new_row(s)))
                .collect(),
            shape,
        )
        .unwrap(),
    )
}
#[test]
fn standard_vectors_cellstr_and_default_delimiters() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
    let input = strings(&["a", "b", "c"], vec![3, 1]);
    assert_eq!(
        block_on(strjoin_builtin_source(input.clone(), vec![])).unwrap(),
        Value::String("a b c".into())
    );
    assert_eq!(
        block_on(strjoin_builtin_source(
            input.clone(),
            vec![Value::from("-")]
        ))
        .unwrap(),
        Value::String("a-b-c".into())
    );
    assert_eq!(
        block_on(strjoin_builtin_source(
            input,
            vec![strings(&["-", ":"], vec![1, 2])]
        ))
        .unwrap(),
        Value::String("a-b:c".into())
    );
    assert_eq!(
        block_on(strjoin_builtin_source(
            cellstr(&["a", "b"], vec![2, 1]),
            vec![]
        ))
        .unwrap(),
        Value::CharArray(CharArray::new_row("a b"))
    );
}
#[test]
fn rowwise_extension_preserves_matrix_column_and_character_behavior() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let delimiter = || vec![Value::from("-")];
    let input = strings(&["a", "c", "b", "d"], vec![2, 2]);
    assert_eq!(
        block_on(strjoin_builtin_source(input, delimiter())).unwrap(),
        strings(&["a-b", "c-d"], vec![2, 1])
    );
    let column = strings(&["a", "b"], vec![2, 1]);
    assert_eq!(
        block_on(strjoin_builtin_source(column.clone(), delimiter())).unwrap(),
        column
    );
    let chars =
        Value::CharArray(CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap());
    assert_eq!(
        block_on(strjoin_builtin_source(chars, delimiter())).unwrap(),
        strings(&["ab", "cd"], vec![2, 1])
    );
}
#[test]
fn unsupported_rank_rejects_without_silent_data_loss_in_both_modes() {
    for enabled in [false, true] {
        let _mode = crate::compatibility::push_runmat_extensions_enabled(enabled);
        for value in [
            strings(&["a", "b"], vec![1, 1, 2]),
            Value::CharArray(
                CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
            ),
        ] {
            let error =
                block_on(strjoin_builtin_source(value, vec![Value::from("-")])).unwrap_err();
            assert_eq!(error.identifier(), STRJOIN_ERROR_INVALID_INPUT.identifier);
        }
    }
}
#[test]
fn empty_modes_and_delimiter_length_are_distinct() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = strings(&[], vec![0, 3]);
    assert_eq!(
        block_on(strjoin_builtin_source(input.clone(), vec![])).unwrap(),
        Value::String(String::new().into())
    );
    assert_eq!(
        block_on(strjoin_builtin_source(input, vec![Value::from("-")])).unwrap(),
        strings(&[], vec![0, 0])
    );
    assert_eq!(
        block_on(strjoin_builtin_source(cellstr(&[], vec![0, 0]), vec![])).unwrap(),
        Value::CharArray(CharArray::new_row(""))
    );
    let error = block_on(strjoin_builtin_source(
        strings(&["a", "b"], vec![1, 2]),
        vec![strings(&["-", ":", "."], vec![1, 3])],
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        STRJOIN_ERROR_INVALID_DELIMITER.identifier
    );
}
#[test]
fn strict_matrix_numeric_provider_and_output_count_reject_before_work() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(strjoin_builtin_source(
        strings(&["a", "b", "c", "d"], vec![2, 2]),
        vec![Value::from("-")],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), STRJOIN_ERROR_INVALID_INPUT.identifier);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let resident = Value::GpuTensor(allocation.handle.clone());
    for value in [
        Value::Num(1.0),
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        resident,
    ] {
        assert_eq!(
            block_on(strjoin_builtin_source(value, vec![]))
                .unwrap_err()
                .identifier(),
            STRJOIN_ERROR_INVALID_INPUT.identifier
        );
    }
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    assert_eq!(
        block_on(strjoin_builtin_source(Value::from("a"), vec![]))
            .unwrap_err()
            .identifier(),
        STRJOIN_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}
