use super::*;
use runmat_value::{IntValue, IntegerStorage};

fn member(name: &str) -> Value {
    crate::object::indexing::standard_substruct_fixture_from_parts(
        crate::OBJECT_INDEX_MEMBER,
        Value::from(name),
    )
    .expect("member path")
}

fn key(value: Value) -> Value {
    crate::object::indexing::standard_substruct_fixture_from_parts(
        OBJECT_INDEX_PAREN,
        Value::Cell(CellArray::new(vec![value], 1, 1).expect("one key")),
    )
    .expect("key path")
}

#[test]
fn table_methods_preserve_typed_variable_and_return_updated_value() {
    let values = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]),
            vec![2, 1],
        )
        .expect("typed column"),
    );
    let table = block_on(table_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            values,
            Value::from("VariableNames"),
            Value::Cell(CellArray::new(vec![Value::from("Count")], 1, 1).expect("name")),
        ]),
    ))
    .expect("table");
    let selected =
        block_on(table_subsref(None, table.clone(), member("Count"))).expect("read variable");
    let Value::Tensor(selected) = selected.value() else {
        panic!("expected numeric column");
    };
    assert_eq!(
        selected.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]))
    );

    let replacement = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![2, 1]).expect("replacement"),
    );
    let updated = block_on(table_subsasgn(
        None,
        table.clone(),
        member("Count"),
        replacement,
    ))
    .expect("replace variable");
    let original = block_on(table_subsref(None, table, member("Count"))).expect("original read");
    let modified = updated
        .with_owned(|value| block_on(table_subsref(None, value, member("Count"))))
        .expect("updated read");
    let Value::Tensor(original) = original.value() else {
        panic!("expected original column");
    };
    let Value::Tensor(modified) = modified.value() else {
        panic!("expected updated column");
    };
    assert_eq!(
        original.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]))
    );
    assert_eq!(
        modified.integer_storage(),
        Some(&IntegerStorage::U64(vec![1, 2]))
    );
}

#[test]
fn dictionary_methods_compare_wide_keys_and_keep_original_value() {
    let large_key = Value::Int(IntValue::U64(9_007_199_254_740_993));
    let dictionary = block_on(dictionary_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            large_key.clone(),
            Value::Int(IntValue::U64(u64::MAX)),
        ]),
    ))
    .expect("dictionary");
    let selected = block_on(dictionary_subsref(
        None,
        dictionary.clone(),
        key(large_key.clone()),
    ))
    .expect("lookup");
    assert_eq!(selected, Value::Int(IntValue::U64(u64::MAX)));
    let updated = block_on(dictionary_subsasgn(
        None,
        dictionary.clone(),
        key(large_key.clone()),
        Value::Int(IntValue::U64(3)),
    ))
    .expect("replace");
    assert_eq!(
        block_on(dictionary_subsref(None, dictionary, key(large_key.clone()))).expect("original"),
        Value::Int(IntValue::U64(u64::MAX)),
    );
    assert_eq!(
        updated
            .with_owned(|value| block_on(dictionary_subsref(None, value, key(large_key))))
            .expect("updated"),
        Value::Int(IntValue::U64(3)),
    );
}

#[test]
fn method_receivers_reject_wrong_class_and_invalid_path() {
    let wrong = Value::Num(1.0);
    let valid_path = member("Count");
    assert!(block_on(table_subsref(None, wrong.clone(), valid_path.clone())).is_err());
    assert!(block_on(dictionary_subsref(None, wrong.clone(), valid_path.clone())).is_err());
    assert!(block_on(table_subsasgn(
        None,
        wrong.clone(),
        valid_path.clone(),
        Value::Num(2.0)
    ))
    .is_err());
    assert!(block_on(dictionary_subsasgn(
        None,
        wrong,
        valid_path,
        Value::Num(2.0)
    ))
    .is_err());

    let dictionary = block_on(dictionary_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("empty dictionary");
    assert!(block_on(dictionary_subsref(None, dictionary, Value::Num(1.0))).is_err());
}
