use super::*;
#[test]
fn public_subsref_and_subsasgn_use_default_numeric_indexing() {
    let source =
        Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).expect("tensor"));
    let descriptor = substruct(
        crate::OBJECT_INDEX_PAREN,
        cell(vec![Value::Num(2.0), Value::Num(1.0)]),
    );
    assert_eq!(
        block_on(subsref_builtin(source.clone(), descriptor.clone())).expect("subsref"),
        Value::Num(20.0)
    );

    let updated =
        block_on(subsasgn_builtin(source, descriptor, Value::Num(99.0))).expect("subsasgn");
    let Value::Tensor(updated) = updated else {
        panic!("numeric parentheses assignment must preserve tensor storage");
    };
    assert_eq!(
        (0..updated.len())
            .map(|index| updated.numeric_value_at(index).unwrap().materialize_f64())
            .collect::<Vec<_>>(),
        vec![10.0, 99.0, 30.0, 40.0]
    );
}

#[test]
fn typed_structure_array_carries_multiple_subscript_levels() {
    let level = |kind: &str, subs: Value| {
        let mut value = StructValue::new();
        value.insert("type", Value::String(kind.to_string()));
        value.insert("subs", subs);
        value
    };
    let descriptor = Value::StructArray(
        StructArray::new(
            vec![
                level(crate::OBJECT_INDEX_MEMBER, Value::String("field".into())),
                level(crate::OBJECT_INDEX_PAREN, cell(vec![Value::Num(1.0)])),
            ],
            vec![1, 2],
        )
        .expect("subscript structure array"),
    );
    let levels = parse_subscript_levels(&descriptor).expect("parse subscript levels");
    assert_eq!(levels.len(), 2);
    assert_eq!(levels[0].kind, crate::OBJECT_INDEX_MEMBER);
    assert_eq!(levels[1].kind, crate::OBJECT_INDEX_PAREN);

    let legacy_cell = cell(vec![substruct(
        crate::OBJECT_INDEX_MEMBER,
        Value::String("field".into()),
    )]);
    assert!(parse_subscript_levels(&legacy_cell).is_err());
}
