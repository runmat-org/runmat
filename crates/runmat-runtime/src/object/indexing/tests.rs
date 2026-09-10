use super::*;

#[test]
fn standard_substruct_path_is_an_ordered_typed_structure_array() {
    let path = ObjectSubscriptPath::new(vec![
        ObjectSubscript::member("payload"),
        ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: vec![Value::Num(2.0).into(), ObjectIndexComponent::Colon],
        }),
    ])
    .unwrap();
    let Value::StructArray(array) = path.to_standard_substruct_value().unwrap() else {
        panic!("a multi-step subscript must remain a typed structure array");
    };
    assert_eq!(array.shape(), &[1, 2]);
    assert_eq!(
        array.field_names().map(String::as_str).collect::<Vec<_>>(),
        ["type", "subs"]
    );
    assert_eq!(
        array.field_values("type").unwrap(),
        [Value::String(".".into()), Value::String("()".into())]
    );
    assert_eq!(
        array.field_values("subs").unwrap()[0],
        Value::String("payload".into())
    );
}

#[test]
fn object_subscript_paths_are_nonempty_and_preserve_multiple_steps() {
    let error = ObjectSubscriptPath::new(Vec::new()).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:InvalidObjectSubscriptPath")
    );

    let path = ObjectSubscriptPath::new(vec![
        ObjectSubscript::member("first"),
        ObjectSubscript::member("second"),
    ])
    .unwrap();
    let encoded = path.to_standard_substruct_value().unwrap();
    let decoded = parse_standard_substruct(&encoded).unwrap();
    assert_eq!(decoded.steps().len(), 2);
}
