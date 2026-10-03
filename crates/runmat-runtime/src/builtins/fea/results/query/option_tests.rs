use super::*;

#[test]
fn aliases_and_punctuation_decode_to_one_typed_selector() {
    let query = results_query_from_args(&[
        Value::String("FIELDS".into()),
        Value::String("stress".into()),
        Value::String("Field-Values".into()),
        Value::Bool(false),
    ])
    .expect("known aliases");
    assert_eq!(query.include_fields, vec!["stress"]);
    assert!(!query.include_field_values);

    let error = results_query_from_args(&[
        Value::String("fields".into()),
        Value::String("stress".into()),
        Value::String("Include_Fields".into()),
        Value::String("strain".into()),
    ])
    .expect_err("canonical duplicate");
    assert_eq!(error.identifier(), ERROR_INPUT.identifier);
    assert!(error
        .message()
        .contains("duplicate fea.results option `includefields`"));
}

#[test]
fn unknown_option_error_and_duplicate_order_are_retained() {
    let error = results_query_from_args(&[Value::String("not.option".into()), Value::Bool(true)])
        .expect_err("unknown selector");
    assert!(error
        .message()
        .contains("unsupported fea.results option `notoption`"));
}
