use super::*;

#[test]
fn java_control_decoder_accepts_only_nonempty_text_scalar_shapes() {
    assert_eq!(
        string_argument(Some(Value::String("java.lang.String".into())), "class").unwrap(),
        "java.lang.String"
    );
    let text = "java.lang.String";
    let trailing =
        CharArray::new_with_shape(text.chars().collect(), vec![1, text.chars().count(), 1])
            .unwrap();
    assert_eq!(
        string_argument(Some(Value::CharArray(trailing)), "class").unwrap(),
        text
    );
    let pages = CharArray::new_with_shape(text.chars().collect(), vec![1, 1, text.chars().count()])
        .unwrap();
    for invalid in [
        Value::Num(7.0),
        Value::Bool(true),
        Value::String(" ".into()),
        Value::CharArray(pages),
    ] {
        assert!(string_argument(Some(invalid), "class").is_err());
    }
}

#[test]
fn java_feature_identity_is_shared_with_catalog() {
    assert_eq!(
        runmat_builtins::JavaFeature::parse("JVM"),
        Some(runmat_builtins::JavaFeature::Jvm)
    );
    assert_eq!(
        runmat_builtins::JavaFeature::parse("aWt"),
        Some(runmat_builtins::JavaFeature::Awt)
    );
    assert_eq!(runmat_builtins::JavaFeature::parse("other"), None);
}
