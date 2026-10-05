use super::{CallableForm, CallableValue};
use runmat_types::FunctionSourceRole;

fn callable() -> CallableValue {
    CallableValue {
        owner_identity: "bound:7".into(),
        qualified_name: "f".into(),
        callable_digest: CallableValue::identity_digest(
            "bound:7",
            "f",
            CallableForm::Closure,
            FunctionSourceRole::Anonymous,
            Some("/origin/f.m"),
        ),
        form: CallableForm::Closure,
        source_role: FunctionSourceRole::Anonymous,
        source_file: Some("/origin/f.m".into()),
        captures: Vec::new(),
    }
}

#[test]
fn callable_form_and_role_are_required_revision_fields() {
    let value = callable();
    assert!(value.validate_identity().is_ok());
    let mut old = serde_json::to_value(&value).unwrap();
    let object = old.as_object_mut().unwrap();
    object.remove("form");
    object.remove("source_role");
    object.remove("source_file");
    assert!(serde_json::from_value::<CallableValue>(old).is_err());

    let mut old_source_file = serde_json::to_value(&value).unwrap();
    old_source_file
        .as_object_mut()
        .unwrap()
        .remove("source_file");
    assert!(serde_json::from_value::<CallableValue>(old_source_file).is_err());

    let mut explicit_absent_file = serde_json::to_value(&value).unwrap();
    explicit_absent_file["source_file"] = serde_json::Value::Null;
    let decoded: CallableValue =
        serde_json::from_value(explicit_absent_file).expect("explicit null is a valid field");
    assert_eq!(decoded.source_file, None);
    assert!(decoded.validate_identity().is_err());

    let mut invalid_file_type = serde_json::to_value(&value).unwrap();
    invalid_file_type["source_file"] = serde_json::json!(7);
    assert!(serde_json::from_value::<CallableValue>(invalid_file_type).is_err());

    let mut wrong_form = value.clone();
    wrong_form.form = CallableForm::Handle;
    assert!(wrong_form.validate_identity().is_err());
    let mut wrong_file = value.clone();
    wrong_file.source_file = Some("/destination/unrelated.m".into());
    assert!(wrong_file.validate_identity().is_err());
    let mut wrong_role = value;
    wrong_role.source_role = FunctionSourceRole::Local;
    assert!(wrong_role.validate_identity().is_err());
}
