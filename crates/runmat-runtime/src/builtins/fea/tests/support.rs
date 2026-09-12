use runmat_value::Value;

use super::super::contracts::identities::{
    FEA_PAYLOAD_JSON_PROPERTY, FEA_STUDY_SPEC_JSON_PROPERTY, FEA_SWEEP_SPEC_JSON_PROPERTY,
};

pub(super) fn assert_object_class(value: &Value, expected: runmat_types::StaticClassIdentity) {
    let Value::Object(object) = value else {
        panic!("expected object value");
    };
    assert!(object.class_name.is(expected));
    assert!(
        object.properties.contains_key(FEA_PAYLOAD_JSON_PROPERTY)
            || object.properties.contains_key(FEA_STUDY_SPEC_JSON_PROPERTY)
            || object.properties.contains_key(FEA_SWEEP_SPEC_JSON_PROPERTY)
    );
}
