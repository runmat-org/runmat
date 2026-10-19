use super::*;
use runmat_types::{standard, DynamicReason, ObjectFact};
use std::collections::BTreeMap;

#[test]
fn unknown_name_shape_is_not_inferred_as_scalar() {
    let result = infer(
        "isenv",
        vec![ValueFact::unknown(DynamicReason::DynamicDispatch)],
        RequestedOutputCount::One,
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn setenv_recognizes_dictionary_identity_without_source_string_matching() {
    let dictionary = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(standard::DICTIONARY.owned()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    }));
    let accepted = infer("setenv", vec![dictionary], RequestedOutputCount::Zero);
    assert!(accepted.diagnostics.is_empty());

    let unrelated = ValueFact::unknown(DynamicReason::RuntimeValue);
    let dynamic = infer("setenv", vec![unrelated], RequestedOutputCount::Zero);
    assert!(dynamic.diagnostics.is_empty());
}
