use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, ObjectFact, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn network(name: &str) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(runmat_types::ClassIdentity::new(name).unwrap()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    }))
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, count: usize) -> CallInference {
    crate::infer_catalog_call(
        &EXPORT_ONNX_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
    )
}

#[test]
fn known_network_and_zero_output_have_filesystem_effect() {
    let result = infer(
        vec![
            network("dlnetwork"),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![],
        0,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(result.outputs.is_empty());
    assert!(result
        .effects
        .0
        .contains(&runmat_types::EffectKind::FilesystemWrite));
    assert_eq!(
        EXPORT_ONNX_CATALOG_ENTRY.placement.accelerator,
        BuiltinAcceleratorPolicy::Forbidden
    );
}

#[test]
fn impossible_output_and_known_wrong_class_are_rejected() {
    let arguments = vec![network("dlarray"), ValueFact::scalar(ValueKindFact::String)];
    assert!(!infer(arguments.clone(), vec![], 0).diagnostics.is_empty());
    assert!(!infer(arguments, vec![], 1).diagnostics.is_empty());
}

#[test]
fn options_are_validated_without_claiming_dynamic_values() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let arguments = vec![
        network("dlnetwork"),
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::String),
        unknown,
    ];
    let valid = infer(
        arguments.clone(),
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("OpsetVersion".into()),
            LiteralValue::Unknown,
        ],
        0,
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    let invalid = infer(
        arguments,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("NoSuchOption".into()),
            LiteralValue::Unknown,
        ],
        0,
    );
    assert!(!invalid.diagnostics.is_empty());
}
