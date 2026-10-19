use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, ObjectFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
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
fn option_schema_round_trips_and_agrees_with_public_declarations() {
    let signature = &EXPORT_ONNX_DESCRIPTOR.signatures[0];
    let option_description = signature.inputs[2].description;
    let options_section = documentation::DOCUMENTATION
        .sections
        .iter()
        .find(|section| section.heading == "Options and effects")
        .expect("public option documentation");
    for key in ExportOnnxOptionKey::ALL {
        let name = key.canonical_name();
        assert_eq!(ExportOnnxOptionKey::from_name(name), Some(key));
        assert_eq!(
            ExportOnnxOptionKey::from_name(&name.to_ascii_uppercase()),
            Some(key)
        );
        assert!(
            option_description.contains(name),
            "signature omitted {name}"
        );
        assert!(
            options_section
                .paragraphs
                .iter()
                .any(|paragraph| paragraph.contains(name)),
            "documentation omitted {name}"
        );
        assert_eq!(
            key.accepts_integer_extension(),
            key.value_role() != ExportOnnxOptionValueRole::SingleName
        );
    }
    assert_eq!(ExportOnnxOptionKey::from_name("unknown"), None);
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

#[test]
fn numeric_controls_reject_logical_but_verbose_accepts_it() {
    for name in ["OpsetVersion", "BatchSize"] {
        let result = infer(
            vec![
                network("dlnetwork"),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String(name.into()),
                LiteralValue::Bool(true),
            ],
            0,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-DL-EXPORTONNX-OPTION-VALUE"));
    }
    let verbose = infer(
        vec![
            network("dlnetwork"),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("Verbose".into()),
            LiteralValue::Bool(true),
        ],
        0,
    );
    assert!(verbose.diagnostics.is_empty(), "{:?}", verbose.diagnostics);
}

#[test]
fn one_element_string_array_name_is_admitted_without_guessing_contents() {
    let one_string = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer(
        vec![
            network("SeriesNetwork"),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            one_string,
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("InputNames".into()),
            LiteralValue::Unknown,
        ],
        0,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
}
