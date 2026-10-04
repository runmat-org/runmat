use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn status_and_message_outputs_follow_requested_count() {
    let args = vec![
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    ];
    let none = inference::infer(&request(args.clone(), 0), &XLSWRITE_CATALOG_ENTRY);
    assert!(none.outputs.is_empty());
    let one = inference::infer(&request(args.clone(), 1), &XLSWRITE_CATALOG_ENTRY);
    assert!(matches!(one.outputs[0].kind, ValueKindFact::Logical));
    let two = inference::infer(&request(args, 2), &XLSWRITE_CATALOG_ENTRY);
    let ValueKindFact::Struct(fields) = &two.outputs[1].kind else {
        panic!("message must be scalar struct")
    };
    assert!(fields.fields.contains_key("message"));
    assert!(fields.fields.contains_key("identifier"));
    assert!(two.effects.0.contains(&EffectKind::FilesystemWrite));
}

#[test]
fn known_invalid_path_and_complex_data_have_useful_diagnostics() {
    let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    let known_bad = inference::infer(
        &request(vec![ValueFact::scalar(ValueKindFact::Logical), complex], 1),
        &XLSWRITE_CATALOG_ENTRY,
    );
    assert!(known_bad
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-XLSWRITE-PATH"));
    assert!(known_bad
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-XLSWRITE-DATA"));
    let dynamic = inference::infer(
        &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2], 1),
        &XLSWRITE_CATALOG_ENTRY,
    );
    assert!(dynamic.diagnostics.is_empty());
}

#[test]
fn descriptor_and_runtime_output_limits_agree() {
    assert!(XLSWRITE_DESCRIPTOR
        .signatures
        .iter()
        .any(|sig| sig.outputs.len() == 1));
    assert!(XLSWRITE_DESCRIPTOR
        .signatures
        .iter()
        .any(|sig| sig.outputs.len() == 2));
    assert!(!XLSWRITE_DESCRIPTOR
        .signatures
        .iter()
        .any(|sig| sig.outputs.len() > 2));
}
