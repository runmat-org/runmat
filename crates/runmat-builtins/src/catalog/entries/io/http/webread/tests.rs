use super::*;
use runmat_types::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_unsupported_url_scheme_is_rejected_by_shared_http_decoder() {
    let request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext::new(vec![LiteralValue::String(
            "mailto:alice@example.test".into(),
        )]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer(&request, &WEBREAD_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-WEBREAD-URL"));
}

fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &WEBREAD_CATALOG_ENTRY,
    )
}

#[test]
fn webread_preserves_dynamic_response_and_network_effects() {
    let result = call(
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::String("https://example.invalid/data".into())],
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert!(result.effects.0.contains(&EffectKind::Network));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));
}

#[test]
fn webread_rejects_proven_nontext_url_but_not_unknown() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(!call(vec![numeric], vec![LiteralValue::Unknown])
        .diagnostics
        .is_empty());
    assert!(call(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![LiteralValue::Unknown]
    )
    .diagnostics
    .is_empty());
}

#[test]
fn literal_content_type_proves_exact_binary_class_but_dynamic_tail_invalidates_it() {
    let url = ValueFact::scalar(ValueKindFact::String);
    let name = ValueFact::scalar(ValueKindFact::String);
    let value = ValueFact::scalar(ValueKindFact::String);
    let known = call(
        vec![url.clone(), name.clone(), value.clone()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("ContentType".into()),
            LiteralValue::String("binary".into()),
        ],
    );
    assert_eq!(
        known.outputs[0].numeric().unwrap().class,
        NumericClass::UInt8
    );
    assert_eq!(known.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
    let dynamic_tail = call(
        vec![url, name.clone(), value.clone(), name, value],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("ContentType".into()),
            LiteralValue::String("binary".into()),
            LiteralValue::Unknown,
            LiteralValue::Unknown,
        ],
    );
    assert_eq!(dynamic_tail.outputs[0].kind, ValueKindFact::Unknown);
}
