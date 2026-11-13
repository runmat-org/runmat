use super::*;
use runmat_types::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_unsupported_url_scheme_is_rejected_by_shared_http_decoder() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::String("ftp://example.test/item".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer(&request, &WEBWRITE_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-WEBWRITE-URL"));
}

#[test]
fn webwrite_requires_url_and_data_without_guessing_response_class() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer(&request, &WEBWRITE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert!(result.effects.0.contains(&EffectKind::Network));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));
    let missing = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String)],
        ..request
    };
    assert!(!inference::infer(&missing, &WEBWRITE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn literal_binary_response_is_double_not_webread_uint8() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("ContentType".into()),
            LiteralValue::String("binary".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer(&request, &WEBWRITE_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
}
