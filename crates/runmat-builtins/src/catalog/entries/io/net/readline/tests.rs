use super::*;
use runmat_types::*;

#[test]
fn inference_keeps_peer_dependent_string_or_timeout_double_dynamic_but_host_owned() {
    let request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::super::TCPCLIENT_CLASS.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }))],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &READLINE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(
        result.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::ExternalData)
    );
}

#[test]
fn known_invalid_client_and_extra_control_diagnose_without_fabricating_result_class() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &READLINE_CATALOG_ENTRY);
    assert_eq!(result.diagnostics.len(), 2);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}
