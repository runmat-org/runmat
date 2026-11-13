use super::*;
use runmat_types::{CallRequest, LiteralContext, OutputSelection, RequestedOutputCount};

#[test]
fn close_contract_has_statement_scalar_and_variadic_forms() {
    let labels: Vec<_> = CLOSE_DESCRIPTOR
        .signatures
        .iter()
        .map(|form| form.label)
        .collect();
    assert_eq!(
        labels,
        [
            "close",
            "status = close()",
            "status = close(target)",
            "status = close(targets...)"
        ]
    );
    assert_eq!(CLOSE_CATALOG_ENTRY.bindings.len(), 1);
    assert_eq!(CLOSE_CATALOG_ENTRY.extensions.len(), 4);
    assert_eq!(
        CLOSE_NETWORK_SELECTOR_EXTENSION.mode,
        BuiltinExtensionMode::RunMatOnly
    );
    assert_eq!(
        CLOSE_TCP_CELL_EXTENSION.mode,
        BuiltinExtensionMode::RunMatOnly
    );
}

#[test]
fn requested_output_and_unknown_input_facts_remain_sound() {
    let request = CallRequest {
        arguments: vec![],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inferred = inference::infer(&request, &CLOSE_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs.len(), 1);
    let excess = CallRequest {
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        ..request
    };
    assert!(!inference::infer(&excess, &CLOSE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    let request = |argument| CallRequest {
        arguments: vec![argument],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let tcp = request(runmat_types::ValueFact::scalar(
        runmat_types::ValueKindFact::Object(runmat_types::ObjectFact {
            class: None,
            runtime_class: Some(crate::TCPCLIENT_CLASS.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }),
    ));
    let inferred = inference::infer(&tcp, &CLOSE_CATALOG_ENTRY);
    assert!(inferred.effects.0.contains(&EffectKind::Network));
    for effect in [
        EffectKind::UserInterface,
        EffectKind::HostCallback,
        EffectKind::Unknown,
    ] {
        assert!(!inferred.effects.0.contains(&effect));
    }
    let unknown = request(runmat_types::ValueFact::unknown(
        runmat_types::DynamicReason::RuntimeValue,
    ));
    let inferred = inference::infer(&unknown, &CLOSE_CATALOG_ENTRY);
    assert!(inferred.effects.0.contains(&EffectKind::HostCallback));
    assert!(inferred.effects.0.contains(&EffectKind::Unknown));
}
