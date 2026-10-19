use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn drawnow_inference_keeps_statement_and_runmat_status_forms() {
    assert_eq!(DRAWNOW_DESCRIPTOR.signatures.len(), 6);
    assert_eq!(
        DRAWNOW_DESCRIPTOR
            .signatures
            .iter()
            .filter(|signature| !signature.outputs.is_empty())
            .count(),
        3
    );
    let text = ValueFact::scalar(ValueKindFact::Character);
    for count in [0, 1, 2] {
        let result = inference::infer(
            &request(
                vec![text.clone()],
                vec![LiteralValue::String("limitrate".into())],
                count,
            ),
            &DRAWNOW_CATALOG_ENTRY,
        );
        assert_eq!(!result.diagnostics.is_empty(), count > 1);
        assert_eq!(result.outputs.len(), count);
        if count > 0 {
            assert_eq!(result.outputs[0], ValueFact::scalar(ValueKindFact::Logical));
        }
        if count > 1 {
            assert_eq!(
                result.outputs[1],
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            );
        }
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
        assert!(result.effects.0.contains(&EffectKind::HostCallback));
    }
    assert_eq!(
        DRAWNOW_STATUS_OUTPUT_EXTENSION.mode,
        BuiltinExtensionMode::RunMatOnly
    );
}

#[test]
fn drawnow_inference_rejects_known_bad_controls_but_not_dynamic_text() {
    let text = ValueFact::scalar(ValueKindFact::Character);
    for tokens in [
        vec!["bogus"],
        vec!["limitrate", "limitrate"],
        vec!["nocallbacks", "limitrate"],
    ] {
        let result = inference::infer(
            &request(
                tokens.iter().map(|_| text.clone()).collect(),
                tokens
                    .iter()
                    .map(|token| LiteralValue::String((*token).into()))
                    .collect(),
                0,
            ),
            &DRAWNOW_CATALOG_ENTRY,
        );
        assert!(!result.diagnostics.is_empty());
    }
    let unknown = inference::infer(
        &request(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            vec![LiteralValue::Unknown],
            0,
        ),
        &DRAWNOW_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
}
