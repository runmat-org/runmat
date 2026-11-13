use super::*;
use runmat_types::*;

#[test]
fn hold_inference_keeps_effects_without_outputs_and_rejects_explicit_outputs() {
    for count in [0, 1, 2] {
        let request = CallRequest {
            arguments: vec![],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let result = inference::infer(&request, &HOLD_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), count > 0);
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
        if count == 0 {
            assert!(result.outputs.is_empty());
        }
    }
}

#[test]
fn hold_mode_inference_is_literal_sensitive_and_unknown_conservative() {
    for (literal, invalid) in [
        (LiteralValue::Unknown, false),
        (LiteralValue::String(" ON ".into()), false),
        (LiteralValue::String("toggle".into()), true),
        (LiteralValue::Number(2.0), true),
        (LiteralValue::Number(1.0), false),
    ] {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: LiteralContext::new(vec![literal]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        };
        assert_eq!(
            !inference::infer(&request, &HOLD_CATALOG_ENTRY)
                .diagnostics
                .is_empty(),
            invalid
        );
    }
}
