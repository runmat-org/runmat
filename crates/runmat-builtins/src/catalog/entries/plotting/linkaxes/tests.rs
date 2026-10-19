use super::*;
use runmat_types::*;
#[test]
fn linkaxes_inference_has_no_dummy_string_and_retains_effects() {
    for count in [0, 1, 2] {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let result = inference::infer(&request, &LINKAXES_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), count > 0);
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
        if count == 0 {
            assert!(result.outputs.is_empty());
        }
    }
}
#[test]
fn linkaxes_literal_mode_validation_does_not_invent_handle_values() {
    for (mode, invalid) in [(" xy ", false), ("off", false), ("z", true)] {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String(mode.into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        };
        assert_eq!(
            !inference::infer(&request, &LINKAXES_CATALOG_ENTRY)
                .diagnostics
                .is_empty(),
            invalid
        );
    }
}
