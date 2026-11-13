use super::*;
use runmat_types::*;

#[test]
fn colorbar_output_does_not_invent_a_profile_specific_class() {
    let result = inference::infer(&request(vec![], vec![], 1), &COLORBAR_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].shape.element_count(), Some(1));
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn colorbar_literal_modes_and_device_controls_match_runtime_admission() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    for (mode, invalid) in [(" ON ", false), ("not-a-mode", true)] {
        let result = inference::infer(
            &request(
                vec![unknown.clone()],
                vec![LiteralValue::String(mode.into())],
                0,
            ),
            &COLORBAR_CATALOG_ENTRY,
        );
        assert_eq!(!result.diagnostics.is_empty(), invalid);
    }
    let mut device = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    device.residency = ResidencyFact::Device { provider: None };
    assert!(
        !inference::infer(&request(vec![device], vec![], 0), &COLORBAR_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn colorbar_output_count_and_effects_are_independent_of_discarded_results() {
    for count in [0, 1, 2] {
        let result = inference::infer(&request(vec![], vec![], count), &COLORBAR_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), count > 1);
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
        assert!(result.effects.0.contains(&EffectKind::HostCallback));
        assert!(result.effects.0.contains(&EffectKind::MayThrow));
        if count == 0 {
            assert!(result.outputs.is_empty());
        }
    }
}

#[test]
fn colorbar_unknown_input_stays_conservative_and_dense_text_is_rejected() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(
        inference::infer(&request(vec![unknown], vec![], 1), &COLORBAR_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let text_array = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    assert!(!inference::infer(
        &request(vec![text_array], vec![], 1),
        &COLORBAR_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
}
