use super::*;
use runmat_types::*;

#[test]
fn cla_literal_modes_and_device_controls_match_runtime_admission() {
    for (class, invalid) in [(standard::AXES, false), (standard::FIGURE, true)] {
        let target = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.into()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }));
        let call = CallRequest {
            arguments: vec![target],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            !inference::infer(&call, &CLA_CATALOG_ENTRY)
                .diagnostics
                .is_empty(),
            invalid
        );
    }
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    for (mode, invalid) in [(" RESET ", false), ("not-a-mode", true)] {
        let result = inference::infer(
            &request(
                vec![unknown.clone()],
                vec![LiteralValue::String(mode.into())],
                0,
            ),
            &CLA_CATALOG_ENTRY,
        );
        assert_eq!(!result.diagnostics.is_empty(), invalid);
    }
    let mut device = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    device.residency = ResidencyFact::Device { provider: None };
    assert!(
        !inference::infer(&request(vec![device], vec![], 0), &CLA_CATALOG_ENTRY)
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
fn cla_output_count_and_effects_are_independent_of_discarded_results() {
    for count in [0, 1, 2] {
        let result = inference::infer(&request(vec![], vec![], count), &CLA_CATALOG_ENTRY);
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
fn cla_unknown_input_stays_conservative_and_dense_text_is_rejected() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(
        inference::infer(&request(vec![unknown], vec![], 1), &CLA_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let text_array = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    assert!(
        !inference::infer(&request(vec![text_array], vec![], 1), &CLA_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
