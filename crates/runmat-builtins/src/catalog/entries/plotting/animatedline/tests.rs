use super::*;
use runmat_types::*;

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        },
        &ANIMATEDLINE_CATALOG_ENTRY,
    )
}

#[test]
fn animatedline_unknown_and_empty_calls_have_scalar_handle_and_effects() {
    for arguments in [
        vec![],
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
    ] {
        let result = infer(arguments, vec![]);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0],
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        );
        for effect in [
            EffectKind::UserInterface,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
        ] {
            assert!(result.effects.0.contains(&effect));
        }
    }
}

#[test]
fn animatedline_requested_outputs_do_not_erase_effects() {
    for count in [0, 1, 2] {
        let result = inference::infer(
            &CallRequest {
                arguments: vec![],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
            },
            &ANIMATEDLINE_CATALOG_ENTRY,
        );
        assert_eq!(!result.diagnostics.is_empty(), count > 1);
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
    }
}

#[test]
fn animatedline_known_property_controls_and_nd_follow_runtime() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let numeric = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    for (dims, rejected) in [
        (vec![1, 2, 3], false),
        (vec![2, 1, 3], false),
        (vec![2, 3], true),
        (vec![0, 0], true),
        (vec![1, 0], false),
    ] {
        let result = infer(
            vec![
                text.clone(),
                ValueFact::proven(
                    numeric.clone(),
                    ShapeFact::from(dims.into_iter().map(Some).collect::<Vec<_>>()),
                    StorageFact::Dense,
                ),
            ],
            vec![
                LiteralValue::Character("XData".into()),
                LiteralValue::Unknown,
            ],
        );
        assert_eq!(!result.diagnostics.is_empty(), rejected);
    }
    for (value, rejected) in [
        (0.0, true),
        (-1.0, true),
        (2.5, true),
        (2.0, false),
        (f64::INFINITY, false),
    ] {
        let result = infer(
            vec![text.clone(), ValueFact::scalar(numeric.clone())],
            vec![
                LiteralValue::Character("MaximumNumPoints".into()),
                LiteralValue::Number(value),
            ],
        );
        assert_eq!(!result.diagnostics.is_empty(), rejected);
    }
    assert!(!infer(
        vec![text.clone()],
        vec![LiteralValue::Character("Color".into())]
    )
    .diagnostics
    .is_empty());
    assert!(!infer(
        vec![text, ValueFact::scalar(ValueKindFact::Logical)],
        vec![
            LiteralValue::Character("XData".into()),
            LiteralValue::Unknown
        ]
    )
    .diagnostics
    .is_empty());
}
