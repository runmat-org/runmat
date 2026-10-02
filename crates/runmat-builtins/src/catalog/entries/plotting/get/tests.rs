use super::*;
use runmat_types::*;

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        },
        &GET_CATALOG_ENTRY,
    )
}

#[test]
fn get_declares_suspension_and_host_property_facts() {
    let handle = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let all = infer(vec![handle.clone()], vec![]);
    assert!(matches!(all.outputs[0].kind, ValueKindFact::Struct(_)));
    assert_eq!(all.outputs[0].residency, ResidencyFact::Host);
    assert!(all.effects.0.contains(&EffectKind::MaySuspend));
    let one = infer(
        vec![handle, ValueFact::scalar(ValueKindFact::String)],
        vec![],
    );
    assert!(matches!(
        one.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
    let typed = infer(
        vec![
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![LiteralValue::Unknown, LiteralValue::String("Type".into())],
    );
    assert!(typed.diagnostics.is_empty());
    assert_eq!(typed.outputs[0].kind, ValueKindFact::String);
}

#[test]
fn get_rejects_known_resident_handle_without_proven_output() {
    let mut handle = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    handle.residency = ResidencyFact::Device { provider: None };
    let inferred = infer(vec![handle], vec![]);
    assert!(!inferred.diagnostics.is_empty());
    assert!(matches!(
        inferred.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
}
