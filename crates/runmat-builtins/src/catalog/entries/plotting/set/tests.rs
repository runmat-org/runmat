use super::*;
use runmat_types::*;

#[test]
fn set_inference_preserves_effects_and_status_without_proving_invalid_calls() {
    let handle = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let name = ValueFact::scalar(ValueKindFact::String);
    let value = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let valid = inference::infer(
        &CallRequest {
            arguments: vec![handle.clone(), name.clone(), value.clone()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        },
        &SET_CATALOG_ENTRY,
    );
    assert!(valid.diagnostics.is_empty());
    assert_eq!(valid.outputs[0].kind, ValueKindFact::String);
    assert!(valid.effects.0.contains(&EffectKind::UserInterface));
    assert!(valid.effects.0.contains(&EffectKind::MaySuspend));
    let invalid = inference::infer(
        &CallRequest {
            arguments: vec![handle, name, value.clone(), value],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        },
        &SET_CATALOG_ENTRY,
    );
    assert!(!invalid.diagnostics.is_empty());
    assert!(matches!(
        invalid.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
}

#[test]
fn set_accepts_known_pan_control_object_and_rejects_unrelated_class() {
    let object = |runtime_class| {
        ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class,
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }))
    };
    let name = ValueFact::scalar(ValueKindFact::String);
    let value = ValueFact::scalar(ValueKindFact::Logical);
    let call = |handle| {
        inference::infer(
            &CallRequest {
                arguments: vec![handle, name.clone(), value.clone()],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &SET_CATALOG_ENTRY,
        )
    };
    assert!(call(object(Some(standard::PAN_CONTROL.owned())))
        .diagnostics
        .is_empty());
    assert!(!call(object(Some(standard::TABLE.owned())))
        .diagnostics
        .is_empty());
}
