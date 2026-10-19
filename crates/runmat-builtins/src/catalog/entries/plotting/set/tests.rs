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

#[test]
fn set_rejects_known_multirow_property_name_without_proven_status() {
    let handle = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut property = ValueFact::scalar(ValueKindFact::Character);
    property.shape = ShapeFact::from(vec![Some(2), Some(4)]);
    let result = inference::infer(
        &CallRequest {
            arguments: vec![handle, property, ValueFact::scalar(ValueKindFact::Logical)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        },
        &SET_CATALOG_ENTRY,
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(
        result.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
}

#[test]
fn control_objects_share_scalar_text_admission_with_partial_shapes() {
    let value = ValueFact::scalar(ValueKindFact::Logical);
    for class in [
        standard::ZOOM_CONTROL,
        standard::PAN_CONTROL,
        standard::DATA_CURSOR_CONTROL,
    ] {
        let handle = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }));
        for (shape, should_reject) in [
            (ShapeFact::from(vec![Some(1), Some(1)]), false),
            (ShapeFact::from(vec![Some(2), None]), true),
            (ShapeFact::from(vec![None, Some(0)]), true),
            (ShapeFact::from(vec![None, None]), false),
        ] {
            let mut name = ValueFact::scalar(ValueKindFact::String);
            name.shape = shape;
            let result = inference::infer(
                &CallRequest {
                    arguments: vec![handle.clone(), name, value.clone()],
                    literals: LiteralContext::default(),
                    outputs: OutputSelection::default(),
                },
                &SET_CATALOG_ENTRY,
            );
            assert_eq!(!result.diagnostics.is_empty(), should_reject);
            if should_reject {
                assert!(matches!(
                    result.outputs[0].certainty,
                    CertaintyFact::Dynamic(_)
                ));
            }
        }
    }
}
