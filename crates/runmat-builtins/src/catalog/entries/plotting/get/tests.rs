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

#[test]
fn get_accepts_known_graphics_control_object_and_keeps_unknown_object_conservative() {
    let object = |runtime_class| {
        ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class,
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }))
    };
    let known = infer(vec![object(Some(standard::ZOOM_CONTROL.owned()))], vec![]);
    assert!(known.diagnostics.is_empty());
    assert!(matches!(known.outputs[0].kind, ValueKindFact::Struct(_)));
    let unknown = infer(vec![object(None)], vec![]);
    assert!(unknown.diagnostics.is_empty());
    assert!(matches!(
        unknown.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
    let unknown_type = infer(
        vec![object(None), ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::Unknown, LiteralValue::String("Type".into())],
    );
    assert!(unknown_type.diagnostics.is_empty());
    assert!(matches!(
        unknown_type.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
    let unrelated = infer(vec![object(Some(standard::TABLE.owned()))], vec![]);
    assert!(!unrelated.diagnostics.is_empty());
}

#[test]
fn control_object_type_is_rejected_and_shared_enable_is_string() {
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
        let name = ValueFact::scalar(ValueKindFact::String);
        let type_result = infer(
            vec![handle.clone(), name.clone()],
            vec![LiteralValue::Unknown, LiteralValue::String("Type".into())],
        );
        assert!(!type_result.diagnostics.is_empty());
        assert!(matches!(
            type_result.outputs[0].certainty,
            CertaintyFact::Dynamic(_)
        ));
        let enable = infer(
            vec![handle.clone(), name.clone()],
            vec![LiteralValue::Unknown, LiteralValue::String("Enable".into())],
        );
        assert!(enable.diagnostics.is_empty());
        assert_eq!(enable.outputs[0].kind, ValueKindFact::String);
        let other = infer(
            vec![handle, name],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("NotAProperty".into()),
            ],
        );
        assert!(matches!(
            other.outputs[0].certainty,
            CertaintyFact::Dynamic(_)
        ));
    }
}

#[test]
fn get_property_text_geometry_rejects_known_arrays_but_keeps_unknown_shapes() {
    let handle = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut scalar_string = ValueFact::scalar(ValueKindFact::String);
    assert!(infer(vec![handle.clone(), scalar_string.clone()], vec![])
        .diagnostics
        .is_empty());
    scalar_string.shape = ShapeFact::from(vec![Some(1), Some(2)]);
    assert!(!infer(vec![handle.clone(), scalar_string], vec![])
        .diagnostics
        .is_empty());
    let mut characters = ValueFact::scalar(ValueKindFact::Character);
    characters.shape = ShapeFact::from(vec![Some(2), Some(2)]);
    assert!(!infer(vec![handle.clone(), characters.clone()], vec![])
        .diagnostics
        .is_empty());
    characters.shape = ShapeFact::Unknown;
    assert!(infer(vec![handle, characters], vec![])
        .diagnostics
        .is_empty());
}

#[test]
fn control_object_get_uses_partial_scalar_text_geometry() {
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
        for (kind, shape, invalid) in [
            (
                ValueKindFact::String,
                ShapeFact::from(vec![Some(1), Some(1)]),
                false,
            ),
            (
                ValueKindFact::String,
                ShapeFact::from(vec![Some(2), None]),
                true,
            ),
            (
                ValueKindFact::String,
                ShapeFact::from(vec![None, Some(0)]),
                true,
            ),
            (
                ValueKindFact::String,
                ShapeFact::from(vec![None, None]),
                false,
            ),
            (
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(2), None]),
                true,
            ),
            (ValueKindFact::Character, ShapeFact::Unknown, false),
        ] {
            let mut name = ValueFact::scalar(kind);
            name.shape = shape;
            let inferred = infer(vec![handle.clone(), name], vec![]);
            assert_eq!(!inferred.diagnostics.is_empty(), invalid);
            if invalid {
                assert!(matches!(
                    inferred.outputs[0].certainty,
                    CertaintyFact::Dynamic(_)
                ));
            }
        }
    }
}
