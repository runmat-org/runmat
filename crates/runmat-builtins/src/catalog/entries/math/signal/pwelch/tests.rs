use super::*;
use runmat_types::*;

fn signal(class: NumericClass, domain: NumericDomain, residency: ResidencyFact) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(vec![Some(128), Some(2)]),
        StorageFact::Dense,
    );
    fact.residency = residency;
    fact
}

fn infer(input: ValueFact, outputs: RequestedOutputCount) -> CallInference {
    crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(outputs),
        },
    )
}

#[test]
fn parent_literal_requires_a_scalar_axes_candidate() {
    let signal = signal(
        NumericClass::Double,
        NumericDomain::Real,
        ResidencyFact::Host,
    );
    let name = ValueFact::scalar(ValueKindFact::String);
    let invalid_target = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let result = crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![signal, name, invalid_target],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("Parent".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PWELCH-PARENT"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn plotting_effect_is_conditional_on_requested_outputs_and_parent() {
    let x = signal(
        NumericClass::Double,
        NumericDomain::Real,
        ResidencyFact::Host,
    );
    let numeric = infer(x.clone(), RequestedOutputCount::Exactly(2));
    assert!(!numeric.effects.0.contains(&EffectKind::UserInterface));
    assert!(!numeric
        .capabilities
        .0
        .contains(&CapabilityRequirement::UserInterface));
    let plotted = infer(x.clone(), RequestedOutputCount::Exactly(0));
    assert!(plotted.effects.0.contains(&EffectKind::UserInterface));
    let parent = crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                x,
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("Parent".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    assert!(parent.effects.0.contains(&EffectKind::UserInterface));
}

#[test]
fn positional_limit_counts_controls_after_a_known_parent_pair() {
    let x = signal(
        NumericClass::Double,
        NumericDomain::Real,
        ResidencyFact::Host,
    );
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let controls = [
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("onesided".into()),
        LiteralValue::String("psd".into()),
        LiteralValue::String("mean".into()),
        LiteralValue::String("maxhold".into()),
    ];
    let mut arguments = vec![x.clone()];
    arguments.extend((0..4).map(|_| numeric.clone()));
    arguments.extend((0..4).map(|_| ValueFact::scalar(ValueKindFact::String)));
    let mut literals = vec![LiteralValue::Unknown];
    literals.extend(controls);
    let invalid = crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PWELCH-ARITY"));
    assert_eq!(invalid.outputs[0].kind, ValueKindFact::Unknown);

    let mut arguments = vec![x, ValueFact::scalar(ValueKindFact::String), numeric.clone()];
    arguments.extend((0..4).map(|_| numeric.clone()));
    arguments.extend((0..3).map(|_| ValueFact::scalar(ValueKindFact::String)));
    let valid = crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("Parent".into()),
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("onesided".into()),
                LiteralValue::String("psd".into()),
                LiteralValue::String("mean".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(!valid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PWELCH-ARITY"));
}

#[test]
fn output_class_is_signal_derived_and_residency_is_not_proven() {
    for class in [
        NumericClass::Single,
        NumericClass::Double,
        NumericClass::Int64,
    ] {
        let input = signal(class, NumericDomain::Real, ResidencyFact::Host);
        let facts = infer(input, RequestedOutputCount::Exactly(3));
        let expected = if class == NumericClass::Single {
            NumericClass::Single
        } else {
            NumericClass::Double
        };
        assert_eq!(facts.outputs[0].numeric().unwrap().class, expected);
        assert_eq!(facts.outputs[1].numeric().unwrap().class, expected);
        assert_eq!(
            facts.outputs[0].shape.known_dims(),
            Some(vec![None, Some(2)])
        );
        assert_eq!(
            facts.outputs[1].shape.known_dims(),
            Some(vec![None, Some(1)])
        );
        assert_eq!(facts.outputs[2].kind, ValueKindFact::Unknown);
        assert!(!facts.outputs[2].is_scalar());
    }
    let input = signal(
        NumericClass::Single,
        NumericDomain::Complex,
        ResidencyFact::Device {
            provider: Some("owner".into()),
        },
    );
    let facts = infer(input, RequestedOutputCount::Exactly(2));
    assert!(facts
        .outputs
        .iter()
        .all(|fact| fact.residency == ResidencyFact::Unknown));
    assert!(facts
        .outputs
        .iter()
        .all(|fact| fact.numeric().unwrap().domain == NumericDomain::Real));
}

#[test]
fn invalid_signal_does_not_prove_outputs() {
    let input = signal(
        NumericClass::Int16,
        NumericDomain::Complex,
        ResidencyFact::Host,
    );
    let facts = infer(input, RequestedOutputCount::One);
    assert!(!facts.diagnostics.is_empty());
    assert_eq!(facts.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn known_invalid_control_does_not_prove_result() {
    let result = crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                signal(
                    NumericClass::Single,
                    NumericDomain::Real,
                    ResidencyFact::Host,
                ),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(-1.0)]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn typed_selector_tail_rejects_known_numeric_after_keyword() {
    let result = crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                signal(
                    NumericClass::Double,
                    NumericDomain::Real,
                    ResidencyFact::Host,
                ),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("maxhold".into()),
                LiteralValue::Number(2.0),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PWELCH-RANGE"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn selector_identity_and_order_are_shared_with_runtime() {
    assert_eq!(PwelchKeyword::decode(" MaxHold "), PwelchKeyword::MaxHold);
    assert_eq!(PwelchKeyword::decode("Parent"), PwelchKeyword::Parent);
    let mut cursor = PwelchControlCursor::default();
    assert_eq!(cursor.next(None), PwelchControlRole::Window);
    assert_eq!(
        cursor.next(Some(&PwelchKeyword::MaxHold)),
        PwelchControlRole::Option
    );
    assert_eq!(cursor.next(None), PwelchControlRole::Option);
}

#[test]
fn complex_onesided_selector_has_no_success_fact() {
    let input = signal(
        NumericClass::Double,
        NumericDomain::Complex,
        ResidencyFact::Host,
    );
    let result = crate::infer_catalog_call(
        &PWELCH_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                input,
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(8.0),
                LiteralValue::Number(0.0),
                LiteralValue::Number(8.0),
                LiteralValue::String("onesided".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PWELCH-RANGE"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn descriptor_and_docs_are_canonical() {
    assert!(crate::validate_builtin_catalog(&[&PWELCH_CATALOG_ENTRY]).is_empty());
    assert_eq!(PWELCH_CATALOG_ENTRY.identity.name, "pwelch");
    assert_eq!(
        PWELCH_CATALOG_ENTRY
            .documentation
            .examples
            .iter()
            .map(|example| example.id)
            .collect::<Vec<_>>(),
        vec![
            "example-1",
            "example-2",
            "example-3",
            "example-4",
            "example-5",
            "example-6",
            "example-7",
            "example-8",
            "example-9",
            "example-10",
            "example-11"
        ]
    );
    assert_eq!(
        PWELCH_CATALOG_ENTRY.descriptor.signatures[1].outputs.len(),
        2
    );
}
