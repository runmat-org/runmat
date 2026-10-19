use super::*;

fn fact(class: NumericClass, shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}
fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        },
        &super::super::DIAG_CATALOG_ENTRY,
    )
}
#[test]
fn known_vector_and_matrix_forms_have_distinct_shapes() {
    let vector = call(
        vec![fact(NumericClass::Single, vec![Some(1), Some(4)])],
        vec![],
    );
    assert_eq!(
        vector.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(4)])
    );
    assert_eq!(
        vector.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    let matrix = call(
        vec![
            fact(NumericClass::Double, vec![Some(3), Some(4)]),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ],
        vec![LiteralValue::Unknown, LiteralValue::Number(-1.0)],
    );
    assert_eq!(
        matrix.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}
#[test]
fn literal_size_and_vector_options_do_not_guess_dynamic_extents() {
    let input = fact(NumericClass::Double, vec![Some(1), None]);
    let size = call(
        vec![
            input.clone(),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(1), Some(2)]),
                StorageFact::Dense,
            ),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(4.0)]),
        ],
    );
    assert_eq!(size.outputs[0].shape, ShapeFact::Unknown);
    let vector = call(
        vec![input, ValueFact::scalar(ValueKindFact::Character)],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("vector".into()),
        ],
    );
    assert_eq!(vector.outputs[0].shape, ShapeFact::Unknown);
}
#[test]
fn invalid_input_and_offset_are_diagnosed() {
    assert!(!call(vec![], vec![]).diagnostics.is_empty());
    assert!(
        !call(vec![ValueFact::scalar(ValueKindFact::String)], vec![])
            .diagnostics
            .is_empty()
    );
    assert!(!call(
        vec![
            fact(NumericClass::Double, vec![Some(1), Some(2)]),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            }))
        ],
        vec![LiteralValue::Unknown, LiteralValue::Number(1.5)]
    )
    .diagnostics
    .is_empty());
}

#[test]
fn dynamic_controls_and_like_do_not_claim_a_real_dense_result() {
    let mut complex = fact(NumericClass::Single, vec![Some(1), Some(2)]);
    complex.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Complex,
    });
    let double = call(
        vec![complex.clone(), ValueFact::scalar(ValueKindFact::Character)],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("double".into()),
        ],
    );
    assert_eq!(
        double.outputs[0].numeric(),
        Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })
    );
    let dynamic = call(
        vec![
            complex.clone(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
    );
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Unknown);
    let mut resident_prototype = ValueFact::unknown(DynamicReason::RuntimeValue);
    resident_prototype.residency = ResidencyFact::Device {
        provider: Some("provider".into()),
    };
    let dynamic_option_then_prototype = call(
        vec![
            fact(NumericClass::Double, vec![Some(1), Some(2)]),
            ValueFact::unknown(DynamicReason::RuntimeValue),
            resident_prototype,
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
        ],
    );
    assert!(dynamic_option_then_prototype.diagnostics.is_empty());
    assert_eq!(
        dynamic_option_then_prototype.outputs[0].residency,
        ResidencyFact::Unknown
    );
    let like = call(
        vec![
            complex,
            ValueFact::scalar(ValueKindFact::Character),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("like".into()),
            LiteralValue::Unknown,
        ],
    );
    assert_eq!(like.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(like.outputs[0].storage, StorageFact::Unknown);
    let zero_then_one = call(
        vec![
            fact(NumericClass::Double, vec![Some(1), Some(2)]),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(0.0),
            LiteralValue::Number(1.0),
        ],
    );
    assert!(!zero_then_one.diagnostics.is_empty());
    let input = fact(NumericClass::Double, vec![Some(1), Some(2)]);
    for singleton in [
        LiteralValue::Vector(vec![LiteralValue::Number(1.0)]),
        LiteralValue::Matrix(vec![vec![LiteralValue::Number(1.0)]]),
    ] {
        let offset = call(
            vec![
                input.clone(),
                fact(NumericClass::Double, vec![Some(1), Some(1)]),
            ],
            vec![LiteralValue::Unknown, singleton],
        );
        assert!(offset.diagnostics.is_empty());
        assert_eq!(
            offset.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(3)])
        );
    }
}
