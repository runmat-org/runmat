use super::*;

#[test]
fn log1p_contract_tracks_value_dependent_real_to_complex_promotion() {
    use runmat_types::{
        CallRequest, CertaintyFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
        NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
        ValueFact, ValueKindFact,
    };

    let entry = builtin_catalog_entry_by_name("log1p").expect("log1p entry");
    assert!(matches!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Logarithm(LogarithmKind::OnePlus))
    ));

    let infer = |input: ValueFact, literal: LiteralValue| {
        infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::new(vec![literal]),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        )
    };
    let single = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    let promoted = infer(single.clone(), LiteralValue::Number(-2.0));
    assert!(promoted.diagnostics.is_empty());
    assert_eq!(
        promoted.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
    let boundary = infer(single, LiteralValue::Number(-1.0));
    assert_eq!(
        boundary.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );

    let vector = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let vector_result = infer(
        vector,
        LiteralValue::Vector(vec![
            LiteralValue::Number(0.0),
            LiteralValue::Number(-2.0),
            LiteralValue::Number(1.0),
        ]),
    );
    assert_eq!(
        vector_result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })
    );
    assert_eq!(
        vector_result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );

    let signed_dynamic = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Int64,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(2), Some(4)]),
                StorageFact::Dense,
            )],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(signed_dynamic.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(
        signed_dynamic.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(4)])
    );
    assert_eq!(
        signed_dynamic.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
    );

    let unsigned = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }))],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(
        unsigned.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
}

#[test]
fn log1p_contract_rejects_unsupported_representations_without_fabricating_facts() {
    use runmat_types::{
        CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    let entry = builtin_catalog_entry_by_name("log1p").expect("log1p entry");
    let request_for = |input| CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let complex_integer = infer_catalog_call(
        entry,
        &request_for(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Complex,
        }))),
    );
    assert!(complex_integer
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOG1P-COMPLEX-INTEGER"));

    let sparse = infer_catalog_call(
        entry,
        &request_for(ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(3), Some(3)]),
            StorageFact::Sparse,
        )),
    );
    assert!(sparse
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOG1P-SPARSE"));

    let too_many = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(too_many
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOG1P-ARITY"));
}
