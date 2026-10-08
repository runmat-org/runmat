use super::*;

#[test]
fn known_pairwise_form_proves_scalar_class_and_rejects_second_output() {
    let result = call(
        numeric(NumericClass::Double, ShapeFact::Scalar),
        vec![LiteralValue::Number(2.0)],
        2,
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn pairwise_same_class_broadcast_is_proven_after_typed_option_admission() {
    let mut request = CallRequest {
        arguments: vec![
            numeric(
                NumericClass::UInt64,
                ShapeFact::from(vec![Some(1), Some(3)]),
            ),
            numeric(
                NumericClass::UInt64,
                ShapeFact::from(vec![Some(2), Some(1)]),
            ),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("ComparisonMethod".into()),
            LiteralValue::String("magnitude".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &MAX_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );

    request.arguments.truncate(3);
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("omitnan".into()),
    ]);
    let admitted = inference::infer(&request, &MAX_CATALOG_ENTRY);
    assert_eq!(
        admitted.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        admitted.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );

    request
        .arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("omitnan".into()),
        LiteralValue::Unknown,
    ]);
    let invalid = inference::infer(&request, &MAX_CATALOG_ENTRY);
    assert!(matches!(
        invalid.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
}

#[test]
fn pairwise_known_shape_mismatch_is_diagnosed_without_proving_result() {
    let request = CallRequest {
        arguments: vec![
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(2), Some(3)]),
            ),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(4), Some(3)]),
            ),
        ],
        literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Unknown]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &MAX_CATALOG_ENTRY);
    assert!(matches!(
        result.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-BROADCAST"));
}

#[test]
fn same_class_single_pairwise_preserves_class_with_unknown_or_scalar_shape() {
    for left_shape in [ShapeFact::Unknown, ShapeFact::Scalar] {
        let request = CallRequest {
            arguments: vec![
                numeric(NumericClass::Single, left_shape.clone()),
                numeric(NumericClass::Single, ShapeFact::Scalar),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Real {
                    text: "2".into(),
                    class: NumericClass::Single,
                },
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let result = inference::infer(&request, &MAX_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
        assert_eq!(result.outputs[0].shape, left_shape);
    }
}
