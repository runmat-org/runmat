use super::*;

#[test]
fn literal_zero_is_pairwise_operand_but_not_reduction_axis() {
    let shape = ShapeFact::from(vec![Some(2), Some(3)]);
    let pairwise = call(
        numeric(NumericClass::Double, shape.clone()),
        vec![LiteralValue::Number(0.0)],
        1,
    );
    assert_eq!(pairwise.outputs[0].shape, shape);
    assert!(!pairwise
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-REDUCTION-DIMENSION"));

    let invalid = call(
        numeric(NumericClass::Double, shape),
        vec![LiteralValue::Empty, LiteralValue::Number(0.0)],
        1,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-REDUCTION-DIMENSION"));
}

#[test]
fn known_empty_placeholder_selects_reduction_and_unknown_rhs_does_not() {
    let shape = ShapeFact::from(vec![Some(3), Some(4)]);
    let reduction = call(
        numeric(NumericClass::Single, shape.clone()),
        vec![LiteralValue::Empty, LiteralValue::Number(2.0)],
        2,
    );
    assert_eq!(
        reduction.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert_eq!(
        reduction.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );

    let linear = call(
        numeric(NumericClass::Double, shape.clone()),
        vec![LiteralValue::Empty, LiteralValue::String("linear".into())],
        2,
    );
    assert_eq!(linear.outputs[0].shape, ShapeFact::Scalar);

    let unresolved = call(
        numeric(NumericClass::Single, shape),
        vec![LiteralValue::Unknown],
        2,
    );
    assert!(matches!(
        unresolved.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
    assert!(matches!(
        unresolved.outputs[1].certainty,
        CertaintyFact::Dynamic(_)
    ));
}

#[test]
fn zero_sized_nonplaceholder_rhs_does_not_claim_reduction() {
    let left = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(1)]),
    );
    let empty_shape = ShapeFact::from(vec![Some(0), Some(1)]);
    let object = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: None,
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: None,
        }),
        empty_shape.clone(),
        StorageFact::Opaque,
    );
    let mut sparse = numeric(NumericClass::Double, empty_shape);
    sparse.storage = StorageFact::Sparse;
    for right in [object, sparse] {
        let result = inference::infer(
            &CallRequest {
                arguments: vec![left.clone(), right],
                literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Unknown]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            },
            &MAX_CATALOG_ENTRY,
        );
        assert!(result
            .outputs
            .iter()
            .all(|output| matches!(output.certainty, CertaintyFact::Dynamic(_))));
    }
}

#[test]
fn conflicting_reduction_axes_leave_facts_dynamic() {
    let result = call(
        numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(3), Some(4)]),
        ),
        vec![
            LiteralValue::Empty,
            LiteralValue::Number(2.0),
            LiteralValue::String("all".into()),
        ],
        1,
    );
    assert!(matches!(
        result.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));

    let empty_then_all = call(
        numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(3), Some(4)]),
        ),
        vec![
            LiteralValue::Empty,
            LiteralValue::Empty,
            LiteralValue::String("all".into()),
        ],
        1,
    );
    assert!(matches!(
        empty_then_all.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));

    let placeholder_then_all = call(
        numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(3), Some(4)]),
        ),
        vec![LiteralValue::Empty, LiteralValue::String("all".into())],
        1,
    );
    assert_eq!(placeholder_then_all.outputs[0].shape, ShapeFact::Scalar);
}
