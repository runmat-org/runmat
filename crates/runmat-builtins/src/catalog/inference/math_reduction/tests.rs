use super::*;
use runmat_types::{
    LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount,
};

fn request(shape: ShapeFact, literals: Vec<LiteralValue>) -> runmat_types::CallRequest {
    let argument_count = literals.len().max(1);
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    let mut arguments = vec![input];
    arguments.resize_with(argument_count, || {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    });
    runmat_types::CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn logical_reductions_preserve_rank_and_reduce_known_dimensions() {
    let shape = ShapeFact::from(vec![Some(2), Some(3), Some(4)]);
    let default = infer_logical(
        &request(shape.clone(), vec![LiteralValue::Unknown]),
        &crate::ALL_CATALOG_ENTRY,
        LogicalReductionKind::All,
    );
    assert_eq!(
        default.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3), Some(4)])
    );

    let along_two = infer_logical(
        &request(
            shape,
            vec![LiteralValue::Unknown, LiteralValue::Number(2.0)],
        ),
        &crate::ANY_CATALOG_ENTRY,
        LogicalReductionKind::Any,
    );
    assert_eq!(
        along_two.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1), Some(4)])
    );
}

#[test]
fn all_selector_produces_a_logical_scalar() {
    let result = infer_logical(
        &request(
            ShapeFact::from(vec![Some(2), Some(3)]),
            vec![LiteralValue::Unknown, LiteralValue::String("all".into())],
        ),
        &crate::ANY_CATALOG_ENTRY,
        LogicalReductionKind::Any,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
}

#[test]
fn vecdim_reduces_each_selected_dimension_without_changing_rank() {
    let result = infer_logical(
        &request(
            ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
            vec![
                LiteralValue::Unknown,
                LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(3.0)]),
            ],
        ),
        &crate::ALL_CATALOG_ENTRY,
        LogicalReductionKind::All,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3), Some(1)])
    );
}

#[test]
fn dynamic_dimension_preserves_rank_without_claiming_extents() {
    let result = infer_logical(
        &request(
            ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
            vec![LiteralValue::Unknown, LiteralValue::Unknown],
        ),
        &crate::ANY_CATALOG_ENTRY,
        LogicalReductionKind::Any,
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Ranked { rank: 3 });
}

#[test]
fn invalid_input_and_arity_are_diagnostics_not_invented_facts() {
    let mut request = runmat_types::CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 4]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let result = infer_logical(
        &request,
        &crate::ALL_CATALOG_ENTRY,
        LogicalReductionKind::All,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGICAL-REDUCTION-INPUT"));
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGICAL-REDUCTION-ARITY"));
    assert!(!result.diagnostics.is_empty());

    request.arguments.clear();
    let missing = infer_logical(
        &request,
        &crate::ANY_CATALOG_ENTRY,
        LogicalReductionKind::Any,
    );
    assert!(missing
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGICAL-REDUCTION-ARITY"));
}
