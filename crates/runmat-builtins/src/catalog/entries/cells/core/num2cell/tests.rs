use super::*;
use crate::{infer_catalog_call, BuiltinInferenceRule};
use runmat_types::{
    CallRequest, DimensionFact, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, ValueFact, ValueKindFact,
};

fn numeric(shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        runmat_types::StorageFact::Dense,
    )
}

#[test]
fn entry_owns_identity_inference_and_catalog_documentation() {
    assert!(matches!(
        NUM2CELL_CATALOG_ENTRY.contract.inference_rule,
        BuiltinInferenceRule::Identity(_)
    ));
    assert_eq!(NUM2CELL_CATALOG_ENTRY.documentation.examples.len(), 7);
    assert!(NUM2CELL_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !matches!(
            example.verification,
            crate::BuiltinExampleVerification::Succeeds
        )));
}

#[test]
fn scalar_partition_preserves_outer_shape_and_numeric_element_kind() {
    let request = CallRequest {
        arguments: vec![numeric(vec![Some(2), Some(3)])],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let output = &infer_catalog_call(&NUM2CELL_CATALOG_ENTRY, &request).outputs[0];
    assert_eq!(output.shape, ShapeFact::from(vec![Some(2), Some(3)]));
    let ValueKindFact::Cell(cell) = &output.kind else {
        panic!("expected cell fact");
    };
    assert!(matches!(cell.element.kind, ValueKindFact::Numeric(_)));
    assert_eq!(cell.element.shape, ShapeFact::Scalar);
}

#[test]
fn literal_dimension_order_produces_exact_outer_and_block_shapes() {
    let request = CallRequest {
        arguments: vec![
            numeric(vec![Some(2), Some(3)]),
            numeric(vec![Some(1), Some(2)]),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(1.0)]),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(&NUM2CELL_CATALOG_ENTRY, &request);
    assert!(inference.diagnostics.is_empty());
    let output = &inference.outputs[0];
    assert_eq!(
        output.shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(1)]
        }
    );
    let ValueKindFact::Cell(cell) = &output.kind else {
        panic!("expected cell fact");
    };
    assert_eq!(cell.element.shape, ShapeFact::from(vec![Some(3), Some(2)]));
}

#[test]
fn matrix_and_duplicate_dimension_literals_are_diagnosed() {
    for literal in [
        LiteralValue::Matrix(vec![
            vec![LiteralValue::Number(1.0), LiteralValue::Number(2.0)],
            vec![LiteralValue::Number(2.0), LiteralValue::Number(1.0)],
        ]),
        LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(1.0)]),
        LiteralValue::Vector(vec![LiteralValue::Number(-1.0)]),
        LiteralValue::Vector(vec![LiteralValue::Number(1.5)]),
        LiteralValue::Vector(vec![LiteralValue::String("1".into())]),
    ] {
        let request = CallRequest {
            arguments: vec![numeric(vec![Some(2), Some(2)]), numeric(vec![None, None])],
            literals: LiteralContext::new(vec![LiteralValue::Unknown, literal]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        assert!(!infer_catalog_call(&NUM2CELL_CATALOG_ENTRY, &request)
            .diagnostics
            .is_empty());
    }
}

#[test]
fn singleton_degenerate_dimension_shapes_are_admitted_at_any_rank() {
    let request = CallRequest {
        arguments: vec![
            numeric(vec![Some(2), Some(3), Some(4)]),
            numeric(vec![Some(1), Some(2), Some(1)]),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Vector(vec![LiteralValue::Number(2.0)]),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    assert!(infer_catalog_call(&NUM2CELL_CATALOG_ENTRY, &request)
        .diagnostics
        .is_empty());
}
