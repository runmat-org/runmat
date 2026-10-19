use super::*;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

#[test]
fn literal_partitions_produce_exact_cell_grid_and_typed_element() {
    let request = CallRequest {
        arguments: vec![
            numeric(vec![Some(4), Some(4)]),
            numeric(vec![Some(1), Some(2)]),
            numeric(vec![Some(1), Some(2)]),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(2.0)]),
            LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(3.0)]),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(&MAT2CELL_CATALOG_ENTRY, &request);
    assert!(inference.diagnostics.is_empty());
    assert_eq!(
        inference.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    let ValueKindFact::Cell(cell) = &inference.outputs[0].kind else {
        panic!("cell output")
    };
    assert!(matches!(
        cell.element.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            ..
        })
    ));
}

#[test]
fn missing_partition_is_diagnosed() {
    let request = CallRequest {
        arguments: vec![numeric(vec![Some(2), Some(2)])],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    assert!(!infer_catalog_call(&MAT2CELL_CATALOG_ENTRY, &request)
        .diagnostics
        .is_empty());
}
