use super::*;
use crate::infer_catalog_call;
use runmat_types::{
    CallInference, CallRequest, CellFact, LiteralContext, LiteralValue, OutputSelection,
    RequestedOutputCount,
};

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    infer_catalog_call(
        &CELL_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn literal_sizes_produce_exact_cell_and_empty_element_shapes() {
    let result = infer(
        vec![numeric(vec![Some(1), Some(3)])],
        vec![LiteralValue::Vector(vec![
            LiteralValue::Number(2.0),
            LiteralValue::Number(3.0),
            LiteralValue::Number(1.0),
        ])],
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    let ValueKindFact::Cell(CellFact { element, .. }) = &result.outputs[0].kind else {
        panic!("expected cell fact");
    };
    assert_eq!(element.shape, ShapeFact::from(vec![Some(0), Some(0)]));
}

#[test]
fn like_without_sizes_uses_prototype_shape_and_element_representation() {
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(4)]),
        StorageFact::Dense,
    );
    let result = infer(
        vec![character_row(), logical],
        vec![LiteralValue::Keyword("like".into()), LiteralValue::Unknown],
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(4)])
    );
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("expected cell");
    };
    assert!(matches!(cell.element.kind, ValueKindFact::Logical));
}

#[test]
fn invalid_size_and_like_forms_are_diagnostics() {
    let fractional = infer(
        vec![numeric(vec![Some(1), Some(1)])],
        vec![LiteralValue::Number(2.5)],
    );
    assert!(!fractional.diagnostics.is_empty());
    let missing = infer(
        vec![character_row()],
        vec![LiteralValue::Keyword("like".into())],
    );
    assert!(!missing.diagnostics.is_empty());
    let trailing = infer(
        vec![
            character_row(),
            numeric(vec![Some(1), Some(1)]),
            numeric(vec![Some(1), Some(1)]),
        ],
        vec![
            LiteralValue::Keyword("like".into()),
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
        ],
    );
    assert!(!trailing.diagnostics.is_empty());
}

#[test]
fn like_element_facts_describe_new_empty_values_not_prototype_payloads() {
    let string_array = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let string_result = infer(
        vec![character_row(), string_array],
        vec![LiteralValue::Keyword("like".into()), LiteralValue::Unknown],
    );
    let ValueKindFact::Cell(string_cell) = &string_result.outputs[0].kind else {
        panic!("expected cell");
    };
    assert_eq!(
        string_cell.element.shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );

    let prototype_cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(numeric(vec![Some(1), Some(1)])),
            elements: vec![numeric(vec![Some(1), Some(1)])],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    let nested_result = infer(
        vec![character_row(), prototype_cell],
        vec![LiteralValue::Keyword("like".into()), LiteralValue::Unknown],
    );
    let ValueKindFact::Cell(outer) = &nested_result.outputs[0].kind else {
        panic!("expected outer cell");
    };
    let ValueKindFact::Cell(inner) = &outer.element.kind else {
        panic!("expected empty cell element");
    };
    assert!(inner.elements.is_empty());
    assert!(inner.elements_complete);
}
