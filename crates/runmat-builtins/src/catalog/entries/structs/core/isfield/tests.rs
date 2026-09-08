use super::*;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, CellFact, DimensionFact, DynamicReason, LiteralContext, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    infer_catalog_call(
        &ISFIELD_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

#[test]
fn scalar_text_query_produces_a_logical_scalar() {
    let result = infer(vec![unknown(), ValueFact::scalar(ValueKindFact::String)]);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].storage, StorageFact::Scalar);
}

#[test]
fn string_and_cell_collections_preserve_query_shape() {
    let shape = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(2), DimensionFact::Known(3)],
    };
    let strings = ValueFact::proven(ValueKindFact::String, shape.clone(), StorageFact::Dense);
    assert_eq!(infer(vec![unknown(), strings]).outputs[0].shape, shape);

    let cells = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::scalar(ValueKindFact::Character)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), Some(4)]),
        StorageFact::Dense,
    );
    assert_eq!(
        infer(vec![unknown(), cells]).outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
}

#[test]
fn invalid_name_types_and_character_matrices_are_diagnostics() {
    let logical = infer(vec![unknown(), ValueFact::scalar(ValueKindFact::Logical)]);
    assert!(!logical.diagnostics.is_empty());
    let matrix = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    assert!(!infer(vec![unknown(), matrix]).diagnostics.is_empty());
}
