use super::*;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, CellFact, DimensionFact, DynamicReason, LiteralContext, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(input: ValueFact) -> runmat_types::CallInference {
    infer_catalog_call(
        &CELLSTR_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

fn shaped(kind: ValueKindFact, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        kind,
        ShapeFact::Shaped {
            dims: dims.iter().copied().map(DimensionFact::Known).collect(),
        },
        StorageFact::Dense,
    )
}

#[test]
fn character_rows_become_a_cell_column() {
    let result = infer(shaped(ValueKindFact::Character, &[3, 8]));
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(3), DimensionFact::Known(1)]
        }
    );
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("expected cell output");
    };
    assert_eq!(cell.element.kind, ValueKindFact::Character);
}

#[test]
fn string_and_cell_inputs_preserve_outer_shape() {
    let shape = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(2), DimensionFact::Known(3)],
    };
    let string = infer(ValueFact::proven(
        ValueKindFact::String,
        shape.clone(),
        StorageFact::Dense,
    ));
    assert_eq!(string.outputs[0].shape, shape);

    let cell = infer(ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::scalar(ValueKindFact::String)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape.clone(),
        StorageFact::Dense,
    ));
    assert_eq!(cell.outputs[0].shape, shape);
    assert!(cell.diagnostics.is_empty());
}

#[test]
fn known_invalid_input_and_cell_contents_are_diagnostics() {
    assert_eq!(
        infer(ValueFact::scalar(ValueKindFact::Logical))
            .diagnostics
            .len(),
        1
    );
    let bad_cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::scalar(ValueKindFact::Logical)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    assert_eq!(infer(bad_cell).diagnostics.len(), 1);

    let nonscalar_string = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(shaped(ValueKindFact::String, &[1, 2])),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    assert_eq!(infer(nonscalar_string).diagnostics.len(), 1);
}

#[test]
fn dynamic_input_retains_a_conservative_cell_contract() {
    let result = infer(ValueFact::unknown(DynamicReason::RuntimeValue));
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn unknown_character_row_count_is_not_rejected_early() {
    let character = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![None, Some(8)]),
                StorageFact::Dense,
            )),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    assert!(infer(character).diagnostics.is_empty());
}
