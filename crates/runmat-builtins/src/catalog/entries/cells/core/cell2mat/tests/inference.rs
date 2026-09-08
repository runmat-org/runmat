use super::*;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, CellFact, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn request(argument: ValueFact) -> CallRequest {
    CallRequest {
        arguments: vec![argument],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn cell(element: ValueFact, shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

#[test]
fn numeric_cells_preserve_known_class_and_domain() {
    let numeric = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer_catalog_call(
        &CELL2MAT_CATALOG_ENTRY,
        &request(cell(numeric, vec![Some(1), Some(2)])),
    );
    assert!(result.diagnostics.is_empty());
    assert!(matches!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real
        })
    ));
}

#[test]
fn empty_cell_shape_is_exact_double_empty() {
    let unknown = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    let output = &infer_catalog_call(
        &CELL2MAT_CATALOG_ENTRY,
        &request(cell(unknown, vec![Some(0), Some(0)])),
    )
    .outputs[0];
    assert_eq!(output.shape, ShapeFact::from(vec![Some(0), Some(0)]));
}

#[test]
fn non_cell_input_is_diagnosed() {
    let input = ValueFact::scalar(ValueKindFact::Logical);
    assert!(
        !infer_catalog_call(&CELL2MAT_CATALOG_ENTRY, &request(input))
            .diagnostics
            .is_empty()
    );
}
