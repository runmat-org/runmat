use super::*;
use runmat_types::*;

#[test]
fn source_dependent_result_is_cell_row_and_arguments_reject() {
    let mut request = CallRequest {
        arguments: Vec::new(),
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &LOCALFUNCTIONS_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("cell output")
    };
    assert!(matches!(cell.element.kind, ValueKindFact::Callable(_)));
    request
        .arguments
        .push(ValueFact::scalar(ValueKindFact::Logical));
    assert!(!inference::infer(&request, &LOCALFUNCTIONS_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
