use super::*;
use runmat_types::*;

#[test]
fn hierarchy_result_is_host_cell_of_character_rows_not_strings() {
    let mut request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let output = &inference::infer(&request, &SUPERCLASSES_CATALOG_ENTRY).outputs[0];
    assert_eq!(output.residency, ResidencyFact::Host);
    assert_eq!(output.storage, StorageFact::Dense);
    assert_eq!(output.shape, ShapeFact::from(vec![Some(1), None]));
    let ValueKindFact::Cell(cell) = &output.kind else {
        panic!("cell");
    };
    assert_eq!(cell.element.kind, ValueKindFact::Character);
    assert!(output
        .invalidation
        .0
        .contains(&InvalidationCause::ClassHierarchyChanged));
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&request, &SUPERCLASSES_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
