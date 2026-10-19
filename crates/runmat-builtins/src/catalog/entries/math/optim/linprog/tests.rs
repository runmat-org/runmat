use super::*;
use runmat_types::*;

#[test]
fn positional_facts_and_bounded_outputs_preserve_failure_geometry() {
    let real = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let empty = ValueFact::proven(
        real.kind.clone(),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    let mut request = CallRequest {
        arguments: vec![real, empty.clone(), empty],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(4)),
    };
    let result = inference::infer(&request, &LINPROG_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    assert_eq!(result.outputs[1].storage, StorageFact::Unknown);
    assert!(matches!(result.outputs[3].kind, ValueKindFact::Struct(_)));
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(usize::MAX));
    let invalid = inference::infer(&request, &LINPROG_CATALOG_ENTRY);
    assert!(invalid.outputs.is_empty());
    assert!(!invalid.diagnostics.is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(1));
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&request, &LINPROG_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
