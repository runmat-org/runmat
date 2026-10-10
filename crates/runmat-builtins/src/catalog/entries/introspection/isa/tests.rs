use super::*;
use runmat_types::*;

fn infer(designator: ValueFact) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue), designator],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &ISA_CATALOG_ENTRY,
    )
}

#[test]
fn known_unknown_and_invalid_designator_geometry() {
    let scalar = infer(ValueFact::scalar(ValueKindFact::String));
    assert!(scalar.diagnostics.is_empty());
    assert_eq!(scalar.outputs[0], ValueFact::scalar(ValueKindFact::Logical));
    let unknown = infer(ValueFact::unknown(DynamicReason::RuntimeValue));
    assert!(unknown.diagnostics.is_empty());
    let invalid = infer(ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(4)]),
        StorageFact::Dense,
    ));
    assert!(!invalid.diagnostics.is_empty());
    assert_eq!(invalid.outputs[0].kind, ValueKindFact::Unknown);
}
