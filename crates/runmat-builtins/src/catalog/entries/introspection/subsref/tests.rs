use super::*;
use runmat_types::*;

#[test]
fn unknown_paths_and_overloads_do_not_invent_output_facts() {
    let mut request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
    };
    let result = inference::infer(&request, &SUBSREF_CATALOG_ENTRY);
    assert_eq!(result.outputs.len(), 3);
    assert!(result
        .outputs
        .iter()
        .all(|v| v.kind == ValueKindFact::Unknown && v.residency == ResidencyFact::Unknown));
    assert!(result.effects.0.contains(&EffectKind::Unknown));
    request.arguments[1] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&request, &SUBSREF_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    let result = inference::infer(&request, &SUBSREF_CATALOG_ENTRY);
    assert!(result.outputs.is_empty());
    assert!(result.effects.0.contains(&EffectKind::HostCallback));
}
