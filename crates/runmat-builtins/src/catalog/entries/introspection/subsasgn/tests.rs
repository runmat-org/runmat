use super::*;
use runmat_types::*;

#[test]
fn unknown_or_overloaded_assignment_does_not_preserve_stale_receiver_shape() {
    let mut request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); 3],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    request.arguments[0].shape = ShapeFact::from(vec![Some(1), Some(2)]);
    let result = inference::infer(&request, &SUBSASGN_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    assert!(result.effects.0.contains(&EffectKind::Unknown));
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!inference::infer(&request, &SUBSASGN_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
