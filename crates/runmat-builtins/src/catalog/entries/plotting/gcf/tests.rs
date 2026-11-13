use super::*;
use runmat_types::*;

#[test]
fn gcf_inference_is_host_scalar_with_conditional_creation_effects() {
    for count in [0, 1, 2] {
        let request = CallRequest {
            arguments: vec![],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let result = inference::infer(&request, &GCF_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), count > 1);
        if count == 1 {
            assert_eq!(
                result.outputs[0],
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }))
            );
        }
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
        assert!(result.effects.0.contains(&EffectKind::HostCallback));
    }
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::default(),
    };
    assert!(!inference::infer(&request, &GCF_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
