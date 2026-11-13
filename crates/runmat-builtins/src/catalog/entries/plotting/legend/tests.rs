use super::*;
use runmat_types::*;

#[test]
fn legend_inference_retains_host_graphics_handle_and_rejects_resident_controls() {
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &LEGEND_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert!(result.effects.0.contains(&EffectKind::UserInterface));

    let mut invalid = request;
    invalid.arguments[0].residency = ResidencyFact::Device { provider: None };
    assert!(inference::infer(&invalid, &LEGEND_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LEGEND-ARGUMENT"));
    assert_eq!(LEGEND_DESCRIPTOR.signatures.len(), 9);
    assert_eq!(documentation::DOCUMENTATION.examples.len(), 3);
    assert_eq!(documentation::DOCUMENTATION.faqs.len(), 3);
}
