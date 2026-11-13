use super::*;
use runmat_types::*;
fn sample() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(10), Some(1)]),
        StorageFact::Dense,
    )
}
#[test]
fn double_estimates_positive_output_effects_and_native_point_class_are_truthful() {
    let mut request = CallRequest {
        arguments: vec![sample()],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
    };
    let result = inference::infer(&request, &KSDENSITY_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(100)])
    );
    for output in &result.outputs {
        assert!(matches!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        ));
    }
    assert!(!result.effects.0.contains(&EffectKind::UserInterface));
    assert!(!result
        .capabilities
        .0
        .contains(&CapabilityRequirement::UserInterface));
    request.arguments.push(sample());
    let result = inference::infer(&request, &KSDENSITY_CATALOG_ENTRY);
    assert_eq!(result.outputs[1].kind, request.arguments[1].kind);
    assert_eq!(result.outputs[0].shape, request.arguments[1].shape);
    request.arguments[1].shape = ShapeFact::Unknown;
    assert!(matches!(
        inference::infer(&request, &KSDENSITY_CATALOG_ENTRY).outputs[1].kind,
        ValueKindFact::Unknown
    ));
    request.outputs = OutputSelection::new(RequestedOutputCount::Zero);
    let result = inference::infer(&request, &KSDENSITY_CATALOG_ENTRY);
    assert!(result.effects.0.contains(&EffectKind::UserInterface));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::UserInterface));
    assert_eq!(KsdensityOptionName::from_public("Weights"), None);
    assert_eq!(KsdensityFunction::from_public("icdf"), None);
    assert_eq!(KSDENSITY_CATALOG_ENTRY.documentation.examples.len(), 2);
}
