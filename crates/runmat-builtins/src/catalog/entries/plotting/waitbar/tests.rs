use super::*;
use runmat_types::*;

#[test]
fn waitbar_inference_validates_progress_and_preserves_scalar_handle() {
    let progress = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![progress.clone()],
        literals: LiteralContext::new(vec![LiteralValue::Number(0.25)]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &WAITBAR_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert!(result.effects.0.contains(&EffectKind::UserInterface));

    for invalid in [
        LiteralValue::Number(f64::INFINITY),
        LiteralValue::Number(f64::NAN),
    ] {
        let mut request = request.clone();
        request.literals = LiteralContext::new(vec![invalid]);
        assert!(inference::infer(&request, &WAITBAR_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-WAITBAR-PROGRESS"));
    }
    let mut resident = request;
    resident.arguments[0].residency = ResidencyFact::Device { provider: None };
    assert!(inference::infer(&resident, &WAITBAR_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-WAITBAR-PROGRESS"));
    assert_eq!(WAITBAR_EXTENSIONS.len(), 2);
    assert_eq!(WAITBAR_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(documentation::DOCUMENTATION.examples.len(), 1);
}
