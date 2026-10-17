use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn cylinder_inference_tracks_profile_resolution_and_requested_outputs() {
    let radius = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let n = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let inferred = inference::infer(
        &request(
            vec![radius.clone(), n],
            vec![LiteralValue::Number(0.0), LiteralValue::Number(12.0)],
            3,
        ),
        &CYLINDER_CATALOG_ENTRY,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs.len(), 3);
    assert!(inferred.outputs.iter().all(|fact| fact.shape
        == ShapeFact::from(vec![Some(3), Some(13)])
        && fact.residency == ResidencyFact::Host));
    let plotted = inference::infer(&request(vec![radius], vec![], 0), &CYLINDER_CATALOG_ENTRY);
    assert!(plotted.outputs.is_empty());
    assert!(plotted.effects.0.contains(&EffectKind::UserInterface));
}

#[test]
fn cylinder_static_invalid_and_unknown_facts_do_not_claim_geometry() {
    let mut device = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    device.residency = ResidencyFact::Device { provider: None };
    assert!(
        !inference::infer(&request(vec![device], vec![], 1), &CYLINDER_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let inferred = inference::infer(&request(vec![unknown], vec![], 1), &CYLINDER_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![None, Some(21)])
    );
}
