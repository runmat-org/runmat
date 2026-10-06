use super::*;
use runmat_types::*;

fn real() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn request(arguments: Vec<ValueFact>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn descriptor_and_binding_cover_public_output_forms() {
    assert!(!SUBTITLE_DESCRIPTOR.signatures.is_empty());
    assert!(SUBTITLE_DESCRIPTOR
        .signatures
        .iter()
        .any(|signature| signature.outputs.is_empty()));
    assert_eq!(SUBTITLE_CATALOG_ENTRY.bindings.len(), 1);
    assert_eq!(SUBTITLE_CATALOG_ENTRY.bindings[0].variant, "default");
    let accepted = inference::infer(
        &request(vec![ValueFact::scalar(ValueKindFact::String)], 1),
        &SUBTITLE_CATALOG_ENTRY,
    );
    assert!(
        accepted.diagnostics.is_empty(),
        "{:?}",
        accepted.diagnostics
    );
    assert_eq!(accepted.outputs[0].kind, real().kind);
    assert!(accepted.effects.0.contains(&EffectKind::UserInterface));
    assert!(accepted.effects.0.contains(&EffectKind::HostCallback));
    let discarded = inference::infer(
        &request(vec![ValueFact::scalar(ValueKindFact::String)], 0),
        &SUBTITLE_CATALOG_ENTRY,
    );
    assert!(discarded.diagnostics.is_empty());
    let excess = inference::infer(
        &request(vec![ValueFact::scalar(ValueKindFact::String)], 2),
        &SUBTITLE_CATALOG_ENTRY,
    );
    assert!(!excess.diagnostics.is_empty());
}

#[test]
fn known_invalid_input_rejects_but_unknown_stays_conservative() {
    let invalid = inference::infer(
        &request(vec![ValueFact::scalar(ValueKindFact::Logical)], 1),
        &SUBTITLE_CATALOG_ENTRY,
    );
    assert!(!invalid.diagnostics.is_empty());
    let unknown = inference::infer(
        &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1),
        &SUBTITLE_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty(), "{:?}", unknown.diagnostics);
}

#[test]
fn known_resident_control_rejects_before_graphics_execution() {
    let mut resident = ValueFact::scalar(ValueKindFact::String);
    for residency in [
        ResidencyFact::Device { provider: None },
        ResidencyFact::Remote { pool: None },
    ] {
        resident.residency = residency;
        let result = inference::infer(&request(vec![resident.clone()], 1), &SUBTITLE_CATALOG_ENTRY);
        assert!(!result.diagnostics.is_empty());
    }
}
