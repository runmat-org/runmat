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
    assert!(!TEXT_DESCRIPTOR.signatures.is_empty());
    assert!(TEXT_DESCRIPTOR
        .signatures
        .iter()
        .any(|signature| signature.outputs.is_empty()));
    assert_eq!(TEXT_CATALOG_ENTRY.bindings.len(), 1);
    assert_eq!(TEXT_CATALOG_ENTRY.bindings[0].variant, "default");
    let accepted = inference::infer(
        &request(
            vec![real(), real(), ValueFact::scalar(ValueKindFact::String)],
            1,
        ),
        &TEXT_CATALOG_ENTRY,
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
        &request(
            vec![real(), real(), ValueFact::scalar(ValueKindFact::String)],
            0,
        ),
        &TEXT_CATALOG_ENTRY,
    );
    assert!(discarded.diagnostics.is_empty());
    let excess = inference::infer(
        &request(
            vec![real(), real(), ValueFact::scalar(ValueKindFact::String)],
            2,
        ),
        &TEXT_CATALOG_ENTRY,
    );
    assert!(!excess.diagnostics.is_empty());
}

#[test]
fn known_invalid_input_rejects_but_unknown_stays_conservative() {
    let invalid = inference::infer(
        &request(
            vec![
                ValueFact::scalar(ValueKindFact::Logical),
                real(),
                ValueFact::scalar(ValueKindFact::String),
            ],
            1,
        ),
        &TEXT_CATALOG_ENTRY,
    );
    assert!(!invalid.diagnostics.is_empty());
    let unknown = inference::infer(
        &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1),
        &TEXT_CATALOG_ENTRY,
    );
    assert!(!unknown.diagnostics.is_empty()); // Fewer than three coordinates/labels.
}
