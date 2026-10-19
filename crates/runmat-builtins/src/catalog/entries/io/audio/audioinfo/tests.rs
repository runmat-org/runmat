use super::*;
use runmat_types::*;

fn request(input: ValueFact, literal: Option<&str>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![input],
        literals: LiteralContext::new(vec![
            literal.map_or(LiteralValue::Unknown, |s| LiteralValue::String(s.into()))
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn metadata_fields_and_effects_are_typed() {
    let inferred = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::String),
            Some("sample.wav"),
            1,
        ),
        &AUDIOINFO_CATALOG_ENTRY,
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    let ValueKindFact::Struct(fields) = &inferred.outputs[0].kind else {
        panic!("metadata should be struct")
    };
    assert!(fields.fields.contains_key("SampleRate"));
    assert!(fields.fields.contains_key("Format"));
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
    assert!(inferred.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(inferred
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn discarded_metadata_retains_filesystem_effect_and_validation() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let inferred = inference::infer(
        &request(text.clone(), Some("sample.wav"), 0),
        &AUDIOINFO_CATALOG_ENTRY,
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert!(inferred.outputs.is_empty());
    assert!(inferred.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(inferred
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(
        !inference::infer(&request(text, Some(" "), 0), &AUDIOINFO_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}

#[test]
fn invalid_scalar_text_and_output_count_are_diagnosed() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(
        !inference::infer(&request(numeric, None, 1), &AUDIOINFO_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let text = ValueFact::scalar(ValueKindFact::String);
    assert!(!inference::infer(
        &request(text.clone(), Some(" "), 1),
        &AUDIOINFO_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    assert!(
        !inference::infer(&request(text, Some("x.wav"), 2), &AUDIOINFO_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
