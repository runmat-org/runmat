use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn formatted_write_has_double_byte_count_and_transitive_effects() {
    let inferred = inference::infer(
        &request(vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })),
        ]),
        &FPRINTF_CATALOG_ENTRY,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert!(inferred.effects.0.contains(&EffectKind::UserInterface));
    assert!(!inferred.effects.0.contains(&EffectKind::FilesystemWrite));
    assert!(!inferred
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));

    let file = inference::infer(
        &request(vec![
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            ValueFact::scalar(ValueKindFact::String),
        ]),
        &FPRINTF_CATALOG_ENTRY,
    );
    assert!(file.effects.0.contains(&EffectKind::FilesystemWrite));
    assert!(file
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn literal_standard_stream_ids_do_not_require_filesystem() {
    for id in [1.0, 2.0] {
        let mut request = request(vec![
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            ValueFact::scalar(ValueKindFact::String),
        ]);
        request.literals.literal_args = vec![LiteralValue::Number(id), LiteralValue::Unknown];
        let inferred = inference::infer(&request, &FPRINTF_CATALOG_ENTRY);
        assert!(!inferred.effects.0.contains(&EffectKind::FilesystemWrite));
        assert!(!inferred
            .capabilities
            .0
            .contains(&CapabilityRequirement::Filesystem));
    }
}

#[test]
fn absent_or_definitely_unformattable_arguments_are_diagnosed() {
    assert!(
        !inference::infer(&request(Vec::new()), &FPRINTF_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let invalid = inference::infer(
        &request(vec![ValueFact::scalar(ValueKindFact::Void)]),
        &FPRINTF_CATALOG_ENTRY,
    );
    assert!(!invalid.diagnostics.is_empty());
}

#[test]
fn dynamic_format_position_is_not_guessed() {
    let inferred = inference::infer(
        &request(vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::String),
        ]),
        &FPRINTF_CATALOG_ENTRY,
    );
    assert!(inferred.diagnostics.is_empty());
}
