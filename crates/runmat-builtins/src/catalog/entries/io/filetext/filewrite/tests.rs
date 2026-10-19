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
fn native_write_has_double_count_and_filesystem_effect() {
    let inferred = inference::infer(
        &request(vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::UInt64,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(1), Some(4)]),
                StorageFact::Dense,
            ),
        ]),
        &FILEWRITE_CATALOG_ENTRY,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert!(inferred
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(inferred.effects.0.contains(&EffectKind::FilesystemWrite));
}

#[test]
fn invalid_controls_and_complex_data_are_diagnosed() {
    let bad_path = inference::infer(
        &request(vec![
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::String),
        ]),
        &FILEWRITE_CATALOG_ENTRY,
    );
    assert!(!bad_path.diagnostics.is_empty());
    let bad_data = inference::infer(
        &request(vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            })),
        ]),
        &FILEWRITE_CATALOG_ENTRY,
    );
    assert!(!bad_data.diagnostics.is_empty());
}

#[test]
fn character_row_path_and_dynamic_data_remain_admitted() {
    let inferred = inference::infer(
        &request(vec![
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(1), Some(9)]),
                StorageFact::Dense,
            ),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ]),
        &FILEWRITE_CATALOG_ENTRY,
    );
    assert!(inferred.diagnostics.is_empty());
}

fn option_request(literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String); literals.len()],
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn known_invalid_option_grammar_is_diagnosed_without_rejecting_dynamic_forms() {
    let prefix = [LiteralValue::Unknown, LiteralValue::Unknown];
    for suffix in [
        vec![LiteralValue::String("Encoding".into())],
        vec![
            LiteralValue::String("bogus".into()),
            LiteralValue::String("ascii".into()),
        ],
        vec![
            LiteralValue::String("Encoding".into()),
            LiteralValue::String("wrong".into()),
        ],
        vec![
            LiteralValue::String("Encoding".into()),
            LiteralValue::String("ascii".into()),
            LiteralValue::String("Encoding".into()),
            LiteralValue::String("raw".into()),
        ],
    ] {
        let mut literals = prefix.to_vec();
        literals.extend(suffix);
        assert!(
            !inference::infer(&option_request(literals), &FILEWRITE_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
    }
    for suffix in [
        vec![LiteralValue::Unknown],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
        ],
    ] {
        let mut literals = prefix.to_vec();
        literals.extend(suffix);
        assert!(
            inference::infer(&option_request(literals), &FILEWRITE_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
    }
}
