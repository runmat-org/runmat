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
                    class: NumericClass::Uint64,
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
