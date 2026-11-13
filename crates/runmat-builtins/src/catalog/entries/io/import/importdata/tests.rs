use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

#[test]
fn external_payload_remains_dynamic_but_delimiter_and_header_classes_are_known() {
    let result = inference::infer(
        &request(vec![text()], vec![LiteralValue::Unknown], 3),
        &IMPORTDATA_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 3);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(
        result.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::ExternalData)
    );
    assert_eq!(result.outputs[1].kind, ValueKindFact::Character);
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(
        result.outputs[2].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    );
    assert!(result.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn known_bad_controls_reject_without_guessing_file_contents() {
    let invalid_path = request(
        vec![ValueFact::scalar(ValueKindFact::Logical)],
        vec![LiteralValue::Unknown],
        1,
    );
    assert!(!inference::infer(&invalid_path, &IMPORTDATA_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let invalid_header = request(
        vec![
            text(),
            text(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(-1.0),
        ],
        1,
    );
    assert!(
        !inference::infer(&invalid_header, &IMPORTDATA_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let dynamic_header = request(
        vec![
            text(),
            text(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        vec![LiteralValue::Unknown; 3],
        1,
    );
    assert!(inference::infer(&dynamic_header, &IMPORTDATA_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let paged_name = request(
        vec![ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(1), Some(2)]),
            StorageFact::Dense,
        )],
        vec![LiteralValue::Unknown],
        1,
    );
    assert!(!inference::infer(&paged_name, &IMPORTDATA_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
