use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: LiteralContext) -> CallRequest {
    CallRequest {
        arguments,
        literals,
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn scalar_text_and_unknown_paths_have_character_row_result_and_filesystem_capability() {
    for input in [
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    ] {
        let inferred = inference::infer(
            &request(vec![input], LiteralContext::default()),
            &FILEREAD_CATALOG_ENTRY,
        );
        assert!(inferred.diagnostics.is_empty());
        assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Character));
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
        assert!(inferred
            .capabilities
            .0
            .contains(&CapabilityRequirement::Filesystem));
    }
}

#[test]
fn known_invalid_path_and_named_option_are_diagnosed() {
    let invalid = inference::infer(
        &request(
            vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))],
            LiteralContext::default(),
        ),
        &FILEREAD_CATALOG_ENTRY,
    );
    assert!(!invalid.diagnostics.is_empty());
    let text = ValueFact::scalar(ValueKindFact::String);
    let bad_option = inference::infer(
        &request(
            vec![text.clone(), text.clone(), text],
            LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("Other".into()),
            ]),
        ),
        &FILEREAD_CATALOG_ENTRY,
    );
    assert!(!bad_option.diagnostics.is_empty());
}
