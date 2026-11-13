use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn fid() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn text_and_file_forms_keep_host_cell_fact_and_distinct_capabilities() {
    let text_call = inference::infer(&request(vec![text(), text()], 1), &TEXTSCAN_CATALOG_ENTRY);
    assert!(text_call.diagnostics.is_empty());
    assert!(matches!(text_call.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(
        text_call.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(text_call.outputs[0].residency, ResidencyFact::Host);
    assert!(!text_call
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(!text_call.effects.0.contains(&EffectKind::FilesystemRead));

    let file_call = inference::infer(&request(vec![fid(), text()], 1), &TEXTSCAN_CATALOG_ENTRY);
    assert!(file_call.diagnostics.is_empty());
    assert!(file_call
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));

    let dynamic = inference::infer(
        &request(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue), text()],
            1,
        ),
        &TEXTSCAN_CATALOG_ENTRY,
    );
    assert!(dynamic
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));

    // Catalog-only summaries have no first-argument fact to narrow the file form.
    assert!(TEXTSCAN_CATALOG_ENTRY
        .contract
        .capability_set()
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn known_invalid_roles_and_output_arity_are_diagnosed_without_guessing_format() {
    let bad_input = inference::infer(
        &request(vec![ValueFact::scalar(ValueKindFact::Logical), text()], 1),
        &TEXTSCAN_CATALOG_ENTRY,
    );
    assert!(!bad_input.diagnostics.is_empty());
    let bad_format = inference::infer(&request(vec![text(), fid()], 1), &TEXTSCAN_CATALOG_ENTRY);
    assert!(!bad_format.diagnostics.is_empty());
    let too_many = inference::infer(&request(vec![text(), text()], 2), &TEXTSCAN_CATALOG_ENTRY);
    assert!(!too_many.diagnostics.is_empty());
    let unknown_option = inference::infer(
        &request(
            vec![
                text(),
                text(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
                fid(),
            ],
            1,
        ),
        &TEXTSCAN_CATALOG_ENTRY,
    );
    assert!(unknown_option.diagnostics.is_empty());
}
