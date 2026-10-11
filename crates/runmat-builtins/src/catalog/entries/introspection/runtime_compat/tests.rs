use super::*;
use runmat_types::*;
#[test]
fn unknown_version_controls_remain_dynamic_inputs_with_logical_output() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_ver_less_than(&request, &VER_LESS_THAN_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
}
#[test]
fn compatibility_forms_have_distinct_outputs_and_errors() {
    let zero = CallRequest {
        arguments: vec![],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    assert_eq!(
        inference::infer_isdeployed(&zero, &ISDEPLOYED_CATALOG_ENTRY).outputs[0].kind,
        ValueKindFact::Logical
    );
    assert_eq!(
        inference::infer_version(&zero, &VERSION_CATALOG_ENTRY).outputs[0].kind,
        ValueKindFact::String
    );
    let bad = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
        ..zero
    };
    assert!(
        !inference::infer_ver_less_than(&bad, &VER_LESS_THAN_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}

#[test]
fn version_does_not_claim_two_outputs() {
    let request = CallRequest {
        arguments: vec![],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let result = inference::infer_version(&request, &VERSION_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
}
