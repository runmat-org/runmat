use super::*;
use runmat_types::*;
#[test]
fn unknown_clearvars_word_retains_dynamic_option_admission() {
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_clearvars(&request, &CLEARVARS_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
}
#[test]
fn clearvars_infers_empty_host_result_and_rejects_numeric_target() {
    let request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_clearvars(&request, &CLEARVARS_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    assert!(!result.diagnostics.is_empty());
}
