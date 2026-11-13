use super::*;
use runmat_types::*;
use std::collections::BTreeSet;
#[test]
fn metadata_query_returns_host_string_scalar_and_diagnoses_arity() {
    let request = |arguments| CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let input = ValueFact::unknown(DynamicReason::RuntimeValue);
    let args = vec![input];
    let inferred = inference::infer(&request(args), &UNDERLYING_TYPE_CATALOG_ENTRY);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0],
        ValueFact::scalar(ValueKindFact::String)
    );
    assert!(
        !inference::infer(&request(vec![]), &UNDERLYING_TYPE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}

#[test]
fn underlying_type_output_selection_keeps_string_carrier() {
    let input = ValueFact::unknown(DynamicReason::RuntimeValue);
    let infer = |outputs| {
        inference::infer(
            &CallRequest {
                arguments: vec![input.clone()],
                literals: LiteralContext::default(),
                outputs,
            },
            &UNDERLYING_TYPE_CATALOG_ENTRY,
        )
    };
    assert!(infer(OutputSelection::new(RequestedOutputCount::Zero))
        .outputs
        .is_empty());
    let selected = infer(OutputSelection {
        requested: RequestedOutputCount::One,
        discarded: BTreeSet::from([0]),
    });
    assert_eq!(
        selected.outputs,
        vec![ValueFact::scalar(ValueKindFact::String)]
    );
    assert_eq!(selected.discarded, vec![0]);
    let too_many = infer(OutputSelection::new(RequestedOutputCount::Exactly(2)));
    assert!(too_many
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}
