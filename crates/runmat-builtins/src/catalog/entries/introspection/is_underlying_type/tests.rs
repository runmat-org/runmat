use super::*;
use runmat_types::*;
#[test]
fn metadata_query_returns_host_logical_scalar_and_diagnoses_arity() {
    let request = |arguments| CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let input = ValueFact::unknown(DynamicReason::RuntimeValue);
    let mut args = vec![input];
    args.push(ValueFact::scalar(ValueKindFact::String));
    let inferred = inference::infer(&request(args), &IS_UNDERLYING_TYPE_CATALOG_ENTRY);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0],
        ValueFact::scalar(ValueKindFact::Logical)
    );
    assert!(
        !inference::infer(&request(vec![]), &IS_UNDERLYING_TYPE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
