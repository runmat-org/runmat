use super::*;
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

#[test]
fn cputime_fact_is_scalar_double_and_rejects_inputs() {
    let request = CallRequest {
        arguments: Vec::new(),
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inferred = inference::infer(&request, &CPUTIME_CATALOG_ENTRY);
    assert_eq!(inferred.outputs.len(), 1);
    assert!(matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(_)
    ));
    let invalid = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Unknown)],
        ..request
    };
    assert!(inference::infer(&invalid, &CPUTIME_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-CPUTIME-ARITY"));
}
