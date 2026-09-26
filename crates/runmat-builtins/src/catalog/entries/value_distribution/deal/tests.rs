use super::*;
use runmat_types::*;
fn request(arguments: Vec<ValueFact>, requested: RequestedOutputCount) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(requested),
    }
}
#[test]
fn replication_preserves_full_facts_and_discarded_slots() {
    let value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
        StorageFact::Dense,
    );
    let mut call = request(vec![value.clone()], RequestedOutputCount::Exactly(3));
    call.outputs.discarded.insert(1);
    let result = infer_catalog_call(&DEAL_CATALOG_ENTRY, &call);
    assert_eq!(result.outputs, vec![value; 3]);
    assert_eq!(result.discarded, vec![1]);
    assert!(result.diagnostics.is_empty());
}
#[test]
fn positional_counts_zero_and_dynamic_requests_remain_distinct() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    for (inputs, outputs, invalid) in [
        (0, 0, false),
        (0, 1, true),
        (1, 0, false),
        (1, 3, false),
        (2, 1, true),
        (2, 2, false),
        (2, 0, true),
    ] {
        let result = infer_catalog_call(
            &DEAL_CATALOG_ENTRY,
            &request(
                vec![unknown.clone(); inputs],
                RequestedOutputCount::Exactly(outputs),
            ),
        );
        assert_eq!(
            !result.diagnostics.is_empty(),
            invalid,
            "{inputs}/{outputs}"
        );
    }
    let result = infer_catalog_call(
        &DEAL_CATALOG_ENTRY,
        &request(vec![unknown], RequestedOutputCount::CurrentFunctionNargout),
    );
    assert!(result.dynamic_outputs);
    assert!(result.diagnostics.is_empty());
}
