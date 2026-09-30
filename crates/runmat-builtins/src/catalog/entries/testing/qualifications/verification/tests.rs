use super::*;
use runmat_types::{
    DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, ValueFact, ValueKindFact,
};

struct InferenceCase {
    entry: &'static BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
}

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

#[test]
fn verification_entries_infer_empty_double_and_method_only_host_execution() {
    let cases = &[
        InferenceCase {
            entry: &VERIFY_EMPTY_CATALOG_ENTRY,
            arguments: vec![unknown(), unknown()],
            infer: inference::verify_empty,
        },
        InferenceCase {
            entry: &VERIFY_EQUAL_CATALOG_ENTRY,
            arguments: vec![unknown(), unknown(), unknown()],
            infer: inference::verify_equal,
        },
        InferenceCase {
            entry: &VERIFY_FALSE_CATALOG_ENTRY,
            arguments: vec![unknown(), unknown()],
            infer: inference::verify_false,
        },
        InferenceCase {
            entry: &VERIFY_GREATER_THAN_CATALOG_ENTRY,
            arguments: vec![unknown(), unknown(), unknown()],
            infer: inference::verify_greater_than,
        },
        InferenceCase {
            entry: &VERIFY_LESS_THAN_CATALOG_ENTRY,
            arguments: vec![unknown(), unknown(), unknown()],
            infer: inference::verify_less_than,
        },
        InferenceCase {
            entry: &VERIFY_NOT_EMPTY_CATALOG_ENTRY,
            arguments: vec![unknown(), unknown()],
            infer: inference::verify_not_empty,
        },
        InferenceCase {
            entry: &VERIFY_NOT_EQUAL_CATALOG_ENTRY,
            arguments: vec![unknown(), unknown(), unknown()],
            infer: inference::verify_not_equal,
        },
        InferenceCase {
            entry: &VERIFY_THAT_CATALOG_ENTRY,
            arguments: vec![unknown(), unknown(), unknown()],
            infer: inference::verify_that,
        },
        InferenceCase {
            entry: &VERIFY_TRUE_CATALOG_ENTRY,
            arguments: vec![unknown(), unknown()],
            infer: inference::verify_true,
        },
    ];
    for InferenceCase {
        entry,
        arguments,
        infer,
    } in cases
    {
        assert_eq!(
            entry.descriptor.completion_policy,
            BuiltinCompletionPolicy::MethodOnly
        );
        assert_eq!(entry.placement.residency, BuiltinResidencyPolicy::Host);
        let result = infer(&call(arguments.to_vec(), 1), entry);
        assert!(
            result.diagnostics.is_empty(),
            "{}: {:#?}",
            entry.identity.name,
            result.diagnostics
        );
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
        assert!(matches!(
            &result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        ));
        assert_eq!(
            result.outputs[0].residency,
            runmat_types::ResidencyFact::Host
        );
        assert!(infer(&call(arguments.to_vec(), 0), entry)
            .outputs
            .is_empty());
    }
    assert_eq!(
        VERIFY_EMPTY_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::MaySuspend
    );
    assert_eq!(
        VERIFY_EQUAL_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::NeverSuspends
    );
    assert_eq!(
        VERIFY_NOT_EMPTY_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::MaySuspend
    );
    assert_eq!(
        VERIFY_THAT_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::NeverSuspends
    );
}

#[test]
fn verification_arity_and_receiver_diagnostics_do_not_confuse_failed_predicates_with_bad_calls() {
    assert!(!inference::verify_equal(
        &call(vec![unknown(), unknown()], 1),
        &VERIFY_EQUAL_CATALOG_ENTRY,
    )
    .diagnostics
    .is_empty());
    let known_non_receiver = ValueFact::scalar(ValueKindFact::Logical);
    assert!(inference::verify_false(
        &call(vec![known_non_receiver, unknown()], 1),
        &VERIFY_FALSE_CATALOG_ENTRY,
    )
    .diagnostics
    .iter()
    .any(|diagnostic| diagnostic.code == "RM-TESTING-QUALIFICATION-RECEIVER"));

    // A nonnumeric comparison operand is a failed verification at runtime,
    // not a static argument error. The receiver and arity remain valid.
    let result = inference::verify_greater_than(
        &call(
            vec![
                unknown(),
                ValueFact::scalar(ValueKindFact::String),
                unknown(),
            ],
            1,
        ),
        &VERIFY_GREATER_THAN_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());

    let non_constraint = ValueFact::scalar(ValueKindFact::Logical);
    let invalid_constraint = inference::verify_that(
        &call(vec![unknown(), unknown(), non_constraint], 1),
        &VERIFY_THAT_CATALOG_ENTRY,
    );
    assert!(invalid_constraint
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TESTING-QUALIFICATION-CONSTRAINT"));

    let nonnumeric_comparison = inference::verify_less_than(
        &call(
            vec![
                unknown(),
                ValueFact::scalar(ValueKindFact::String),
                unknown(),
            ],
            1,
        ),
        &VERIFY_LESS_THAN_CATALOG_ENTRY,
    );
    assert!(nonnumeric_comparison.diagnostics.is_empty());
}
