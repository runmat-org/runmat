use super::*;
use runmat_types::{
    DynamicReason, LiteralContext, NumericClass, OutputSelection, RequestedOutputCount, ShapeFact,
    ValueFact, ValueKindFact,
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

fn test_case() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(runmat_types::ObjectFact {
        class: None,
        runtime_class: Some(runmat_types::standard::UNIT_TEST_CASE.owned()),
        properties: Default::default(),
        properties_complete: false,
        handle_semantics: None,
    }))
}

#[test]
fn five_qualifications_return_host_empty_double_when_used_as_expressions() {
    let receiver = test_case();
    let actual = ValueFact::scalar(ValueKindFact::Logical);
    let expected = actual.clone();
    let constraint = ValueFact::unknown(DynamicReason::RuntimeValue);
    let cases = &[
        InferenceCase {
            entry: &ASSERT_EQUAL_CATALOG_ENTRY,
            arguments: vec![receiver.clone(), actual.clone(), expected.clone()],
            infer: inference::assert_equal,
        },
        InferenceCase {
            entry: &ASSERT_THAT_CATALOG_ENTRY,
            arguments: vec![receiver.clone(), actual.clone(), constraint],
            infer: inference::assert_that,
        },
        InferenceCase {
            entry: &ASSERT_TRUE_CATALOG_ENTRY,
            arguments: vec![receiver.clone(), actual.clone()],
            infer: inference::assert_true,
        },
        InferenceCase {
            entry: &ASSUME_EQUAL_CATALOG_ENTRY,
            arguments: vec![receiver.clone(), actual.clone(), expected],
            infer: inference::assume_equal,
        },
        InferenceCase {
            entry: &ASSUME_TRUE_CATALOG_ENTRY,
            arguments: vec![receiver, actual],
            infer: inference::assume_true,
        },
    ];
    for InferenceCase {
        entry,
        arguments,
        infer,
    } in cases
    {
        let result = infer(&call(arguments.to_vec(), 1), entry);
        assert!(
            result.diagnostics.is_empty(),
            "{}: {:#?}",
            entry.identity.name,
            result.diagnostics
        );
        assert_eq!(result.outputs.len(), 1, "{}", entry.identity.name);
        let output = &result.outputs[0];
        assert_eq!(output.shape, ShapeFact::from(vec![Some(0), Some(0)]));
        assert_eq!(output.residency, runmat_types::ResidencyFact::Host);
        assert!(
            matches!(&output.kind, ValueKindFact::Numeric(numeric) if numeric.class == NumericClass::Double)
        );
        assert!(infer(&call(arguments.to_vec(), 0), entry)
            .outputs
            .is_empty());
    }
}

#[test]
fn receiver_and_constraint_are_diagnosed_only_when_their_kind_is_proven_invalid() {
    let number = ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
        class: NumericClass::Double,
        domain: runmat_types::NumericDomain::Real,
    }));
    let receiver = test_case();
    let equal = inference::assert_equal(
        &call(vec![number.clone(), unknown(), unknown()], 1),
        &ASSERT_EQUAL_CATALOG_ENTRY,
    );
    assert!(equal
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TESTING-QUALIFICATION-RECEIVER"));

    let invalid_constraint = inference::assert_that(
        &call(vec![receiver.clone(), unknown(), number], 1),
        &ASSERT_THAT_CATALOG_ENTRY,
    );
    assert!(invalid_constraint
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TESTING-QUALIFICATION-CONSTRAINT"));

    // A custom TestCase or Constraint subclass cannot be ruled out from a
    // generic object fact without a proven class hierarchy.
    let unknown_object = ValueFact::scalar(ValueKindFact::Object(runmat_types::ObjectFact {
        class: None,
        runtime_class: None,
        properties: Default::default(),
        properties_complete: false,
        handle_semantics: None,
    }));
    let maybe_custom = inference::assert_that(
        &call(vec![unknown_object.clone(), unknown(), unknown_object], 1),
        &ASSERT_THAT_CATALOG_ENTRY,
    );
    assert!(maybe_custom.diagnostics.is_empty());
}

#[test]
fn required_arity_is_checked_without_rejecting_variadic_diagnostics() {
    let receiver = test_case();
    assert!(
        !inference::assert_true(&call(vec![receiver.clone()], 1), &ASSERT_TRUE_CATALOG_ENTRY,)
            .diagnostics
            .is_empty()
    );
    assert!(!inference::assume_equal(
        &call(vec![receiver.clone(), unknown()], 1),
        &ASSUME_EQUAL_CATALOG_ENTRY,
    )
    .diagnostics
    .is_empty());
    let result = inference::assume_equal(
        &call(
            vec![
                receiver,
                unknown(),
                unknown(),
                ValueFact::scalar(ValueKindFact::String),
            ],
            1,
        ),
        &ASSUME_EQUAL_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
}
