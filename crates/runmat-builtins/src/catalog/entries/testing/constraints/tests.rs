use super::*;
use runmat_types::{
    DynamicReason, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact, ValueFact,
    ValueKindFact,
};

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn assert_object_class(result: CallInference, class: runmat_types::StaticClassIdentity) {
    assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        result.outputs[0].residency,
        runmat_types::ResidencyFact::Host
    );
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("constraint constructor must infer an object");
    };
    assert_eq!(object.runtime_class, Some(class.owned()));
    assert_eq!(object.handle_semantics, None);
}

#[test]
fn constructors_infer_concrete_scalar_host_objects() {
    let expected = ValueFact::scalar(ValueKindFact::String);
    let inferred =
        inference::is_equal_to(&call(vec![expected.clone()], 1), &IS_EQUAL_TO_CATALOG_ENTRY);
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("expected object");
    };
    assert_eq!(object.properties.get("Expected"), Some(&expected));
    assert_object_class(inferred, runmat_types::standard::UNIT_TEST_IS_EQUAL_TO);
    assert_object_class(
        inference::is_true(&call(vec![], 1), &IS_TRUE_CATALOG_ENTRY),
        runmat_types::standard::UNIT_TEST_IS_TRUE,
    );
    assert_object_class(
        inference::is_false(&call(vec![], 1), &IS_FALSE_CATALOG_ENTRY),
        runmat_types::standard::UNIT_TEST_IS_FALSE,
    );
}

#[test]
fn expected_unknown_remains_unknown_and_discarded_output_is_honored() {
    let expected = ValueFact::unknown(DynamicReason::RuntimeValue);
    let inferred =
        inference::is_equal_to(&call(vec![expected.clone()], 1), &IS_EQUAL_TO_CATALOG_ENTRY);
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("expected object");
    };
    assert_eq!(object.properties.get("Expected"), Some(&expected));
    assert!(inference::is_true(&call(vec![], 0), &IS_TRUE_CATALOG_ENTRY)
        .outputs
        .is_empty());
}

#[test]
fn invalid_arity_is_rejected_statically() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    for inferred in [
        inference::is_equal_to(&call(vec![], 1), &IS_EQUAL_TO_CATALOG_ENTRY),
        inference::is_equal_to(
            &call(vec![unknown.clone(), unknown.clone()], 1),
            &IS_EQUAL_TO_CATALOG_ENTRY,
        ),
        inference::is_true(&call(vec![unknown.clone()], 1), &IS_TRUE_CATALOG_ENTRY),
        inference::is_false(&call(vec![unknown], 1), &IS_FALSE_CATALOG_ENTRY),
    ] {
        assert!(!inferred.diagnostics.is_empty());
    }
}
