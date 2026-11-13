use super::*;
use runmat_types::{
    DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, ValueFact, ValueKindFact,
};

fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

#[test]
fn constructors_preserve_their_concrete_class_and_scalar_host_shape() {
    for (entry, result, class) in [
        (
            &TEST_CASE_CATALOG_ENTRY,
            inference::test_case(&call(vec![], vec![], 1), &TEST_CASE_CATALOG_ENTRY),
            runmat_types::standard::UNIT_TEST_CASE,
        ),
        (
            &STRING_DIAGNOSTIC_CATALOG_ENTRY,
            inference::string_diagnostic(
                &call(vec![ValueFact::scalar(ValueKindFact::String)], vec![], 1),
                &STRING_DIAGNOSTIC_CATALOG_ENTRY,
            ),
            runmat_types::standard::UNIT_TEST_STRING_DIAGNOSTIC,
        ),
    ] {
        assert!(result.diagnostics.is_empty(), "{}", entry.identity.name);
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(
            result.outputs[0].residency,
            runmat_types::ResidencyFact::Host
        );
        let ValueKindFact::Object(object) = &result.outputs[0].kind else {
            panic!("object result required")
        };
        assert_eq!(object.runtime_class, Some(class.owned()));
        assert_eq!(object.handle_semantics, None);
    }
}

#[test]
fn constructors_reject_known_invalid_inputs_and_support_discarded_output() {
    assert!(
        !inference::test_case(&call(vec![unknown()], vec![], 1), &TEST_CASE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let invalid = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(!inference::string_diagnostic(
        &call(vec![invalid], vec![], 1),
        &STRING_DIAGNOSTIC_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    assert!(inference::string_diagnostic(
        &call(vec![unknown()], vec![], 1),
        &STRING_DIAGNOSTIC_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    assert!(
        inference::test_case(&call(vec![], vec![], 0), &TEST_CASE_CATALOG_ENTRY)
            .outputs
            .is_empty()
    );
}

#[test]
fn log_effects_output_and_supported_forms_are_explicit() {
    let receiver = inference::test_case(&call(vec![], vec![], 1), &TEST_CASE_CATALOG_ENTRY)
        .outputs
        .remove(0);
    let diagnostic = inference::string_diagnostic(
        &call(vec![ValueFact::scalar(ValueKindFact::String)], vec![], 1),
        &STRING_DIAGNOSTIC_CATALOG_ENTRY,
    )
    .outputs
    .remove(0);
    for (arguments, literals) in [
        (
            vec![receiver.clone(), ValueFact::scalar(ValueKindFact::String)],
            vec![],
        ),
        (
            vec![
                receiver.clone(),
                ValueFact::scalar(ValueKindFact::String),
                diagnostic,
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("verbose".into()),
            ],
        ),
    ] {
        let result = inference::log(&call(arguments, literals, 1), &TEST_CASE_LOG_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
        assert!(result
            .effects
            .0
            .contains(&runmat_types::EffectKind::EnvironmentWrite));
    }
    assert!(inference::log(
        &call(vec![receiver, unknown()], vec![], 0),
        &TEST_CASE_LOG_CATALOG_ENTRY
    )
    .outputs
    .is_empty());
}

#[test]
fn log_rejects_invalid_receiver_verbosity_and_diagnostic() {
    let receiver = inference::test_case(&call(vec![], vec![], 1), &TEST_CASE_CATALOG_ENTRY)
        .outputs
        .remove(0);
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let text = ValueFact::scalar(ValueKindFact::String);
    for request in [
        call(vec![numeric.clone(), text.clone()], vec![], 1),
        call(vec![receiver.clone(), numeric.clone()], vec![], 1),
        call(
            vec![receiver.clone(), numeric.clone(), text.clone()],
            vec![LiteralValue::Unknown, LiteralValue::Number(0.0)],
            1,
        ),
        call(
            vec![receiver.clone(), text.clone(), text.clone()],
            vec![LiteralValue::Unknown, LiteralValue::String("trace".into())],
            1,
        ),
        call(vec![receiver], vec![], 1),
    ] {
        assert!(!inference::log(&request, &TEST_CASE_LOG_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
}
