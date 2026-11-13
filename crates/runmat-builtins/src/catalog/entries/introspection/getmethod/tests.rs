use super::*;
use runmat_types::*;

fn infer(
    receiver: ValueFact,
    name: ValueFact,
    literal_name: &str,
    outputs: usize,
) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![receiver, name],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String(literal_name.into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &GETMETHOD_CATALOG_ENTRY,
    )
}

fn receiver() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: None,
        properties: Default::default(),
        properties_complete: false,
        handle_semantics: None,
    }))
}

#[test]
fn callable_result_retains_dynamic_target_and_fixed_outer_output_count() {
    let output = infer(
        receiver(),
        ValueFact::scalar(ValueKindFact::String),
        "compute",
        1,
    );
    assert!(output.diagnostics.is_empty());
    let ValueKindFact::Callable(callable) = &output.outputs[0].kind else {
        panic!("callable")
    };
    assert!(callable.identity.is_none());
    assert!(!callable.outputs_complete);
    assert!(!callable.captures_complete);
    assert!(!infer(
        receiver(),
        ValueFact::scalar(ValueKindFact::String),
        "compute",
        2
    )
    .diagnostics
    .is_empty());
}

#[test]
fn integer_receiver_integer_name_and_trimmed_empty_name_reject() {
    let integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    assert!(!infer(
        integer.clone(),
        ValueFact::scalar(ValueKindFact::String),
        "compute",
        1
    )
    .diagnostics
    .is_empty());
    assert!(!infer(receiver(), integer, "compute", 1)
        .diagnostics
        .is_empty());
    assert!(!infer(
        receiver(),
        ValueFact::scalar(ValueKindFact::String),
        "  ",
        1
    )
    .diagnostics
    .is_empty());
    assert_eq!(GETMETHOD_EXTENSION.mode, BuiltinExtensionMode::RunMatOnly);
    assert_eq!(
        GETMETHOD_DESCRIPTOR.completion_policy,
        BuiltinCompletionPolicy::Public
    );
}
