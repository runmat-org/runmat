use super::*;
use runmat_types::{
    standard, CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &OPTIMIZABLE_VARIABLE_CATALOG_ENTRY,
    )
}

fn valid_arguments() -> Vec<ValueFact> {
    vec![
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        ),
    ]
}

#[test]
fn public_variable_identities_are_closed_and_case_insensitive() {
    for option in [
        OptimizableVariableOption::Optimize,
        OptimizableVariableOption::Transform,
        OptimizableVariableOption::Type,
    ] {
        assert_eq!(
            OptimizableVariableOption::from_public_name(option.public_name()),
            Some(option)
        );
    }
    assert_eq!(
        OptimizableVariableOption::from_public_name(" TRANSFORM "),
        Some(OptimizableVariableOption::Transform)
    );
    assert_eq!(OptimizableVariableOption::from_public_name("Unknown"), None);
    assert_eq!(
        OptimizationVariableType::from_public_name("INTEGER"),
        Some(OptimizationVariableType::Integer)
    );
    assert_eq!(
        OptimizationVariableTransform::from_public_name("LoG"),
        Some(OptimizationVariableTransform::Log)
    );
    assert_eq!(
        standard::OPTIMIZABLE_VARIABLE.display_name(),
        "optimizableVariable"
    );
}

#[test]
fn known_valid_unknown_and_invalid_calls_have_sound_result_facts() {
    let valid = call(valid_arguments(), vec![]);
    assert!(valid.diagnostics.is_empty());
    assert!(matches!(valid.outputs[0].kind, ValueKindFact::Object(_)));

    let mut unknown_arguments = valid_arguments();
    unknown_arguments[1] = ValueFact::unknown(DynamicReason::RuntimeValue);
    let unknown = call(unknown_arguments, vec![]);
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);

    let mut unknown_option_arguments = valid_arguments();
    unknown_option_arguments.extend([
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::Logical),
    ]);
    let unknown_option = call(
        unknown_option_arguments,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Bool(true),
        ],
    );
    assert_eq!(unknown_option.outputs[0].kind, ValueKindFact::Unknown);

    let mut invalid_arguments = valid_arguments();
    invalid_arguments.extend([
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::String),
    ]);
    let invalid = call(
        invalid_arguments,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("Type".into()),
            LiteralValue::String("not-a-type".into()),
        ],
    );
    assert!(!invalid.diagnostics.is_empty());
    assert_eq!(invalid.outputs[0].kind, ValueKindFact::Unknown);
}
