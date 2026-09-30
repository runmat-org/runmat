use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &STATGET_CATALOG_ENTRY,
    )
}

fn options(fields: &[(&str, ValueFact)]) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        fields
            .iter()
            .map(|(name, fact)| ((*name).into(), fact.clone()))
            .collect::<BTreeMap<_, _>>(),
        true,
    )))
}

#[test]
fn known_field_and_unique_prefix_return_exact_field_fact() {
    let integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let result = call(
        vec![
            options(&[("MaxIter", integer.clone())]),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![LiteralValue::Unknown, LiteralValue::String("maxi".into())],
        1,
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
}

#[test]
fn known_empty_field_returns_exact_wide_integer_default() {
    let options = super::super::statset::empty_options_fact();
    let default = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let result = call(
        vec![options, ValueFact::scalar(ValueKindFact::String), default],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("TolFun".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
}

#[test]
fn unknown_or_ambiguous_field_is_empty_and_ignores_default() {
    let result = call(
        vec![
            options(&[]),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Tol".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert_eq!(result.outputs[0].shape.element_count(), Some(0));
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn dynamic_field_and_invalid_argument_do_not_claim_a_selected_class() {
    let unknown = call(
        vec![options(&[]), ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
        1,
    );
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);
    let invalid = call(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![LiteralValue::Unknown],
        2,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.STATGET.INVALID_ARGUMENT"));
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}
