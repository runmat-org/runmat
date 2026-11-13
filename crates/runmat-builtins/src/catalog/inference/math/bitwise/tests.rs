use super::*;
use crate::SWAPBYTES_CATALOG_ENTRY;
use runmat_types::{
    AliasFact, CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

mod basic_operations;
mod shape;

fn request(argument: ValueFact) -> CallRequest {
    CallRequest {
        arguments: vec![argument],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn request_many(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn numeric(class: NumericClass, shape: impl Into<ShapeFact>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape.into(),
        StorageFact::Dense,
    )
}

#[test]
fn direct_bitwise_rules_retain_the_data_class() {
    let value = numeric(NumericClass::Int32, vec![Some(1), Some(4)]);
    let control = numeric(NumericClass::UInt8, ShapeFact::Scalar);
    for (rule, entry, arguments) in [
        (
            crate::BitwiseInferenceRule::Complement,
            &crate::BITCMP_CATALOG_ENTRY,
            vec![value.clone()],
        ),
        (
            crate::BitwiseInferenceRule::Get,
            &crate::BITGET_CATALOG_ENTRY,
            vec![value.clone(), control.clone()],
        ),
        (
            crate::BitwiseInferenceRule::Set,
            &crate::BITSET_CATALOG_ENTRY,
            vec![value.clone(), control.clone(), control.clone()],
        ),
        (
            crate::BitwiseInferenceRule::Shift,
            &crate::BITSHIFT_CATALOG_ENTRY,
            vec![value.clone(), control.clone()],
        ),
    ] {
        let result = infer(rule, &request_many(arguments), entry);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, value.kind);
        assert_eq!(result.outputs[0].shape, value.shape);
    }
}

#[test]
fn direct_single_and_logical_inputs_infer_actual_double_results() {
    let shape = ShapeFact::from(vec![Some(2), Some(1), Some(3)]);
    for value in [
        numeric(NumericClass::Single, shape.clone()),
        ValueFact::proven(ValueKindFact::Logical, shape.clone(), StorageFact::Dense),
    ] {
        for (rule, entry, args) in [
            (
                crate::BitwiseInferenceRule::Complement,
                &crate::BITCMP_CATALOG_ENTRY,
                vec![value.clone()],
            ),
            (
                crate::BitwiseInferenceRule::Get,
                &crate::BITGET_CATALOG_ENTRY,
                vec![
                    value.clone(),
                    numeric(NumericClass::Double, ShapeFact::Scalar),
                ],
            ),
            (
                crate::BitwiseInferenceRule::Shift,
                &crate::BITSHIFT_CATALOG_ENTRY,
                vec![
                    value.clone(),
                    numeric(NumericClass::Double, ShapeFact::Scalar),
                ],
            ),
        ] {
            let result = infer(rule, &request_many(args), entry);
            assert!(result.diagnostics.is_empty());
            assert_eq!(
                result.outputs[0].numeric().unwrap().class,
                NumericClass::Double
            );
            assert_eq!(result.outputs[0].shape, shape);
        }
    }
}

#[test]
fn bitset_text_assumed_type_does_not_broadcast_as_value_control() {
    let shape = ShapeFact::from(vec![Some(2), Some(3)]);
    let mut call = request_many(vec![
        numeric(NumericClass::Double, shape.clone()),
        numeric(NumericClass::Double, ShapeFact::Scalar),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(5)]),
            StorageFact::Dense,
        ),
    ]);
    call.literals = LiteralContext::new(vec![
        runmat_types::LiteralValue::Unknown,
        runmat_types::LiteralValue::Number(1.0),
        runmat_types::LiteralValue::Character("uint8".into()),
    ]);
    let result = infer(
        crate::BitwiseInferenceRule::Set,
        &call,
        &crate::BITSET_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, shape);
    call.arguments[2] = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    call.literals = LiteralContext::default();
    let result = infer(
        crate::BitwiseInferenceRule::Set,
        &call,
        &crate::BITSET_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn assumed_type_and_control_diagnostics_preserve_unknown_possibilities() {
    let mut call = request_many(vec![
        numeric(NumericClass::UInt16, ShapeFact::Scalar),
        ValueFact::scalar(ValueKindFact::String),
    ]);
    call.literals = LiteralContext::new(vec![
        runmat_types::LiteralValue::Unknown,
        runmat_types::LiteralValue::String("uint8".into()),
    ]);
    assert!(!infer(
        crate::BitwiseInferenceRule::Complement,
        &call,
        &crate::BITCMP_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    call.arguments[1] = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    call.literals = LiteralContext::default();
    assert!(infer(
        crate::BitwiseInferenceRule::Complement,
        &call,
        &crate::BITCMP_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    call.arguments[1] = ValueFact::scalar(ValueKindFact::String);
    assert!(!infer(
        crate::BitwiseInferenceRule::Shift,
        &call,
        &crate::BITSHIFT_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
}
