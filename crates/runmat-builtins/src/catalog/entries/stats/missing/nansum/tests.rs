use super::*;
use runmat_types::*;

#[test]
fn nansum_owns_a_data_only_sum_preparation_form() {
    assert!(crate::validate_builtin_preparation(&NANSUM_CATALOG_ENTRY).is_empty());
    let BuiltinOperandPreparation::Forms([form]) = NANSUM_CATALOG_ENTRY.placement.preparation
    else {
        panic!("nansum owns exactly one data-only preparation form")
    };
    assert_eq!(form.signatures, &[0]);
    assert_eq!(form.data_arguments, &[0]);
    assert_eq!(
        form.operation,
        BuiltinPreparationOperation::Reduction(BuiltinPreparationReduction::Sum)
    );
    assert_eq!(
        crate::sum_invocation_entry(SumInvocation::NanSum).identity,
        NANSUM_CATALOG_ENTRY.identity
    );
}

fn call(input: ValueFact, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &NANSUM_CATALOG_ENTRY,
    )
}
#[test]
fn known_single_input_keeps_shape_but_not_unproven_output_class() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(2)]),
        StorageFact::Dense,
    );
    let inferred = call(input, 1);
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
}
#[test]
fn all_selector_proves_scalar_output_shape() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(2)]),
        StorageFact::Dense,
    );
    let inferred = inference::infer(
        &CallRequest {
            arguments: vec![input, ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("all".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANSUM_CATALOG_ENTRY,
    );
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
}
#[test]
fn known_nonnumeric_input_has_a_type_diagnostic() {
    assert!(call(ValueFact::scalar(ValueKindFact::String), 1)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.MISSING.UNSUPPORTED_TYPE"));
}
#[test]
fn unknown_input_and_excess_outputs_do_not_claim_success() {
    let inferred = call(ValueFact::unknown(DynamicReason::RuntimeValue), 2);
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn integer_output_class_obeys_native_and_default_options() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    assert_eq!(
        call(input.clone(), 1).outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let native = inference::infer(
        &CallRequest {
            arguments: vec![input.clone(), ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("native".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANSUM_CATALOG_ENTRY,
    );
    assert_eq!(
        native.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    let like = inference::infer(
        &CallRequest {
            arguments: vec![input, ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("like".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANSUM_CATALOG_ENTRY,
    );
    assert_eq!(like.outputs[0].kind, ValueKindFact::Unknown);
}
