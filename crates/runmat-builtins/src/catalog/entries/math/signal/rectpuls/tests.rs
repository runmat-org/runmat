use super::*;
use runmat_types::*;

fn real(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &RECTPULS_CATALOG_ENTRY,
    )
}

#[test]
fn real_single_nd_times_produce_double_same_shape_with_unknown_residency() {
    let times = real(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(0), Some(3)]),
    );
    let out = infer(vec![times.clone()], vec![LiteralValue::Unknown], 1);
    assert!(out.diagnostics.is_empty());
    assert_eq!(out.outputs[0].shape, times.shape);
    assert_eq!(
        out.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(out.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn width_and_complex_times_diagnose_without_success_facts() {
    let times = real(NumericClass::Double, ShapeFact::Scalar);
    let width = real(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    let out = infer(
        vec![times.clone(), width],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
        1,
    );
    assert!(!out.diagnostics.is_empty());
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
    let out = infer(
        vec![times.clone(), real(NumericClass::Double, ShapeFact::Scalar)],
        vec![LiteralValue::Unknown, LiteralValue::Number(-1.0)],
        1,
    );
    assert!(!out.diagnostics.is_empty());
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
    let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    let out = infer(vec![complex], vec![LiteralValue::Unknown], 1);
    assert!(!out.diagnostics.is_empty());
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn unknown_time_and_discarded_output_do_not_claim_values() {
    let out = infer(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![LiteralValue::Unknown],
        1,
    );
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
    let out = infer(
        vec![real(NumericClass::Double, ShapeFact::Scalar)],
        vec![LiteralValue::Unknown],
        0,
    );
    assert!(out.outputs.is_empty());
}
