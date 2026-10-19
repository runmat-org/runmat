use super::*;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn number(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
fn call(arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            literals: LiteralContext::new(Vec::new()),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &REFLINE_CATALOG_ENTRY,
    )
}

#[test]
fn vector_coefficients_prove_host_double_handle_class_not_count() {
    let result = call(vec![number(ShapeFact::from(vec![Some(1), Some(2)]))]);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn axes_ambiguous_and_unknown_inputs_still_have_conditional_handle_class() {
    let scalars = call(vec![number(ShapeFact::Scalar), number(ShapeFact::Scalar)]);
    assert_eq!(
        scalars.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(scalars.outputs[0].shape, ShapeFact::Unknown);
    let unknown = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    let possible_axes = call(vec![
        unknown,
        number(ShapeFact::from(vec![Some(1), Some(2)])),
    ]);
    assert!(possible_axes.diagnostics.is_empty());
    assert_eq!(
        possible_axes.outputs[0].numeric().unwrap().domain,
        NumericDomain::Real
    );
    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let invalid = call(vec![complex]);
    assert!(!invalid.diagnostics.is_empty());
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));
}
