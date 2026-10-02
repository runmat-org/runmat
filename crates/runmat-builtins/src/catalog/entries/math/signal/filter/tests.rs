use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, outputs: usize) -> CallInference {
    let literal_count = arguments.len();
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(vec![LiteralValue::Unknown; literal_count]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &FILTER_CATALOG_ENTRY,
    )
}

#[test]
fn signal_class_shape_and_complex_domain_are_retained() {
    let coeff = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let signal = numeric(
        NumericClass::Single,
        NumericDomain::Complex,
        ShapeFact::from(vec![Some(3), Some(2), Some(4)]),
    );
    let result = infer(vec![coeff.clone(), coeff, signal.clone()], 2);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, signal.shape);
    assert_eq!(
        result.outputs[0].numeric(),
        Some(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
    assert_eq!(result.outputs[1].numeric(), result.outputs[0].numeric());
    assert_eq!(result.outputs[1].shape, ShapeFact::Unknown);
}

#[test]
fn integer_signal_floating_result_and_unknown_shape_are_sound() {
    let coeff = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let signal = numeric(
        NumericClass::UInt64,
        NumericDomain::Real,
        ShapeFact::Unknown,
    );
    let result = infer(vec![coeff.clone(), coeff, signal], 1);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
}

#[test]
fn known_invalid_inputs_do_not_receive_proven_output_facts() {
    let coeff = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let signal = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(2), Some(2)]),
    );
    let result = infer(vec![signal.clone(), coeff.clone(), signal.clone()], 1);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);

    let mut sparse = signal;
    sparse.storage = StorageFact::Sparse;
    let result = infer(vec![coeff.clone(), coeff, sparse], 1);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn discarded_output_and_surplus_padding_do_not_invent_numeric_facts() {
    let coeff = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let empty = infer(vec![coeff.clone(), coeff.clone(), coeff.clone()], 0);
    assert!(empty.outputs.is_empty());
    let surplus = infer(vec![coeff.clone(), coeff.clone(), coeff], 3);
    assert_eq!(surplus.outputs.len(), 3);
    assert_eq!(surplus.outputs[2].kind, ValueKindFact::Unknown);
}

#[test]
fn complex_double_singleton_is_a_scalar_but_single_is_a_tensor() {
    let coeff = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let complex_double = numeric(
        NumericClass::Double,
        NumericDomain::Complex,
        ShapeFact::Scalar,
    );
    let result = infer(vec![coeff.clone(), coeff.clone(), complex_double], 1);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].storage, StorageFact::Scalar);
    let complex_single = numeric(
        NumericClass::Single,
        NumericDomain::Complex,
        ShapeFact::Scalar,
    );
    let result = infer(vec![coeff.clone(), coeff, complex_single], 1);
    assert!(result.diagnostics.is_empty());
    assert_ne!(result.outputs[0].storage, StorageFact::Scalar);
}
