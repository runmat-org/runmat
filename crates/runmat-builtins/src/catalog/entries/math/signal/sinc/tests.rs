use super::*;
use runmat_types::*;

fn input(class: NumericClass, domain: NumericDomain, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(value: ValueFact, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![value],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &SINC_CATALOG_ENTRY,
    )
}

#[test]
fn floating_real_and_complex_outputs_keep_class_and_shape() {
    for class in [NumericClass::Double, NumericClass::Single] {
        for domain in [NumericDomain::Real, NumericDomain::Complex] {
            let x = input(
                class,
                domain,
                ShapeFact::from(vec![Some(2), Some(0), Some(3)]),
            );
            let result = infer(x.clone(), 1);
            assert!(result.diagnostics.is_empty());
            assert_eq!(result.outputs[0].shape, x.shape);
            assert_eq!(
                result.outputs[0].numeric().unwrap(),
                NumericFact { class, domain }
            );
            assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
        }
    }
}

#[test]
fn integer_promotes_double_but_complex_integer_is_invalid() {
    let integer = input(NumericClass::UInt64, NumericDomain::Real, ShapeFact::Scalar);
    let result = infer(integer, 1);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );

    let invalid = input(
        NumericClass::Int16,
        NumericDomain::Complex,
        ShapeFact::Scalar,
    );
    let result = infer(invalid, 1);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn unknown_and_discarded_outputs_are_conservative() {
    let result = infer(ValueFact::unknown(DynamicReason::RuntimeValue), 1);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    let result = infer(
        input(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
        0,
    );
    assert!(result.outputs.is_empty());
}

#[test]
fn known_sparse_input_is_diagnosed() {
    let mut value = input(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(2), Some(2)]),
    );
    value.storage = StorageFact::Sparse;
    let result = infer(value, 1);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}
