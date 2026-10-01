use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn vector(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(args: Vec<ValueFact>, count: usize) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        &POLYDER_CATALOG_ENTRY,
        &CallRequest {
            arguments: args,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
    )
}

#[test]
fn host_real_derivative_keeps_known_orientation_but_not_trimmed_length() {
    let column = vector(ShapeFact::from(vec![Some(4), Some(1)]));
    let result = infer(vec![column], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn binary_form_switches_from_product_to_quotient_with_requested_count() {
    let left = vector(ShapeFact::from(vec![Some(1), Some(3)]));
    let right = vector(ShapeFact::from(vec![Some(4), Some(1)]));
    let product = infer(vec![left.clone(), right.clone()], 1);
    assert_eq!(product.outputs.len(), 1);
    assert_eq!(
        product.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    let quotient = infer(vec![left, right], 3);
    assert_eq!(
        quotient.outputs[1].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert_eq!(quotient.outputs[2].shape, ShapeFact::Scalar);
}

#[test]
fn invalid_quotient_arity_and_matrix_are_diagnosed() {
    let matrix = vector(ShapeFact::from(vec![Some(2), Some(3)]));
    let invalid = infer(vec![matrix], 2);
    assert_eq!(invalid.diagnostics.len(), 2);
    let unknown = infer(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1);
    assert!(unknown.diagnostics.is_empty());
}

#[test]
fn resident_single_input_does_not_invent_fallback_class_or_owner() {
    let mut single = vector(ShapeFact::from(vec![Some(1), Some(3)]));
    single.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    });
    single.residency = ResidencyFact::Device { provider: None };
    let result = infer(vec![single], 1);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
}
