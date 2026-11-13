use super::*;
use runmat_types::*;
fn infer(arguments: Vec<ValueFact>, count: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        &NEW_OBJECT_CATALOG_ENTRY,
    )
}
#[test]
fn fixed_arity_and_single_output_are_owned() {
    let args = vec![ValueFact::scalar(ValueKindFact::String); 1];
    assert!(infer(args.clone(), 1).diagnostics.is_empty());
    assert!(!infer(args, 2).diagnostics.is_empty());
    assert!(!infer(vec![ValueFact::scalar(ValueKindFact::String); 2], 1)
        .diagnostics
        .is_empty());
}
#[test]
fn legacy_scalar_integer_formatting_does_not_admit_integer_arrays() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    assert!(infer(vec![scalar.clone(); 1], 1).diagnostics.is_empty());
    let mut array = scalar;
    array.shape = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(1), DimensionFact::Known(2)],
    };
    assert!(!infer(vec![array; 1], 1).diagnostics.is_empty());
}
