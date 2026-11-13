use super::*;
use runmat_types::{
    CallInference, CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};
fn numeric(class: NumericClass, shape: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}
fn invoke(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &NNZ_CATALOG_ENTRY,
    )
}
fn scalar() -> ValueFact {
    numeric(NumericClass::Double, &[1, 1])
}
#[test]
fn canonical_binding_and_independent_examples_are_explicit() {
    assert_eq!(
        NNZ_CATALOG_ENTRY.bindings,
        REQUIRED_DEFAULT_BINDING.as_slice()
    );
    assert_eq!(NNZ_CATALOG_ENTRY.documentation.examples.len(), 5);
    assert_eq!(
        NNZ_CATALOG_ENTRY.placement.fusion,
        BuiltinFusionPolicy::Boundary
    );
    assert!(NNZ_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !matches!(example.verification, BuiltinExampleVerification::Succeeds)));
}
#[test]
fn count_is_host_double_and_dimension_preserves_character_pages() {
    let mut a = numeric(NumericClass::UInt64, &[2, 3]);
    a.residency = ResidencyFact::Device { provider: None };
    let result = invoke(vec![a], vec![LiteralValue::Unknown], 1);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let a = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
        StorageFact::Dense,
    );
    let result = invoke(
        vec![a, scalar()],
        vec![LiteralValue::Unknown, LiteralValue::Number(2.)],
        1,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1), Some(4)])
    );
    assert!(!invoke(
        vec![scalar(), scalar()],
        vec![LiteralValue::Unknown, LiteralValue::Number(0.)],
        1
    )
    .diagnostics
    .is_empty());
    assert!(!invoke(vec![scalar()], vec![LiteralValue::Unknown], 2)
        .diagnostics
        .is_empty());
}
