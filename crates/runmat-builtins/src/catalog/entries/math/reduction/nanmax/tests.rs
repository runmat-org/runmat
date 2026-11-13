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
        &NANMAX_CATALOG_ENTRY,
    )
}

#[test]
fn canonical_binding_and_independent_examples_are_explicit() {
    assert_eq!(
        NANMAX_CATALOG_ENTRY.bindings,
        REQUIRED_DEFAULT_BINDING.as_slice()
    );
    assert_eq!(NANMAX_CATALOG_ENTRY.documentation.examples.len(), 2);
    assert_eq!(
        NANMAX_CATALOG_ENTRY.placement.fusion,
        BuiltinFusionPolicy::Boundary
    );
    assert!(NANMAX_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !matches!(example.verification, BuiltinExampleVerification::Succeeds)));
}
#[test]
fn omit_nan_maximum_retains_host_boundary_and_pairwise_output_bound() {
    let a = numeric(NumericClass::Single, &[2, 3]);
    let result = invoke(vec![a.clone()], vec![LiteralValue::Unknown], 1);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let pair = invoke(
        vec![a.clone(), a],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
        2,
    );
    assert!(!pair.diagnostics.is_empty());
}
