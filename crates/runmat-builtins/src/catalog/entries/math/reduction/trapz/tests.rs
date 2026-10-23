use super::*;
use runmat_types::{
    CallInference, CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
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
        &TRAPZ_CATALOG_ENTRY,
    )
}
fn scalar() -> ValueFact {
    numeric(NumericClass::Double, &[1, 1])
}
#[test]
fn canonical_binding_and_independent_examples_are_explicit() {
    assert_eq!(
        TRAPZ_CATALOG_ENTRY.bindings,
        REQUIRED_DEFAULT_BINDING.as_slice()
    );
    assert_eq!(TRAPZ_CATALOG_ENTRY.documentation.examples.len(), 4);
    assert_eq!(
        TRAPZ_CATALOG_ENTRY.placement.fusion,
        BuiltinFusionPolicy::Boundary
    );
    assert!(TRAPZ_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !matches!(example.verification, BuiltinExampleVerification::Succeeds)));
}
#[test]
fn final_integral_uses_y_class_and_selected_axis_not_spacing_class() {
    let a = numeric(NumericClass::Single, &[2, 3]);
    let result = invoke(
        vec![a.clone(), scalar()],
        vec![LiteralValue::Unknown, LiteralValue::Number(2.)],
        1,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    let result = invoke(
        vec![scalar(), a.clone()],
        vec![LiteralValue::Number(0.5), LiteralValue::Unknown],
        1,
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    let beyond = invoke(
        vec![a.clone(), scalar()],
        vec![LiteralValue::Unknown, LiteralValue::Number(100000.)],
        1,
    );
    assert_eq!(beyond.outputs[0].shape, a.shape);
    assert!(!invoke(vec![a], vec![LiteralValue::Unknown], 2)
        .diagnostics
        .is_empty());
}
