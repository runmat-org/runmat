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
        &PROD_CATALOG_ENTRY,
    )
}
fn scalar() -> ValueFact {
    numeric(NumericClass::Double, &[1, 1])
}
#[test]
fn canonical_binding_and_independent_examples_are_explicit() {
    assert_eq!(
        PROD_CATALOG_ENTRY.bindings,
        REQUIRED_DEFAULT_BINDING.as_slice()
    );
    assert_eq!(PROD_CATALOG_ENTRY.documentation.examples.len(), 7);
    assert_eq!(
        PROD_CATALOG_ENTRY.placement.fusion,
        BuiltinFusionPolicy::Boundary
    );
    assert!(PROD_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !matches!(example.verification, BuiltinExampleVerification::Succeeds)));
}
#[test]
fn product_classes_axes_and_output_bounds_are_proven() {
    let a = numeric(NumericClass::Single, &[3, 4]);
    let result = invoke(vec![a.clone()], vec![LiteralValue::Unknown], 1);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
    let result = invoke(
        vec![a, ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::Unknown, LiteralValue::String("double".into())],
        1,
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let a = numeric(NumericClass::Int8, &[2, 2]);
    let native = invoke(
        vec![a.clone(), ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::Unknown, LiteralValue::String("native".into())],
        1,
    );
    assert_eq!(
        native.outputs[0].numeric().unwrap().class,
        NumericClass::Int8
    );
    let ordinary = invoke(vec![a.clone()], vec![LiteralValue::Unknown], 1);
    assert_eq!(
        ordinary.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert!(invoke(vec![a.clone()], vec![LiteralValue::Unknown], 0)
        .diagnostics
        .is_empty());
    assert!(!invoke(vec![a], vec![LiteralValue::Unknown], 2)
        .diagnostics
        .is_empty());
}
#[test]
fn empty_selector_grammar_and_beyond_rank_are_not_fabricated() {
    let a = numeric(NumericClass::Double, &[2, 3]);
    let result = invoke(
        vec![a.clone(), numeric(NumericClass::Double, &[0, 0])],
        vec![LiteralValue::Unknown, LiteralValue::Empty],
        1,
    );
    assert!(!result.diagnostics.is_empty());
    let result = invoke(
        vec![a.clone(), numeric(NumericClass::Double, &[1, 0])],
        vec![LiteralValue::Unknown, LiteralValue::Vector(vec![])],
        1,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    let result = invoke(
        vec![a.clone(), scalar()],
        vec![LiteralValue::Unknown, LiteralValue::Number(8.)],
        1,
    );
    assert_eq!(result.outputs[0].shape, a.shape);
}
