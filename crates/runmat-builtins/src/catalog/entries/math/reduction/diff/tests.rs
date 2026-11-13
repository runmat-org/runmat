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
        &DIFF_CATALOG_ENTRY,
    )
}
fn scalar() -> ValueFact {
    numeric(NumericClass::Double, &[1, 1])
}
#[test]
fn canonical_binding_and_independent_examples_are_explicit() {
    assert_eq!(
        DIFF_CATALOG_ENTRY.bindings,
        REQUIRED_DEFAULT_BINDING.as_slice()
    );
    assert_eq!(DIFF_CATALOG_ENTRY.documentation.examples.len(), 7);
    assert_eq!(
        DIFF_CATALOG_ENTRY.placement.fusion,
        BuiltinFusionPolicy::Boundary
    );
    assert!(DIFF_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !matches!(example.verification, BuiltinExampleVerification::Succeeds)));
}
#[test]
fn difference_keeps_exact_class_and_empty_selected_extent() {
    let a = numeric(NumericClass::Int8, &[0, 5]);
    let result = invoke(
        vec![a, scalar(), scalar()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.),
            LiteralValue::Number(2.),
        ],
        1,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(3)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Int8
    );
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let identity = invoke(
        vec![logical.clone(), scalar()],
        vec![LiteralValue::Unknown, LiteralValue::Number(0.)],
        1,
    );
    assert_eq!(identity.outputs[0].kind, ValueKindFact::Logical);
    let positive = invoke(vec![logical], vec![LiteralValue::Unknown], 1);
    assert_eq!(
        positive.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert!(!invoke(vec![scalar()], vec![LiteralValue::Unknown], 2)
        .diagnostics
        .is_empty());
}
#[test]
fn symbolic_order_and_variable_roles_accept_both_public_orders() {
    for reversed in [false, true] {
        let symbol = ValueFact::scalar(ValueKindFact::Symbolic);
        let text = ValueFact::scalar(ValueKindFact::String);
        let (arguments, literals) = if reversed {
            (
                vec![symbol, scalar(), text],
                vec![
                    LiteralValue::Unknown,
                    LiteralValue::Number(2.),
                    LiteralValue::String("x".into()),
                ],
            )
        } else {
            (
                vec![symbol, text, scalar()],
                vec![
                    LiteralValue::Unknown,
                    LiteralValue::String("x".into()),
                    LiteralValue::Number(2.),
                ],
            )
        };
        let result = invoke(arguments, literals, 1);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Symbolic);
    }
}
