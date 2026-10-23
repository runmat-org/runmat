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
        &CUMTRAPZ_CATALOG_ENTRY,
    )
}
fn scalar() -> ValueFact {
    numeric(NumericClass::Double, &[1, 1])
}
#[test]
fn canonical_binding_and_independent_examples_are_explicit() {
    assert_eq!(
        CUMTRAPZ_CATALOG_ENTRY.bindings,
        REQUIRED_DEFAULT_BINDING.as_slice()
    );
    assert_eq!(CUMTRAPZ_CATALOG_ENTRY.documentation.examples.len(), 4);
    assert_eq!(
        CUMTRAPZ_CATALOG_ENTRY.placement.fusion,
        BuiltinFusionPolicy::Boundary
    );
    assert!(CUMTRAPZ_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !matches!(example.verification, BuiltinExampleVerification::Succeeds)));
}
#[test]
fn cumulative_integral_keeps_y_full_shape_and_native_precision() {
    let mut a = numeric(NumericClass::Single, &[2, 3, 4]);
    a.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Complex,
    });
    let result = invoke(
        vec![a.clone(), scalar()],
        vec![LiteralValue::Unknown, LiteralValue::Number(3.)],
        1,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, a.shape);
    assert_eq!(result.outputs[0].kind, a.kind);
    let result = invoke(
        vec![scalar(), a.clone()],
        vec![LiteralValue::Number(0.5), LiteralValue::Unknown],
        1,
    );
    assert_eq!(result.outputs[0].kind, a.kind);
    assert!(!invoke(vec![a], vec![LiteralValue::Unknown], 2)
        .diagnostics
        .is_empty());
}
