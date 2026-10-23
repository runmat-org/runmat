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
        &GRADIENT_CATALOG_ENTRY,
    )
}
fn scalar() -> ValueFact {
    numeric(NumericClass::Double, &[1, 1])
}
#[test]
fn canonical_binding_and_independent_examples_are_explicit() {
    assert_eq!(
        GRADIENT_CATALOG_ENTRY.bindings,
        REQUIRED_DEFAULT_BINDING.as_slice()
    );
    assert_eq!(GRADIENT_CATALOG_ENTRY.documentation.examples.len(), 5);
    assert_eq!(
        GRADIENT_CATALOG_ENTRY.placement.fusion,
        BuiltinFusionPolicy::Boundary
    );
    assert!(GRADIENT_CATALOG_ENTRY
        .documentation
        .examples
        .iter()
        .all(|example| !matches!(example.verification, BuiltinExampleVerification::Succeeds)));
}
#[test]
fn gradient_full_shape_native_complex_class_and_available_outputs_are_known() {
    let mut a = numeric(NumericClass::Single, &[3, 4, 2]);
    a.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Complex,
    });
    let result = invoke(vec![a.clone()], vec![LiteralValue::Unknown], 3);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 3);
    for output in result.outputs {
        assert_eq!(output.shape, a.shape);
        assert_eq!(output.kind, a.kind);
    }
    assert!(!invoke(vec![a], vec![LiteralValue::Unknown], 4)
        .diagnostics
        .is_empty());
    let row = numeric(NumericClass::Single, &[4]);
    let result = invoke(vec![row], vec![LiteralValue::Unknown], 1);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
    assert!(!invoke(
        vec![scalar(), scalar()],
        vec![LiteralValue::Unknown, LiteralValue::Number(0.)],
        1
    )
    .diagnostics
    .is_empty());
}
