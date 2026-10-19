use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn input(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(value: ValueFact) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        &ROOTS_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![value],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn scalar_input_proves_empty_host_double_column() {
    let result = infer(input(NumericClass::Double, ShapeFact::Scalar));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let output = &result.outputs[0];
    assert_eq!(output.shape, ShapeFact::from(vec![Some(0), Some(1)]));
    assert_eq!(output.numeric().unwrap().class, NumericClass::Double);
    assert_eq!(output.residency, ResidencyFact::Host);
}

#[test]
fn degree_does_not_prove_root_count_or_complexity_without_coefficients() {
    let result = infer(input(
        NumericClass::Single,
        ShapeFact::from(vec![Some(1), Some(4)]),
    ));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn invalid_matrix_and_nonnumeric_input_are_diagnosed() {
    let matrix = infer(input(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(3)]),
    ));
    assert_eq!(matrix.diagnostics.len(), 1);
    let text = infer(ValueFact::scalar(ValueKindFact::String));
    assert_eq!(text.diagnostics.len(), 1);
    let unknown = infer(ValueFact::unknown(DynamicReason::RuntimeValue));
    assert!(unknown.diagnostics.is_empty());
}

#[test]
fn n_d_singleton_extended_vector_is_admitted() {
    let result = infer(input(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(4), Some(1)]),
    ));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
}
