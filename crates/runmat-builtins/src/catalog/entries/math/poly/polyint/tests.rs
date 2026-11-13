use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, domain: NumericDomain, length: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(vec![Some(length), Some(1)]),
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        &POLYINT_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn known_single_column_produces_single_row_geometry() {
    let result = infer(vec![numeric(NumericClass::Single, NumericDomain::Real, 3)]);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let output = &result.outputs[0];
    assert_eq!(output.shape, ShapeFact::from(vec![Some(1), Some(4)]));
    assert_eq!(output.numeric().unwrap().class, NumericClass::Single);
    assert_eq!(output.residency, runmat_types::ResidencyFact::Unknown);
}

#[test]
fn complex_constant_promotes_real_single_coefficients_to_complex_double() {
    let result = infer(vec![
        numeric(NumericClass::Single, NumericDomain::Real, 2),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
    ]);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().domain,
        NumericDomain::Complex
    );
}

#[test]
fn unknown_coefficients_retain_runtime_unknown_class() {
    let result = infer(vec![ValueFact::unknown(DynamicReason::RuntimeValue)]);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn invalid_integer_or_nonscalar_constant_is_diagnosed() {
    let result = infer(vec![
        numeric(NumericClass::UInt64, NumericDomain::Real, 3),
        numeric(NumericClass::Double, NumericDomain::Real, 2),
    ]);
    assert_eq!(result.diagnostics.len(), 2);
}

#[test]
fn documented_examples_have_assertions_and_resident_example_has_wgpu_gate() {
    let docs = &POLYINT_CATALOG_ENTRY.documentation;
    assert_eq!(docs.examples.len(), 6);
    assert!(docs.examples.iter().all(|example| matches!(
        example.verification,
        crate::BuiltinExampleVerification::Assertions { .. }
    )));
    assert!(docs.examples.iter().any(|example| {
        example.id == "gpu" && example.harness == crate::BuiltinExampleHarness::Wgpu
    }));
}
