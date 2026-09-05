use crate::{
    builtin_catalog_entry_by_name, infer_catalog_call, BuiltinInferenceRule, MathInferenceRule,
};
use runmat_types::{
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};
use runmat_types::{OutputSelection, RequestedOutputCount};

fn numeric(class: NumericClass, domain: NumericDomain, shape: ShapeFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        storage,
    )
}

fn infer(arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    let entry = builtin_catalog_entry_by_name("complex").expect("complex catalog entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::ComplexConstruction)
    );
    infer_catalog_call(
        entry,
        &runmat_types::CallRequest {
            arguments,
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn catalog_owns_signatures_integer_capabilities_and_documentation() {
    let entry = builtin_catalog_entry_by_name("complex").expect("complex catalog entry");
    let labels = entry
        .descriptor
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert_eq!(labels, vec!["Z = complex(A)", "Z = complex(A, B)"]);
    assert_eq!(entry.integer_capabilities.len(), 2);
    assert_eq!(entry.documentation.examples.len(), 6);
    assert!(entry.documentation.examples.iter().all(|example| matches!(
        example.verification,
        crate::BuiltinExampleVerification::Assertions { .. }
    )));
}

#[test]
fn unary_preserves_class_and_shape_while_selecting_complex_domain() {
    for class in [
        NumericClass::Double,
        NumericClass::Single,
        NumericClass::UInt64,
    ] {
        let result = infer(vec![numeric(
            class,
            NumericDomain::Real,
            ShapeFact::from(vec![Some(2), Some(3)]),
        )]);
        let output = &result.outputs[0];
        assert_eq!(output.numeric().expect("numeric").class, class);
        assert_eq!(
            output.numeric().expect("numeric").domain,
            NumericDomain::Complex
        );
        assert_eq!(output.shape, ShapeFact::from(vec![Some(2), Some(3)]));
        assert!(result.diagnostics.is_empty());
    }
}

#[test]
fn unary_complex_input_is_an_identity_fact() {
    let mut input = numeric(
        NumericClass::Single,
        NumericDomain::Complex,
        ShapeFact::Scalar,
    );
    input.residency = ResidencyFact::Device {
        provider: Some("test".into()),
    };
    let result = infer(vec![input.clone()]);
    assert_eq!(result.outputs[0], input);
}

#[test]
fn binary_applies_scalar_only_shape_and_class_rules() {
    let array = ShapeFact::from(vec![Some(2), Some(3)]);
    let mut resident = numeric(NumericClass::UInt64, NumericDomain::Real, array.clone());
    resident.residency = ResidencyFact::Device {
        provider: Some("test".into()),
    };
    let result = infer(vec![
        resident,
        numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
    ]);
    assert_eq!(result.outputs[0].shape, array);
    assert_eq!(
        result.outputs[0].numeric().expect("numeric").class,
        NumericClass::UInt64
    );
    assert_eq!(
        result.outputs[0].numeric().expect("numeric").domain,
        NumericDomain::Complex
    );
    assert_eq!(
        result.outputs[0].residency,
        ResidencyFact::Device {
            provider: Some("test".into())
        }
    );
    assert!(result.diagnostics.is_empty());

    let mismatch = infer(vec![
        numeric(
            NumericClass::Double,
            NumericDomain::Real,
            ShapeFact::from(vec![Some(2), Some(1)]),
        ),
        numeric(
            NumericClass::Double,
            NumericDomain::Real,
            ShapeFact::from(vec![Some(1), Some(3)]),
        ),
    ]);
    assert!(!mismatch.diagnostics.is_empty());
}

#[test]
fn binary_rejects_complex_character_and_invalid_integer_peers() {
    let complex = infer(vec![
        numeric(
            NumericClass::Double,
            NumericDomain::Complex,
            ShapeFact::Scalar,
        ),
        numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
    ]);
    assert!(!complex.diagnostics.is_empty());

    let character = infer(vec![
        ValueFact::scalar(ValueKindFact::Character),
        numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar),
    ]);
    assert!(!character.diagnostics.is_empty());

    let mixed_integer = infer(vec![
        numeric(NumericClass::Int16, NumericDomain::Real, ShapeFact::Scalar),
        numeric(NumericClass::UInt16, NumericDomain::Real, ShapeFact::Scalar),
    ]);
    assert!(!mixed_integer.diagnostics.is_empty());

    let nonscalar_double_peer = infer(vec![
        numeric(NumericClass::Int16, NumericDomain::Real, ShapeFact::Scalar),
        numeric(
            NumericClass::Double,
            NumericDomain::Real,
            ShapeFact::from(vec![Some(1), Some(2)]),
        ),
    ]);
    assert!(!nonscalar_double_peer.diagnostics.is_empty());

    let logical_peer = infer(vec![
        numeric(NumericClass::Int16, NumericDomain::Real, ShapeFact::Scalar),
        ValueFact::scalar(ValueKindFact::Logical),
    ]);
    assert!(!logical_peer.diagnostics.is_empty());
}
