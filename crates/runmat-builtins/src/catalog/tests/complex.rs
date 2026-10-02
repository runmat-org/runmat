use super::*;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn fact(class: NumericClass, shape: ShapeFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    )
}

fn call(arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("complex").unwrap(),
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn complex_canonical_contract_is_pure_with_no_unknown_effect_or_capability() {
    let entry = builtin_catalog_entry_by_name("complex").unwrap();
    assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Complete);
    assert_eq!(entry.contract.purity, BuiltinPurity::Pure);
    assert_eq!(entry.contract.effects, &[EffectKind::MayThrow]);
    assert!(entry.contract.capabilities.is_empty());
    assert!(entry.contract.workspace_effect.is_none());
    assert!(entry.contract.environment_effect.is_none());
    assert_eq!(entry.bindings[0].identity.variant, "default");
    assert_eq!(entry.descriptor.signatures.len(), 2);
    assert_eq!(entry.integer_capabilities.len(), 2);
}

#[test]
fn complex_scalar_and_shape_inference_uses_canonical_authority() {
    let scalar = fact(NumericClass::Double, ShapeFact::Scalar);
    let matrix = fact(NumericClass::Double, vec![Some(2), Some(3)].into());
    for args in [
        vec![scalar.clone(), scalar.clone()],
        vec![matrix.clone(), matrix.clone()],
        vec![scalar.clone(), matrix.clone()],
        vec![matrix.clone(), scalar.clone()],
    ] {
        let expected = if args.iter().all(|arg| arg.shape.element_count() == Some(1)) {
            ShapeFact::Scalar
        } else {
            matrix.shape.clone()
        };
        let result = call(args);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, expected);
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            })
        );
    }
    let mismatch = call(vec![
        fact(NumericClass::Double, vec![Some(2), Some(1)].into()),
        fact(NumericClass::Double, vec![Some(1), Some(3)].into()),
    ]);
    assert!(!mismatch.diagnostics.is_empty());
    assert_eq!(mismatch.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn complex_retains_integer_and_single_classes_without_unlike_integer_promotion() {
    let integer = fact(NumericClass::UInt64, vec![Some(2), Some(3)].into());
    for args in [
        vec![integer.clone()],
        vec![integer.clone(), integer.clone()],
        vec![
            integer.clone(),
            fact(NumericClass::Double, ShapeFact::Scalar),
        ],
    ] {
        let result = call(args);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Complex,
            })
        );
        assert_eq!(result.outputs[0].shape, integer.shape);
    }
    assert!(!call(vec![
        integer.clone(),
        fact(NumericClass::Int64, integer.shape.clone())
    ])
    .diagnostics
    .is_empty());
    assert!(!call(vec![
        integer.clone(),
        fact(NumericClass::Double, integer.shape.clone())
    ])
    .diagnostics
    .is_empty());
    let result = call(vec![
        fact(NumericClass::Single, ShapeFact::Scalar),
        fact(NumericClass::Double, ShapeFact::Scalar),
    ]);
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
}

#[test]
fn complex_rejects_invalid_arity_and_binary_complex_components() {
    assert!(!call(vec![]).diagnostics.is_empty());
    let real = fact(NumericClass::Double, ShapeFact::Scalar);
    assert!(!call(vec![real.clone(), real.clone(), real.clone()])
        .diagnostics
        .is_empty());
    let mut complex = real.clone();
    complex.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    });
    assert!(call(vec![complex.clone()]).diagnostics.is_empty());
    assert!(!call(vec![real, complex]).diagnostics.is_empty());
    assert!(!call(vec![ValueFact::scalar(ValueKindFact::String)])
        .diagnostics
        .is_empty());
}
