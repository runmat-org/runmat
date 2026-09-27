use super::super::inference::test_support::{numeric, request};
use super::PLUS_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{
    LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, ValueFact,
    ValueKindFact,
};

#[test]
fn integer_inference_rejects_non_numeric_and_complex_pairs() {
    let integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    for name in ["plus", "minus", "times", "rdivide", "ldivide", "power"] {
        let entry = crate::builtin_catalog_entry_by_name(name).unwrap();
        for other in [
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::Character),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            })),
        ] {
            for pair in [
                vec![integer.clone(), other.clone()],
                vec![other, integer.clone()],
            ] {
                assert!(
                    infer_catalog_call(entry, &request(pair))
                        .diagnostics
                        .iter()
                        .any(|d| d.code == "RM-CATALOG-BINARY-ARITHMETIC-INPUT"),
                    "{name}"
                );
            }
        }
        let result = infer_catalog_call(entry, &request(vec![integer.clone(), integer.clone()]));
        assert!(result.diagnostics.is_empty(), "{name}");
        assert_eq!(result.outputs[0].kind, integer.kind, "{name}");
        let unknown_shape = numeric(NumericClass::Double, NumericDomain::Real, &[None, None]);
        let result = infer_catalog_call(entry, &request(vec![integer.clone(), unknown_shape]));
        assert!(
            result.diagnostics.is_empty(),
            "{name}: unknown shape may be scalar"
        );
    }
}

#[test]
fn sparse_storage_is_not_inferred_as_a_dense_or_scalar_buffer() {
    use runmat_types::StorageFact;
    for name in ["plus", "minus", "times"] {
        let entry = crate::builtin_catalog_entry_by_name(name).unwrap();
        for shape in [&[Some(1), Some(1)][..], &[Some(2), Some(3)][..]] {
            let mut sparse = numeric(NumericClass::Double, NumericDomain::Real, shape);
            sparse.storage = StorageFact::Sparse;
            let inferred =
                infer_catalog_call(entry, &request(vec![sparse.clone(), sparse.clone()]));
            assert_eq!(inferred.outputs[0].storage, StorageFact::Sparse, "{name}");
            let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }));
            let mixed = infer_catalog_call(entry, &request(vec![sparse, scalar]));
            assert_eq!(mixed.outputs[0].storage, StorageFact::Unknown, "{name}");
        }
        let mut unknown = numeric(NumericClass::Double, NumericDomain::Real, &[None, None]);
        unknown.storage = StorageFact::Sparse;
        let inferred = infer_catalog_call(entry, &request(vec![unknown.clone(), unknown]));
        assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown, "{name}");
    }
}

#[test]
fn tracks_broadcast_class_domain_and_residency() {
    let mut left = numeric(
        NumericClass::Single,
        NumericDomain::Real,
        &[Some(3), Some(1)],
    );
    left.residency = ResidencyFact::Device {
        provider: Some("p".into()),
    };
    let right = numeric(
        NumericClass::Double,
        NumericDomain::Complex,
        &[Some(1), Some(4)],
    );
    let inferred = infer_catalog_call(&PLUS_CATALOG_ENTRY, &request(vec![left, right]));
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4)])
    );
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn preserves_integer_and_applies_typed_like_placement() {
    let integer = numeric(
        NumericClass::UInt64,
        NumericDomain::Real,
        &[Some(1), Some(2)],
    );
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut prototype = scalar.clone();
    prototype.residency = ResidencyFact::Device {
        provider: Some("gpu".into()),
    };
    let mut request = request(vec![
        integer,
        scalar,
        ValueFact::scalar(ValueKindFact::String),
        prototype,
    ]);
    request.literals.literal_args = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Keyword("like".into()),
        LiteralValue::Unknown,
    ];
    let inferred = infer_catalog_call(&PLUS_CATALOG_ENTRY, &request);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::UInt64)
    );
    assert_eq!(
        inferred.outputs[0].residency,
        ResidencyFact::Device {
            provider: Some("gpu".into()),
        }
    );
}

#[test]
fn dynamic_like_keyword_is_not_a_proven_argument_error() {
    for name in ["plus", "minus", "times", "rdivide", "ldivide", "power"] {
        let entry = crate::builtin_catalog_entry_by_name(name).expect("binary entry");
        let operand = numeric(
            NumericClass::Double,
            NumericDomain::Real,
            &[Some(2), Some(3)],
        );
        for option in [
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        ] {
            let call = request(vec![
                operand.clone(),
                operand.clone(),
                option,
                operand.clone(),
            ]);
            let inferred = infer_catalog_call(entry, &call);
            assert!(
                inferred.diagnostics.is_empty(),
                "{name}: {:?}",
                inferred.diagnostics
            );
        }
        let mut call = request(vec![
            operand.clone(),
            operand.clone(),
            ValueFact::scalar(ValueKindFact::String),
            operand.clone(),
        ]);
        call.literals.literal_args = vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Keyword("not-like".into()),
            LiteralValue::Unknown,
        ];
        assert!(
            infer_catalog_call(entry, &call)
                .diagnostics
                .iter()
                .any(|d| d.code == "RM-CATALOG-BINARY-ARITHMETIC-OPTION"),
            "{name}"
        );
        call.arguments[2] = operand;
        call.literals = Default::default();
        assert!(
            infer_catalog_call(entry, &call)
                .diagnostics
                .iter()
                .any(|d| d.code == "RM-CATALOG-BINARY-ARITHMETIC-OPTION"),
            "{name}"
        );
    }
}
