use super::super::inference::test_support::{numeric, request};
use super::PLUS_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{
    LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, ValueFact,
    ValueKindFact,
};

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
    assert_eq!(
        inferred.outputs[0].residency,
        ResidencyFact::Device {
            provider: Some("p".into()),
        }
    );
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
