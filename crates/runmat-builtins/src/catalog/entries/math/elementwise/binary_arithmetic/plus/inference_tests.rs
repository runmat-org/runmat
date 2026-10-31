use super::super::inference::test_support::{numeric, request};
use super::PLUS_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{
    LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, ValueFact,
    ValueKindFact,
};

mod string_addition;

#[test]
fn storage_facts_distinguish_unknown_representation_single_and_placement() {
    use runmat_types::StorageFact;
    let mut unknown = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        &[Some(2), Some(3)],
    );
    unknown.storage = StorageFact::Unknown;
    let zero = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let inferred = infer_catalog_call(&PLUS_CATALOG_ENTRY, &request(vec![unknown, zero.clone()]));
    assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown);
    let single = numeric(
        NumericClass::Single,
        NumericDomain::Real,
        &[Some(1), Some(1)],
    );
    let inferred = infer_catalog_call(
        &PLUS_CATALOG_ENTRY,
        &request(vec![single.clone(), zero.clone()]),
    );
    assert_eq!(inferred.outputs[0].storage, StorageFact::Dense);
    let mut call = request(vec![
        single.clone(),
        single.clone(),
        ValueFact::scalar(ValueKindFact::String),
        zero,
    ]);
    call.literals.literal_args = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Keyword("like".into()),
        LiteralValue::Unknown,
    ];
    assert_eq!(
        infer_catalog_call(&PLUS_CATALOG_ENTRY, &call).outputs[0].storage,
        StorageFact::Dense
    );
    call.arguments[3].residency = ResidencyFact::Device {
        provider: Some("p".into()),
    };
    assert_eq!(
        infer_catalog_call(&PLUS_CATALOG_ENTRY, &call).outputs[0].storage,
        StorageFact::Opaque
    );
    let inferred = infer_catalog_call(&PLUS_CATALOG_ENTRY, &request(vec![single.clone(), single]));
    // Known floating carriers stay materialized even when automatic placement
    // cannot promise a host or device result.
    assert_eq!(inferred.outputs[0].storage, StorageFact::Dense);
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);
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
fn mixed_real_single_array_and_double_scalar_share_operator_class_rule() {
    let single = numeric(
        NumericClass::Single,
        NumericDomain::Real,
        &[Some(2), Some(1)],
    );
    let double = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        &[Some(1), Some(1)],
    );
    let catalog = infer_catalog_call(
        &PLUS_CATALOG_ENTRY,
        &request(vec![single.clone(), double.clone()]),
    );
    let operator = runmat_types::infer_binary(runmat_types::OperatorKind::Add, &single, &double);
    assert!(catalog.diagnostics.is_empty());
    assert!(operator.diagnostics.is_empty());
    assert_eq!(catalog.outputs[0].kind, operator.fact.kind);
    assert_eq!(catalog.outputs[0].shape, operator.fact.shape);
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
