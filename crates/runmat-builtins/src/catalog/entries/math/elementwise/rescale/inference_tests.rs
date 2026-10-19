use super::RESCALE_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn numeric(class: NumericClass, domain: NumericDomain, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn preserves_single_class_and_broadcasts_all_explicit_bounds() {
    let scalar = numeric(NumericClass::Double, NumericDomain::Real, &[1, 1]);
    let inferred = infer_catalog_call(
        &RESCALE_CATALOG_ENTRY,
        &request(
            vec![
                numeric(NumericClass::Single, NumericDomain::Real, &[4, 2]),
                scalar.clone(),
                scalar,
                ValueFact::scalar(ValueKindFact::String),
                numeric(NumericClass::Double, NumericDomain::Real, &[1, 2]),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("inputmin".into()),
                LiteralValue::Unknown,
            ],
        ),
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(2)])
    );
}

#[test]
fn logical_input_produces_dense_double_with_option_bound_shape() {
    let inferred = infer_catalog_call(
        &RESCALE_CATALOG_ENTRY,
        &request(
            vec![
                ValueFact::proven(
                    ValueKindFact::Logical,
                    ShapeFact::from(vec![Some(3), Some(1)]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::String),
                numeric(NumericClass::Double, NumericDomain::Real, &[1, 4]),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("InputMax".into()),
                LiteralValue::Unknown,
            ],
        ),
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::Double)
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4)])
    );
}

#[test]
fn diagnoses_complex_input_unknown_options_and_shape_mismatch() {
    let inferred = infer_catalog_call(
        &RESCALE_CATALOG_ENTRY,
        &request(
            vec![
                numeric(NumericClass::Double, NumericDomain::Complex, &[2, 2]),
                ValueFact::scalar(ValueKindFact::String),
                numeric(NumericClass::Double, NumericDomain::Real, &[1, 3]),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("Range".into()),
                LiteralValue::Unknown,
            ],
        ),
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-RESCALE-INPUT"));
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-RESCALE-OPTION"));
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-RESCALE-SIZE"));
}

#[test]
fn preserves_one_known_provider_and_diagnoses_mixed_owners() {
    let mut source = numeric(NumericClass::Double, NumericDomain::Real, &[1, 3]);
    source.residency = ResidencyFact::Device {
        provider: Some("left".into()),
    };
    let mut bound = numeric(NumericClass::Double, NumericDomain::Real, &[1, 1]);
    bound.residency = ResidencyFact::Device {
        provider: Some("right".into()),
    };
    let inferred = infer_catalog_call(
        &RESCALE_CATALOG_ENTRY,
        &request(
            vec![
                source,
                bound,
                numeric(NumericClass::Double, NumericDomain::Real, &[1, 1]),
            ],
            vec![LiteralValue::Unknown; 3],
        ),
    );
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-RESCALE-PROVIDER"));
}
