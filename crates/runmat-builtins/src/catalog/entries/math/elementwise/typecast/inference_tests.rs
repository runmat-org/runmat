use super::TYPECAST_CATALOG_ENTRY;
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
fn infers_class_width_and_row_orientation() {
    let inferred = infer_catalog_call(
        &TYPECAST_CATALOG_ENTRY,
        &request(
            vec![
                numeric(NumericClass::UInt32, NumericDomain::Real, &[1, 3]),
                ValueFact::scalar(ValueKindFact::String),
            ],
            vec![LiteralValue::Unknown, LiteralValue::String("uint8".into())],
        ),
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::UInt8)
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(12)])
    );
}

#[test]
fn infers_complex_like_class_and_component_pairing() {
    let inferred = infer_catalog_call(
        &TYPECAST_CATALOG_ENTRY,
        &request(
            vec![
                numeric(NumericClass::Int16, NumericDomain::Real, &[4, 1]),
                ValueFact::scalar(ValueKindFact::String),
                numeric(NumericClass::Int16, NumericDomain::Complex, &[1, 1]),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Keyword("like".into()),
                LiteralValue::Unknown,
            ],
        ),
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int16,
            domain: NumericDomain::Complex,
        })
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}

#[test]
fn preserves_known_provider_for_supported_direct_form() {
    let mut source = numeric(NumericClass::UInt64, NumericDomain::Real, &[1, 2]);
    source.residency = ResidencyFact::Device {
        provider: Some("gpu".into()),
    };
    let inferred = infer_catalog_call(
        &TYPECAST_CATALOG_ENTRY,
        &request(
            vec![source, ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::Unknown, LiteralValue::String("uint8".into())],
        ),
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].residency,
        ResidencyFact::Device {
            provider: Some("gpu".into())
        }
    );
}

#[test]
fn diagnoses_nonvector_and_indivisible_known_forms() {
    let matrix = infer_catalog_call(
        &TYPECAST_CATALOG_ENTRY,
        &request(
            vec![
                numeric(NumericClass::UInt8, NumericDomain::Real, &[2, 2]),
                ValueFact::scalar(ValueKindFact::String),
            ],
            vec![LiteralValue::Unknown, LiteralValue::String("uint16".into())],
        ),
    );
    assert!(matrix
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TYPECAST-SHAPE"));

    let bytes = infer_catalog_call(
        &TYPECAST_CATALOG_ENTRY,
        &request(
            vec![
                numeric(NumericClass::UInt8, NumericDomain::Real, &[1, 1]),
                ValueFact::scalar(ValueKindFact::String),
            ],
            vec![LiteralValue::Unknown, LiteralValue::String("uint16".into())],
        ),
    );
    assert!(bytes
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TYPECAST-BYTE-COUNT"));
}
