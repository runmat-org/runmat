use super::*;
use crate::PolyfitStatField;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, StructFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn numeric(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn request(arguments: Vec<ValueFact>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn host_real_output_preserves_query_geometry_and_padding() {
    let points = numeric(ShapeFact::from(vec![Some(2), Some(3), Some(4)]));
    let coeffs = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    let result =
        crate::infer_catalog_call(&POLYVAL_CATALOG_ENTRY, &request(vec![coeffs, points], 1));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3), Some(4)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn delta_requires_s_and_closed_stats_fields() {
    let coeffs = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    let points = numeric(ShapeFact::Scalar);
    let missing = crate::infer_catalog_call(
        &POLYVAL_CATALOG_ENTRY,
        &request(vec![coeffs.clone(), points.clone()], 2),
    );
    assert!(missing
        .diagnostics
        .iter()
        .any(|error| error.code == "RM-CATALOG-POLYVAL-DELTA-S"));
    let mut fields = BTreeMap::new();
    fields.insert(
        PolyfitStatField::R.public_name().into(),
        numeric(ShapeFact::from(vec![Some(3), Some(3)])),
    );
    let structure = ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(fields, true)),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let result = crate::infer_catalog_call(
        &POLYVAL_CATALOG_ENTRY,
        &request(vec![coeffs, points, structure], 3),
    );
    assert_eq!(
        result
            .diagnostics
            .iter()
            .filter(|error| error.code == "RM-CATALOG-POLYVAL-S-FIELD")
            .count(),
        2
    );
    assert_eq!(result.outputs[1].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[2].shape, ShapeFact::Scalar);
}

#[test]
fn unknown_placement_does_not_prove_result_class_or_owner() {
    let coeffs = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    let mut points = numeric(ShapeFact::from(vec![Some(1), Some(4)]));
    points.residency = ResidencyFact::Device { provider: None };
    let result =
        crate::infer_catalog_call(&POLYVAL_CATALOG_ENTRY, &request(vec![coeffs, points], 1));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    let unknown = crate::infer_catalog_call(
        &POLYVAL_CATALOG_ENTRY,
        &request(
            vec![
                ValueFact::unknown(DynamicReason::RuntimeValue),
                numeric(ShapeFact::Scalar),
            ],
            1,
        ),
    );
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn host_complex_operands_prove_complex_double_output() {
    let mut coefficients = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    coefficients.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    });
    let points = numeric(ShapeFact::from(vec![Some(2), Some(2)]));
    let result = crate::infer_catalog_call(
        &POLYVAL_CATALOG_ENTRY,
        &request(vec![coefficients, points], 1),
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().domain,
        NumericDomain::Complex
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn invalid_coefficient_matrix_and_short_mu_are_diagnosed() {
    let coeffs = numeric(ShapeFact::from(vec![Some(2), Some(2)]));
    let points = numeric(ShapeFact::Scalar);
    let empty = numeric(ShapeFact::from(vec![Some(0), Some(0)]));
    let mu = numeric(ShapeFact::Scalar);
    let result = crate::infer_catalog_call(
        &POLYVAL_CATALOG_ENTRY,
        &request(vec![coeffs, points, empty, mu], 1),
    );
    assert_eq!(result.diagnostics.len(), 2);
}
