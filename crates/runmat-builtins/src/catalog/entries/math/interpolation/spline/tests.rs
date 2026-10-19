use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn double(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

#[test]
fn endpoint_slope_y_still_produces_piecewise_polynomial() {
    let inferred = inference::infer(
        &request(vec![
            double(ShapeFact::from(vec![Some(1), Some(3)])),
            double(ShapeFact::from(vec![Some(1), Some(5)])),
        ]),
        &SPLINE_CATALOG_ENTRY,
    );
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Struct(_)));
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
}

#[test]
fn scalar_query_with_vector_samples_is_double_scalar() {
    let inferred = inference::infer(
        &request(vec![
            double(ShapeFact::from(vec![Some(1), Some(3)])),
            double(ShapeFact::from(vec![Some(1), Some(5)])),
            double(ShapeFact::Scalar),
        ]),
        &SPLINE_CATALOG_ENTRY,
    );
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn array_components_do_not_falsely_prove_query_shape() {
    let inferred = inference::infer(
        &request(vec![
            double(ShapeFact::from(vec![Some(1), Some(3)])),
            double(ShapeFact::from(vec![Some(2), Some(5)])),
            double(ShapeFact::from(vec![Some(2), Some(2)])),
        ]),
        &SPLINE_CATALOG_ENTRY,
    );
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
}
