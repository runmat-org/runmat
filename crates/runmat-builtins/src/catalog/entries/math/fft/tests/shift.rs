use super::*;

#[test]
fn shift_float_dimension_above_platform_range_is_an_out_of_rank_noop() {
    let input = numeric(NumericClass::Double, vec![Some(1), Some(2)]);
    let dimension = numeric(NumericClass::Double, vec![Some(1), Some(1)]);
    let request = call(
        vec![input.clone(), dimension],
        vec![LiteralValue::Unknown, LiteralValue::Number(1.0e30)],
    );
    for (transform, entry) in [
        (inference::Transform::Fftshift, &FFTSHIFT_CATALOG_ENTRY),
        (inference::Transform::Ifftshift, &IFFTSHIFT_CATALOG_ENTRY),
    ] {
        let inferred = inference::infer(transform, &request, entry);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:#?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs[0].shape, input.shape);
    }
}

#[test]
fn shift_preserves_logical_shape_and_invalid_input_is_diagnosed() {
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(1), Some(5)]),
        StorageFact::Dense,
    );
    let result = inference::infer(
        inference::Transform::Fftshift,
        &call(vec![logical.clone()], vec![]),
        &FFTSHIFT_CATALOG_ENTRY,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(result.outputs[0].shape, logical.shape);
    let bad = ValueFact::scalar(ValueKindFact::String);
    let result = inference::infer(
        inference::Transform::Ifftshift,
        &call(vec![bad], vec![]),
        &IFFTSHIFT_CATALOG_ENTRY,
    );
    assert!(!result.diagnostics.is_empty());
}

#[test]
fn shift_dimension_vectors_reject_zero_in_every_literal_carrier() {
    let input = numeric(NumericClass::Double, vec![Some(2), Some(3)]);
    for (class, zero) in [
        (NumericClass::Double, LiteralValue::Number(0.0)),
        (
            NumericClass::Single,
            LiteralValue::Real {
                text: "0".into(),
                class: NumericClass::Single,
            },
        ),
        (
            NumericClass::UInt64,
            LiteralValue::Integer {
                text: "0".into(),
                class: NumericClass::UInt64,
            },
        ),
    ] {
        for literal in [
            LiteralValue::Vector(vec![LiteralValue::Number(1.0), zero.clone()]),
            LiteralValue::Matrix(vec![vec![LiteralValue::Number(1.0)], vec![zero]]),
        ] {
            let control = numeric(class, vec![Some(1), Some(2)]);
            let request = call(
                vec![input.clone(), control],
                vec![LiteralValue::Unknown, literal],
            );
            for (transform, entry) in [
                (inference::Transform::Fftshift, &FFTSHIFT_CATALOG_ENTRY),
                (inference::Transform::Ifftshift, &IFFTSHIFT_CATALOG_ENTRY),
            ] {
                assert!(!inference::infer(transform, &request, entry)
                    .diagnostics
                    .is_empty());
            }
        }
    }
}
