use super::*;
fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::default(),
    }
}
fn matrix(rows: usize, cols: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(cols)]),
        StorageFact::Dense,
    )
}
#[test]
fn known_and_unknown_coefficients_return_host_value_objects() {
    for input in [
        matrix(0, 3),
        matrix(2, 2),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    ] {
        let result = constructor(
            &request(vec![input]),
            &super::super::DECOMPOSITION_CATALOG_ENTRY,
        );
        assert!(is_object(&result.outputs[0]));
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    }
}
#[test]
fn options_and_invalid_inputs_are_checked() {
    let mut req = request(vec![matrix(2, 3), ValueFact::scalar(ValueKindFact::String)]);
    req.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("chol".into()),
    ]);
    assert!(
        !constructor(&req, &super::super::DECOMPOSITION_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    for kind in ["qr", "COD"] {
        req.literals.literal_args[1] = LiteralValue::String(kind.into());
        for storage in [StorageFact::Dense, StorageFact::Sparse] {
            req.arguments[0].storage = storage;
            req.arguments[0].shape = ShapeFact::from(vec![Some(2), Some(3), Some(1)]);
            assert!(
                constructor(&req, &super::super::DECOMPOSITION_CATALOG_ENTRY)
                    .diagnostics
                    .is_empty()
            );
        }
    }
    req.arguments[0].shape = ShapeFact::from(vec![Some(2), None]);
    req.literals.literal_args[1] = LiteralValue::String("chol".into());
    assert!(
        constructor(&req, &super::super::DECOMPOSITION_CATALOG_ENTRY)
            .diagnostics
            .is_empty(),
        "an unknown column count cannot prove a nonsquare matrix"
    );
    let text = ValueFact::scalar(ValueKindFact::String);
    let mut dynamic_triangle = request(vec![matrix(2, 2), text.clone(), text.clone()]);
    dynamic_triangle.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("chol".into()),
        LiteralValue::Unknown,
    ]);
    assert!(constructor(
        &dynamic_triangle,
        &super::super::DECOMPOSITION_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    let mut tolerance = request(vec![matrix(2, 3), text.clone(), text, matrix(1, 1)]);
    tolerance.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("COD".into()),
        LiteralValue::String("RankTolerance".into()),
        LiteralValue::Real {
            text: "-1".into(),
            class: NumericClass::Double,
        },
    ]);
    assert!(
        constructor(&tolerance, &super::super::DECOMPOSITION_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-DECOMPOSITION-TOLERANCE")
    );
    tolerance.literals.literal_args[3] = LiteralValue::Unknown;
    tolerance.arguments[3] = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    assert!(
        constructor(&tolerance, &super::super::DECOMPOSITION_CATALOG_ENTRY)
            .diagnostics
            .is_empty(),
        "complex storage alone cannot prove a nonzero imaginary value"
    );
    assert!(
        !constructor(&request(vec![]), &super::super::DECOMPOSITION_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    assert!(!constructor(
        &request(vec![ValueFact::scalar(ValueKindFact::String)]),
        &super::super::DECOMPOSITION_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
}
#[test]
fn right_division_does_not_confuse_scaling_with_solving() {
    let entry = &super::super::DECOMPOSITION_MRDIVIDE_CATALOG_ENTRY;
    let scaled = right_divide(&request(vec![object(), matrix(1, 1)]), entry);
    assert!(is_object(&scaled.outputs[0]));
    let solved = right_divide(&request(vec![matrix(1, 2), object()]), entry);
    assert_eq!(solved.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(solved.outputs[0].residency, ResidencyFact::Unknown);
}
#[test]
fn output_selection_retains_validation() {
    let mut req = request(vec![]);
    req.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(
        !constructor(&req, &super::super::DECOMPOSITION_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    req.arguments.push(matrix(1, 1));
    req.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(
        !constructor(&req, &super::super::DECOMPOSITION_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
