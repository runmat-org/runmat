use super::*;
use runmat_types::*;

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

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn host_vector_and_array_forms_keep_correct_query_geometry() {
    let y = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    let xq = numeric(ShapeFact::from(vec![Some(2), Some(2)]));
    let implicit = inference::infer(
        &request(vec![y, xq.clone()], vec![]),
        &INTERP1_CATALOG_ENTRY,
    );
    assert_eq!(implicit.outputs[0].shape, xq.shape);
    assert_eq!(
        implicit.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );

    let x = numeric(ShapeFact::from(vec![Some(3), Some(1)]));
    let y = numeric(ShapeFact::from(vec![Some(3), Some(2), Some(4)]));
    let explicit = inference::infer(&request(vec![x, y, xq], vec![]), &INTERP1_CATALOG_ENTRY);
    assert_eq!(
        explicit.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2), Some(2), Some(4)])
    );
}

#[test]
fn static_method_disambiguates_but_unknown_third_argument_does_not() {
    let y = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    let query = numeric(ShapeFact::from(vec![Some(1), Some(2)]));
    let option = ValueFact::unknown(DynamicReason::RuntimeValue);
    let known_method = inference::infer(
        &request(
            vec![y.clone(), query.clone(), option.clone()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("spline".into()),
            ],
        ),
        &INTERP1_CATALOG_ENTRY,
    );
    assert_eq!(known_method.outputs[0].shape, query.shape);
    let dynamic = inference::infer(
        &request(vec![y, query, option], vec![]),
        &INTERP1_CATALOG_ENTRY,
    );
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn resident_samples_do_not_claim_host_class_or_residency() {
    let mut y = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    y.residency = ResidencyFact::Device { provider: None };
    let query = numeric(ShapeFact::from(vec![Some(1), Some(2)]));
    let inferred = inference::infer(
        &request(vec![y, query.clone()], vec![]),
        &INTERP1_CATALOG_ENTRY,
    );
    assert_eq!(inferred.outputs[0].shape, query.shape);
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn missing_and_complex_inputs_are_not_silently_proven() {
    let missing = inference::infer(&request(vec![], vec![]), &INTERP1_CATALOG_ENTRY);
    assert!(missing
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTERP1-ARITY"));
    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let result = inference::infer(
        &request(vec![complex, numeric(ShapeFact::Scalar)], vec![]),
        &INTERP1_CATALOG_ENTRY,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTERP1-NUMERIC"));
}
