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
fn implicit_and_explicit_queries_keep_shape_and_double_class() {
    let z = numeric(ShapeFact::from(vec![Some(2), Some(2)]));
    let xq = numeric(ShapeFact::Scalar);
    let yq = numeric(ShapeFact::from(vec![Some(2), Some(3)]));
    let implicit = inference::infer(
        &request(vec![z.clone(), xq.clone(), yq.clone()], vec![]),
        &INTERP2_CATALOG_ENTRY,
    );
    assert_eq!(implicit.outputs[0].shape, yq.shape);
    assert_eq!(
        implicit.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let explicit = inference::infer(
        &request(vec![xq.clone(), xq.clone(), z, xq, yq.clone()], vec![]),
        &INTERP2_CATALOG_ENTRY,
    );
    assert_eq!(explicit.outputs[0].shape, yq.shape);
}

#[test]
fn literal_option_disambiguates_five_argument_implicit_form() {
    let z = numeric(ShapeFact::from(vec![Some(2), Some(2)]));
    let query = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    let option = ValueFact::unknown(DynamicReason::RuntimeValue);
    let inferred = inference::infer(
        &request(
            vec![
                z.clone(),
                query.clone(),
                query.clone(),
                option.clone(),
                option.clone(),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("nearest".into()),
            ],
        ),
        &INTERP2_CATALOG_ENTRY,
    );
    assert_eq!(inferred.outputs[0].shape, query.shape);
    let unknown = inference::infer(
        &request(
            vec![z, query.clone(), query, option.clone(), option],
            vec![],
        ),
        &INTERP2_CATALOG_ENTRY,
    );
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn mismatched_queries_do_not_receive_false_shape_fact() {
    let z = numeric(ShapeFact::from(vec![Some(2), Some(2)]));
    let xq = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    let yq = numeric(ShapeFact::from(vec![Some(1), Some(4)]));
    let inferred = inference::infer(&request(vec![z, xq, yq], vec![]), &INTERP2_CATALOG_ENTRY);
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn literal_spline_retains_single_complex_sample_fact_without_claiming_dynamic_method() {
    let samples = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(4), Some(4)]),
        StorageFact::Dense,
    );
    let query = numeric(ShapeFact::Scalar);
    let method = ValueFact::unknown(DynamicReason::RuntimeValue);
    let literal = inference::infer(
        &request(
            vec![
                samples.clone(),
                query.clone(),
                query.clone(),
                method.clone(),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("spline".into()),
            ],
        ),
        &INTERP2_CATALOG_ENTRY,
    );
    assert_eq!(
        literal.outputs[0].numeric(),
        Some(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex
        })
    );
    let dynamic = inference::infer(
        &request(vec![samples, query.clone(), query, method], vec![]),
        &INTERP2_CATALOG_ENTRY,
    );
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Unknown);
    assert!(
        dynamic.diagnostics.is_empty(),
        "a dynamic method may select spline"
    );
}

#[test]
fn later_method_option_controls_spline_class_inference() {
    let samples = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(4), Some(4)]),
        StorageFact::Dense,
    );
    let query = numeric(ShapeFact::Scalar);
    let option = ValueFact::unknown(DynamicReason::RuntimeValue);
    let arguments = vec![samples, query.clone(), query, option.clone(), option];
    let literals = |last| {
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("spline".into()),
            last,
        ]
    };
    let linear = inference::infer(
        &request(
            arguments.clone(),
            literals(LiteralValue::String("linear".into())),
        ),
        &INTERP2_CATALOG_ENTRY,
    );
    assert_eq!(
        linear.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let extrap = inference::infer(
        &request(
            arguments.clone(),
            literals(LiteralValue::String("extrap".into())),
        ),
        &INTERP2_CATALOG_ENTRY,
    );
    assert_eq!(
        extrap.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    let unknown = inference::infer(
        &request(arguments, literals(LiteralValue::Unknown)),
        &INTERP2_CATALOG_ENTRY,
    );
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);
}
