use super::super::*;
use super::support::*;
use runmat_types::*;

fn waypoint_path_request(
    lower: ValueFact,
    upper: ValueFact,
    options: Vec<(LiteralValue, ValueFact)>,
) -> CallRequest {
    let row = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    let mut arguments = vec![callable_with_output(row), lower, upper];
    let mut literals = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Unknown,
    ];
    for (name, value) in options {
        arguments.push(ValueFact::scalar(ValueKindFact::String));
        arguments.push(value);
        literals.push(name);
        literals.push(LiteralValue::Unknown);
    }
    request(arguments, literals, 1)
}

#[test]
fn duplicate_waypoints_and_dynamic_names_follow_last_value_wins_domain() {
    use inference::PathDomain;
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let complex = scalar(NumericClass::Double, NumericDomain::Complex);
    let waypoint = || LiteralValue::String("Waypoints".into());
    let complex_then_real = waypoint_path_request(
        real.clone(),
        real.clone(),
        vec![(waypoint(), complex.clone()), (waypoint(), real.clone())],
    );
    assert_eq!(inference::path_domain(&complex_then_real), PathDomain::Real);
    let inferred_real = inference::infer(&complex_then_real, &INTEGRAL_CATALOG_ENTRY);
    assert!(
        inferred_real.diagnostics.is_empty(),
        "{:?}",
        inferred_real.diagnostics
    );
    assert_eq!(
        inferred_real.outputs[0].numeric().unwrap().domain,
        NumericDomain::Real
    );
    let real_then_complex = waypoint_path_request(
        real.clone(),
        real.clone(),
        vec![(waypoint(), real.clone()), (waypoint(), complex.clone())],
    );
    assert_eq!(
        inference::path_domain(&real_then_complex),
        PathDomain::Complex
    );
    let inferred_complex = inference::infer(&real_then_complex, &INTEGRAL_CATALOG_ENTRY);
    assert!(
        inferred_complex.diagnostics.is_empty(),
        "{:?}",
        inferred_complex.diagnostics
    );
    assert_eq!(
        inferred_complex.outputs[0].numeric().unwrap().domain,
        NumericDomain::Complex
    );
    let dynamic_after_complex = waypoint_path_request(
        real.clone(),
        real.clone(),
        vec![
            (waypoint(), complex.clone()),
            (LiteralValue::Unknown, real.clone()),
        ],
    );
    assert_eq!(
        inference::path_domain(&dynamic_after_complex),
        PathDomain::Unknown
    );
    let inferred_dynamic = inference::infer(&dynamic_after_complex, &INTEGRAL_CATALOG_ENTRY);
    assert!(
        inferred_dynamic.diagnostics.is_empty(),
        "{:?}",
        inferred_dynamic.diagnostics
    );
    assert!(matches!(
        inferred_dynamic.outputs[0].kind,
        ValueKindFact::Unknown
    ));
    let known_after_dynamic = waypoint_path_request(
        real.clone(),
        real.clone(),
        vec![
            (waypoint(), complex.clone()),
            (LiteralValue::Unknown, real.clone()),
            (waypoint(), real.clone()),
        ],
    );
    assert_eq!(
        inference::path_domain(&known_after_dynamic),
        PathDomain::Real
    );
    let complex_bounds = waypoint_path_request(
        complex.clone(),
        real,
        vec![
            (waypoint(), complex),
            (
                LiteralValue::Unknown,
                scalar(NumericClass::Double, NumericDomain::Real),
            ),
        ],
    );
    assert_eq!(inference::path_domain(&complex_bounds), PathDomain::Complex);
}

#[test]
fn unknown_bounds_and_waypoint_modes_do_not_prove_real_result_domain() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let scalar_mode = [
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::Logical),
    ];
    let scalar_literals = [
        LiteralValue::String("Vectorized".into()),
        LiteralValue::Bool(false),
    ];
    let cases = [
        (vec![unknown.clone(), real.clone()], vec![]),
        (
            vec![
                real.clone(),
                real.clone(),
                ValueFact::scalar(ValueKindFact::String),
                unknown.clone(),
            ],
            vec![
                LiteralValue::String("Waypoints".into()),
                LiteralValue::Unknown,
            ],
        ),
        (
            vec![
                real.clone(),
                real.clone(),
                ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                    std::collections::BTreeMap::new(),
                    false,
                ))),
            ],
            vec![LiteralValue::Unknown],
        ),
    ];
    for (mut arguments, mut rest_literals) in cases {
        let struct_option = rest_literals.len() == 1 && arguments.len() == 3;
        if !struct_option {
            arguments.extend(scalar_mode.clone());
            rest_literals.extend(scalar_literals.clone());
        }
        let mut call_args = vec![callable_with_output(real.clone())];
        call_args.extend(arguments);
        let mut literals = vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
        ];
        literals.extend(rest_literals);
        let inferred = inference::infer(&request(call_args, literals, 1), &INTEGRAL_CATALOG_ENTRY);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
    }
}

#[test]
fn unknown_callback_does_not_prove_real_or_complex_result() {
    let call = request(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            scalar(NumericClass::Double, NumericDomain::Real),
            scalar(NumericClass::Double, NumericDomain::Real),
        ],
        vec![],
        1,
    );
    let inferred = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn infinite_real_bound_is_admitted_but_nan_bound_is_diagnostic() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let call = |bound: f64| {
        request(
            vec![
                ValueFact::unknown(DynamicReason::RuntimeValue),
                real.clone(),
                real.clone(),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Number(0.0),
                LiteralValue::Number(bound),
            ],
            1,
        )
    };
    let improper = inference::infer(&call(f64::INFINITY), &INTEGRAL_CATALOG_ENTRY);
    assert!(improper.diagnostics.is_empty());
    assert!(matches!(improper.outputs[0].kind, ValueKindFact::Unknown));
    let invalid = inference::infer(&call(f64::NAN), &INTEGRAL_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTEGRAL-BOUND"));
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn complex_bound_and_incomplete_option_pairs_do_not_prove_success() {
    let mut call = request(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            scalar(NumericClass::Double, NumericDomain::Complex),
            scalar(NumericClass::Double, NumericDomain::Real),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("AbsTol".into()),
        ],
        1,
    );
    let inferred = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(!inferred.diagnostics.is_empty());
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
    call.arguments[1] = scalar(NumericClass::Double, NumericDomain::Real);
    call.arguments
        .push(scalar(NumericClass::Double, NumericDomain::Real));
    call.literals.literal_args.push(LiteralValue::Unknown);
    let valid = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(valid.diagnostics.is_empty());
    assert!(matches!(valid.outputs[0].kind, ValueKindFact::Unknown));
}
