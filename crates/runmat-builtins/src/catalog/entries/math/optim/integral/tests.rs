use super::*;
use runmat_types::*;

fn scalar(class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact { class, domain }))
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn callable_with_output(output: ValueFact) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        parameters_complete: true,
        outputs: vec![output],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }))
}

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
fn vectorized_numeric_unknown_geometry_is_not_a_false_error() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    for shape in [
        ShapeFact::Unknown,
        ShapeFact::Ranked { rank: 2 },
        ShapeFact::from(vec![None, None]),
    ] {
        let output = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        );
        let inferred = inference::infer(
            &request(
                vec![callable_with_output(output), real.clone(), real.clone()],
                vec![],
                1,
            ),
            &INTEGRAL_CATALOG_ENTRY,
        );
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
    }
    let wrong_row = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), None]),
        StorageFact::Dense,
    );
    let inferred = inference::infer(
        &request(
            vec![callable_with_output(wrong_row), real.clone(), real],
            vec![],
            1,
        ),
        &INTEGRAL_CATALOG_ENTRY,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTEGRAL-CALLBACK-RESULT"));
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

#[test]
fn interval_count_is_a_positive_structural_control_not_an_evaluation_alias() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let name = ValueFact::scalar(ValueKindFact::String);
    let call = |count: i64| {
        request(
            vec![
                ValueFact::unknown(DynamicReason::RuntimeValue),
                real.clone(),
                real.clone(),
                name.clone(),
                real.clone(),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("MaxIntervalCount".into()),
                LiteralValue::Number(count as f64),
            ],
            1,
        )
    };
    let invalid = inference::infer(&call(0), &INTEGRAL_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTEGRAL-OPTION-VALUE"));
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));
    let valid = inference::infer(&call(1), &INTEGRAL_CATALOG_ENTRY);
    assert!(valid.diagnostics.is_empty());
    assert!(matches!(valid.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn callback_capability_is_transitive() {
    let mut call = request(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            scalar(NumericClass::Double, NumericDomain::Real),
            scalar(NumericClass::Double, NumericDomain::Real),
        ],
        vec![],
        1,
    );
    call.arguments[0] = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet(std::collections::BTreeSet::from([
            CapabilityRequirement::Filesystem,
        ])),
        parameters: vec![ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(1), DimensionFact::Unknown],
            },
            StorageFact::Dense,
        )],
        parameters_complete: true,
        outputs: vec![ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(1), DimensionFact::Unknown],
            },
            StorageFact::Dense,
        )],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    let inferred = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert!(inferred
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn known_invalid_callback_result_never_proves_a_scalar_integral() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    for output in [ValueFact::scalar(ValueKindFact::String)] {
        let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
            identity: None,
            capabilities: CapabilitySet::default(),
            parameters: vec![real.clone()],
            parameters_complete: true,
            outputs: vec![output],
            outputs_complete: true,
            variadic_inputs: false,
            variadic_outputs: false,
            captures: vec![],
            captures_complete: true,
        }));
        let call = request(vec![callback, real.clone(), real.clone()], vec![], 1);
        let result = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTEGRAL-CALLBACK-RESULT"));
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    }
}

#[test]
fn known_complex_scalar_callback_infers_complex_double() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![real.clone()],
        parameters_complete: true,
        outputs: vec![scalar(NumericClass::Single, NumericDomain::Complex)],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    let call = request(
        vec![
            callback,
            real.clone(),
            real,
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("Vectorized".into()),
            LiteralValue::Bool(false),
        ],
        1,
    );
    let result = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().domain,
        NumericDomain::Complex
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn callback_shape_facts_follow_typed_vectorized_selector() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![real.clone()],
        parameters_complete: true,
        outputs: vec![real.clone()],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    let default_call = request(
        vec![callback.clone(), real.clone(), real.clone()],
        vec![],
        1,
    );
    let default_result = inference::infer(&default_call, &INTEGRAL_CATALOG_ENTRY);
    assert!(!default_result.diagnostics.is_empty());
    assert!(matches!(
        default_result.outputs[0].kind,
        ValueKindFact::Unknown
    ));

    let scalar_call = request(
        vec![
            callback,
            real.clone(),
            real,
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("Vectorized".into()),
            LiteralValue::Bool(false),
        ],
        1,
    );
    let scalar_result = inference::infer(&scalar_call, &INTEGRAL_CATALOG_ENTRY);
    assert!(
        scalar_result.diagnostics.is_empty(),
        "{:?}",
        scalar_result.diagnostics
    );
    assert_eq!(
        scalar_result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn array_valued_callback_proves_its_fixed_output_shape() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let array = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(3)],
        },
        StorageFact::Dense,
    );
    let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![real.clone()],
        parameters_complete: true,
        outputs: vec![array],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    let call = request(
        vec![
            callback,
            real.clone(),
            real,
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("ArrayValued".into()),
            LiteralValue::Bool(true),
        ],
        1,
    );
    let result = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(3)]
        }
    );
}

#[test]
fn single_complex_array_callback_proves_double_complex_singleton_nd_result() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(1), Some(1), Some(1)]),
        StorageFact::Dense,
    );
    let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![real.clone()],
        parameters_complete: true,
        outputs: vec![output.clone()],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    let call = request(
        vec![
            callback,
            real.clone(),
            real,
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("ArrayValued".into()),
            LiteralValue::Bool(true),
        ],
        1,
    );
    let result = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().domain,
        NumericDomain::Complex
    );
    assert_eq!(result.outputs[0].shape, output.shape);
}

#[test]
fn array_valued_singleton_nd_shape_is_not_scalarized() {
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let singleton_nd = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(1), Some(1)]),
        StorageFact::Dense,
    );
    let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![real.clone()],
        parameters_complete: true,
        outputs: vec![singleton_nd.clone()],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    let call = request(
        vec![
            callback,
            real.clone(),
            real,
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("ArrayValued".into()),
            LiteralValue::Bool(true),
        ],
        1,
    );
    let result = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, singleton_nd.shape);
}

#[test]
fn option_identity_is_typed_at_the_public_boundary() {
    for (spelling, identity) in [
        ("AbsTol", IntegralOptionName::AbsTol),
        ("reltol", IntegralOptionName::RelTol),
        ("AbsoluteTolerance", IntegralOptionName::AbsoluteTolerance),
        ("relativetolerance", IntegralOptionName::RelativeTolerance),
        ("MaxFunEvals", IntegralOptionName::MaxFunEvals),
        ("MaxIntervalCount", IntegralOptionName::MaxIntervalCount),
        ("ArrayValued", IntegralOptionName::ArrayValued),
        ("Vectorized", IntegralOptionName::Vectorized),
        ("Waypoints", IntegralOptionName::Waypoints),
    ] {
        assert_eq!(IntegralOptionName::parse(spelling), Some(identity));
    }
    assert_eq!(IntegralOptionName::parse("unknown"), None);
}

#[test]
fn known_invalid_controls_do_not_prove_success_and_last_override_is_respected() {
    let callback = ValueFact::unknown(DynamicReason::RuntimeValue);
    let real = scalar(NumericClass::Double, NumericDomain::Real);
    let text = ValueFact::scalar(ValueKindFact::String);
    for (name, value, fact) in [
        ("MaxFunEvals", LiteralValue::Number(4.0), real.clone()),
        ("ArrayValued", LiteralValue::Number(2.0), real.clone()),
    ] {
        let call = request(
            vec![
                callback.clone(),
                real.clone(),
                real.clone(),
                text.clone(),
                fact,
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String(name.into()),
                value,
            ],
            1,
        );
        let inferred = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTEGRAL-OPTION-VALUE"));
        assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
    }

    let mut call = request(
        vec![
            callback,
            real.clone(),
            real.clone(),
            text.clone(),
            real.clone(),
            text.clone(),
            real.clone(),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("AbsTol".into()),
            LiteralValue::Number(0.0),
            LiteralValue::String("RelTol".into()),
            LiteralValue::Number(0.0),
        ],
        1,
    );
    let invalid = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTEGRAL-TOLERANCES"));
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));

    call.arguments.extend([text.clone(), real]);
    call.literals.literal_args.extend([
        LiteralValue::String("RelativeTolerance".into()),
        LiteralValue::Number(1.0e-6),
    ]);
    let overridden = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(overridden.diagnostics.is_empty());
    assert!(matches!(overridden.outputs[0].kind, ValueKindFact::Unknown));
}
