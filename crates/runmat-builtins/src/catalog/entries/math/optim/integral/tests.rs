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

#[test]
fn finite_scalar_bounds_prove_host_double_result() {
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
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
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
    assert_eq!(
        improper.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let invalid = inference::infer(&call(f64::NAN), &INTEGRAL_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTEGRAL-BOUND"));
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn invalid_complex_bound_and_incomplete_option_pairs_do_not_prove_success() {
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
    assert_eq!(
        valid.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
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
    assert_eq!(
        valid.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
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
    let mut vector = real.clone();
    vector.shape = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(1), DimensionFact::Known(2)],
    };
    for output in [
        ValueFact::scalar(ValueKindFact::String),
        scalar(NumericClass::Double, NumericDomain::Complex),
        vector,
    ] {
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
        LiteralValue::String("RelTol".into()),
        LiteralValue::Number(1.0e-6),
    ]);
    let overridden = inference::infer(&call, &INTEGRAL_CATALOG_ENTRY);
    assert!(overridden.diagnostics.is_empty());
    assert_eq!(
        overridden.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}
