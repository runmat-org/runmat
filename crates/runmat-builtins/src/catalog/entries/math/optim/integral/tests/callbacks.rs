use super::super::*;
use super::support::*;
use runmat_types::*;

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
    let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![real.clone()],
        parameters_complete: true,
        outputs: vec![ValueFact::scalar(ValueKindFact::String)],
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
