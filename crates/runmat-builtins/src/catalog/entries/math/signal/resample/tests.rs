use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

#[test]
fn known_invalid_dimension_kinds_reject_even_with_numeric_literal() {
    let rate = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut resident = rate.clone();
    resident.residency = ResidencyFact::Device { provider: None };
    let invalid = [
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::Logical),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
        resident,
    ];
    for dimension in invalid {
        let request = CallRequest {
            arguments: vec![
                numeric(NumericClass::Double, NumericDomain::Real, &[2, 4]),
                rate.clone(),
                rate.clone(),
                ValueFact::scalar(ValueKindFact::String),
                dimension,
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
                LiteralValue::Number(1.0),
                LiteralValue::String("Dimension".into()),
                LiteralValue::Number(2.0),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let inferred = inference::infer(&request, &RESAMPLE_CATALOG_ENTRY);
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-RESAMPLE-DIMENSION"));
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    }
}

#[test]
fn literal_rates_and_dimension_preserve_single_complex_class() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![
            numeric(NumericClass::Single, NumericDomain::Complex, &[2, 3]),
            scalar.clone(),
            scalar.clone(),
            ValueFact::scalar(ValueKindFact::String),
            scalar,
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::Number(1.0),
            LiteralValue::String("Dimension".into()),
            LiteralValue::Number(2.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let result = inference::infer(&request, &RESAMPLE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric(),
        Some(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex
        })
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(6)])
    );
    assert_eq!(
        result.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn integer_signal_is_rejected_statically() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![
            numeric(NumericClass::Int32, NumericDomain::Real, &[1, 3]),
            scalar.clone(),
            scalar,
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    assert!(!inference::infer(&request, &RESAMPLE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn scalar_and_logical_input_shapes_follow_selected_dimension() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::Logical),
            scalar.clone(),
            scalar.clone(),
            ValueFact::scalar(ValueKindFact::String),
            scalar,
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(3.0),
            LiteralValue::Number(1.0),
            LiteralValue::String("Dimension".into()),
            LiteralValue::Number(2.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &RESAMPLE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
}

#[test]
fn dynamic_dimension_and_default_uncertain_axis_do_not_select_a_false_extent() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut request = CallRequest {
        arguments: vec![
            numeric(NumericClass::Double, NumericDomain::Real, &[2, 4]),
            scalar.clone(),
            scalar.clone(),
            ValueFact::scalar(ValueKindFact::String),
            scalar,
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::Number(1.0),
            LiteralValue::String("Dimension".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&request, &RESAMPLE_CATALOG_ENTRY);
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );

    request.arguments.truncate(3);
    request.arguments[0].shape = ShapeFact::from(vec![None, Some(4), Some(7)]);
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Number(2.0),
        LiteralValue::Number(1.0),
    ]);
    let inferred = inference::infer(&request, &RESAMPLE_CATALOG_ENTRY);
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![None, None, Some(7)])
    );
}

#[test]
fn invalid_dimension_literal_has_diagnostic_and_no_success_shape() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![
            numeric(NumericClass::Double, NumericDomain::Real, &[2, 4]),
            scalar.clone(),
            scalar.clone(),
            ValueFact::scalar(ValueKindFact::String),
            scalar,
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::Number(1.0),
            LiteralValue::String("Dimension".into()),
            LiteralValue::Number(0.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&request, &RESAMPLE_CATALOG_ENTRY);
    assert!(!inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn typed_wide_rates_and_dimension_keep_exact_platform_range() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let integer = |value: u64| LiteralValue::Integer {
        text: value.to_string(),
        class: NumericClass::UInt64,
    };
    let wide = 9_007_199_254_740_993_u64;
    let mut request = CallRequest {
        arguments: vec![
            numeric(NumericClass::Double, NumericDomain::Real, &[1, 2]),
            scalar.clone(),
            scalar.clone(),
            ValueFact::scalar(ValueKindFact::String),
            scalar,
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            integer(wide),
            integer(1),
            LiteralValue::String("Dimension".into()),
            integer(2),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&request, &RESAMPLE_CATALOG_ENTRY);
    if usize::BITS == 64 {
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![
                Some(1),
                Some(usize::try_from(18_014_398_509_481_986_u64).unwrap()),
            ])
        );

        request.literals.literal_args[2] = integer(wide - 1);
        let ratio = inference::infer(&request, &RESAMPLE_CATALOG_ENTRY);
        assert!(ratio.diagnostics.is_empty(), "{:?}", ratio.diagnostics);
        assert_eq!(
            ratio.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(3)])
        );
    } else {
        assert!(inferred
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-RESAMPLE-RATE"));
    }

    request.literals.literal_args[1] = integer(1);
    request.literals.literal_args[2] = integer(1);
    request.literals.literal_args[4] = integer(u64::MAX);
    let inferred = inference::infer(&request, &RESAMPLE_CATALOG_ENTRY);
    if usize::BITS == 64 {
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
    } else {
        assert!(inferred
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-RESAMPLE-DIMENSION"));
    }
}
