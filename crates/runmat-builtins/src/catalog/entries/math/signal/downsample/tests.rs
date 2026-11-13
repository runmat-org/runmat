use super::*;
use runmat_types::*;

#[test]
fn symbolic_leading_extent_does_not_prove_the_operating_axis() {
    let mut input = numeric(NumericClass::Single, NumericDomain::Real, &[1, 4, 7]);
    input.shape = ShapeFact::Shaped {
        dims: vec![
            DimensionFact::Symbolic(DimensionSymbol("sample_count".into())),
            DimensionFact::Known(4),
            DimensionFact::Known(7),
        ],
    };
    for (entry, denominator) in [
        (&DOWNSAMPLE_CATALOG_ENTRY, None),
        (&super::super::upsample::UPSAMPLE_CATALOG_ENTRY, None),
        (&super::super::resample::RESAMPLE_CATALOG_ENTRY, Some(1.0)),
    ] {
        let rate = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        let mut arguments = vec![input.clone(), rate.clone()];
        let mut literals = vec![LiteralValue::Unknown, LiteralValue::Number(2.0)];
        if let Some(denominator) = denominator {
            arguments.push(rate);
            literals.push(LiteralValue::Number(denominator));
        }
        let result = crate::catalog::inference::infer_catalog_call(
            entry,
            &CallRequest {
                arguments,
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![None, None, Some(7)])
        );
    }
}

fn numeric(class: NumericClass, domain: NumericDomain, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

#[test]
fn down_and_up_descriptors_retain_shared_error_identity_and_signature_fields() {
    let down = &DOWNSAMPLE_DESCRIPTOR;
    let up = &super::super::upsample::UPSAMPLE_DESCRIPTOR;
    assert_eq!(down.errors.len(), 7);
    assert_eq!(
        down.errors
            .iter()
            .map(|error| error.code)
            .collect::<Vec<_>>(),
        up.errors.iter().map(|error| error.code).collect::<Vec<_>>()
    );
    for descriptor in [down, up] {
        assert_eq!(descriptor.signatures.len(), 2);
        assert_eq!(
            descriptor.signatures[0]
                .inputs
                .iter()
                .map(|input| input.name)
                .collect::<Vec<_>>(),
            vec!["X", "N"]
        );
        assert_eq!(
            descriptor.signatures[1]
                .inputs
                .iter()
                .map(|input| input.name)
                .collect::<Vec<_>>(),
            vec!["X", "N", "PHASE"]
        );
        assert_eq!(descriptor.signatures[0].outputs[0].name, "Y");
    }
    assert_eq!(
        SAMPLE_ERROR_INVALID_FACTOR.code,
        super::super::upsample::UPSAMPLE_ERROR_INVALID_FACTOR.code
    );
}

#[test]
fn complex_integer_samples_are_rejected_for_both_rate_operators() {
    let input = numeric(NumericClass::UInt64, NumericDomain::Complex, &[1, 2]);
    let factor = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![input, factor],
        literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    for entry in [
        &DOWNSAMPLE_CATALOG_ENTRY,
        &super::super::upsample::UPSAMPLE_CATALOG_ENTRY,
    ] {
        let inferred = crate::catalog::inference::infer_catalog_call(entry, &request);
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.code == "RM-CATALOG-SAMPLE-RATE-COMPLEX-INTEGER" }));
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    }
}

fn request(shape: &[usize], literals: Vec<LiteralValue>) -> CallRequest {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    CallRequest {
        arguments: vec![
            numeric(NumericClass::UInt64, NumericDomain::Real, shape),
            scalar.clone(),
            scalar,
        ],
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn literal_factor_and_phase_preserve_wide_integer_class_and_shape() {
    let result = inference::infer(
        &request(
            &[5, 1],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
                LiteralValue::Number(1.0),
            ],
        ),
        &DOWNSAMPLE_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}

#[test]
fn invalid_phase_does_not_publish_success_facts() {
    let result = inference::infer(
        &request(
            &[5, 1],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
                LiteralValue::Number(2.0),
            ],
        ),
        &DOWNSAMPLE_CATALOG_ENTRY,
    );
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn scalar_upsample_expands_canonical_first_dimension() {
    let mut request = request(
        &[1, 1],
        vec![LiteralValue::Unknown, LiteralValue::Number(3.0)],
    );
    request.arguments[0].shape = ShapeFact::Scalar;
    request.arguments.pop();
    let inferred = crate::catalog::inference::infer_catalog_call(
        &super::super::upsample::UPSAMPLE_CATALOG_ENTRY,
        &request,
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
}

#[test]
fn singleton_single_remains_dense_storage_fact() {
    let mut request = request(
        &[1, 1],
        vec![LiteralValue::Unknown, LiteralValue::Number(1.0)],
    );
    request.arguments[0] = numeric(NumericClass::Single, NumericDomain::Real, &[1, 1]);
    request.arguments.pop();
    let inferred = crate::catalog::inference::infer_catalog_call(
        &super::super::upsample::UPSAMPLE_CATALOG_ENTRY,
        &request,
    );
    assert_eq!(inferred.outputs[0].storage, StorageFact::Dense);
}

#[test]
fn unknown_phase_and_possible_leading_singleton_do_not_invent_extent() {
    let mut unknown_phase = request(
        &[5, 1],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::Unknown,
        ],
    );
    let result = inference::infer(&unknown_phase, &DOWNSAMPLE_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );

    unknown_phase.arguments[0].shape = ShapeFact::from(vec![None, Some(4), Some(7)]);
    unknown_phase.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Number(2.0),
        LiteralValue::Number(0.0),
    ]);
    let result = inference::infer(&unknown_phase, &DOWNSAMPLE_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, None, Some(7)])
    );
}

#[test]
fn typed_wide_structural_controls_are_not_rounded_to_binary64() {
    let wide = 9_007_199_254_740_993_u64;
    let integer = |value: u64| LiteralValue::Integer {
        text: value.to_string(),
        class: NumericClass::UInt64,
    };
    let mut call = request(
        &[2, 1],
        vec![LiteralValue::Unknown, integer(wide), integer(wide - 1)],
    );
    let down = inference::infer(&call, &DOWNSAMPLE_CATALOG_ENTRY);
    if usize::BITS == 64 {
        assert!(down.diagnostics.is_empty(), "{:?}", down.diagnostics);
        assert_eq!(
            down.outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(1)])
        );
    } else {
        assert!(down
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-SAMPLE-RATE-CONTROL"));
    }

    call.arguments.pop();
    call.arguments[0].shape = ShapeFact::from(vec![Some(2), Some(1)]);
    let up = crate::catalog::inference::infer_catalog_call(
        &super::super::upsample::UPSAMPLE_CATALOG_ENTRY,
        &call,
    );
    if usize::BITS == 64 {
        assert!(up.diagnostics.is_empty(), "{:?}", up.diagnostics);
        assert_eq!(
            up.outputs[0].shape,
            ShapeFact::from(vec![
                Some(usize::try_from(18_014_398_509_481_986_u64).unwrap()),
                Some(1),
            ])
        );
    } else {
        assert!(up
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-SAMPLE-RATE-CONTROL"));
    }
}
