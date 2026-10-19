use super::*;
use runmat_types::*;

fn numeric(dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &KSTEST_CATALOG_ENTRY,
    )
}

#[test]
fn admitted_vector_and_named_cdf_have_scalar_host_output_facts() {
    let result = infer(
        vec![
            numeric(&[5, 1]),
            ValueFact::scalar(ValueKindFact::Character),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("CDF".into()),
            LiteralValue::String("normal".into()),
        ],
        4,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    for output in &result.outputs[1..] {
        assert_eq!(output.numeric().unwrap().class, NumericClass::Double);
    }
    assert!(result.outputs.iter().all(|output| {
        output.shape.is_proven_equivalent(&ShapeFact::Scalar)
            && output.residency == ResidencyFact::Host
    }));
}

#[test]
fn known_invalid_shapes_and_controls_remain_dynamic() {
    let matrix = infer(vec![numeric(&[2, 2])], vec![LiteralValue::Unknown], 1);
    assert!(!matrix.diagnostics.is_empty());
    assert_eq!(matrix.outputs[0].kind, ValueKindFact::Unknown);
    let empty = infer(vec![numeric(&[0, 1])], vec![LiteralValue::Unknown], 1);
    assert!(!empty.diagnostics.is_empty());

    for (name, value) in [
        ("Alpha", LiteralValue::Number(1.0)),
        ("Tail", LiteralValue::String("sideways".into())),
        ("CDF", LiteralValue::String("unregistered".into())),
        ("Unknown", LiteralValue::Number(0.5)),
    ] {
        let result = infer(
            vec![
                numeric(&[5, 1]),
                ValueFact::scalar(ValueKindFact::Character),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Character(name.into()),
                value,
            ],
            1,
        );
        assert!(!result.diagnostics.is_empty(), "{name}");
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown, "{name}");
    }

    let complex_cdf = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(3), Some(2)]),
        StorageFact::Dense,
    );
    let result = infer(
        vec![
            numeric(&[5, 1]),
            ValueFact::scalar(ValueKindFact::Character),
            complex_cdf,
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("CDF".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn option_and_tail_aliases_decode_once_into_canonical_identities() {
    assert_eq!(
        KstestOptionName::from_public("cDf"),
        Some(KstestOptionName::Cdf)
    );
    assert_eq!(KstestTail::from_public("both"), Some(KstestTail::Unequal));
    assert_eq!(KstestTail::from_public("right"), Some(KstestTail::Larger));
    assert_eq!(
        KstestNamedCdf::from_public("gaussian"),
        Some(KstestNamedCdf::StandardNormal)
    );
    assert_eq!(KstestNamedCdf::from_public("arbitrary"), None);
}
