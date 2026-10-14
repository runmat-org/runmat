use super::*;
use runmat_types::*;

fn scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn infer(
    order: LiteralValue,
    cutoff_shape: ShapeFact,
    option: Option<&str>,
    outputs: usize,
) -> CallInference {
    let mut arguments = vec![
        scalar(),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            cutoff_shape,
            StorageFact::Dense,
        ),
    ];
    let mut literals = vec![order, LiteralValue::Unknown];
    if let Some(option) = option {
        arguments.push(ValueFact::scalar(ValueKindFact::String));
        literals.push(LiteralValue::String(option.into()));
    }
    crate::infer_catalog_call(
        &BUTTER_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn requested_representation_changes_output_geometry() {
    let cutoff = ShapeFact::Scalar;
    let coefficients = infer(LiteralValue::Number(2.0), cutoff.clone(), None, 2);
    assert_eq!(coefficients.outputs.len(), 2);
    assert_eq!(
        coefficients.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    let roots = infer(LiteralValue::Number(2.0), cutoff.clone(), None, 3);
    assert_eq!(
        roots.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    assert_eq!(
        roots.outputs[1].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    let state = infer(LiteralValue::Number(2.0), cutoff, None, 4);
    assert_eq!(
        state.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert_eq!(state.outputs[3].storage, StorageFact::Scalar);
}

#[test]
fn bandpass_doubles_order_and_analog_lowpass_has_no_zeros() {
    let band = infer(
        LiteralValue::Number(3.0),
        ShapeFact::from(vec![Some(1), Some(2)]),
        Some("bandpass"),
        2,
    );
    assert_eq!(
        band.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(7)])
    );
    let analog = infer(LiteralValue::Number(2.0), ShapeFact::Scalar, Some("s"), 3);
    assert_eq!(
        analog.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(1)])
    );
}

#[test]
fn proven_bad_order_and_output_count_are_diagnosed() {
    let bad_order = infer(LiteralValue::Number(0.0), ShapeFact::Scalar, None, 2);
    assert!(bad_order
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-BUTTER-ORDER"));
    let bad_outputs = infer(LiteralValue::Number(2.0), ShapeFact::Scalar, None, 5);
    assert!(bad_outputs
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-BUTTER-OUTPUTS"));
}

#[test]
fn character_options_unknown_kind_and_literal_cutoff_orientation_match_runtime() {
    for (text, cutoff_shape, width) in [
        ("high", ShapeFact::Scalar, 4),
        ("bandpass", ShapeFact::from(vec![Some(1), Some(2)]), 7),
        ("s", ShapeFact::Scalar, 4),
    ] {
        let option = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(text.len())]),
            StorageFact::Dense,
        );
        let result = crate::infer_catalog_call(
            &BUTTER_CATALOG_ENTRY,
            &CallRequest {
                arguments: vec![
                    scalar(),
                    ValueFact::proven(scalar().kind, cutoff_shape, StorageFact::Dense),
                    option,
                ],
                literals: LiteralContext::new(vec![
                    LiteralValue::Number(3.0),
                    LiteralValue::Unknown,
                    LiteralValue::Character(text.into()),
                ]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            },
        );
        assert!(
            result.diagnostics.is_empty(),
            "{text}: {:?}",
            result.diagnostics
        );
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(width)])
        );
    }

    for cutoff in [
        LiteralValue::Matrix(vec![vec![
            LiteralValue::Number(0.2),
            LiteralValue::Number(0.4),
        ]]),
        LiteralValue::Matrix(vec![
            vec![LiteralValue::Number(0.2)],
            vec![LiteralValue::Number(0.4)],
        ]),
    ] {
        let result = crate::infer_catalog_call(
            &BUTTER_CATALOG_ENTRY,
            &CallRequest {
                arguments: vec![
                    scalar(),
                    ValueFact::proven(
                        scalar().kind,
                        ShapeFact::from(vec![Some(1), Some(2)]),
                        StorageFact::Dense,
                    ),
                ],
                literals: LiteralContext::new(vec![LiteralValue::Number(2.0), cutoff]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            },
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(5)])
        );
    }

    let dynamic = crate::infer_catalog_call(
        &BUTTER_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                scalar(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Number(3.0),
                LiteralValue::Unknown,
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    assert!(dynamic.diagnostics.is_empty());
    assert_eq!(
        dynamic.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(
        dynamic.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn ranked_cutoff_fact_does_not_allocate_rank_sized_dimensions() {
    let result = infer(
        LiteralValue::Number(3.0),
        ShapeFact::Ranked { rank: usize::MAX },
        None,
        2,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
}
