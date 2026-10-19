use super::*;
use runmat_types::{DimensionFact, OutputSelection, RequestedOutputCount};

#[test]
fn floating_selector_uses_shared_numeric_dimension_admission() {
    let mut dimensions = vec![1.0 + f64::EPSILON];
    if usize::BITS < 64 {
        dimensions.push(usize::MAX as f64);
    }
    for value in dimensions {
        let request = options(vec![
            LiteralValue::Unknown,
            LiteralValue::Empty,
            LiteralValue::Number(value),
        ]);
        let mut diagnostics = Vec::new();
        let (selector, _, _) = parse_options(
            &request,
            MomentStatisticKind::MeanAbsoluteDeviation,
            &mut diagnostics,
        );
        assert!(diagnostics.is_empty());
        let expected = runmat_types::decode_shape_dimension(value).unwrap();
        assert!(
            matches!(selector, Selector::Dimensions(ref dimensions) if dimensions == &[expected])
        );
    }
}

#[test]
fn scalar_flag_tolerance_matches_runtime_admission() {
    let request = options(vec![LiteralValue::Unknown, LiteralValue::Number(1e-13)]);
    let mut diagnostics = Vec::new();
    let (selector, flag, _) = parse_options(
        &request,
        MomentStatisticKind::MeanAbsoluteDeviation,
        &mut diagnostics,
    );
    assert!(diagnostics.is_empty());
    assert_eq!(flag, 0);
    assert!(matches!(selector, Selector::Default));
}

fn options(literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); literals.len()],
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn canonical_flag_defaults_and_static_option_errors() {
    for (kind, expected) in [
        (MomentStatisticKind::MeanAbsoluteDeviation, 0),
        (MomentStatisticKind::Skewness, 1),
        (MomentStatisticKind::Kurtosis, 1),
    ] {
        let request = options(vec![LiteralValue::Unknown]);
        let mut diagnostics = Vec::new();
        let (_, flag, missing) = parse_options(&request, kind, &mut diagnostics);
        assert_eq!(flag, expected);
        assert_eq!(missing, MissingMode::Omit);
        assert!(diagnostics.is_empty());

        let request = options(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]);
        let mut diagnostics = Vec::new();
        parse_options(&request, kind, &mut diagnostics);
        assert!(!diagnostics.is_empty(), "{kind:?} accepted invalid flag");
    }
}

#[test]
fn ambiguity_and_large_rank_remain_symbolic() {
    let huge = ShapeFact::Ranked { rank: usize::MAX };
    assert_eq!(reduced_shape(&huge, &Selector::Default), huge);
    assert_eq!(
        reduced_shape(&huge, &Selector::Dimensions(vec![usize::MAX])),
        huge
    );
    assert_eq!(reduced_shape(&huge, &Selector::All), ShapeFact::Scalar);

    let request = options(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Number(2.0),
    ]);
    let mut diagnostics = Vec::new();
    let (selector, _, _) = parse_options(&request, MomentStatisticKind::Skewness, &mut diagnostics);
    assert!(matches!(selector, Selector::Dynamic));
    assert_eq!(reduced_shape(&huge, &selector), ShapeFact::Unknown);
}

#[test]
fn first_scalar_array_is_a_flag_not_a_dimension() {
    let request = options(vec![
        LiteralValue::Unknown,
        LiteralValue::Vector(vec![LiteralValue::Number(1.0)]),
    ]);
    let mut diagnostics = Vec::new();
    let (selector, flag, _) =
        parse_options(&request, MomentStatisticKind::Skewness, &mut diagnostics);
    assert_eq!(flag, 1);
    assert!(matches!(selector, Selector::Default));
    assert!(diagnostics.is_empty());
}

#[test]
fn later_singleton_axes_do_not_consume_following_options() {
    for dimension in [
        LiteralValue::Vector(vec![LiteralValue::Number(2.0)]),
        LiteralValue::Matrix(vec![vec![LiteralValue::Number(2.0)]]),
    ] {
        let request = options(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(1.0),
            dimension.clone(),
            LiteralValue::String("all".into()),
            LiteralValue::String("includenan".into()),
            LiteralValue::String("omitmissing".into()),
        ]);
        let mut diagnostics = Vec::new();
        let (selector, flag, missing) =
            parse_options(&request, MomentStatisticKind::Kurtosis, &mut diagnostics);
        assert!(diagnostics.is_empty());
        assert!(matches!(selector, Selector::All));
        assert_eq!(flag, 1);
        assert_eq!(missing, MissingMode::Omit);

        let request = options(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(1.0),
            dimension,
        ]);
        let (selector, _, _) =
            parse_options(&request, MomentStatisticKind::Kurtosis, &mut diagnostics);
        assert!(matches!(selector, Selector::Dimensions(ref dimensions) if dimensions == &[2]));
    }
}

#[test]
fn first_array_flag_and_logical_dimension_keep_distinct_roles() {
    let request = options(vec![
        LiteralValue::Unknown,
        LiteralValue::Vector(vec![LiteralValue::Number(2.0)]),
    ]);
    let mut diagnostics = Vec::new();
    let (selector, _, _) = parse_options(&request, MomentStatisticKind::Kurtosis, &mut diagnostics);
    assert!(!diagnostics.is_empty());
    assert!(matches!(selector, Selector::Dynamic));

    let request = options(vec![
        LiteralValue::Unknown,
        LiteralValue::Vector(vec![LiteralValue::Bool(true)]),
    ]);
    let mut diagnostics = Vec::new();
    let (selector, _, _) = parse_options(&request, MomentStatisticKind::Kurtosis, &mut diagnostics);
    assert!(diagnostics.is_empty());
    assert!(matches!(selector, Selector::Dimensions(ref dimensions) if dimensions == &[1]));
}

#[test]
fn concrete_reduction_preserves_unselected_nd_extents() {
    let input = ShapeFact::Shaped {
        dims: vec![
            DimensionFact::Known(2),
            DimensionFact::Known(3),
            DimensionFact::Known(4),
        ],
    };
    assert_eq!(
        reduced_shape(&input, &Selector::Dimensions(vec![2, usize::MAX])),
        ShapeFact::Shaped {
            dims: vec![
                DimensionFact::Known(2),
                DimensionFact::Known(1),
                DimensionFact::Known(4),
            ],
        }
    );
}

#[test]
fn single_element_single_remains_a_typed_tensor() {
    let request = CallRequest {
        arguments: vec![ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(3), Some(1)]),
            StorageFact::Dense,
        )],
        literals: LiteralContext::new(vec![LiteralValue::Unknown]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = infer(
        &request,
        &super::super::MAD_CATALOG_ENTRY,
        MomentStatisticKind::MeanAbsoluteDeviation,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
}
