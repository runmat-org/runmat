use super::*;

fn infer_options(shape: ShapeFact, options: Vec<LiteralValue>) -> runmat_types::CallInference {
    let mut literals = vec![LiteralValue::Unknown];
    literals.extend(options);
    infer_logical(
        &request(shape, literals),
        &crate::ALL_CATALOG_ENTRY,
        LogicalReductionKind::All,
    )
}

#[test]
fn unknown_selector_before_or_after_nan_flag_does_not_invent_default_shape() {
    for options in [
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("omitnan".into()),
        ],
        vec![
            LiteralValue::String("includenan".into()),
            LiteralValue::Unknown,
        ],
    ] {
        let result = infer_options(ShapeFact::from(vec![Some(2), Some(3), Some(4)]), options);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
        assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
    }
}

#[test]
fn known_selector_is_not_overwritten_by_unknown_or_empty_controls() {
    for options in [
        vec![LiteralValue::Unknown, LiteralValue::Number(2.0)],
        vec![LiteralValue::Number(2.0), LiteralValue::Unknown],
        vec![
            LiteralValue::Empty,
            LiteralValue::Number(2.0),
            LiteralValue::Vector(vec![]),
        ],
    ] {
        let result = infer_options(ShapeFact::from(vec![Some(2), Some(3), Some(4)]), options);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(1), Some(4)])
        );
        assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    }
}

#[test]
fn repeated_known_selectors_diagnose_in_both_orders_even_without_input_shape() {
    let all = || LiteralValue::String("all".into());
    for options in [
        vec![LiteralValue::Number(1.0), LiteralValue::Number(2.0)],
        vec![LiteralValue::Number(1.0), all()],
        vec![all(), LiteralValue::Number(1.0)],
        vec![all(), all()],
        vec![LiteralValue::Vector(vec![LiteralValue::Unknown]), all()],
    ] {
        let result = infer_options(ShapeFact::Unknown, options);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGICAL-REDUCTION-ARGUMENTS"));
    }
}

#[test]
fn all_proves_scalar_without_input_extents_and_with_unknown_extra_control() {
    let result = infer_options(
        ShapeFact::Unknown,
        vec![LiteralValue::Unknown, LiteralValue::String("all".into())],
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].storage, StorageFact::Scalar);
}

#[test]
fn variadic_unknown_controls_and_missing_literals_do_not_invent_arity_or_extents() {
    let mut call = request(
        ShapeFact::from(vec![Some(2), Some(3)]),
        vec![LiteralValue::Unknown; 5],
    );
    call.literals.literal_args.truncate(1);
    let result = infer_logical(&call, &crate::ANY_CATALOG_ENTRY, LogicalReductionKind::Any);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Ranked { rank: 2 });
    assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
}

#[test]
fn unknown_default_axis_and_partial_dimension_vector_remain_unknown() {
    let result = infer_options(ShapeFact::from(vec![None, Some(3)]), vec![]);
    assert_eq!(result.outputs[0].shape, ShapeFact::Ranked { rank: 2 });
    let result = infer_options(
        ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
        vec![LiteralValue::Vector(vec![
            LiteralValue::Number(1.0),
            LiteralValue::Unknown,
        ])],
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn repeated_flags_and_invalid_known_dimensions_have_option_diagnostics() {
    for options in [
        vec![
            LiteralValue::String("omitnan".into()),
            LiteralValue::String("includenan".into()),
        ],
        vec![LiteralValue::Number(0.0)],
        vec![LiteralValue::Number(-1.0)],
        vec![LiteralValue::String("invalid".into())],
    ] {
        let result = infer_options(ShapeFact::Unknown, options);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGICAL-REDUCTION-ARGUMENTS"));
    }
}

#[test]
fn single_dimension_literals_use_represented_precision() {
    let single = |text: &str| LiteralValue::Real {
        text: text.into(),
        class: NumericClass::Single,
    };
    let input = ShapeFact::from(vec![Some(2), Some(3)]);
    for (literal, expected) in [
        (
            single("1.00000001"),
            ShapeFact::from(vec![Some(1), Some(3)]),
        ),
        (single("16777217"), input.clone()),
    ] {
        let result = infer_options(input.clone(), vec![literal]);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, expected);
        assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    }
    for literal in [
        single("1.00000011920928955078125"),
        LiteralValue::Real {
            text: "1.00000001".into(),
            class: NumericClass::Double,
        },
    ] {
        let result = infer_options(input.clone(), vec![literal]);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGICAL-REDUCTION-ARGUMENTS"));
    }
}
