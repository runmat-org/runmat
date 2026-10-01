use super::*;
use runmat_types::*;

fn infer(
    entry: &BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> CallInference {
    crate::infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

fn numeric(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn sequences(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(numeric(ShapeFact::Unknown)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape,
        StorageFact::Dense,
    )
}

#[test]
fn combvec_uses_sum_rows_product_columns_and_double_host_output() {
    let result = infer(
        &COMBVEC_CATALOG_ENTRY,
        vec![
            numeric(ShapeFact::from(vec![Some(2), Some(2)])),
            numeric(ShapeFact::from(vec![Some(1), Some(3)])),
        ],
        vec![],
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(6)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn combvec_rejects_known_invalid_without_guessing_unknown_shapes() {
    assert!(!infer(&COMBVEC_CATALOG_ENTRY, vec![], vec![])
        .diagnostics
        .is_empty());
    assert!(!infer(
        &COMBVEC_CATALOG_ENTRY,
        vec![numeric(ShapeFact::from(vec![Some(1), Some(0)]))],
        vec![]
    )
    .diagnostics
    .is_empty());
    assert!(!infer(
        &COMBVEC_CATALOG_ENTRY,
        vec![numeric(ShapeFact::Ranked { rank: 3 })],
        vec![]
    )
    .diagnostics
    .is_empty());
    let unknown = infer(
        &COMBVEC_CATALOG_ENTRY,
        vec![numeric(ShapeFact::Unknown)],
        vec![],
    );
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(unknown.outputs[0].shape, ShapeFact::from(vec![None, None]));
}

#[test]
fn combvec_rank_one_input_is_one_row_with_known_column_count() {
    let result = infer(
        &COMBVEC_CATALOG_ENTRY,
        vec![numeric(ShapeFact::from(vec![Some(4)]))],
        vec![],
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
    let unknown_length = infer(
        &COMBVEC_CATALOG_ENTRY,
        vec![numeric(ShapeFact::Ranked { rank: 1 })],
        vec![],
    );
    assert_eq!(
        unknown_length.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
}

#[test]
fn padsequences_default_double_or_explicit_cell_output() {
    let cell = sequences(ShapeFact::from(vec![Some(1), Some(2)]));
    let dim = numeric(ShapeFact::Scalar);
    let default = infer(
        &PADSEQUENCES_CATALOG_ENTRY,
        vec![cell.clone(), dim.clone()],
        vec![],
    );
    assert!(default.diagnostics.is_empty());
    assert_eq!(
        default.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let false_output = infer(
        &PADSEQUENCES_CATALOG_ENTRY,
        vec![
            cell.clone(),
            dim.clone(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::String("UnIfOrMoUtPuT".into()),
            LiteralValue::Bool(false),
        ],
    );
    assert!(false_output.diagnostics.is_empty());
    assert!(matches!(
        false_output.outputs[0].kind,
        ValueKindFact::Cell(_)
    ));
    assert_eq!(false_output.outputs[0].shape, cell.shape);
    let dynamic = infer(
        &PADSEQUENCES_CATALOG_ENTRY,
        vec![
            cell,
            dim,
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::String("UniformOutput".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(matches!(dynamic.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn padsequences_infers_preserved_integer_class_and_second_mask() {
    let integer = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let cells = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(integer),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let result = crate::infer_catalog_call(
        &PADSEQUENCES_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![cells, numeric(ShapeFact::Scalar)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 2);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert!(matches!(result.outputs[1].kind, ValueKindFact::Logical));
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(result.outputs[1].residency, ResidencyFact::Host);
}

#[test]
fn padsequences_diagnoses_known_invalid_cells_dimensions_and_options() {
    let dim = numeric(ShapeFact::Scalar);
    let bad_cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::scalar(ValueKindFact::String)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    assert!(!infer(
        &PADSEQUENCES_CATALOG_ENTRY,
        vec![bad_cell, dim.clone()],
        vec![]
    )
    .diagnostics
    .is_empty());
    assert!(!infer(
        &PADSEQUENCES_CATALOG_ENTRY,
        vec![sequences(ShapeFact::Scalar), dim.clone()],
        vec![LiteralValue::Unknown, LiteralValue::Number(0.0)]
    )
    .diagnostics
    .is_empty());
    assert!(!infer(
        &PADSEQUENCES_CATALOG_ENTRY,
        vec![
            sequences(ShapeFact::Scalar),
            dim,
            ValueFact::scalar(ValueKindFact::String),
            numeric(ShapeFact::Scalar)
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::String("bogus".into()),
            LiteralValue::Number(1.0)
        ]
    )
    .diagnostics
    .is_empty());
}

#[test]
fn padsequences_structured_contract_retains_no_option_and_mask_forms() {
    let labels = PADSEQUENCES_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        [
            "Y = padsequences(sequences, paddingDim)",
            "Y = padsequences(sequences, paddingDim, Name, Value)",
            "[Y, mask] = padsequences(sequences, paddingDim)",
            "[Y, mask] = padsequences(sequences, paddingDim, Name, Value)",
        ]
    );
}

#[test]
fn padsequences_shared_option_tokens_preserve_alias_and_value_grammar() {
    assert_eq!(
        PadOptionKey::parse("DiReCtIoN"),
        Some(PadOptionKey::DirectionAlias)
    );
    assert_eq!(
        PadOptionKey::parse("PaddingDirection"),
        Some(PadOptionKey::PaddingDirection)
    );
    assert_eq!(PadDirection::parse("BoTh"), Some(PadDirection::Both));
    assert_eq!(
        PadLengthKeyword::parse("shortest"),
        Some(PadLengthKeyword::Shortest)
    );
    assert_eq!(
        PadBooleanKeyword::parse("off").map(PadBooleanKeyword::value),
        Some(false)
    );
    assert_eq!(PadOptionKey::parse("unknown"), None);
    assert_eq!(PadDirection::parse("symmetric"), None);
}

#[test]
fn padsequences_known_option_values_and_integer_controls_are_sound() {
    let cell = sequences(ShapeFact::from(vec![Some(1), Some(2)]));
    let dimension = numeric(ShapeFact::Scalar);
    let option_name = ValueFact::scalar(ValueKindFact::String);
    let option_value = ValueFact::scalar(ValueKindFact::String);
    let call = |names: Vec<LiteralValue>, args: Vec<ValueFact>| {
        infer(&PADSEQUENCES_CATALOG_ENTRY, args, names)
    };
    let valid = call(
        vec![
            LiteralValue::Unknown,
            LiteralValue::Integer {
                text: "2".into(),
                class: NumericClass::UInt64,
            },
            LiteralValue::String("Direction".into()),
            LiteralValue::String("left".into()),
            LiteralValue::String("UniformOutput".into()),
            LiteralValue::String("off".into()),
        ],
        vec![
            cell.clone(),
            dimension.clone(),
            option_name.clone(),
            option_value.clone(),
            option_name.clone(),
            option_value.clone(),
        ],
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    assert!(matches!(valid.outputs[0].kind, ValueKindFact::Cell(_)));

    let overwritten = call(
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::String("UniformOutput".into()),
            LiteralValue::String("invalid".into()),
            LiteralValue::String("uniformoutput".into()),
            LiteralValue::String("off".into()),
        ],
        vec![
            cell.clone(),
            dimension.clone(),
            option_name.clone(),
            option_value.clone(),
            option_name.clone(),
            option_value.clone(),
        ],
    );
    assert!(
        overwritten.diagnostics.is_empty(),
        "{:?}",
        overwritten.diagnostics
    );
    assert!(matches!(
        overwritten.outputs[0].kind,
        ValueKindFact::Cell(_)
    ));

    let dynamic_override = call(
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::String("UniformOutput".into()),
            LiteralValue::String("invalid".into()),
            LiteralValue::Unknown,
            LiteralValue::String("off".into()),
        ],
        vec![
            cell.clone(),
            dimension.clone(),
            option_name.clone(),
            option_value.clone(),
            option_name.clone(),
            option_value.clone(),
        ],
    );
    assert!(dynamic_override.diagnostics.is_empty());
    assert!(matches!(
        dynamic_override.outputs[0].kind,
        ValueKindFact::Unknown
    ));

    let nontext_name = call(
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::Number(1.0),
            LiteralValue::String("off".into()),
        ],
        vec![
            cell.clone(),
            dimension.clone(),
            numeric(ShapeFact::Scalar),
            option_value.clone(),
        ],
    );
    assert!(nontext_name
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DL-PADSEQUENCES-OPTION"));

    for invalid in [
        LiteralValue::Integer {
            text: "-1".into(),
            class: NumericClass::Int64,
        },
        LiteralValue::Integer {
            text: "0".into(),
            class: NumericClass::UInt64,
        },
    ] {
        let result = call(
            vec![LiteralValue::Unknown, invalid],
            vec![cell.clone(), dimension.clone()],
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|item| item.code == "RM-DL-PADSEQUENCES-DIMENSION"));
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    }
    let bad_direction = call(
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::String("Direction".into()),
            LiteralValue::String("symmetric".into()),
        ],
        vec![
            cell.clone(),
            dimension.clone(),
            option_name.clone(),
            option_value.clone(),
        ],
    );
    assert!(bad_direction
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DL-PADSEQUENCES-OPTION"));
    assert!(matches!(
        bad_direction.outputs[0].kind,
        ValueKindFact::Unknown
    ));
    let conflicting_aliases = call(
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::String("Direction".into()),
            LiteralValue::String("left".into()),
            LiteralValue::String("PaddingDirection".into()),
            LiteralValue::String("right".into()),
        ],
        vec![
            cell,
            dimension,
            option_name.clone(),
            option_value.clone(),
            option_name,
            option_value,
        ],
    );
    assert!(conflicting_aliases
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DL-PADSEQUENCES-OPTION"));
    assert!(matches!(
        conflicting_aliases.outputs[0].kind,
        ValueKindFact::Unknown
    ));
}
