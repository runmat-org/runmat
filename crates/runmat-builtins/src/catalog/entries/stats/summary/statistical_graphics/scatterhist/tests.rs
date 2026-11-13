use super::*;
use runmat_types::{
    CallRequest, CellFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn real(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

#[test]
fn known_group_cardinality_uses_observation_count_not_character_elements() {
    let name = ValueFact::scalar(ValueKindFact::String);
    let literal = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Group".into()),
        LiteralValue::Unknown,
    ];
    let cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    for group in [
        real(2, 1),
        ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        ),
        cell,
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        ),
    ] {
        let result = infer(
            vec![real(4, 1), real(4, 1), name.clone(), group],
            literal.clone(),
            1,
        );
        assert!(!result.diagnostics.is_empty());
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    }
    let matching_char_rows = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(4), Some(3)]),
        StorageFact::Dense,
    );
    let accepted = infer(
        vec![real(4, 1), real(4, 1), name, matching_char_rows],
        literal,
        1,
    );
    assert!(accepted.diagnostics.is_empty());
}

#[test]
fn scalar_text_option_values_match_runtime_geometry() {
    let name = ValueFact::scalar(ValueKindFact::String);
    let single = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    let multiple = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    for option in [
        "Kernel",
        "Legend",
        "PlotGroup",
        "Location",
        "Direction",
        "Style",
    ] {
        let literals = vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String(option.into()),
            LiteralValue::Unknown,
        ];
        let accepted = infer(
            vec![real(3, 1), real(3, 1), name.clone(), single.clone()],
            literals.clone(),
            1,
        );
        assert!(accepted.diagnostics.is_empty(), "{option}");
        let rejected = infer(
            vec![real(3, 1), real(3, 1), name.clone(), multiple.clone()],
            literals,
            1,
        );
        assert!(!rejected.diagnostics.is_empty(), "{option}");
        assert!(matches!(rejected.outputs[0].kind, ValueKindFact::Unknown));
    }
}

#[test]
fn multirow_character_option_name_does_not_prove_handles() {
    let name = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let result = infer(
        vec![real(2, 1), real(2, 1), name, real(1, 1)],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("NBins".into()),
            LiteralValue::Number(2.0),
        ],
        1,
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

fn infer(
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: usize,
) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            literals: LiteralContext::new(literals),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &SCATTERHIST_CATALOG_ENTRY,
    )
}

#[test]
fn valid_vectors_prove_three_double_axes_handles_and_ui_effect() {
    let result = infer(vec![real(4, 1), real(1, 4)], vec![], 1);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert!(result
        .effects
        .0
        .contains(&runmat_types::EffectKind::UserInterface));
}

#[test]
fn unknown_observation_class_does_not_prove_chart_handles() {
    let unknown = ValueFact::proven(
        ValueKindFact::Unknown,
        ShapeFact::from(vec![Some(4), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer(vec![unknown, real(4, 1)], vec![], 1);
    assert!(result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn bad_matrix_empty_and_mismatched_vectors_do_not_prove_handles() {
    for arguments in [
        vec![real(2, 2), real(4, 1)],
        vec![real(0, 1), real(0, 1)],
        vec![real(3, 1), real(4, 1)],
    ] {
        let result = infer(arguments, vec![], 1);
        assert!(!result.diagnostics.is_empty());
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    }
}

#[test]
fn typed_option_identity_rejects_bad_known_counts_and_unknown_names() {
    let option = ValueFact::scalar(ValueKindFact::String);
    let bad = infer(
        vec![real(3, 1), real(3, 1), option.clone(), real(1, 1)],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("NBins".into()),
            LiteralValue::Number(1.0),
        ],
        1,
    );
    assert!(!bad.diagnostics.is_empty());
    let unknown = infer(
        vec![real(3, 1), real(3, 1), option, real(1, 1)],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("DefinitelyUnknown".into()),
            LiteralValue::Number(2.0),
        ],
        1,
    );
    assert!(!unknown.diagnostics.is_empty());
    assert!(matches!(unknown.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn dynamic_option_name_keeps_output_conservative() {
    let result = infer(
        vec![
            real(3, 1),
            real(3, 1),
            ValueFact::scalar(ValueKindFact::String),
            real(1, 1),
        ],
        vec![LiteralValue::Unknown; 4],
        1,
    );
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn character_row_option_name_is_text_not_a_string_array() {
    let name = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(5)]),
        StorageFact::Dense,
    );
    let literals = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("NBins".into()),
        LiteralValue::Number(2.0),
    ];
    let result = infer(
        vec![real(3, 1), real(3, 1), name.clone(), real(1, 1)],
        literals,
        1,
    );
    assert!(result.diagnostics.is_empty());
    let dynamic = infer(
        vec![real(3, 1), real(3, 1), name, real(1, 1)],
        vec![LiteralValue::Unknown; 4],
        1,
    );
    assert!(dynamic.diagnostics.is_empty());
    assert!(matches!(dynamic.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn unknown_option_class_is_not_rejected_by_character_count() {
    let name = ValueFact::proven(
        ValueKindFact::Unknown,
        ShapeFact::from(vec![Some(1), Some(5)]),
        StorageFact::Dense,
    );
    let result = infer(
        vec![real(3, 1), real(3, 1), name, real(1, 1)],
        vec![LiteralValue::Unknown; 4],
        1,
    );
    assert!(result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}
