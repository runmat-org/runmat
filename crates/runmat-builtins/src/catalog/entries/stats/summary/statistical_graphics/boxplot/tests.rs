use super::*;
use runmat_types::{
    CallRequest, CellFact, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn real(shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

fn call(
    args: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: usize,
) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            arguments: args,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &BOXPLOT_CATALOG_ENTRY,
    )
}

#[test]
fn canonical_entry_preserves_descriptor_extensions_and_integer_forms() {
    assert_eq!(BOXPLOT_CATALOG_ENTRY.extensions, &BOXPLOT_EXTENSIONS);
    assert_eq!(
        BOXPLOT_CATALOG_ENTRY
            .integer_capabilities
            .iter()
            .map(|capability| capability.form)
            .collect::<Vec<_>>(),
        BOXPLOT_INTEGER_CAPABILITIES
            .iter()
            .map(|capability| capability.form)
            .collect::<Vec<_>>()
    );
    assert_eq!(BOXPLOT_CATALOG_ENTRY.identity.name, "boxplot");
}

#[test]
fn known_valid_matrix_proves_row_vector_double_handle_class() {
    let result = call(vec![real(vec![Some(4), Some(2)])], vec![], 1);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
}

#[test]
fn ungrouped_vector_and_matrix_admit_one_factor_controls() {
    let name = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    for shape in [vec![Some(2), Some(1)], vec![Some(2), Some(2)]] {
        let result = call(
            vec![
                real(shape),
                name.clone(),
                real(vec![Some(1), Some(1)]),
                name.clone(),
                real(vec![Some(1), Some(1)]),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("FactorSeparator".into()),
                LiteralValue::Number(1.0),
                LiteralValue::String("FactorGap".into()),
                LiteralValue::Number(10.0),
            ],
            1,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Double
        );
    }
}

#[test]
fn invalid_data_and_group_lengths_do_not_prove_handles() {
    let invalid_rank = call(vec![real(vec![Some(2), Some(2), Some(2)])], vec![], 1);
    assert!(!invalid_rank.diagnostics.is_empty());
    assert!(matches!(
        invalid_rank.outputs[0].kind,
        ValueKindFact::Unknown
    ));

    let bad_group = call(
        vec![real(vec![Some(4), Some(2)]), real(vec![Some(3), Some(1)])],
        vec![],
        1,
    );
    assert!(!bad_group.diagnostics.is_empty());
    assert!(matches!(bad_group.outputs[0].kind, ValueKindFact::Unknown));

    let row_group = call(
        vec![real(vec![Some(4), Some(2)]), real(vec![Some(4), Some(1)])],
        vec![],
        1,
    );
    assert!(row_group.diagnostics.is_empty());
}

#[test]
fn typed_option_name_and_invalid_host_value_are_diagnosed() {
    let text = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let result = call(
        vec![
            real(vec![Some(4), Some(1)]),
            text,
            real(vec![Some(1), Some(1)]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Whisker".into()),
            LiteralValue::Number(-1.0),
        ],
        1,
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn scalar_text_options_reject_known_cell_or_multielement_text_facts() {
    let name = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(
                runmat_types::DynamicReason::RuntimeValue,
            )),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    for (option, value) in [
        ("BoxStyle", cell.clone()),
        (
            "MedianStyle",
            ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(vec![Some(1), Some(2)]),
                StorageFact::Dense,
            ),
        ),
        (
            "FullFactors",
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(2), Some(3)]),
                StorageFact::Dense,
            ),
        ),
    ] {
        let result = call(
            vec![real(vec![Some(4), Some(1)]), name.clone(), value],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String(option.into()),
                LiteralValue::Unknown,
            ],
            1,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "OPTION"));
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    }

    let factor_direction = call(
        vec![real(vec![Some(4), Some(1)]), name, cell],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("FactorDirection".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(factor_direction.diagnostics.is_empty());
    assert_eq!(
        factor_direction.outputs[0].numeric().unwrap().class,
        NumericClass::Double,
        "FactorDirection has a distinct text-vector grammar"
    );
}

#[test]
fn factor_and_visual_numeric_controls_do_not_prove_known_invalid_calls() {
    let name = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let group = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    for (option, value, literal) in [
        (
            "FactorSeparator",
            real(vec![Some(1), Some(1)]),
            LiteralValue::Number(0.0),
        ),
        (
            "FactorSeparator",
            real(vec![Some(1), Some(1)]),
            LiteralValue::Number(2.0),
        ),
        (
            "FactorGap",
            real(vec![Some(1), Some(1)]),
            LiteralValue::Number(-1.0),
        ),
        (
            "FactorGap",
            real(vec![Some(1), Some(2)]),
            LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(2.0)]),
        ),
        (
            "Positions",
            real(vec![Some(1), Some(2)]),
            LiteralValue::Vector(vec![
                LiteralValue::Number(1.0),
                LiteralValue::Number(f64::NAN),
            ]),
        ),
        (
            "Widths",
            real(vec![Some(1), Some(1)]),
            LiteralValue::Number(0.0),
        ),
        (
            "DataLim",
            real(vec![Some(1), Some(3)]),
            LiteralValue::Unknown,
        ),
        (
            "Labels",
            ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::Scalar,
                StorageFact::Scalar,
            ),
            LiteralValue::Bool(true),
        ),
    ] {
        let result = call(
            vec![
                real(vec![Some(2), Some(1)]),
                group.clone(),
                name.clone(),
                value,
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String(option.into()),
                literal,
            ],
            1,
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "OPTION"),
            "{option}: {:#?}",
            result.diagnostics
        );
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    }

    let valid = call(
        vec![
            real(vec![Some(2), Some(1)]),
            group,
            name,
            real(vec![Some(1), Some(1)]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("FactorSeparator".into()),
            LiteralValue::Number(1.0),
        ],
        1,
    );
    assert!(valid.diagnostics.is_empty(), "{:#?}", valid.diagnostics);
}

#[test]
fn unrepresentable_plot_size_is_not_proven_as_an_admitted_option() {
    let text = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let result = call(
        vec![
            real(vec![Some(4), Some(1)]),
            text,
            real(vec![Some(1), Some(1)]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("OutlierSize".into()),
            LiteralValue::Number(f64::MAX),
        ],
        1,
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn unknown_values_and_padded_requested_outputs_remain_conservative() {
    let result = call(
        vec![ValueFact::unknown(
            runmat_types::DynamicReason::RuntimeValue,
        )],
        vec![],
        2,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert!(matches!(result.outputs[1].kind, ValueKindFact::Unknown));
}

#[test]
fn known_text_group_vectors_are_not_misdiagnosed_as_unpaired_options() {
    for (kind, shape) in [
        (ValueKindFact::String, vec![Some(4), Some(1)]),
        (ValueKindFact::Character, vec![Some(4), Some(3)]),
    ] {
        let group = ValueFact::proven(kind, ShapeFact::from(shape), StorageFact::Dense);
        let result = call(
            vec![real(vec![Some(4), Some(1)]), group],
            vec![LiteralValue::Unknown, LiteralValue::Unknown],
            1,
        );
        assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Double
        );
    }
}

#[test]
fn known_character_row_count_mismatch_is_rejected() {
    let group = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(3), Some(2)]),
        StorageFact::Dense,
    );
    let result = call(
        vec![real(vec![Some(4), Some(1)]), group],
        vec![LiteralValue::Unknown; 2],
        1,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "GROUP"));
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn complex_visual_control_is_not_proven_valid() {
    let option_name = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let result = call(
        vec![real(vec![Some(4), Some(1)]), option_name, complex],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Whisker".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "OPTION"));
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn unknown_scalar_text_option_name_does_not_become_a_group() {
    let name = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let result = call(
        vec![
            real(vec![Some(4), Some(1)]),
            name,
            real(vec![Some(1), Some(1)]),
        ],
        vec![LiteralValue::Unknown; 3],
        1,
    );
    assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
}
