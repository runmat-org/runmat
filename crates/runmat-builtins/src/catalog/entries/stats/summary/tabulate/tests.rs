use super::*;
use runmat_types::{
    CallRequest, CellFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn infer(input: ValueFact) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &TABULATE_CATALOG_ENTRY,
    )
}

#[test]
fn numeric_class_and_output_columns_follow_runtime_without_guessing_levels() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(4)]),
        StorageFact::Dense,
    );
    let result = infer(input);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
    assert_eq!(
        result.outputs[0].residency,
        runmat_types::ResidencyFact::Host
    );
}

#[test]
fn wide_integer_and_unknown_contents_do_not_claim_numeric_label_conversion() {
    let integer = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer(integer);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
    assert_eq!(
        infer(ValueFact::unknown(DynamicReason::RuntimeValue)).outputs[0].kind,
        ValueKindFact::Unknown
    );
}

#[test]
fn invalid_input_does_not_prove_a_table() {
    let result = infer(ValueFact::scalar(ValueKindFact::Symbolic));
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn character_row_admission_tracks_runtime_shape() {
    let row = infer(ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(4)]),
        StorageFact::Dense,
    ));
    assert!(row.diagnostics.is_empty());
    assert!(matches!(row.outputs[0].kind, ValueKindFact::Cell(_)));

    let multirow = infer(ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(4)]),
        StorageFact::Dense,
    ));
    assert!(!multirow.diagnostics.is_empty());
    assert_eq!(multirow.outputs[0].kind, ValueKindFact::Unknown);

    let unknown_rows = infer(ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![None, Some(4)]),
        StorageFact::Dense,
    ));
    assert!(unknown_rows.diagnostics.is_empty());
    assert_eq!(unknown_rows.outputs[0].kind, ValueKindFact::Unknown);
}

fn cell_with(element: ValueFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element.clone()),
            elements: vec![element],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Unknown,
    )
}

#[test]
fn complete_cell_contents_are_proven_only_when_label_forms_are_certain() {
    let text = infer(cell_with(ValueFact::scalar(ValueKindFact::String)));
    assert!(text.diagnostics.is_empty());
    assert!(matches!(text.outputs[0].kind, ValueKindFact::Cell(_)));

    let nested = infer(cell_with(ValueFact::scalar(ValueKindFact::Cell(
        CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: true,
        },
    ))));
    assert!(!nested.diagnostics.is_empty());
    assert_eq!(nested.outputs[0].kind, ValueKindFact::Unknown);

    let non_scalar_text = infer(cell_with(ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    )));
    assert!(!non_scalar_text.diagnostics.is_empty());
    assert_eq!(non_scalar_text.outputs[0].kind, ValueKindFact::Unknown);

    let numeric = infer(cell_with(ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))));
    assert!(numeric.diagnostics.is_empty());
    assert_eq!(numeric.outputs[0].kind, ValueKindFact::Unknown);

    let dynamic = infer(ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Unknown,
        StorageFact::Unknown,
    ));
    assert!(dynamic.diagnostics.is_empty());
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Unknown);
}
