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
