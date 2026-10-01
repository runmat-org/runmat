use super::*;
use runmat_types::*;

fn observations(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &TSNE_CATALOG_ENTRY,
    )
}

#[test]
fn output_dimension_and_class_are_known_but_nan_filtered_rows_are_not() {
    let result = infer(
        vec![observations(8, 4), text(), scalar()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("NumDimensions".into()),
            LiteralValue::Integer {
                text: "3".into(),
                class: NumericClass::Int32,
            },
        ],
        2,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(result.outputs[1].shape, ShapeFact::Scalar);
    assert_eq!(
        result.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
    let one = infer(vec![observations(1, 4)], vec![], 1);
    assert_eq!(
        one.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
}

#[test]
fn unknown_and_invalid_options_do_not_invent_shapes() {
    let dynamic_name = infer(
        vec![observations(8, 4), text(), scalar()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(3.0),
        ],
        1,
    );
    assert_eq!(
        dynamic_name.outputs[0].shape,
        ShapeFact::from(vec![None, None])
    );
    let nontext_name = infer(
        vec![observations(8, 4), scalar(), scalar()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(7.0),
            LiteralValue::Number(3.0),
        ],
        1,
    );
    assert_eq!(
        nontext_name.outputs[0].shape,
        ShapeFact::from(vec![None, None])
    );
    assert!(!nontext_name.diagnostics.is_empty());
    let dynamic = infer(
        vec![observations(8, 4), text(), scalar()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("NumDimensions".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::from(vec![None, None]));
    for invalid in [
        LiteralValue::Number(0.0),
        LiteralValue::Number(2.5),
        LiteralValue::Number(129.0),
    ] {
        let result = infer(
            vec![observations(8, 4), text(), scalar()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("NumDimensions".into()),
                invalid,
            ],
            1,
        );
        assert!(!result.diagnostics.is_empty());
    }
    let invalid_metric = infer(
        vec![observations(8, 4), text(), text()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Distance".into()),
            LiteralValue::String("bogus".into()),
        ],
        1,
    );
    assert!(!invalid_metric.diagnostics.is_empty());
    assert_eq!(
        TsneDistance::from_keyword("fasteuclidean"),
        Some(TsneDistance::Euclidean)
    );
}

#[test]
fn current_extra_output_padding_is_reflected_in_facts() {
    let result = infer(vec![observations(8, 4)], vec![], 3);
    assert_eq!(result.outputs.len(), 3);
    assert_eq!(
        result.outputs[2].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[2].shape, ShapeFact::Scalar);
}
